import time
import wave

import pytest
from fastapi.testclient import TestClient

from qvd.schema import JobRequest
from qvd.server import _auth_matches, create_app
from qvd.store import Store


@pytest.fixture()
def client(tmp_path):
    app = create_app({"data_dir": str(tmp_path), "api_key": "test-secret", "engine": "test"})
    with TestClient(app) as value:
        deadline = time.time() + 3
        while app.state.worker.state == "loading" and time.time() < deadline:
            time.sleep(0.01)
        yield value


def auth():
    return {"Authorization": "Bearer test-secret"}


def wait_job(client, jid):
    deadline = time.time() + 3
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{jid}", headers=auth())
        assert response.status_code == 200
        job = response.json()
        if job["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.01)
    pytest.fail("job did not reach a terminal state")


def test_health_authentication_and_capabilities(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/status").status_code == 401
    status = client.get("/v1/status", headers=auth())
    assert status.status_code == 200
    assert status.json()["service"] == "qwen-voice-design"
    caps = client.get("/v1/capabilities", headers=auth()).json()
    assert caps["languages"][-1] == "Auto"
    assert caps["request_schema"]["additionalProperties"] is False


def test_request_boundaries_and_unknown_fields(client):
    base = {"text": "hello", "instruct": "warm voice"}
    assert client.post("/v1/jobs", json={**base, "max_new_tokens": 127}, headers=auth()).status_code == 422
    assert client.post("/v1/jobs", json={**base, "max_new_tokens": 4097}, headers=auth()).status_code == 422
    assert client.post("/v1/jobs", json={**base, "unexpected": 1}, headers=auth()).status_code == 422
    assert client.post("/v1/jobs", json={"text": " ", "instruct": "voice"}, headers=auth()).status_code == 422
    assert client.post("/v1/jobs", json={"text": "文本", "instruct": "温暖、清晰", "language": "Chinese"}, headers=auth()).status_code == 200


def test_idempotency_and_wav_result(client):
    body = {"text": "same request", "instruct": "calm", "seed": 7}
    first = client.post("/v1/jobs", json=body, headers={**auth(), "Idempotency-Key": "idem-1"})
    second = client.post("/v1/jobs", json=body, headers={**auth(), "Idempotency-Key": "idem-1"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["replayed"] is True
    assert second.json()["job"]["id"] == first.json()["job"]["id"]
    conflict = client.post(
        "/v1/jobs",
        json={**body, "seed": 8},
        headers={**auth(), "Idempotency-Key": "idem-1"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"

    job = wait_job(client, first.json()["job"]["id"])
    assert job["state"] == "succeeded"
    result = job["result"]
    assert result["backend"] == "test"
    assert result["sha256"]
    audio = client.get(result["audio_url"], headers=auth())
    assert audio.status_code == 200
    with wave.open(__import__("io").BytesIO(audio.content), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() > 0


def test_cancel_completion_race_and_recovery(tmp_path):
    store = Store(tmp_path)
    queued, _ = store.submit({"text": "q", "instruct": "i"}, "queued")
    cancelled = store.cancel(queued["id"])
    assert cancelled["state"] == "cancelled"

    running, _ = store.submit({"text": "r", "instruct": "i"}, "running")
    assert store.claim()["id"] == running["id"]
    assert store.cancel(running["id"])["state"] == "running"
    assert store.finish(running["id"], "succeeded", result={"sha256": "x"})["state"] == "cancelled"

    interrupted, _ = store.submit({"text": "x", "instruct": "i"}, "interrupt")
    assert store.claim()["id"] == interrupted["id"]
    Store(tmp_path).recover()
    assert Store(tmp_path).get(interrupted["id"])["state"] == "interrupted"


def test_voice_recipe_snapshot_is_idempotent(client):
    response = client.post("/v1/jobs", json={"text": "voice", "instruct": "bright"}, headers=auth())
    job = wait_job(client, response.json()["job"]["id"])
    assert job["state"] == "succeeded"
    payload = {"name": "中文音色", "job_id": job["id"]}
    created = client.post("/v1/voices", json=payload, headers=auth())
    replay = client.post("/v1/voices", json=payload, headers=auth())
    assert created.status_code == 200
    assert replay.status_code == 200 and replay.json()["replayed"] is True
    voice = client.get("/v1/voices/中文音色", headers=auth())
    assert voice.status_code == 200
    assert voice.json()["job_id"] == job["id"]
    assert voice.json()["request"] == job["request"]
    conflict = client.post("/v1/voices", json={"name": "中文音色", "job_id": "other"}, headers=auth())
    assert conflict.status_code == 409


def test_job_request_model_defaults_and_finite_only():
    req = JobRequest(text="x", instruct="i")
    assert req.language == "Chinese"
    assert req.seed == 42
    assert req.temperature == 0.9
    assert req.top_p == 1.0
    assert req.top_k == 50
    assert req.max_new_tokens == 2048


def test_unicode_auth_comparison_is_safe():
    assert _auth_matches("Bearer 密钥", "Bearer 密钥")
    assert not _auth_matches("Bearer 密钥", "Bearer 另一把钥匙")


def test_stop_drains_active_and_preserves_queue(tmp_path, monkeypatch):
    import threading

    from qvd.engine import TestEngine
    from qvd.server import Worker

    began, release = threading.Event(), threading.Event()

    class BlockingEngine(TestEngine):
        def generate(self, request, output, progress):
            began.set()
            assert release.wait(5)
            return super().generate(request, output, progress)

    store = Store(tmp_path)
    body = JobRequest(text="你好", instruct="温暖自然").model_dump()
    active, _ = store.submit(body, "active")
    queued, _ = store.submit(body, "queued")
    worker = Worker({"engine": "test"}, store)
    monkeypatch.setattr(worker, "_engine", lambda: BlockingEngine({}))
    worker.thread.start()
    try:
        assert began.wait(5)
        worker.request_stop()
    finally:
        release.set()
        worker.thread.join(5)
    assert not worker.thread.is_alive()
    assert store.get(active["id"])["state"] == "succeeded"
    assert store.get(queued["id"])["state"] == "queued"
    store.recover()
    assert store.claim()["id"] == queued["id"]
