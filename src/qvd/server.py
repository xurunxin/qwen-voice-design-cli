import asyncio
import hashlib
import hmac
import os
import re
import threading
import time
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .common import Failure, InstanceLock
from .engine import Cancelled, QwenEngine, TestEngine, model_revision
from .schema import JobRequest, VoiceRequest, capabilities
from .store import Store


class Worker:
    def __init__(self, config, store):
        self.config = config
        self.store = store
        self.stop_requested = threading.Event()
        self.control = threading.Lock()
        self.wake = threading.Event()
        self.state = "loading"
        self.error = None
        self.active = None
        self.engine = None
        self.thread = threading.Thread(target=self.run, name="qvd-model-worker", daemon=True)

    def _engine(self):
        return TestEngine(self.config) if self.config["engine"] == "test" else QwenEngine(self.config)

    def request_stop(self):
        with self.control:
            self.stop_requested.set()
        self.wake.set()

    def run(self):
        try:
            self.engine = self._engine()
            self.state = "ready"
        except Exception as exc:  # noqa: BLE001 - isolate arbitrary model import/load failures
            # Do not include request text in this diagnostic. Model libraries can
            # raise useful load errors without exposing user speech content.
            traceback.print_exc()
            code = getattr(exc, "code", "MODEL_LOAD_FAILED")
            self.state = "failed"
            self.error = {"code": code, "message": str(exc)}
            return

        while True:
            with self.control:
                if self.stop_requested.is_set():
                    break
                job = self.store.claim()
                self.active = job["id"] if job else None
            if job is None:
                self.wake.wait(0.5)
                self.wake.clear()
                continue
            jid = job["id"]
            self.active = jid
            output_dir = self.store.directory / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            pending = output_dir / f"{jid}.part.wav"
            output = output_dir / f"{jid}.wav"
            began = time.perf_counter()
            try:
                def progress(value, desc="", job_id=jid):
                    current = self.store.get(job_id)
                    if current["cancel_requested"]:
                        raise Cancelled()
                    self.store.progress(job_id, value, desc)

                progress(0.01, "starting")
                details = self.engine.generate(job["request"], pending, progress)
                progress(0.98, "validating WAV")
                with wave.open(str(pending), "rb") as wav:
                    frames = wav.getnframes()
                    sample_rate = wav.getframerate()
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    duration = frames / sample_rate if sample_rate else 0
                    if frames <= 0 or sample_rate <= 0 or channels <= 0:
                        raise ValueError("模型未生成有效音频")
                elapsed = time.perf_counter() - began
                os.replace(pending, output)
                details = dict(details or {})
                details.update(
                    {
                        "audio_url": f"/v1/jobs/{jid}/audio",
                        "duration_seconds": round(duration, 4),
                        "generation_seconds": round(elapsed, 3),
                        "generated_at": datetime.now(UTC).isoformat(),
                        "rtf": round(elapsed / duration, 3) if duration else None,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "sample_width_bytes": sample_width,
                        "subtype": "PCM_16" if sample_width == 2 else None,
                        "bytes": output.stat().st_size,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "backend": self.config["engine"],
                        "model_id": details.get("model_id") or self.config.get("model_dir") or "test-fixture",
                        "revision": details.get("revision", self.config.get("revision")),
                    }
                )
                final = self.store.finish(jid, "succeeded", result=details)
                if final["state"] != "succeeded":
                    output.unlink(missing_ok=True)
            except Cancelled:
                self.store.finish(jid, "cancelled", error={"code": "CANCELLED", "message": "任务已取消"})
            except Exception as exc:  # noqa: BLE001 - persist arbitrary model failures as terminal result
                traceback.print_exc()
                final = self.store.finish(
                    jid,
                    "failed",
                    error={"code": "GENERATION_FAILED", "message": str(exc)},
                )
                if final["state"] != "succeeded":
                    output.unlink(missing_ok=True)
            finally:
                pending.unlink(missing_ok=True)
                self.active = None
        self.state = "stopped"


def _auth_matches(provided, expected):
    """Compare UTF-8 bytes so non-ASCII secrets never hit str-only errors."""
    try:
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
    except (AttributeError, UnicodeEncodeError):
        return False


def _normal_config(config):
    value = dict(config)
    value.setdefault("data_dir", str(Path.cwd() / ".qvd-data"))
    value.setdefault("api_key", "")
    value.setdefault("host", "127.0.0.1")
    value.setdefault("port", 8000)
    value.setdefault("engine", "qwen")
    value.setdefault("model_dir", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    value.setdefault("revision", None)
    value.setdefault("device", "cuda")
    value.setdefault("dtype", "bfloat16")
    value.setdefault("attention", "sdpa")
    if value["engine"] not in {"qwen", "test"}:
        raise Failure("INVALID_MODEL_CONFIG", "engine 必须是 qwen 或 test")
    if not isinstance(value["api_key"], str) or not value["api_key"]:
        raise Failure("INVALID_MODEL_CONFIG", "api_key 不能为空")
    if value["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise Failure("INVALID_MODEL_CONFIG", "dtype 必须是 bfloat16、float16 或 float32")
    if value["attention"] not in {"sdpa", "eager", "flash_attention_2"}:
        raise Failure("INVALID_MODEL_CONFIG", "attention 配置不受支持")
    return value


def create_app(config, shutdown=None):
    config = _normal_config(config)
    resolved_revision = model_revision(config["model_dir"], config.get("revision"))
    directory = Path(config["data_dir"])
    store = Store(directory)
    worker = Worker(config, store)
    instance = uuid.uuid4().hex
    token = config["api_key"]
    stopping = False

    @asynccontextmanager
    async def lifespan(app):
        with InstanceLock(directory / "service.lock"):
            store.recover()
            worker.thread.start()
            yield
            worker.request_stop()
            # Active GPU work is allowed to reach its next safe checkpoint;
            # queued work remains in SQLite for the next process.
            await asyncio.to_thread(worker.thread.join)

    app = FastAPI(
        title="Qwen VoiceDesign API",
        version="1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.worker = worker
    app.state.store = store
    app.state.config = config

    @app.exception_handler(Failure)
    async def failed(request, exc):
        if exc.code.endswith("_NOT_FOUND"):
            status = 404
        elif exc.code in {"IDEMPOTENCY_CONFLICT", "VOICE_EXISTS", "SERVICE_BUSY", "RESULT_NOT_READY", "QUEUE_FULL"}:
            status = 409
        else:
            status = 422
        body = {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        if exc.details is not None:
            body["error"]["details"] = exc.details
        return JSONResponse(status_code=status, content=body)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        if exc.status_code == 401:
            return JSONResponse(status_code=401, content={"ok": False, "error": {"code": "UNAUTHORIZED", "message": "API key required"}})
        return JSONResponse(status_code=exc.status_code, content={"ok": False, "error": {"code": "HTTP_ERROR", "message": str(exc.detail)}})

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        message = "; ".join(".".join(map(str, e["loc"])) + ": " + e["msg"] for e in exc.errors())
        return JSONResponse(status_code=422, content={"ok": False, "error": {"code": "INVALID_REQUEST", "message": message}})

    async def auth(authorization: str | None = Header(default=None)):
        expected = "Bearer " + token
        if not authorization or not _auth_matches(authorization, expected):
            raise HTTPException(status_code=401, detail="API key required")

    deps = [Depends(auth)]

    @app.get("/health")
    def health():
        return {"service": "qwen-voice-design", "state": "stopping" if stopping else worker.state}

    @app.get("/v1/status", dependencies=deps)
    def status():
        return {
            "service": "qwen-voice-design",
            "instance": instance,
            "state": "stopping" if stopping else worker.state,
            "active_job": worker.active,
            "error": worker.error,
            "backend": config["engine"],
            "model_id": config.get("model_dir"),
            "revision": resolved_revision,
            "device": config.get("device"),
            "dtype": config.get("dtype"),
            "attention": config.get("attention"),
        }

    @app.get("/v1/capabilities", dependencies=deps)
    def caps():
        return capabilities(config["engine"], {**config, "revision": resolved_revision})

    @app.get("/v1/openapi.json", dependencies=deps)
    def openapi():
        return app.openapi()

    @app.post("/v1/jobs", dependencies=deps)
    def submit(body: JobRequest, idempotency_key: str | None = Header(default=None)):
        if stopping or worker.state != "ready":
            raise Failure("SERVICE_BUSY", f"模型状态为 {worker.state}，就绪后再提交")
        if idempotency_key and (len(idempotency_key) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", idempotency_key)):
            raise Failure("INVALID_IDEMPOTENCY_KEY", "幂等键仅接受字母数字及 . : / - _，最多128字符", 2)
        job, replayed = store.submit(body.model_dump(), idempotency_key)
        worker.wake.set()
        return {"job": job, "replayed": replayed}

    @app.get("/v1/jobs", dependencies=deps)
    def jobs(limit: int = Query(default=50, ge=1, le=200)):
        return {"jobs": store.list(limit)}

    @app.get("/v1/jobs/{jid}", dependencies=deps)
    def job(jid: str):
        return store.get(jid)

    @app.post("/v1/jobs/{jid}/cancel", dependencies=deps)
    def cancel(jid: str):
        return store.cancel(jid)

    @app.get("/v1/jobs/{jid}/audio", dependencies=deps)
    def audio(jid: str):
        task = store.get(jid)
        if task["state"] != "succeeded":
            raise Failure("RESULT_NOT_READY", f"任务状态: {task['state']}", 2)
        path = directory / "outputs" / f"{jid}.wav"
        if not path.is_file():
            raise Failure("RESULT_NOT_FOUND", "结果文件不存在", 2)
        return FileResponse(path, media_type="audio/wav", filename=f"{jid}.wav")

    @app.get("/v1/voices", dependencies=deps)
    def voices():
        return {"voices": store.voices()}

    @app.post("/v1/voices", dependencies=deps)
    def add_voice(body: VoiceRequest):
        voice, replayed = store.add_voice(body.name, body.job_id)
        return {**voice, "replayed": replayed}

    @app.get("/v1/voices/{name}", dependencies=deps)
    def voice(name: str):
        return store.get_voice(name)

    @app.post("/v1/service/stop", dependencies=deps)
    def stop_service():
        nonlocal stopping
        was_stopping = stopping
        stopping = True
        worker.request_stop()
        if shutdown and not was_stopping:
            def drain_and_exit():
                worker.thread.join()
                shutdown()

            threading.Thread(target=drain_and_exit, name="qvd-shutdown-drain", daemon=True).start()
        return {"state": "stopping", "active_job": worker.active, "policy": "drain active job; preserve queue"}

    return app


def serve(config):
    import uvicorn

    server = None

    def request_shutdown():
        server.should_exit = True

    app = create_app(config, request_shutdown)
    server = uvicorn.Server(
        uvicorn.Config(app, host=config.get("host", "127.0.0.1"), port=config.get("port", 8000), access_log=False, log_level="warning")
    )
    server.run()
