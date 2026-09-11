import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .common import TERMINAL, Failure


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "jobs.sqlite3"
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    idem TEXT UNIQUE,
                    fingerprint TEXT NOT NULL,
                    request TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created REAL NOT NULL,
                    started REAL,
                    finished REAL,
                    progress REAL NOT NULL DEFAULT 0,
                    phase TEXT NOT NULL DEFAULT 'queued',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    result TEXT
                );
                CREATE TABLE IF NOT EXISTS voices (
                    name TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    request TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs(state, created);
                """
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def decode(cls, row):
        data = dict(row)
        for key in ("request", "error", "result"):
            data[key] = json.loads(data[key]) if data[key] else None
        if "cancel_requested" in data:
            data["cancel_requested"] = bool(data["cancel_requested"])
        return data

    def recover(self):
        """Leave queued jobs durable and make in-flight work explicitly retryable."""
        with self.connect() as db:
            db.execute(
                """UPDATE jobs SET state='interrupted', finished=?, phase='interrupted',
                   error=?, cancel_requested=0 WHERE state='running'""",
                (time.time(), self._json({"code": "WORKER_INTERRUPTED", "message": "服务在生成时退出；请使用新幂等键重试"})),
            )

    def submit(self, request, idem=None):
        raw = self._json(request)
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idem:
                previous = db.execute("SELECT * FROM jobs WHERE idem=?", (idem,)).fetchone()
                if previous is not None:
                    if previous["fingerprint"] != fingerprint:
                        raise Failure("IDEMPOTENCY_CONFLICT", "同一幂等键对应不同请求", 2)
                    return self.decode(previous), True
            count = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
            if count >= 1000:
                raise Failure("QUEUE_FULL", "队列已达到 1000 个任务上限")
            jid = uuid.uuid4().hex
            db.execute(
                "INSERT INTO jobs (id,idem,fingerprint,request,state,created) VALUES (?,?,?,?,'queued',?)",
                (jid, idem, fingerprint, raw, time.time()),
            )
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()), False

    def get(self, jid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise Failure("JOB_NOT_FOUND", "找不到任务", 2)
        return self.decode(row)

    def list(self, limit=50):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self.decode(row) for row in rows]

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created, id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET state='running', started=?, phase='starting' WHERE id=?", (time.time(), row[0]))
        return self.get(row[0])

    def progress(self, jid, value, phase):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET progress=MAX(progress,?), phase=? WHERE id=? AND state='running'",
                (min(0.99, max(0.0, float(value))), str(phase)[:120], jid),
            )

    def finish(self, jid, state, result=None, error=None):
        """Commit terminal state atomically with the cancel flag.

        If cancellation wins the transaction race, a generated success becomes
        cancelled. If finish wins first, cancel is a no-op on the terminal job.
        """
        if state not in TERMINAL:
            raise ValueError(f"invalid terminal state: {state}")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
            if row is None:
                raise Failure("JOB_NOT_FOUND", "找不到任务", 2)
            if row["state"] in TERMINAL:
                return self.decode(row)
            final_state, final_result, final_error = state, result, error
            if state == "succeeded" and row["cancel_requested"]:
                final_state = "cancelled"
                final_result = None
                final_error = {"code": "CANCELLED", "message": "任务已请求取消"}
            db.execute(
                """UPDATE jobs SET state=?, finished=?, progress=?, phase=?, result=?, error=?
                   WHERE id=? AND state='running'""",
                (
                    final_state,
                    time.time(),
                    1.0 if final_state == "succeeded" else 0.0,
                    final_state,
                    self._json(final_result) if final_result is not None else None,
                    self._json(final_error) if final_error is not None else None,
                    jid,
                ),
            )
        return self.get(jid)

    def cancel(self, jid):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()
            if row is None:
                raise Failure("JOB_NOT_FOUND", "找不到任务", 2)
            if row[0] not in TERMINAL:
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (jid,))
                if row[0] == "queued":
                    db.execute(
                        "UPDATE jobs SET state='cancelled', finished=?, progress=0, phase='cancelled', error=? WHERE id=?",
                        (time.time(), self._json({"code": "CANCELLED", "message": "任务已取消"}), jid),
                    )
        return self.get(jid)

    @staticmethod
    def _voice_decode(row):
        result = dict(row)
        result["request"] = json.loads(result["request"])
        result["result"] = json.loads(result["result"])
        result["reference_job_id"] = result["job_id"]
        return result

    def voices(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM voices ORDER BY name").fetchall()
        return [self._voice_decode(row) for row in rows]

    def get_voice(self, name):
        with self.connect() as db:
            row = db.execute("SELECT * FROM voices WHERE name=?", (name,)).fetchone()
        if row is None:
            raise Failure("VOICE_NOT_FOUND", "找不到音色名", 2)
        return self._voice_decode(row)

    def add_voice(self, name, job_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM voices WHERE name=?", (name,)).fetchone()
            if existing is not None:
                if existing["job_id"] != job_id:
                    raise Failure("VOICE_EXISTS", "音色名已存在，请使用另一个名称", 2)
                return self._voice_decode(existing), True
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise Failure("JOB_NOT_FOUND", "找不到任务", 2)
            if job["state"] != "succeeded" or not job["result"]:
                raise Failure("VOICE_SOURCE_NOT_READY", "只能从成功任务创建音色配方", 2)
            db.execute(
                "INSERT INTO voices (name,job_id,request,result,created) VALUES (?,?,?,?,?)",
                (name, job_id, job["request"], job["result"], time.time()),
            )
            row = db.execute("SELECT * FROM voices WHERE name=?", (name,)).fetchone()
            return self._voice_decode(row), False
