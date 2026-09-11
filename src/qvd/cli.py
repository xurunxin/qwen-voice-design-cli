import argparse
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from . import __version__
from .client import Client
from .common import (
    LANGUAGES,
    MODEL,
    ROOT,
    TERMINAL,
    Failure,
    InstanceLock,
    atomic_json,
    read_json,
)


def emit(data):
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Failure("INVALID_ARGUMENT", message, 2)


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数")
    return number


def runtime_arguments(p):
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--runtime-python", type=Path, default=None)
    p.add_argument("--model-dir", default=None, help="本地模型目录或 Hugging Face 模型 ID；默认使用 init 配置")
    p.add_argument("--revision", default=None, help="远程模型的固定提交 SHA；本地目录由 bootstrap 固定")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    p.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    p.add_argument("--engine", choices=["qwen", "test"], default="qwen", help="test 仅测试传输，不生成语音")


def resolve_runtime(args):
    config = read_json(Path(args.home).resolve() / "runtime.json", {})
    defaults = {"runtime_python": ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
                "model_dir": str(ROOT / "models" / "VoiceDesign"), "device": "cuda:0", "dtype": "bfloat16"}
    for key, default in defaults.items():
        if getattr(args, key) is None:
            value = os.getenv("QVD_" + key.upper()) or config.get(key) or default
            setattr(args, key, Path(value) if key == "runtime_python" else value)


def design_arguments(p):
    p.add_argument("--request", type=Path, help="完整请求 JSON，可由后面的参数覆盖")
    tx = p.add_mutually_exclusive_group()
    tx.add_argument("--text")
    tx.add_argument("--text-file", type=Path)
    ins = p.add_mutually_exclusive_group()
    ins.add_argument("--instruct", help="自然语言音色描述：年龄感、音高、质感、口音、语速和情绪")
    ins.add_argument("--instruct-file", type=Path)
    p.add_argument("--voice", help="使用服务端已保存的设计配方；不保证固定声纹")
    p.add_argument("--language", choices=LANGUAGES)
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--idempotency-key", help="同键同请求返回原任务；冲突报错")


def parser():
    p = Parser(prog="qvd", description="Qwen3-TTS 音色设计 CLI；stdout 为 JSON，监控/候选生成为 NDJSON。")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--home", default=os.getenv("QVD_HOME", str(Path.home() / ".qwen-voice-design")))
    p.add_argument("--url", default=os.getenv("QVD_URL"))
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--timeout", type=positive, default=30)
    sub = p.add_subparsers(dest="command", required=True)
    from .init import arguments as init_arguments
    init_arguments(sub)
    from .uninstall import arguments as uninstall_arguments
    uninstall_arguments(sub)
    config = sub.add_parser("config").add_subparsers(dest="action", required=True)
    cp = config.add_parser("set")
    cp.add_argument("--url", dest="config_url", required=True)
    cp.add_argument("--api-key-env", dest="config_key_env", default="QVD_API_KEY")
    config.add_parser("show")
    d = sub.add_parser("doctor", help="检查本地运行时、GPU 和模型目录")
    runtime_arguments(d)
    service = sub.add_parser("service").add_subparsers(dest="action", required=True)
    start = service.add_parser("start", help="启动本机后台常驻服务")
    runtime_arguments(start)
    start.add_argument("--wait", type=positive, default=0)
    service.add_parser("status")
    stop = service.add_parser("stop", help="当前推理结束后退出，保留队列")
    stop.add_argument("--wait", type=positive, default=0)
    logs = service.add_parser("logs")
    logs.add_argument("--tail", type=int, default=60)
    runtime_arguments(sub.add_parser("serve", help="前台服务，可用于远程主机和容器"))
    sub.add_parser("capabilities")
    voices = sub.add_parser("voices", help="保存已选试听的音频及设计配方引用").add_subparsers(dest="action", required=True)
    voices.add_parser("list")
    save = voices.add_parser("save")
    save.add_argument("name")
    save.add_argument("--job", required=True, dest="job_id")
    voices.add_parser("get").add_argument("name")
    export = voices.add_parser("export", help="导出选定参考音频和完整配方，可供 itt 使用")
    export.add_argument("name")
    export.add_argument("--output-dir", type=Path, required=True)
    jobs = sub.add_parser("jobs").add_subparsers(dest="action", required=True)
    submit = jobs.add_parser("submit")
    design_arguments(submit)
    submit.add_argument("--wait", type=positive, default=0)
    submit.add_argument("--output", type=Path)
    jobs.add_parser("list").add_argument("--limit", type=int, default=50)
    for action in ("get", "watch", "cancel", "download"):
        item = jobs.add_parser(action)
        item.add_argument("job_id")
        if action == "watch":
            item.add_argument("--wait", type=positive, default=600)
            item.add_argument("--interval", type=positive, default=1)
        if action == "download":
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--force", action="store_true")
    design = sub.add_parser("design", help="生成多个种子候选，输出试听 WAV 和可恢复清单")
    design_arguments(design)
    design.add_argument("--candidates", type=int, choices=range(1, 9), default=1)
    design.add_argument("--output-dir", type=Path, required=True, help="使用新目录；中断后用 --resume 恢复")
    design.add_argument("--wait", type=positive, default=600, help="每个候选等待秒数")
    design.add_argument("--resume", action="store_true")
    return p


def connection(args):
    home = Path(args.home).resolve()
    config = read_json(home / "client.json", {})
    local = read_json(home / "local-service.json", {})
    url = args.url or config.get("url") or local.get("url", "http://127.0.0.1:8096")
    env_name = args.api_key_env or config.get("api_key_env", "QVD_API_KEY")
    key = os.getenv(env_name, "")
    if not key and url.rstrip("/") == local.get("url", "http://127.0.0.1:8096") and (home / "local-key").is_file():
        key = (home / "local-key").read_text().strip()
    if not key:
        raise Failure("API_KEY_MISSING", f"请设置 {env_name}", 2)
    return Client(url, key, args.timeout)


def runtime_config(args):
    home = Path(args.home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.port <= 65535:
        raise Failure("INVALID_ARGUMENT", "端口必须为1..65535", 2)
    key = os.getenv(args.api_key_env or "QVD_API_KEY", "")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not key:
        raise Failure("API_KEY_MISSING", "监听外部网卡必须通过环境变量提供 API key", 2)
    if not key:
        path = home / "local-key"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_urlsafe(32))
        key = path.read_text().strip()
    if not key or not key.isascii():
        raise Failure("INVALID_API_KEY", "API key 必须为非空 ASCII 字符串", 2)
    model = str(Path(args.model_dir).resolve()) if Path(args.model_dir).exists() else args.model_dir
    return {"data_dir": str(home), "api_key": key, "model_dir": model,
            **{k: getattr(args, k) for k in ("host", "port", "device", "dtype", "attention", "engine", "revision")}}


def background_start(args):
    home = Path(args.home).resolve()
    with InstanceLock(home / "start.lock"):
        cfg = runtime_config(args)
        address = "127.0.0.1" if args.host in {"0.0.0.0", "localhost"} else "::1" if args.host == "::" else args.host
        url = f"http://{'[' + address + ']' if ':' in address else address}:{args.port}"
        client = Client(url, cfg["api_key"], 2)
        try:
            status = client.call("/v1/status")
        except Failure as e:
            if e.code != "CONNECTION_FAILED":
                raise
        else:
            previous = read_json(home / "local-service.json", {})
            if status.get("service") != "qwen-voice-design" or previous.get("url") != url:
                raise Failure("SERVICE_NOT_OWNED", "此端口服务不属于当前 CLI home")
            return {"already_running": True, "url": url, **status}
        with InstanceLock(home / "service.lock"):
            pass
        python = args.runtime_python.resolve()
        if not python.is_file():
            raise Failure("RUNTIME_MISSING", "先运行 scripts/bootstrap.py，或指定 --runtime-python", 2)
        forwarded = ["--home", str(home), "serve"]
        for k in ("host", "port", "model_dir", "device", "dtype", "attention", "engine", "revision"):
            if cfg[k] is not None:
                forwarded.extend(["--" + k.replace("_", "-"), str(cfg[k])])
        log_path = home / "service.log"
        env = {**os.environ, "QVD_API_KEY": cfg["api_key"], "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        creation = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with log_path.open("ab") as log:
            process = subprocess.Popen([str(python), "-m", "qvd.cli", *forwarded], cwd=home,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, **creation)
        info = {"pid": process.pid, "url": url, "log": str(log_path), "runtime_python": str(python)}
        atomic_json(home / "local-service.json", info)
    if args.wait:
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            try:
                status = client.call("/v1/status")
                if status["state"] == "ready":
                    return {**info, **status}
                if status["state"] == "failed":
                    raise Failure("MODEL_LOAD_FAILED", status["error"]["message"], details=info)
            except Failure as e:
                if e.code != "CONNECTION_FAILED":
                    raise
            if process.poll() is not None:
                raise Failure("SERVICE_EXITED", "后台进程已退出；用 service logs 查看原因", details=info)
            time.sleep(0.5)
        raise Failure("WAIT_TIMEOUT", "模型仍在后台加载；用 service status/logs 检查", 3, info)
    return {**info, "state": "starting"}


def watch(client, jid, seconds, interval=1):
    deadline, previous = time.monotonic() + seconds, None
    while True:
        job = client.call("/v1/jobs/" + quote(jid, safe=""))
        fingerprint = (job["state"], job["progress"], job["phase"], job["cancel_requested"])
        if fingerprint != previous:
            emit({"ok": True, "event": "job", "job_id": jid,
                  **{k: job[k] for k in ("state", "progress", "phase", "cancel_requested", "result", "error")}})
            previous = fingerprint
        if job["state"] in TERMINAL:
            if job["state"] != "succeeded":
                raise Failure("JOB_" + job["state"].upper(), f"任务结束: {job['state']}", 4, {"job_id": jid, "error": job["error"]})
            return job
        if time.monotonic() >= deadline:
            raise Failure("WAIT_TIMEOUT", "等待超时；任务继续，可用 jobs watch 恢复监控", 3, {"job_id": jid})
        time.sleep(max(0.1, min(interval, deadline - time.monotonic())))


def request_body(args, client):
    body = {}
    if args.voice:
        body.update(client.call("/v1/voices/" + quote(args.voice, safe=""))["request"])
    if args.request:
        loaded = json.loads(args.request.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            raise Failure("INVALID_REQUEST", "请求文件必须是 JSON 对象", 2)
        body.update(loaded)
    for key in ("text", "instruct", "language", "seed", "temperature", "top_p", "top_k", "max_new_tokens"):
        value = getattr(args, key)
        if value is not None:
            body[key] = value
    for key in ("text", "instruct"):
        path = getattr(args, key + "_file")
        if path:
            body[key] = path.read_text(encoding="utf-8-sig")
    return body


def submit_request(client, body, idem):
    try:
        result = client.call("/v1/jobs", "POST", body, headers={"Idempotency-Key": idem})
    except Failure as e:
        if e.code == "CONNECTION_FAILED":
            raise Failure("SUBMISSION_UNKNOWN", "响应未确认；用同一请求和幂等键重试", 1, {"idempotency_key": idem}) from None
        raise
    return {**result, "idempotency_key": idem}


def design(args, client):
    directory = args.output_dir.resolve()
    manifest = directory / "manifest.json"
    if args.resume:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("service_url") != client.url:
            raise Failure("SERVICE_MISMATCH", "清单属于其他服务地址", 2)
    else:
        body = request_body(args, client)
        seed = body.get("seed", 42)
        if not isinstance(seed, int) or not 0 <= seed <= 4294967295 - args.candidates + 1:
            raise Failure("INVALID_SEED", "候选种子必须在 uint32 范围内", 2)
        directory.mkdir(parents=True, exist_ok=False)
        base = args.idempotency_key or uuid.uuid4().hex
        data = {"version": 1, "service_url": client.url, "note": "设计配方不保证固定声纹；test 后端不是语音", "candidates": [
            {"index": i + 1, "request": {**body, "seed": seed + i}, "idempotency_key": f"{base}.{i + 1}"}
            for i in range(args.candidates)]}
        atomic_json(manifest, data)
    for candidate in data["candidates"]:
        # Re-submit even after response loss: same key/body recovers the same job.
        result = submit_request(client, candidate["request"], candidate["idempotency_key"])
        candidate["job_id"] = result["job"]["id"]
        candidate["effective_request"] = result["job"]["request"]
        atomic_json(manifest, data)
        job = watch(client, candidate["job_id"], args.wait)
        target = directory / f"candidate-{int(candidate['index']):02d}.wav"
        if target.exists():
            import hashlib
            if hashlib.sha256(target.read_bytes()).hexdigest() != job["result"]["sha256"]:
                raise Failure("OUTPUT_EXISTS", "候选文件与服务结果不一致；请保留文件并选择新目录", 2)
        else:
            client.download(job["id"], target)
        candidate.update({"output": str(target), "result": job["result"]})
        atomic_json(manifest, data)
    return {"manifest": str(manifest), "candidates": data["candidates"]}


def run(args):
    home = Path(args.home).resolve()
    if args.command == "uninstall":
        from .uninstall import uninstall
        return uninstall(args)
    if args.command == "init":
        from .init import initialize
        return initialize(args, emit)
    if hasattr(args, "model_dir"):
        resolve_runtime(args)
    if args.command == "config":
        if args.action == "set":
            Client(args.config_url, "")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.config_key_env):
                raise Failure("INVALID_ARGUMENT", "API key 环境变量名不合法", 2)
            atomic_json(home / "client.json", {"url": args.config_url.rstrip("/"), "api_key_env": args.config_key_env})
        return read_json(home / "client.json", {})
    if args.command == "doctor":
        result = {"cli_version": __version__, "python": sys.executable, "model": MODEL,
                  "model_config_exists": (Path(args.model_dir) / "config.json").is_file(),
                  "runtime_python": str(args.runtime_python), "runtime_exists": args.runtime_python.is_file()}
        if shutil.which("nvidia-smi"):
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=False)
            result["gpu"] = gpu.stdout.strip()
        if args.runtime_python.is_file():
            code = "import json,importlib.util; import torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'bf16':torch.cuda.is_bf16_supported(),'qwen_tts':importlib.util.find_spec('qwen_tts') is not None}))"
            check = subprocess.run([str(args.runtime_python), "-c", code], capture_output=True, text=True, timeout=60, check=False)
            result["runtime"] = {"exit_code": check.returncode, "output": check.stdout.strip(), "error": check.stderr.strip()}
        return result
    if args.command == "serve":
        if importlib.util.find_spec("fastapi") is None or (args.engine != "test" and importlib.util.find_spec("qwen_tts") is None):
            python = args.runtime_python.resolve()
            if not python.is_file() or python == Path(sys.executable).resolve():
                raise Failure("RUNTIME_MISSING", "请运行 bootstrap 安装服务/模型运行时", 2)
            result = subprocess.run([str(python), "-m", "qvd.cli", *sys.argv[1:]], check=False)
            if result.returncode:
                raise Failure("SERVICE_EXITED", "服务进程退出", result.returncode)
            return None
        cfg = runtime_config(args)
        from .server import serve
        sys.stdout = sys.stderr
        serve(cfg)
        return None
    if args.command == "service" and args.action == "start":
        return background_start(args)
    if args.command == "service" and args.action == "logs":
        from collections import deque
        path = home / "service.log"
        if not path.exists():
            return {"logs": []}
        with path.open(encoding="utf-8", errors="replace") as f:
            return {"logs": [line.rstrip() for line in deque(f, maxlen=max(1, args.tail))]}
    client = connection(args)
    if args.command == "service":
        if args.action == "status":
            return client.call("/v1/status")
        result = client.call("/v1/service/stop", "POST")
        if args.wait:
            deadline = time.monotonic() + args.wait
            while time.monotonic() < deadline:
                try:
                    client.call("/v1/status")
                except Failure as e:
                    if e.code == "CONNECTION_FAILED":
                        return {"state": "stopped"}
                    raise
                time.sleep(0.5)
            raise Failure("WAIT_TIMEOUT", "服务仍在完成当前推理", 3)
        return result
    if args.command == "capabilities":
        return client.call("/v1/capabilities")
    if args.command == "design":
        return design(args, client)
    if args.command == "voices":
        if args.action == "list":
            return client.call("/v1/voices")
        if args.action == "save":
            return client.call("/v1/voices", "POST", {"name": args.name, "job_id": args.job_id})
        voice = client.call("/v1/voices/" + quote(args.name, safe=""))
        if args.action == "get":
            return voice
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        audio = client.download(voice["job_id"], directory / "reference.wav")
        atomic_json(directory / "voice.json", voice)
        (directory / "reference.txt").write_text(voice["request"]["text"], encoding="utf-8")
        return {"voice": voice, "audio": audio, "manifest": str(directory / "voice.json")}
    if args.command == "jobs":
        if args.action == "submit":
            if args.output and not args.wait:
                raise Failure("INVALID_ARGUMENT", "--output 需要 --wait", 2)
            if args.output and args.output.exists():
                raise Failure("OUTPUT_EXISTS", "输出已存在", 2)
            result = submit_request(client, request_body(args, client), args.idempotency_key or uuid.uuid4().hex)
            if args.wait:
                emit({"ok": True, **result})
                job = watch(client, result["job"]["id"], args.wait)
                return client.download(job["id"], args.output) if args.output else {"job": job}
            return result
        if args.action == "list":
            return client.call("/v1/jobs?limit=" + str(args.limit))
        if args.action == "download":
            return client.download(args.job_id, args.output, args.force)
        if args.action == "watch":
            return {"job": watch(client, args.job_id, args.wait, args.interval)}
        path = "/v1/jobs/" + quote(args.job_id, safe="")
        return client.call(path + "/cancel", "POST") if args.action == "cancel" else client.call(path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = run(parser().parse_args())
        if result is not None:
            emit({"ok": True, **result})
    except Failure as e:
        emit({"ok": False, "error": {"code": e.code, "message": str(e), "details": e.details}})
        return e.exit_code
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        emit({"ok": False, "error": {"code": "LOCAL_ERROR", "message": str(e)}})
        return 2
    except KeyboardInterrupt:
        emit({"ok": False, "error": {"code": "MONITOR_INTERRUPTED", "message": "停止等待；已提交任务继续"}})
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
