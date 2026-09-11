# Qwen Voice Design CLI (`qvd`)

使用自然语言设计音色，生成多个试听候选，保存选中的参考音频和设计配方，并导出给 IndexTTS 等后续语音工具。设计沿用本机 `itt` 的轻量 HTTP 客户端、常驻模型服务、单 GPU 队列、持久任务、幂等重试和 JSON 输出约定。

模型使用官方 `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`。不需要参考音频；主要输入是试听台词 `text` 和音色描述 `instruct`。VoiceDesign 与 Base 模型的声音克隆是不同能力：配方复用会重新设计声音，不保证跨文本、种子或设备维持同一声纹。要固定选中的声音，请导出参考 WAV，再交给 IndexTTS 或 Qwen Base 克隆。本工具不加载 Base 模型。

## 初始化与启动

已安装轻量 CLI 的机器无需预装 uv、模型 Python、PyTorch 或 Hugging Face CLI：

```powershell
qvd init --dry-run
qvd init --install-dir 'G:\TTS\qvd-runtime'
qvd init --check
qvd service start --wait 600
```

`init` 默认创建 `<home>/runtime`，自动补齐 uv、Python 3.11、独立虚拟环境、推理依赖和固定版本模型。安装完成后保存 `<home>/runtime.json`，`service start`、`serve`、`doctor` 自动读取；显式命令参数优先，其次环境变量，再次为保存的路径。原源码目录的 `.venv` 和模型不会被覆盖。

`--check`、`--dry-run` 不写文件、不联网下载。`--check` 返回 `ready` 和缺失项；`--dry-run` 只列计划，不导入模型运行时。`--skip-models` 只安装运行环境，返回模型未就绪；重新执行同一 `init` 命令可继续安装。`--device cpu` 使用 CPU PyTorch，服务默认切换到 CPU/float32；CUDA 模式要求已有 NVIDIA 驱动，不自动改动系统驱动。`--start` 在初始化完成后启动服务。非空陌生目录或 CPU/CUDA 安装配置冲突会明确报错，应选择新目录。

初始化 stdout 为 NDJSON 阶段事件，详细安装日志写入 `<home>/init.log`，最近阶段写入 `init-state.json`。模型按 revision 下载并校验必需文件清单和长度；下载失败后重试复用缓存，不把残缺文件标记为 ready。磁盘建议预留至少15 GB；下载需要访问 PyPI、PyTorch 和 Hugging Face。

### 没有 Python 的 Windows 机器

将源码包解压后，在 PowerShell 7 运行：

```powershell
pwsh.exe -NoProfile -File .\scripts\install.ps1 -InstallDir 'G:\TTS\qvd-runtime'
```

入口会按 SHA-256 校验固定版本 uv 的官方 PyPI 二进制，再让 uv 安装 Python、CLI 并调用 `qvd init`。无需管理员权限，不修改系统 Python。`-Package <wheel路径>` 可改为安装携带的 CLI wheel（模型及依赖仍需联网），`-SkipModels` 先装运行时，`-SkipInit` 只装 CLI。脚本最后打印 CLI 的绝对路径；若新终端 PATH 尚未包含该目录，可直接使用打印的路径。

### 兼容的 Python 安装入口

```powershell
Set-Location 'G:\Projects\AIGC_Tools\Qwen3-TTS-VoiceDesign'
python scripts/bootstrap.py
qvd doctor
qvd service start --wait 600
qvd service status
qvd capabilities
```

`scripts/bootstrap.py` 现在是 `qvd init` 的兼容入口，自动准备 uv 并安装 CLI，不再维护另一套部署逻辑。`--cpu` 对应 `init --device cpu`；其余初始化参数可直接传入。固定 Qwen SDK 0.1.1、PyTorch/torchaudio 2.8.0 CUDA 12.8。默认 SDPA，不依赖 Windows 上额外编译 Flash Attention。

全局客户端没有 torch/FastAPI 依赖。单独部署客户端可 `uv tool install <wheel路径>`；执行 `init` 后后台启动自动读取运行环境。使用已有外部环境时可显式指定 `--runtime-python <模型环境python>`，或设置 `QVD_RUNTIME_PYTHON`。服务使用独立目录 `~/.qwen-voice-design`，默认端口 8096；`itt` 的目录和端口不变。全局选项 `--home`、`--url` 等放在子命令前。

## 卸载

```powershell
qvd uninstall --dry-run
qvd service stop --wait 600
qvd uninstall --yes
```

卸载只删除 `init` 标记归属的运行环境及模型，并清除指向该目录的 `runtime.json`。任务、音色、日志、导出的音频以及共享 uv/Python/Hugging Face 缓存保留。可用 `--install-dir <目录>` 清理失败的受管安装；陌生目录、符号链接/junction 和正在使用的服务环境会被拒绝。`--dry-run` 不写文件；实际删除必须带 `--yes`。请从轻量全局客户端执行，不要从待删除环境内运行命令。

运行环境卸载后，如还需移除通过 uv 安装的全局客户端：`uv tool uninstall qwen-voice-design-cli`。安装时使用其他包管理器则通过对应工具移除客户端。用户数据可继续保留供重新安装使用。

## 设计与试听

```powershell
qvd design --text '你好，欢迎来到今天的声音实验室。让我们一起发现声音的可能。' --instruct '年轻成年女性，中低音，温暖自然，轻微气声，普通话清晰，语速舒缓，像朋友聊天，避免播音腔。' --language Chinese --seed 42 --candidates 3 --output-dir outputs/warm-voice --wait 600
```

输出 `candidate-01.wav` 等文件以及 `manifest.json`，包含每个候选的请求、种子、任务 ID、实际音频参数、耗时和 SHA-256。改变种子用于探索候选。试听品质和是否符合描述需要实际听音判断。

清单在提交前写入稳定幂等键。中断或等待超时后，后台任务仍会运行；恢复会重新查询同一个任务，并校验已下载文件：

```powershell
qvd design --output-dir outputs/warm-voice --resume --wait 600
```

恢复以清单中的请求/种子/候选数为准，CLI 新传的设计参数不生效。不同服务地址不能复用此清单。新设计默认拒绝已有输出目录，防止覆盖素材。

支持 `--text-file`、`--instruct-file`（UTF-8）、`--request request.json`、`--temperature`、`--top-p`、`--top-k`、`--max-new-tokens`。完整字段和边界以 `qvd capabilities` 返回的 JSON Schema 为准。使用自然语言描述语速、语气、口音、音高、年龄感、共鸣和质感；这些不是精确数值控制，不提供 IndexTTS 专属情绪向量或时长倍率。

支持 Chinese、English、Japanese、Korean、German、French、Russian、Portuguese、Spanish、Italian、Auto。

## 保存、复用、导出

用清单内选中候选的实际 `job_id`：

```powershell
qvd voices save warm-narrator --job <job_id>
qvd voices list
qvd voices get warm-narrator
qvd voices export warm-narrator --output-dir outputs/warm-reference
qvd jobs submit --voice warm-narrator --text '这是新的试听台词。' --wait 600 --output outputs/new-preview.wav
```

导出含 `reference.wav`、`reference.txt` 和 `voice.json`。导出目录必须为新目录；音色名称不可覆写为其他任务。后续使用 IndexTTS：

```powershell
itt voices add warm-narrator --audio outputs/warm-reference/reference.wav
itt jobs submit --voice warm-narrator --text '用选中的声音朗读新的内容。' --wait 600 --output outputs/itt-result.wav
```

这些 `itt` 命令需已有 IndexTTS 服务可用；`qvd` 不自动启动或停止另一个模型，避免并发占用显存。

## Agent 任务接口

```powershell
qvd jobs submit --text '欢迎收听。' --instruct '成熟男声，低沉温和，吐字清晰。' --idempotency-key scene01-v1
qvd jobs list --limit 20
qvd jobs get <job_id>
qvd jobs watch <job_id> --wait 600
qvd jobs download <job_id> --output outputs/scene01.wav
qvd jobs cancel <job_id>
qvd service logs --tail 40
qvd service stop --wait 600
```

普通命令 stdout 为单行 JSON；`watch`、`submit --wait` 和 `design` 为 NDJSON，包含状态事件和最终结果。模型日志在服务日志/stderr。退出码：0 成功，1 连接/运行错误，2 参数或本地文件错误，3 等待超时（任务继续），4 任务失败/取消/中断，130 停止等待。同键同请求返回原任务，不同请求返回冲突；失败任务不会自动重做，需新幂等键。种子不保证不同软硬件环境逐字节复现。

排队取消立即生效；运行中取消需等待当前同步模型调用结束，丢弃其结果。服务停止会排空当前推理并保留排队任务；崩溃恢复将 running 标为 interrupted，queued 继续。单服务同一时间仅执行一个模型任务。进度表示阶段，不是预计剩余时间。

## 远程服务与 Docker

部署机安装运行环境后，通过环境变量提供 `QVD_API_KEY`，运行：

```powershell
qvd serve --host 0.0.0.0 --model-dir /models/VoiceDesign
```

客户端通过 `qvd config set --url https://your-host --api-key-env QVD_API_KEY` 保存地址；密钥只从环境变量读取，不存远程配置、不放命令行。外部监听必须提供密钥，所有 `/v1/` 接口认证；公网应通过 HTTPS 反向代理。所有任务路径属于服务端，下载/导出目录属于客户端。

Docker：先下载模型，设置 `QVD_API_KEY`，再运行 `docker compose up --build -d`。Compose 默认仅映射本机 8096，使用持久化数据卷和只读模型目录。容器构建/远程 GPU 验收状态见 [docs/validation.md](docs/validation.md)。

## 开发验证与交付

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src scripts tests
python scripts/package.py
```

`--engine test` 仅产生显式标记的测试音，绝不作为模型失败的回退。真实语音验证、依赖和硬件限制见 [验证记录](docs/validation.md)。

官方依据：[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[VoiceDesign 调用示例](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_voice_design.py)。
