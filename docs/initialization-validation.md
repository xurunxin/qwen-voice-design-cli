# 初始化验收（2026-09-11）

Windows / NVIDIA CUDA 实测从空的 `managed-runtime` 目录安装 Python 3.11、独立环境、PyTorch、Qwen SDK 和固定 revision 模型，原有 `.venv`、模型目录保留。`qvd init --check` 返回 ready=true；重复初始化复用已校验的模型清单。

新环境启动后真实生成任务 `e379aced95754808831ebe5029ddbe3e` 成功，输出 `outputs/init-managed-check.wav`：3.6 秒、24 kHz、单声道 PCM16，生成耗时 14.175 秒，SHA-256 `9c3c99fdb500b9a2b9c325d76b13a827e8da2e19d585dd1957678e4434fa0520`。

缺少 uv 分支实际下载并校验 uv 0.12.9。Windows `install.ps1 -SkipInit` 在 PATH 仅含 System32、无法找到 Python/uv 的子进程中成功安装独立 CLI；此测试仍可访问本机 Python 下载缓存，不等于全新操作系统虚拟机验收。

25 项测试通过，覆盖只读检查、目录归属、运行时参数优先级、下载损坏检测、失败重试、打包锁文件和无 uv 引导。CPU 完整推理、Linux、Docker 和远程 GPU 部署未在此次验收中执行。CUDA 驱动需事先安装；安装器不会修改系统驱动。

首次启动建议使用 `qvd init --start --wait 600`。等待超时不会终止后台加载，使用 `qvd service status` 查询。验收后服务停止以释放显存，运行时配置保持可用。
