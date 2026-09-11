# 验证记录 · 2026-09-11

项目位置：`G:\Projects\AIGC_Tools\Qwen3-TTS-VoiceDesign`。该目录原为空且不是 Git 仓库；本次创建文件并安装 CLI，没有提交或推送。原 IndexTTS 项目未修改。

## 实际运行环境

- Windows / NVIDIA GeForce RTX 3070 8 GiB；启动前空闲约 6.2 GiB。
- Python 3.11.15，torch 2.8.0+cu128，torchaudio 2.8.0+cu128，qwen-tts 0.1.1。
- 官方 VoiceDesign 模型 revision：`5ecdb67327fd37bb2e042aab12ff7391903235d3`。
- BF16、SDPA、单 worker；没有使用 Flash Attention、自定义 CUDA 核或其他模型替代。
- CLI 经 `uv tool install --editable .` 安装为 `qvd`，客户端本身不安装 GPU 依赖。

## 真实模型验收

`qvd design` 以同一中文台词和音色描述、种子42/43生成两份候选：

| 候选 | 任务 ID | 时长 | 推理耗时 | 格式 | 剪裁样本 |
| --- | --- | ---: | ---: | --- | ---: |
| 1 | `78b4b9e671904176bae274699c6d3b4e` | 5.36 s | 12.816 s | 24 kHz mono PCM16 WAV | 0 |
| 2 | `13771e174fe648abb1354ab36edc2943` | 5.20 s | 14.120 s | 24 kHz mono PCM16 WAV | 0 |

两份哈希不同；下载 SHA-256 与服务记录一致，音频非空、有限数值、非静音。`design --resume` 返回原任务，未重复推理。已保存 `warm-demo` 并导出 `reference.wav` / `reference.txt` / `voice.json`。

最终源码重启后补测英文男声设计成功：任务 `5f5b8f17bfe34664a92bf0d0cb4687f3`，5.04秒音频、13.433秒推理、24 kHz mono PCM16，证据在 `outputs/final-runtime-check/`。最终服务保持 `ready`，可直接调用。

证据：`outputs/real-voice-design/manifest.json`、`audio-report.json`、`index.html` 和 `outputs/warm-demo-reference/`。参数标签不代表主观符合度已验收；未进行盲听评级或语音识别准确率测试。

## 自动化与部署范围

最终自动化结果：`14 passed`（2条上游测试依赖弃用提醒）；Ruff 全部通过。包含阻塞推理期间请求停止、排空当前任务、queued 留存和恢复后重新 claim 的回归测试。源码包和轻量 wheel 构建成功，wheel 在 `uvx --from` 的独立环境执行版本命令成功，仅需安装一个客户端包。

测试覆盖：请求边界、鉴权、幂等、任务恢复、取消、成功任务音色快照；真实 TCP 客户端/服务分离、候选与恢复、导出、配方复用和拒绝覆盖。测试后端明确为 test，其音频不是语音。

Docker Compose 配置解析通过。容器镜像构建、容器 GPU 推理、独立远程 GPU 主机未运行；不将本机 TCP 通过当作这些部署验收。

运行时提示缺少 Flash Attention 和系统 SoX。此 VoiceDesign 直接生成流程已在该状态下成功，不需要调用 SoX；若扩展音频转换路径应另行验证。首次加载超过45秒，随后进入 ready；默认文档采用600秒启动等待。

## 能力边界

设计配方重新生成不能保证声纹固定；固定声音应使用选定参考音频交给克隆模型。支持语言列表来自官方模型，真实推理测试了中文和英文（其他语言未经本次音频验收）。运行中取消在同步推理返回后生效；没有逐 token ETA。幂等键属于同一服务数据目录，服务更换模型后旧键仍返回原结果；新一轮模型实验应使用新键。模型、SDK和硬件改变时种子不保证逐字节复现。
