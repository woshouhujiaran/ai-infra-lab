# 每日 AI Infra 组件实验（每天 14:00 由 Windows 计划任务触发）

你在 `ai-infra-lab` 仓库根目录工作。目标：今天完成 **1 个** 新的、可在本机复现的 AI infra 小组件实验，按步骤 commit，并推送到 `origin`（公开仓库 woshouhujiaran/ai-infra-lab）。

**每天只做 1 个实验。** 如果 registry 里今天的日期已经有一条记录，直接结束，什么都不做。

## 硬性约束

1. **只针对单个组件**：一个上游项目里的一个模块/算子/数据结构（KV cache、调度器、采样器、量化层、通信原语、tokenizer、kernel 等），不做端到端大模型训练/推理。
2. **只在本机运行**：RTX 4060 Laptop 8GB 或 CPU。禁止租卡、云 GPU、远程算力。GPU 实验必须先调用 `lab/gpu_guard.acquire()`；单次实验总运行时间 ≤ 30 分钟；模型/数据下载 ≤ 1GB（优先合成数据或小模型）。
3. **不影响其它仓库**：上游项目只能以 git submodule 形式加到 `upstream/<name>`，加入后立刻执行 `git -C upstream/<name> remote set-url --push origin DISABLED_read_only_upstream`。禁止向任何上游仓库 push、提 PR/issue/评论、在 GitHub 上 fork 或 star。不要修改 submodule 内的文件；需要改动时，把补丁/替代实现放到实验目录里。
4. **不重复**：先读 `experiments/registry.json`。新实验的 (upstream, component, topic) 不能与任何已有条目（包括 `failed`）重复；同一组件只有在研究问题明显不同的情况下才能再做。
5. 只 push 到本仓库的 `origin main`，不改写历史，不 force push。不提交 `.venv/`、`logs/`、模型权重或大于 5MB 的文件。

## 步骤

1. `git pull --ff-only origin main`，读 `README.md`、`experiments/registry.json`、`experiments/BACKLOG.md`。
2. 如果 `experiments/` 下有已存在但不在 registry 里的实验目录（例如只有 `run.py`、标着 `in progress`），今天就把它做完：沿用它的编号和目录，从下面第 4 步的“实际运行”开始。否则，从 BACKLOG 里挑第一个未完成且依赖可在 Windows 上安装的条目；都不合适时，自行提出一个满足约束的新实验，并把它加进 BACKLOG。
3. 新实验的编号 = registry 中最大 id 和已有实验目录编号中的较大者 + 1（4 位数字），目录为 `experiments/<id>-<slug>/`。
4. 按以下顺序分步 commit（每步一个 commit，message 用英文祈使句）：
   - 如果用到了新上游：添加 submodule 并禁用推送 → commit。
   - 写 `run.py`（固定随机种子，结果写入 `results.json` 和 `results.md`）→ commit。
   - 用 `.venv\Scripts\python.exe` 实际运行，修复问题直到跑通；把结果加上 `README.md`（问题、设置、结论、复现命令）→ commit。
   - 更新 `experiments/registry.json`（status `done`）和根目录 `README.md` 实验表、`BACKLOG.md` 勾选 → commit。
5. 需要新的 Python 依赖时，用 `.venv\Scripts\python.exe -m pip install --proxy http://127.0.0.1:7897 ...` 安装，并加到 `requirements.txt`。
6. `git push origin main`。
7. 如果 30 分钟内跑不通：在 registry 里记为 `failed` 并写上原因（下次不要再选同样的方案），把已有代码和失败说明 commit 并 push。

最后输出一段简短总结：今天的实验、关键结论、commit 列表。
