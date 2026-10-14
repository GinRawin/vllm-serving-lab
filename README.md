# Qwen3 本地服务

当前模型：`Qwen/Qwen3-0.6B`。启动：

```bash
bash scripts/start-qwen.sh
```

服务地址为 `http://127.0.0.1:8000`，上下文长度 4096，GPU 显存使用比例 0.8。
脚本使用项目中的 `.venv/bin/vllm`；额外启动参数可追加在脚本命令后。

验证健康检查、模型列表、贪心生成及 top-k/top-p 随机采样：

```bash
.venv/bin/python scripts/smoke-test.py
```

## 启动故障修复

首次报错 `Failed to find C compiler`，通过在 Ubuntu 环境安装
`build-essential` 补齐 GCC/G++。

随后报错 `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。
该错误来自 FlashInfer 采样模块的 CUDA JIT 编译。启动脚本设置
`VLLM_USE_FLASHINFER_SAMPLER=0`，启用 vLLM 支持的备用采样路径。
FlashAttention 和 torch.compile/CUDA Graph 配置仍使用默认设置。

本次调试启动的后台服务日志位于 `logs/qwen-server.log`，API Server PID
记录在 `logs/qwen-server.pid`。服务运行时直接执行验证命令即可；再次启动前先停止原服务。

查看日志：

```bash
tail -f logs/qwen-server.log
```

停止本次后台服务：

```bash
kill "$(cat logs/qwen-server.pid)"
```

修改采样后端可能影响吞吐量；此配置用于让当前没有 CUDA Toolkit 的环境正常运行，
未进行性能基准测试。
