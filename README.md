# support-assistant

Python 3.12 + FastAPI + LangChain 的单进程电商客服：SSE 多轮对话与独立售后结构化提取。
包含浏览器聊天页面；本章不含工具调用或 Agent 循环。售后提取为独立接口，聊天页面尚未调用它。

## 启动

需要 Python 3.12 和 uv。仓库按维护者约定不提交 `pyproject.toml`、`uv.lock`、`.python-version` 及初始化示例包，因此不能在全新克隆后直接运行 `uv sync`。

仅在全新克隆、尚无 `pyproject.toml` 时，在项目根目录创建本地配置：

```powershell
uv init --bare --python 3.12 --vcs none
uv add fastapi uvicorn pydantic-settings langchain langchain-openai
uv add --dev pytest requests
```

本地生成的配置和锁文件会被忽略。这种方式解析当前可用版本，不保证重现原开发环境的精确依赖版本。已有本地配置时不要重复初始化，直接执行 `uv sync`。

创建 `.env`：已有文件时保留原配置；没有文件时复制 `.env.example`，填入真实模型名、上游地址及密钥。然后启动：

```powershell
uv sync
.\scripts\dev.ps1
```

Git Bash / Linux：

```bash
bash scripts/dev.sh
```

聊天页面位于 `http://127.0.0.1:8000/`，接口文档位于 `/docs`。保持同一页面可连续追问，点击“新对话”会更换会话 ID；页面刷新也会开启新会话。服务没有自动重载，修改代码或配置后需要重启。
现有 `.env` 已迁移为 `CHAT_API_KEY` / `CHAT_BASE_URL` / `CHAT_MODEL`，保留原有配置值。
新环境可复制 `.env.example` 后填写。`CHAT_BASE_URL` 必须包含上游要求的 API 路径，应用不自动追加 `/v1`。
`EMBED_*` 和 `RERANK_*` 本章不使用，允许为空。修改 `.env` 后重启服务。
`.env` 已被 Git 忽略；不要在日志或截图中展示密钥。

## curl 演示（Windows PowerShell）

另开终端，进入项目根目录，按顺序执行。使用 UTF-8 请求文件避免 Windows 命令行 JSON 转义问题。

```powershell
curl.exe -N -sS -H "Content-Type: application/json" --data-binary "@scripts/examples/chat-first.json" http://127.0.0.1:8000/api/chat
curl.exe -N -sS -H "Content-Type: application/json" --data-binary "@scripts/examples/chat-second.json" http://127.0.0.1:8000/api/chat
curl.exe -sS -H "Content-Type: application/json" --data-binary "@scripts/examples/extract.json" http://127.0.0.1:8000/api/extract
```

Git Bash 使用同样参数，将 `curl.exe` 改为 `curl`。

对话请求为 `{"session_id":"abc","message":"你好"}`。
每个文本增量按 `data: {"delta":"..."}` 加空行分帧，成功以 `data: [DONE]` 结束。
增量粒度由上游决定，一个 chunk 可能包含多个 token 或字符。
上游失败发送 `event: error` 后关闭，不保存不完整轮次。

提取请求为 `{"text":"商品坏了，我要退款"}`，响应包含 `order_id`、`request_type`、`expected_solution`。
缺少订单号返回 JSON `null`；上游或解析失败返回 502。
空白输入和超出对话输入预算的消息返回 422。

## 验证

```powershell
uv run python -m pytest tests -q
# 先启动服务；此命令调用真实模型，会产生上游调用费用。
uv run python scripts/evaluate.py
```

单测采用 FakeLLM，不需要密钥或真实模型。
评估脚本包含五条标注售后样例和两轮对话；报告写入 `dev-notes/ch01-evaluation.json`。
结构化订单号和诉求类型精确匹配，方案通过关键字检查并结合人工语义核对。
评估报告保留在本地，不提交 Git。运行评估前需确认 `tests/fixtures/extract_cases.json` 存在；若仓库尚在分批提交中，应先补齐该样例文件再运行。

## 上下文边界

`LLM_TOKEN_BUDGET=2000` 限制输入上下文：system + 历史 + 当前消息。
使用 `ZH_CHARS_PRE_TOKEN=1.2` 与消息开销估算 token，不能保证真实模型 token 数严格小于该值；它不是输出 token 上限。
优先保留最近完整问答轮次；当前消息不静默裁掉。
特别长的完整回复若无法放入历史预算，该问答轮次不会保留。
服务为单进程内存会话，重启丢失历史，不使用多个 worker；本章未实现会话过期、鉴权或限流。

## 开发记录

设计、计划、开发记录和会话交接资料由维护者保存在本地 `specs/`、`dev-notes/`，不随公开仓库发布。

## 第一次阅读代码的建议顺序

1. `app/main.py`：应用如何启动、路由如何注册、首页如何挂载。
2. `app/config.py` → `app/schemas/`：配置和请求数据如何转换成有类型的对象。
3. `app/core/prompts.py` → `app/core/llm.py`：提示词如何和模型连接，结构化结果如何校验。
4. `app/core/memory.py`：消息角色、预算估算、完整轮次裁剪与会话锁。
5. `app/api/chat.py`：从请求到 SSE、流取消、失败回滚及历史保存。
6. `app/api/extract.py`：非流式结构化接口及基础事实核验。
7. `app/static/index.html`：POST 流读取、SSE 分帧、逐字队列与多轮 session_id。
8. `tests/` 和 `scripts/evaluate.py`：FakeLLM 回归测试与真实模型评估的区别。

生产化时应分别考虑会话归属与鉴权、存储和过期、跨进程协调、限流、输入输出限制、监控以及更全面的评估。当前注释标明边界，并不表示这些功能已经实现。
