<h1 align="center">
  <a href="https://github.com/miaomiao-star/support-assistant">Support Assistant</a>
</h1>

Support Assistant 是一个采用**工作流 + LLM Agent 混合架构**构建的电商智能客服助手，围绕购物咨询、售后服务和业务协助等场景，逐步探索大模型在客服系统中的安全应用。

## Get Start

```bash
git clone https://github.com/miaomiao-star/support-assistant.git
cd support-assistant

uv init --bare --python 3.12 --vcs none
uv add fastapi uvicorn pydantic-settings langchain langchain-openai
uv add --dev pytest requests

# Copy the configuration template and fill in your model credentials.
cp .env.example .env

uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```
