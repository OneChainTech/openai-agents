# 智能地理信息助手

这是一个基于 Streamlit 构建的智能地理信息助手应用。它可以帮助用户查询天气信息、获取地址详情、规划旅行路线以及其他与地理相关的查询。

## 功能特性

- **多种地理信息查询**:
    - 天气查询
    - 地址详情
    - 旅行路线规划
- **集成大语言模型 (LLM)**:
    - 支持 DeepSeek API
    - 支持 阿里云通义千问 (Qwen) API (通过兼容 OpenAI 的模式)
    - 可以在代码中切换使用不同的 LLM。
- **通过 MCP (Model Calling Protocol) 与高德地图服务交互**:
    - 利用外部服务进行地理编码、路线规划等操作。
- **用户友好的 Web 界面**:
    - 使用 Streamlit 构建，界面简洁易用。
- **支持重试机制**:
    - 使用 `backoff` 库处理 API 调用可能发生的临时性错误。
- **（可选）追踪功能**:
    - 集成了 `openai-agents` SDK 的追踪功能，可以记录和分析 Agent 的运行流程。

## 技术栈

- Python
- Streamlit (Web 界面)
- OpenAI Python SDK (用于与 DeepSeek 和 Qwen API 交互)
- `openai-agents` SDK (核心 Agent 逻辑)
- `nest_asyncio` (支持 Streamlit 中的异步操作)
- `backoff` (指数退避重试)

## 安装与启动

### 1. 先决条件

- Python 3.8+
- pip (Python 包管理器)

### 2. 克隆或下载项目

如果项目在版本控制中，请克隆仓库。否则，请确保您拥有 `app.py` 文件和相关的 `agents` 库（如果它是自定义的本地库）。

### 3. 安装依赖

在项目根目录下，通过 pip 安装所需的 Python 包：

```bash
pip install streamlit openai backoff nest_asyncio
# 如果 'agents' 是一个公开发布的包，也请安装它：
# pip install agents-sdk # (假设包名为 agents-sdk，请根据实际情况修改)
# 如果 'agents' 是项目内的本地模块，请确保它在 PYTHONPATH 中或与 app.py 在同一级别或子目录中。

### 4. 配置 API 密钥
# app.py
@st.cache_resource
def get_openai_client():
    return AsyncOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key="YOUR_DEEPSEEK_API_KEY" # 替换为您的 DeepSeek API Key
    )

# app.py
@st.cache_resource
def get_qwen_client():
    return AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_QWEN_API_KEY" # 替换为您的 Qwen API Key
    )

### 5. 选择使用的大模型
# app.py
async def process_query(user_query):
    # 设置为 True 使用 Qwen API, 设置为 False 使用 DeepSeek API
    USE_QWEN_API = True # 或 False
    # ...

### 6. 运行应用
streamlit run app.py

### 7. （可选）追踪功能
export OPENAI_API_KEY="your_actual_openai_api_key"
streamlit run app.py

追踪功能日志 https://platform.openai.com/traces