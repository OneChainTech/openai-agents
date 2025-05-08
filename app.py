import streamlit as st
import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, trace # 添加 trace 导入
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings
import nest_asyncio
from functools import partial
import backoff

# 启用嵌套事件循环支持
nest_asyncio.apply()

# 设置页面配置
st.set_page_config(
    page_title="地理信息助手",
    page_icon="🗺️",
    layout="wide"
)

# 添加标题和说明
st.title("🗺️ 智能地理信息助手")
st.markdown("""
这是一个智能地理信息助手，可以帮助您：
- 查询天气信息
- 获取地址详情
- 规划旅行路线
- 更多地理相关查询...
""")

# 配置DeepSeek API
@st.cache_resource
def get_openai_client():
    return AsyncOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-43c7996606b341c790eb888fedaad049"
    )

@st.cache_resource
def get_qwen_client():
    return AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-5a75c9c5920146488f518469d6dee532"
    )

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def run_with_retry(agent, query):
    return await Runner.run(agent, query)

async def process_query(user_query):
    # 设置为 True 使用 Qwen API, 设置为 False 使用 DeepSeek API
    USE_QWEN_API = True 

    if USE_QWEN_API:
        selected_model_name = "qwen-turbo"  # 更改回 "qwen-turbo" 或尝试 "qwen-plus", "qwen-max"
        selected_client = get_qwen_client()
        st.info("当前使用 Qwen API (模型: " + selected_model_name + ")")
    else:
        selected_model_name = "deepseek-chat"
        selected_client = get_openai_client()
        st.info("当前使用 DeepSeek API (模型: " + selected_model_name + ")")

    st.info("准备连接到 MCP 服务器...")
    # 使用 trace 包裹核心处理流程
    with trace("Geographic Info Query Workflow", metadata={"user_query": user_query}):
        async with MCPServerSse(
            name="Amap Maps Server",
            params={
                "url": "https://mcp-e7501f2d-826a-4be5.api-inference.modelscope.cn/sse",
                "timeout": 30  # 增加超时时间到30秒
            }
        ) as mcp_server:
            try:
                st.info("正在连接 MCP 服务器...")
                await mcp_server.connect()
                st.success("MCP 服务器连接成功！")
                
                agent = Agent(
                    name="Assistant",
                    instructions="我是一个可以帮助查询地理信息的助手,我可以帮你查询天气、地址、路线等信息。",
                    model=OpenAIChatCompletionsModel(
                        model=selected_model_name, 
                        openai_client=selected_client
                    ),
                    mcp_servers=[mcp_server],
                    model_settings=ModelSettings(tool_choice="required")
                )
                
                st.info("Agent 已配置，准备通过 MCP 服务器处理查询...")
                result = await run_with_retry(agent, user_query)
                return result.final_output
            except Exception as e:
                st.error(f"MCP 处理或 Agent 运行失败：{str(e)}")
                # 确保在 MCPServerSse 的 __aexit__ 被调用前抛出异常，以便正确关闭
                raise

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# 禁用跟踪
set_tracing_disabled(disabled=False)

# 创建输入框
user_query = st.text_input("请输入您的问题", placeholder="例如：北京的天气怎么样？或者 北京到上海怎么走？")

# 创建发送按钮
if st.button("发送", type="primary"):
    if user_query:
        # 显示加载动画
        with st.spinner("正在思考中..."):
            try:
                # 运行查询
                result = run_async(process_query(user_query))
                # 显示结果
                st.success("查询完成！")
                st.write("回答：", result)
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
    else:
        st.warning("请输入问题后再发送")

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 如何使用
    1. 在输入框中输入您的问题
    2. 点击"发送"按钮
    3. 等待系统响应
    
    ### 示例问题
    - 北京今天的天气怎么样？
    - 北京到上海的高铁路线
    - 杭州西湖附近的景点
    - 上海东方明珠的具体地址
    """)

# 添加页脚
st.markdown("---")
st.markdown("Made with ❤️ by Your Team")