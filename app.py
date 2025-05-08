import streamlit as st
import asyncio
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, trace # 添加 trace 导入
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings
# nest_asyncio.apply() # Temporarily comment out to check context var issues
from functools import partial
import backoff

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
        api_key="sk-43c7996606b341c790eb888fedaad049",
        timeout=60.0
    )

@st.cache_resource
def get_qwen_client():
    return AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-5a75c9c5920146488f518469d6dee532",
        timeout=60.0
    )

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def run_streamed_with_retry(agent, query):
    return Runner.run_streamed(agent, query)

async def generate_agent_response_stream(user_query_param: str):
    """
    Sets up the agent, connects to MCP, runs the query, and yields response chunks.
    This function is designed to be used with st.write_stream.
    """
    # 设置为 True 使用 Qwen API, 设置为 False 使用 DeepSeek API
    USE_QWEN_API = True 

    if USE_QWEN_API:
        selected_model_name = "qwen-turbo"
        selected_client = get_qwen_client()
        # UI updates like st.info should ideally be outside the generator if they need to appear before streaming.
        # However, for simplicity in this refactor, keeping some logic here.
        # Consider moving pre-stream UI updates to the button click handler.
    else:
        selected_model_name = "deepseek-chat"
        selected_client = get_openai_client()

    # Note: st.info calls within an async generator passed to st.write_stream might not behave as expected
    # regarding timing with the spinner. It's generally better to do initial UI updates before st.write_stream.
    # For this example, we'll proceed, but be mindful of this in complex apps.

    # Temporarily remove the trace context manager to isolate the ContextVar issue
    # with trace("Geographic Info Query Workflow", metadata={"user_query": user_query_param}):
    async with MCPServerSse(
        name="Amap Maps Server",
        params={
            "url": "https://mcp-e7501f2d-826a-4be5.api-inference.modelscope.cn/sse",
            "timeout": 60  # Increased from 30 to 60 seconds
        }
    ) as mcp_server:
        try:
            # Consider moving st.info/success messages for connection to before st.write_stream
            # await mcp_server.connect()

            connect_attempts = 3
            connect_success = False
            last_connect_error = None
            for attempt in range(connect_attempts):
                try:
                    # st.info(f"Attempting to connect to MCP server (attempt {attempt + 1}/{connect_attempts})...")
                    await mcp_server.connect()
                    # st.success("MCP Server connected successfully!")
                    connect_success = True
                    break # Exit loop on success
                except Exception as connect_err:
                    # st.warning(f"MCP connection attempt {attempt + 1} failed: {str(connect_err)}")
                    last_connect_error = connect_err
                    if attempt < connect_attempts - 1:
                        await asyncio.sleep(min(2 ** attempt, 10)) # Exponential backoff: 1, 2, 4, capped at 10s
                    else:
                        # st.error("Failed to connect to MCP server after multiple attempts.")
                        print(f"Failed to connect to MCP server after {connect_attempts} attempts. Last error: {last_connect_error}")
                        raise # Re-raise the last exception to be caught by the outer handler
            
            if not connect_success:
                # This path should ideally not be reached if the raise above works, but as a failsafe.
                # st.error("MCP Server connection failed definitively after retries.")
                # We re-raised in the loop, so this part might not be strictly necessary
                # depending on how st.write_stream handles exceptions from the generator.
                # For clarity, ensure an exception is raised if connection fails.
                if last_connect_error:
                    raise Exception(f"Failed to connect to MCP server after {connect_attempts} attempts.") from last_connect_error
                else:
                    raise Exception(f"Failed to connect to MCP server after {connect_attempts} attempts for an unknown reason.")

            agent = Agent(
                name="Assistant",
                instructions="我是一个可以帮助查询地理信息的助手,我可以帮你查询天气、地址、路线等信息。",
                model=OpenAIChatCompletionsModel(
                    model=selected_model_name, 
                    openai_client=selected_client
                ),
                mcp_servers=[mcp_server],
                model_settings=ModelSettings(tool_choice="required") 
                # Note: tool_choice="required" means the agent MUST use a tool.
                # Streaming text deltas (ResponseTextDeltaEvent) typically happens
                # when the LLM generates a direct text response or the text part of a response after tool use.
                # Ensure the agent's workflow and the events from Runner.run_streamed align with this expectation.
            )
            
            result_stream_handler = await run_streamed_with_retry(agent, user_query_param)
            
            async for event in result_stream_handler.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Based on user's example: event.data is ResponseTextDeltaEvent and has a 'delta' attribute
                    if hasattr(event.data, 'delta') and event.data.delta is not None:
                        yield event.data.delta
                # Other event types (tool_calls, tool_outputs, final_output_event) could be handled here
                # if more detailed streaming (e.g., "using tool X...") is needed.
                # For now, focusing on streaming the text delta as per the example.

        except Exception as e:
            # Handle exceptions that occur within the generator
            # This error might not be directly visible in st.error in the main UI easily
            # st.write_stream will raise the exception on the main thread.
            print(f"Error in stream generator: {str(e)}") # Log error
            # Re-raise to be caught by st.write_stream's error handling
            # or the try-except block in the button click handler
            raise 

# Removed run_async function as st.write_stream handles async generator execution.

# 禁用跟踪
set_tracing_disabled(disabled=True) # Disable tracing

# 创建输入框
user_query = st.text_input("请输入您的问题", placeholder="例如：北京的天气怎么样？或者 北京到上海怎么走？")

# 创建发送按钮
if st.button("发送", type="primary"):
    if user_query:
        # Display initial status messages before starting the stream
        st.info(f"收到问题：'{user_query}'. 开始处理...")
        if True: # Simplified from USE_QWEN_API for this example message
            st.info("当前使用 Qwen API")
        else:
            st.info("当前使用 DeepSeek API")
        st.info("准备连接到 MCP 服务器...")

        with st.spinner("正在思考中..."):
            try:
                # Directly use st.write_stream with the async generator function
                st.write("回答：") # Label for the answer area
                # Pass the user_query to the generator function
                st.write_stream(generate_agent_response_stream(user_query))
                st.success("查询完成！") # Appears after the stream finishes
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