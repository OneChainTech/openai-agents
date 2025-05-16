import streamlit as st
import asyncio
from openai import AsyncOpenAI
from datetime import timedelta
from openai.types.responses import ResponseTextDeltaEvent # Added
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, ModelSettings # Added Agent, Runner, ModelSettings
from agents.mcp import MCPServerStreamableHttp
# nest_asyncio.apply()
from functools import partial
import backoff

# 设置页面配置
st.set_page_config(
    page_title="MCP 智能助手", # Changed title
    page_icon="🤖",       # Changed icon
    layout="wide"
)

# 添加标题和说明
st.title("🤖 MCP 智能助手") # Changed title
st.markdown("""
本助手可以连接到指定的 MCP 服务器，查询其可用工具，并允许您通过AI智能体与之交互来解决问题。
""")

# 配置API客户端 (保留)
@st.cache_resource
def get_openai_client():
    return AsyncOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-43c7996606b341c790eb888fedaad049", # 请替换为您的有效API密钥
        timeout=60.0
    )

@st.cache_resource
def get_qwen_client():
    return AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-5a75c9c5920146488f518469d6dee532", # 请替换为您的有效API密钥
        timeout=60.0
    )

@st.cache_resource
def get_siliconflow_client():
    return AsyncOpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-owuhchfalamktrcqsbhhuuxcgfdwcwekrrggmpoomggifotn", # 请替换为您的有效API密钥
        timeout=60.0
    )

# MCP 工具查询函数 (保留)
async def fetch_mcp_tools(mcp_url: str):
    """
    Connects to the MCP server and lists available tools.
    """
    async with MCPServerStreamableHttp(
        params={
            "url": mcp_url,
            "timeout": timedelta(seconds=60),
            "sse_read_timeout": timedelta(seconds=300)
        }
    ) as mcp_server:
        await mcp_server.connect()
        tools = await mcp_server.list_tools()
        return tools

# 智能体响应生成器
async def generate_agent_response_stream(
    user_query_param: str,
    selected_api_provider_param: str,
    mcp_url_param: str
):
    """
    Sets up the agent, connects to MCP, runs the query, and yields response chunks.
    """
    # 1. Select LLM client
    if selected_api_provider_param == "Qwen":
        selected_model_name = "qwen-turbo"
        selected_client = get_qwen_client()
    elif selected_api_provider_param == "DeepSeek":
        selected_model_name = "deepseek-chat"
        selected_client = get_openai_client()
    elif selected_api_provider_param == "SiliconFlow":
        selected_model_name = "Qwen/Qwen3-235B-A22B"
        selected_client = get_siliconflow_client()
    else:
        yield f"错误：未知的API服务商 '{selected_api_provider_param}'. 将默认使用DeepSeek。\n"
        selected_model_name = "deepseek-chat"
        selected_client = get_openai_client()

    # 2. Set up MCP Server
    try:
        async with MCPServerStreamableHttp(
            params={
                "url": mcp_url_param,
                "timeout": timedelta(seconds=300),
                "sse_read_timeout": timedelta(seconds=300)
            }
        ) as mcp_server_instance:
            connect_attempts = 3
            for attempt in range(connect_attempts):
                try:
                    yield f"正在连接 MCP 服务器 (尝试 {attempt + 1}/{connect_attempts})...\n"
                    await mcp_server_instance.connect()
                    yield "MCP 服务器连接成功!\n"
                    break
                except Exception as connect_err:
                    yield f"MCP 连接尝试 {attempt + 1} 失败: {connect_err}\n"
                    if attempt < connect_attempts - 1:
                        await asyncio.sleep(min(2 ** attempt, 5))
                    else:
                        yield f"连接 MCP 服务器 {mcp_url_param} 失败 {connect_attempts} 次。请检查服务状态和地址。\n"
                        return

            agent = Agent(
                name="MCPToolAgent",
                instructions="你是一个乐于助人的AI助手。请使用你可用的工具来回答用户的问题。",
                model=OpenAIChatCompletionsModel(
                    model=selected_model_name,
                    openai_client=selected_client
                ),
                mcp_servers=[mcp_server_instance],
                model_settings=ModelSettings(tool_choice="auto")
            )

            # 4. Run agent and stream results
            yield "智能体正在处理请求...\n"
            result_stream_handler = Runner.run_streamed(agent, user_query_param)
            
            async for event in result_stream_handler.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    if hasattr(event.data, 'delta') and event.data.delta is not None:
                        yield event.data.delta
                elif event.type == "tool_calls":
                    for tool_call in event.data.tool_calls:
                         yield f"\n正在调用工具: `{tool_call.name}` (参数: `{tool_call.arguments}`)\n"
                elif event.type == "tool_outputs":
                     for tool_output in event.data.outputs:
                         yield f"\n工具 `{tool_output.name}` 返回: `{tool_output.output}`\n"
                elif event.type == "final_output_event":
                    if event.data.final_output:
                         yield f"\n最终输出: {event.data.final_output}\n"
    except Exception as e:
        yield f"\n处理请求时发生错误: {str(e)}\n"
        import traceback
        yield traceback.format_exc()

# 禁用跟踪
set_tracing_disabled(disabled=True)

# --- UI Layout ---
with st.sidebar:
    st.header("⚙️ 配置")
    mcp_server_url = st.text_input("MCP 服务器地址", value="http://0.0.0.0:3000/mcp")
    model_options = ["Qwen", "DeepSeek", "SiliconFlow"]
    selected_model_provider = st.selectbox("选择大模型服务商:", model_options, index=0)

    if st.button("查询 MCP 工具列表"):
        if mcp_server_url:
            with st.spinner(f"正在查询 MCP 服务器 {mcp_server_url} 上的工具..."):
                try:
                    tools = asyncio.run(fetch_mcp_tools(mcp_server_url))
                    if tools:
                        st.success("MCP 可用工具:")
                        for tool in tools:
                            tool_info = f"- **{tool.name}**"
                            if hasattr(tool, 'description') and tool.description:
                                tool_info += f": {tool.description}"
                            st.markdown(tool_info)
                            if hasattr(tool, 'parameters') and tool.parameters:
                                with st.expander("参数详情"):
                                    st.json(tool.parameters)
                    else:
                        st.warning("未查询到任何工具。请检查服务器地址或日志。")
                except Exception as e:
                    st.error(f"查询 MCP 工具失败: {e}")
                    st.info("请确保 MCP 服务器已启动，地址正确，且实现了 Streamable HTTP 协议。")
        else:
            st.warning("请输入 MCP 服务器地址。")
    
    with st.expander("使用说明"):
        st.markdown("""
        ### 如何使用
        1.  在侧边栏配置 **MCP 服务器地址**。
        2.  (可选) 点击 **查询 MCP 工具列表** 来查看服务器提供的工具。
        3.  在侧边栏选择 **大模型服务商**。
        4.  在主聊天界面输入您的问题，智能体将尝试使用 MCP 工具来回答。
        
        ### 注意事项
        -   确保您的 API 密钥已在代码中正确配置 (get_openai_client, get_qwen_client, get_siliconflow_client)。
        -   MCP 服务器必须正在运行，并且可以从运行此助手的地方访问。
        -   MCP 服务器必须实现 Streamable HTTP transport 协议。
        """)

st.header("💬 与智能体对话")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("请输入您的问题，智能体将尝试使用 MCP 工具..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            stream_generator = generate_agent_response_stream(
                user_query,
                selected_model_provider,
                mcp_server_url
            )
            full_response = st.write_stream(stream_generator)
            if full_response:
                 st.session_state.messages.append({"role": "assistant", "content": full_response})
            # If full_response is None (e.g. generator yielded nothing), we don't add empty message.

        except Exception as e:
            error_message = f"处理您的请求时出错: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

st.markdown("---")
st.markdown("Made with ❤️")