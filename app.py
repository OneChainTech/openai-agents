import streamlit as st
import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
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

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def run_with_retry(agent, query):
    return await Runner.run(agent, query)

async def process_query(user_query):
    async with MCPServerSse(
        name="Amap Maps Server",
        params={
            "url": "https://mcp-e7501f2d-826a-4be5.api-inference.modelscope.cn/sse",
            "timeout": 30  # 增加超时时间到30秒
        }
    ) as mcp_server:
        await mcp_server.connect()
        
        agent = Agent(
            name="Assistant",
            instructions="我是一个可以帮助查询地理信息的助手,我可以帮你查询天气、地址、路线等信息。",
            model=OpenAIChatCompletionsModel(
                model="deepseek-chat", 
                openai_client=get_openai_client()
            ),
            mcp_servers=[mcp_server],
            model_settings=ModelSettings(tool_choice="required")
        )
        
        try:
            result = await run_with_retry(agent, user_query)
            return result.final_output
        except Exception as e:
            st.error(f"查询失败，正在重试... 错误信息: {str(e)}")
            raise

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# 禁用跟踪
set_tracing_disabled(disabled=True)

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