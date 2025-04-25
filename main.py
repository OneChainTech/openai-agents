import asyncio
import os
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings

# 配置DeepSeek API
client = AsyncOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-43c7996606b341c790eb888fedaad049"
)

# 禁用跟踪,因为我们使用自定义API
set_tracing_disabled(disabled=True)

async def main():
    # 配置高德地图MCP服务器
    async with MCPServerSse(
        name="Amap Maps Server",
        params={
            "url": "https://mcp-e7501f2d-826a-4be5.api-inference.modelscope.cn/sse"
        }
    ) as mcp_server:
        # 创建智能体
        agent = Agent(
            name="Assistant",
            instructions="我是一个可以帮助查询地理信息的助手,我可以帮你查询天气、地址、路线等信息。",
            model=OpenAIChatCompletionsModel(
                model="deepseek-chat", 
                openai_client=client
            ),
            mcp_servers=[mcp_server],
            model_settings=ModelSettings(tool_choice="required")
        )

        # 测试查询
        queries = [
            "北京的天气怎么样?",
            "北京到昆明的旅行规划?"
        ]

        for query in queries:
            print(f"\n执行查询: {query}")
            try:
                result = await Runner.run(agent, query)
                print(f"回答: {result.final_output}")
            except Exception as e:
                print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 