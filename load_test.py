import asyncio
import httpx

API_URL = "http://127.0.0.1:8000"
AGENT_ID = "test_agent"

async def create_agent():
    async with httpx.AsyncClient() as client:
        await client.post(f"{API_URL}/agents/", json={"agent_id": AGENT_ID})

async def ask_question():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/ask/", json={"agent_id": AGENT_ID, "prompt": "Tell me a joke."})
        print(response.json())

async def load_test():
    await create_agent()
    tasks = [ask_question() for _ in range(15)]  # Exceed rate limit
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(load_test())