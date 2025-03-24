from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import time
from collections import deque, defaultdict
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY  # Import API key from config.py

# Initialize FastAPI
app = FastAPI()


class LLMClient:
    """Handles communication with LLM backends, agent-wise rate limiting, cost tracking, and logging"""

    cost_per_1000_tokens = {"openai": 0.002, "langchain": 0.002}  # Example cost

    def __init__(self, backend="openai", api_key=None, rate_limit=10):
        self.backend = backend
        self.api_key = api_key or OPENAI_API_KEY
        self.rate_limit = rate_limit  # Max requests per minute per agent
        self.requests_log = defaultdict(deque)  # Track request timestamps per agent
        self.prompts_log = defaultdict(list)  # Store prompts per agent
        self.total_costs = defaultdict(float)  # Track cost per agent

        if backend == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
        elif backend == "langchain":
            self.client = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.api_key)

    def enforce_rate_limit(self, agent_id):
        """Ensures each agent does not exceed their rate limit"""
        current_time = time.time()
        while self.requests_log[agent_id] and self.requests_log[agent_id][0] < current_time - 60:
            self.requests_log[agent_id].popleft()
        if len(self.requests_log[agent_id]) >= self.rate_limit:
            raise Exception(f"Rate limit exceeded for agent {agent_id}")
        self.requests_log[agent_id].append(current_time)

    def request(self, agent_id, prompt):
        """Handles request-response and tracks costs and logs per agent"""
        self.enforce_rate_limit(agent_id)  # Enforce agent-wise rate limit
        self.prompts_log[agent_id].append(prompt)

        if self.backend == "openai":
            response, tokens_used = self._request_openai(prompt)
        elif self.backend == "langchain":
            response, tokens_used = self._request_langchain(prompt)
        else:
            raise ValueError("Unsupported backend")

        cost = (tokens_used / 1000) * self.cost_per_1000_tokens[self.backend]
        self.total_costs[agent_id] += cost

        return response

    def _request_openai(self, prompt):
        """Handles OpenAI API request"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            tokens_used = response.usage.total_tokens
            return response.choices[0].message.content, tokens_used
        except openai.RateLimitError:
            print("Rate limit exceeded. Retrying after 10 seconds...")
            time.sleep(10)
            return self._request_openai(prompt)  # Retry request

    def _request_langchain(self, prompt):
        """Handles LangChain API request"""
        response = self.client.invoke(prompt)
        return response.content, len(prompt.split())  # Approximate token count

    def get_cost(self, agent_id):
        """Returns total cost for a specific agent"""
        return self.total_costs[agent_id]

    def get_all_prompts(self, agent_id):
        """Returns all prompts submitted by a specific agent"""
        return self.prompts_log[agent_id]


class Agent:
    """Thin wrapper that just calls LLMClient"""

    def __init__(self, name, llm_client):
        self.name = name
        self.llm_client = llm_client

    def ask(self, prompt):
        """Just calls the LLM client"""
        return self.llm_client.request(self.name, prompt)


# Initialize LLM Client (shared across all agents)
api_key = OPENAI_API_KEY
llm_client = LLMClient(backend="langchain", api_key=api_key, rate_limit=10)

# Dictionary to store multiple agents
agents = {}


# Request Models
class AgentCreateRequest(BaseModel):
    agent_id: str


class PromptRequest(BaseModel):
    agent_id: str
    prompt: str


# API Endpoints

@app.post("/agents/")
def create_agent(request: AgentCreateRequest):
    """Dynamically create a new agent"""
    if request.agent_id in agents:
        raise HTTPException(status_code=400, detail="Agent already exists")

    agents[request.agent_id] = Agent(name=request.agent_id, llm_client=llm_client)
    return {"message": f"Agent {request.agent_id} created successfully"}


@app.post("/ask/")
def ask_question(request: PromptRequest):
    """Process a prompt for a specific agent"""
    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        response = agents[request.agent_id].ask(request.prompt)
        return {"agent_id": request.agent_id, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cost/{agent_id}")
def get_cost(agent_id: str):
    """Retrieve total cost for a specific agent"""
    return {"agent_id": agent_id, "total_cost": llm_client.get_cost(agent_id)}


@app.get("/prompts/{agent_id}")
def get_all_prompts(agent_id: str):
    """Retrieve all prompts submitted by a specific agent"""
    return {"agent_id": agent_id, "prompts": llm_client.get_all_prompts(agent_id)}


@app.get("/agents/")
def list_agents():
    """List all registered agents"""
    return {"agents": list(agents.keys())}


# Run using: uvicorn filename:app --host 0.0.0.0 --port 8000