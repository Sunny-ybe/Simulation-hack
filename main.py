import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Literal

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini API key
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
MODELS = [
    {
        "id": "gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash",
        "input_price_per_million": 0.10,
        "output_price_per_million": 0.40,
    },
    {
        "id": "gemini-2.0-flash-lite",
        "display_name": "Gemini 2.0 Flash-Lite",
        "input_price_per_million": 0.075,
        "output_price_per_million": 0.30,
    },
    {
        "id": "gemini-1.5-flash",
        "display_name": "Gemini 1.5 Flash",
        "input_price_per_million": 0.075,
        "output_price_per_million": 0.30,
    },
    {
        "id": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro",
        "input_price_per_million": 1.25,
        "output_price_per_million": 5.00,
    },
    {
        "id": "gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "input_price_per_million": 1.25,
        "output_price_per_million": 10.00,
    },
]
VALID_MODEL_IDS = {m["id"] for m in MODELS}

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "you are now",
    "disregard your",
    "forget your instructions",
    "new persona",
    "act as",
    "pretend you are",
    "system prompt",
    "override your",
    "your instructions are",
]


def contains_injection(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
RATE_LIMIT_WINDOW = 600  # 10 minutes
RATE_LIMIT_MAX = 5

rate_limit_store: dict[str, list[float]] = defaultdict(list)
rate_limit_lock = asyncio.Lock()


async def check_rate_limit(ip: str) -> bool:
    now = time.time()
    async with rate_limit_lock:
        timestamps = [t for t in rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
        rate_limit_store[ip] = timestamps
        if len(timestamps) >= RATE_LIMIT_MAX:
            return False
        rate_limit_store[ip].append(now)
        return True


def get_client_ip(request: Request) -> str:
    trust_proxy = os.environ.get("TRUST_PROXY_IP", "false").lower() == "true"
    if trust_proxy:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    name: str
    system_prompt: str


@dataclass
class FeedEntry:
    agent_name: str
    round_number: int
    content: str
    timestamp: float
    is_user_injection: bool = False


@dataclass
class SimulationState:
    simulation_id: str
    creator_ip: str
    status: Literal["running", "complete", "error"]
    model_id: str
    agents: list[AgentConfig]
    total_rounds: int
    current_round: int
    feed: list[FeedEntry]
    article: str  # NEVER logged
    error_message: str | None
    created_at: float
    completed_at: float | None
    context_truncated: bool = False


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------
simulations: dict[str, SimulationState] = {}


def cleanup_old_simulations() -> None:
    cutoff = time.time() - 7200  # 2 hours
    stale = [sid for sid, s in simulations.items() if s.created_at < cutoff]
    for sid in stale:
        del simulations[sid]


# ---------------------------------------------------------------------------
# Feed context helpers
# ---------------------------------------------------------------------------
MAX_FEED_CONTEXT = 30


def get_feed_for_context(feed: list[FeedEntry]) -> tuple[list[FeedEntry], bool]:
    if len(feed) <= MAX_FEED_CONTEXT:
        return feed, False
    return feed[-MAX_FEED_CONTEXT:], True


def build_agent_prompt(article: str, feed: list[FeedEntry], agent_name: str) -> str:
    lines = [f"TOPIC / ARTICLE:\n{article}\n", "---", "DISCUSSION SO FAR (oldest to newest):"]
    if not feed:
        lines.append("(You are the first to speak. No prior discussion.)")
    else:
        for entry in feed:
            if entry.is_user_injection:
                lines.append(
                    f"[Round {entry.round_number}] [You] (direct question): {entry.content}"
                )
            else:
                lines.append(
                    f"[Round {entry.round_number}] {entry.agent_name}: {entry.content}"
                )
    lines += [
        "---",
        f"You are {agent_name}. Respond in character. Keep your response under 80 words. "
        "React to what others have said if relevant, especially any direct questions from [You].",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------
async def call_gemini(model_id: str, system_prompt: str, user_message: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=system_prompt,
    )
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                user_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.9,
                ),
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini API call timed out after 30 seconds.")

    text = getattr(response, "text", None)
    return text.strip() if text else "[No response generated]"


# ---------------------------------------------------------------------------
# Background simulation runner
# ---------------------------------------------------------------------------
async def run_simulation(sim_id: str) -> None:
    sim = simulations.get(sim_id)
    if sim is None:
        return

    try:
        for round_num in range(1, sim.total_rounds + 1):
            sim.current_round = round_num
            for agent in sim.agents:
                feed_slice, truncated = get_feed_for_context(sim.feed)
                if truncated:
                    sim.context_truncated = True
                prompt = build_agent_prompt(sim.article, feed_slice, agent.name)
                content = await call_gemini(sim.model_id, agent.system_prompt, prompt)
                sim.feed.append(
                    FeedEntry(
                        agent_name=agent.name,
                        round_number=round_num,
                        content=content,
                        timestamp=time.time(),
                    )
                )
        sim.status = "complete"
        sim.completed_at = time.time()
    except Exception as exc:
        sim.status = "error"
        sim.error_message = str(exc)
        logger.error("Simulation %s failed: %s", sim_id, exc)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    name: str
    system_prompt: str


class SimulateRequest(BaseModel):
    article: str
    model_id: str
    agents: list[AgentRequest]
    rounds: int


class InjectRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Multi-Agent Opinion Simulator")

allowed_origin = os.environ.get("ALLOWED_ORIGIN", "http://localhost:8000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/models")
async def get_models():
    return MODELS


@app.post("/simulate")
async def start_simulation(
    payload: SimulateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    ip = get_client_ip(request)

    # Rate limit
    if not await check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. You can start at most 5 simulations per 10 minutes.",
        )

    # Validate article
    if not payload.article.strip():
        raise HTTPException(status_code=422, detail="Article must not be empty.")
    if len(payload.article) > 5000:
        raise HTTPException(
            status_code=422,
            detail=f"Article exceeds 5000 character limit (got {len(payload.article)}).",
        )

    # Validate model
    if payload.model_id not in VALID_MODEL_IDS:
        raise HTTPException(status_code=422, detail=f"Unknown model: {payload.model_id!r}.")

    # Validate agents
    if len(payload.agents) < 2 or len(payload.agents) > 50:
        raise HTTPException(
            status_code=422, detail="Agent count must be between 2 and 50."
        )
    names_seen: set[str] = set()
    for i, agent in enumerate(payload.agents):
        if not agent.name.strip():
            raise HTTPException(
                status_code=422, detail=f"Agent {i + 1} has an empty name."
            )
        if len(agent.name) > 50:
            raise HTTPException(
                status_code=422,
                detail=f"Agent {i + 1} name exceeds 50 characters.",
            )
        if agent.name in names_seen:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate agent name: {agent.name!r}. All agent names must be unique.",
            )
        names_seen.add(agent.name)
        if len(agent.system_prompt) > 500:
            raise HTTPException(
                status_code=422,
                detail=f"Agent {agent.name!r} system prompt exceeds 500 characters.",
            )

    # Validate rounds
    if payload.rounds < 1 or payload.rounds > 20:
        raise HTTPException(
            status_code=422, detail="Rounds must be between 1 and 20."
        )

    # Prompt injection check
    if contains_injection(payload.article):
        logger.warning(
            "Prompt injection attempt detected: ip_hash=%s field=article",
            hash_ip(ip),
        )
        raise HTTPException(
            status_code=400, detail="Input contains disallowed content."
        )
    for i, agent in enumerate(payload.agents):
        if contains_injection(agent.system_prompt):
            logger.warning(
                "Prompt injection attempt detected: ip_hash=%s field=agent[%d].system_prompt",
                hash_ip(ip),
                i,
            )
            raise HTTPException(
                status_code=400, detail="Input contains disallowed content."
            )

    # Cleanup stale simulations
    cleanup_old_simulations()

    # Create simulation
    sim_id = uuid.uuid4().hex
    sim = SimulationState(
        simulation_id=sim_id,
        creator_ip=ip,
        status="running",
        model_id=payload.model_id,
        agents=[AgentConfig(name=a.name, system_prompt=a.system_prompt) for a in payload.agents],
        total_rounds=payload.rounds,
        current_round=0,
        feed=[],
        article=payload.article,
        error_message=None,
        created_at=time.time(),
        completed_at=None,
    )
    simulations[sim_id] = sim

    logger.info(
        "Simulation started: id=%s model=%s agents=%d rounds=%d ip_hash=%s",
        sim_id,
        payload.model_id,
        len(payload.agents),
        payload.rounds,
        hash_ip(ip),
    )

    background_tasks.add_task(run_simulation, sim_id)
    return {"simulation_id": sim_id}


@app.get("/status/{simulation_id}")
async def get_status(simulation_id: str, request: Request):
    ip = get_client_ip(request)
    sim = simulations.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    if sim.creator_ip != ip:
        raise HTTPException(status_code=403, detail="Access denied.")

    feed_data = []
    for entry in sim.feed:
        feed_data.append(
            {
                "agent_name": entry.agent_name,
                "round_number": entry.round_number,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "is_user_injection": entry.is_user_injection,
            }
        )

    return {
        "status": sim.status,
        "current_round": sim.current_round,
        "total_rounds": sim.total_rounds,
        "feed": feed_data,
        "context_truncated": sim.context_truncated,
        "error_message": sim.error_message,
    }


@app.post("/inject/{simulation_id}")
async def inject_message(simulation_id: str, payload: InjectRequest, request: Request):
    ip = get_client_ip(request)
    sim = simulations.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    if sim.creator_ip != ip:
        raise HTTPException(status_code=403, detail="Access denied.")
    if sim.status != "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot inject into a simulation that is not running.",
        )

    if not payload.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty.")
    if len(payload.message) > 500:
        raise HTTPException(
            status_code=422,
            detail=f"Message exceeds 500 character limit (got {len(payload.message)}).",
        )
    if contains_injection(payload.message):
        logger.warning(
            "Prompt injection attempt in inject: ip_hash=%s", hash_ip(ip)
        )
        raise HTTPException(status_code=400, detail="Input contains disallowed content.")

    sim.feed.append(
        FeedEntry(
            agent_name="[You]",
            round_number=sim.current_round,
            content=payload.message.strip(),
            timestamp=time.time(),
            is_user_injection=True,
        )
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
