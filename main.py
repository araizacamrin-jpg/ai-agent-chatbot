import os
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Service Chatbot API",
    description="AI-powered customer service chatbot backed by Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

# In-memory stats store (replace with Supabase/DB for persistence across restarts)
stats_store: dict[str, dict] = defaultdict(lambda: {
    "total_questions": 0,
    "total_tokens_used": 0,
    "last_activity": None,
    "conversations": [],
})


# --------------------------------------------------------------------------- #
# Request / Response schemas
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    business_id: str
    question: str
    knowledge_base: str


class ChatResponse(BaseModel):
    business_id: str
    question: str
    answer: str
    tokens_used: int
    response_time_ms: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str


class StatsResponse(BaseModel):
    business_id: str
    total_questions: int
    total_tokens_used: int
    last_activity: Optional[str]
    recent_conversations: list


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _call_claude(system_prompt: str, user_message: str) -> tuple[str, int]:
    """Call the Claude API and return (answer_text, tokens_used)."""
    if not CLAUDE_API_KEY:
        raise HTTPException(status_code=500, detail="CLAUDE_API_KEY is not configured")

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Claude API request timed out")
        raise HTTPException(status_code=504, detail="Claude API timed out")
    except requests.exceptions.RequestException as exc:
        logger.error("Claude API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc}")

    data = response.json()
    answer = data["content"][0]["text"]
    tokens_used = data.get("usage", {}).get("output_tokens", 0)
    return answer, tokens_used


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a customer question using the provided knowledge base."""
    logger.info("Chat request — business_id=%s question=%r", request.business_id, request.question[:80])

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not request.knowledge_base.strip():
        raise HTTPException(status_code=400, detail="knowledge_base cannot be empty")

    system_prompt = (
        "You are a helpful customer service assistant. "
        "Answer questions using only the knowledge base provided below. "
        "If the answer is not in the knowledge base, politely say you don't have that information "
        "and suggest the customer contact support directly.\n\n"
        f"KNOWLEDGE BASE:\n{request.knowledge_base}"
    )

    start = time.monotonic()
    answer, tokens_used = _call_claude(system_prompt, request.question)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Update in-memory stats
    entry = stats_store[request.business_id]
    entry["total_questions"] += 1
    entry["total_tokens_used"] += tokens_used
    entry["last_activity"] = datetime.utcnow().isoformat()
    entry["conversations"].append({
        "question": request.question,
        "answer": answer,
        "tokens_used": tokens_used,
        "timestamp": entry["last_activity"],
    })
    # Keep only the last 50 conversations in memory
    entry["conversations"] = entry["conversations"][-50:]

    logger.info("Chat response — business_id=%s tokens=%d ms=%d", request.business_id, tokens_used, elapsed_ms)

    return ChatResponse(
        business_id=request.business_id,
        question=request.question,
        answer=answer,
        tokens_used=tokens_used,
        response_time_ms=elapsed_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness/readiness probe for Railway and load balancers."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.get("/stats/{business_id}", response_model=StatsResponse)
async def stats(business_id: str):
    """Return conversation statistics for a given business."""
    entry = stats_store.get(business_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No stats found for business_id '{business_id}'")

    return StatsResponse(
        business_id=business_id,
        total_questions=entry["total_questions"],
        total_tokens_used=entry["total_tokens_used"],
        last_activity=entry["last_activity"],
        recent_conversations=entry["conversations"][-10:],
    )


# --------------------------------------------------------------------------- #
# Global error handlers
# --------------------------------------------------------------------------- #

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )
