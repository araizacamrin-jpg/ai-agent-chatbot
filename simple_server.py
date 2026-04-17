"""
simple_server.py - Production FastAPI server for the customer service chatbot.

Run locally:   python simple_server.py
Railway:       web: uvicorn simple_server:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from main import CustomerServiceAgent

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Service Chatbot API",
    description="AI-powered 24/7 customer service for multiple businesses",
    version="2.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc at /redoc
)

# CORS — allow all origins so client websites can call the API directly.
# Tighten this (allow_origins=[...]) once you know your client domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Agent — created once at startup, shared across requests
# ---------------------------------------------------------------------------
_agent: Optional[CustomerServiceAgent] = None


def get_agent() -> CustomerServiceAgent:
    """Lazy-initialise the agent so the server still boots without an API key."""
    global _agent
    if _agent is None:
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="CLAUDE_API_KEY is not set. Add it in Railway → Variables.",
            )
        _agent = CustomerServiceAgent(api_key=api_key)
    return _agent


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter  (20 requests / 60 seconds per IP)
# ---------------------------------------------------------------------------
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW = 60  # seconds


def _check_rate_limit(client_ip: str) -> None:
    """Raise 429 if the IP has exceeded the rate limit."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Drop timestamps outside the window
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > window_start]
    if len(_rate_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit: {RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW}s.",
        )
    _rate_store[client_ip].append(now)


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic validates automatically)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    business_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique identifier for the business (e.g. 'acme-pizza')",
        examples=["acme-pizza"],
    )
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="The customer's question",
        examples=["What are your opening hours?"],
    )
    knowledge_base: str = Field(
        ..., min_length=10, max_length=50000,
        description="Full text knowledge base for this business",
        examples=["We are open Mon-Fri 9am-5pm. Delivery is free over $50."],
    )


class ChatResponse(BaseModel):
    conversation_id: str
    business_id: str
    response: str
    confidence: int
    escalated: bool
    escalation_reason: Optional[str]
    timestamp: str
    response_time_ms: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str


class StatsResponse(BaseModel):
    business_id: str
    total_conversations: int
    escalation_count: int
    escalation_rate_pct: float
    avg_confidence: float
    avg_response_time_ms: float


# ---------------------------------------------------------------------------
# Middleware — log every request
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "%s %s → %d (%dms) [%s]",
        request.method, request.url.path,
        response.status_code, elapsed_ms,
        request.client.host if request.client else "?",
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness / readiness probe.

    Railway (and load balancers) hit this to confirm the app is up.
    Returns HTTP 200 with status "ok".
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(body: ChatRequest, request: Request):
    """
    Answer a customer question using the provided knowledge base.

    - Searches the knowledge base for relevant context.
    - Calls Claude to generate a response with a confidence score (0-100).
    - If confidence < 80 %, sets `escalated: true` so your system can
      route the conversation to a human agent.
    - Stores the conversation for stats and retrieval.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    agent = get_agent()
    result = agent.process_question(
        business_id=body.business_id,
        question=body.question,
        knowledge_base=body.knowledge_base,
    )

    return ChatResponse(
        conversation_id=result["conversation_id"],
        business_id=result["business_id"],
        response=result["response"],
        confidence=result["confidence"],
        escalated=result["escalated"],
        escalation_reason=result.get("escalation_reason"),
        timestamp=result["timestamp"],
        response_time_ms=result["response_time_ms"],
    )


@app.get("/stats/{business_id}", response_model=StatsResponse, tags=["Analytics"])
async def stats(business_id: str):
    """
    Return aggregate statistics for a business.

    Includes total conversations, escalation count & rate,
    average confidence score, and average response time.
    """
    agent = get_agent()
    return agent.get_stats(business_id)


@app.get("/conversations/{business_id}", tags=["Analytics"])
async def list_conversations(business_id: str, limit: int = 100):
    """
    Return the most recent conversations for a business (max 100).

    Use the `limit` query param to request fewer: /conversations/acme?limit=10
    """
    agent = get_agent()
    convs = agent.get_conversations(business_id, limit=min(limit, 100))
    return {
        "business_id": business_id,
        "count": len(convs),
        "conversations": convs,
    }


@app.get("/conversations/{business_id}/{conversation_id}", tags=["Analytics"])
async def get_conversation(business_id: str, conversation_id: str):
    """
    Return full details for a single conversation.

    Returns 404 if the conversation_id doesn't exist for that business.
    """
    agent = get_agent()
    record = agent.get_conversation(business_id, conversation_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found for business '{business_id}'.",
        )
    return record


# ---------------------------------------------------------------------------
# Global error handler — never expose stack traces to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Entry point — run with: python simple_server.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    logger.info("Starting server on port %d (reload=%s)", port, reload)
    uvicorn.run("simple_server:app", host="0.0.0.0", port=port, reload=reload)
