"""
main.py - CustomerServiceAgent

Core AI agent that powers the chatbot. Import this in simple_server.py
or run directly (python main.py) for a quick demo.

Architecture:
  CustomerServiceAgent   — orchestrates every request end-to-end
  ConversationStore      — thread-safe in-memory store (swap for DB later)
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Optional

from anthropic import Anthropic, APIError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — one format shared across the whole package
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConversationStore — in-memory dict, easy to swap for Supabase/Postgres
# ---------------------------------------------------------------------------
class ConversationStore:
    """
    Stores conversations keyed by (business_id, conversation_id).

    Replace the internal dict with DB calls to make storage persistent
    across server restarts.
    """

    def __init__(self) -> None:
        # { business_id: { conversation_id: conversation_dict } }
        self._data: dict[str, dict[str, dict]] = {}

    def save(self, business_id: str, conversation_id: str, record: dict) -> None:
        self._data.setdefault(business_id, {})[conversation_id] = record

    def get_one(self, business_id: str, conversation_id: str) -> Optional[dict]:
        return self._data.get(business_id, {}).get(conversation_id)

    def get_many(self, business_id: str, limit: int = 100) -> list[dict]:
        records = list(self._data.get(business_id, {}).values())
        records.sort(key=lambda r: r["timestamp"], reverse=True)
        return records[:limit]

    def stats(self, business_id: str) -> dict:
        records = list(self._data.get(business_id, {}).values())
        total = len(records)
        if total == 0:
            return {
                "business_id": business_id,
                "total_conversations": 0,
                "escalation_count": 0,
                "escalation_rate_pct": 0.0,
                "avg_confidence": 0.0,
                "avg_response_time_ms": 0.0,
            }
        escalations = sum(1 for r in records if r.get("escalated"))
        avg_conf = sum(r.get("confidence", 0) for r in records) / total
        avg_rt = sum(r.get("response_time_ms", 0) for r in records) / total
        return {
            "business_id": business_id,
            "total_conversations": total,
            "escalation_count": escalations,
            "escalation_rate_pct": round(escalations / total * 100, 1),
            "avg_confidence": round(avg_conf, 1),
            "avg_response_time_ms": round(avg_rt, 1),
        }


# Shared store — one instance used by the whole process
store = ConversationStore()


# ---------------------------------------------------------------------------
# CustomerServiceAgent
# ---------------------------------------------------------------------------
class CustomerServiceAgent:
    """
    AI-powered customer service agent backed by Claude.

    Usage:
        agent = CustomerServiceAgent(api_key="sk-ant-...")
        result = agent.process_question(
            business_id="acme-corp",
            question="What are your hours?",
            knowledge_base="We are open Mon-Fri 9am-5pm.",
        )

    The agent:
      1. Searches the knowledge base for relevant context (keyword match).
      2. Calls Claude with that context and asks for a JSON response that
         includes the answer text AND a 0-100 confidence score.
      3. Escalates to human review when confidence < ESCALATION_THRESHOLD.
      4. Persists the full conversation record in ConversationStore.
      5. Returns a rich dict with everything the API layer needs.
    """

    ESCALATION_THRESHOLD = 80   # Escalate when confidence < this value
    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 1024
    MAX_KB_CHARS = 8000         # Trim enormous knowledge bases to save tokens

    def __init__(self, api_key: str) -> None:
        self.client = Anthropic(api_key=api_key)
        logger.info("CustomerServiceAgent ready (model=%s)", self.MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_question(
        self,
        business_id: str,
        question: str,
        knowledge_base: str,
    ) -> dict:
        """
        Full pipeline: search → generate → score → store → return.

        Returns a dict with these keys:
          conversation_id, business_id, question, response, confidence,
          escalated, escalation_reason, timestamp, response_time_ms,
          kb_match_score, tokens_used
        """
        start = time.monotonic()
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        logger.info(
            "New question | business=%s | conv=%s | q=%r",
            business_id, conversation_id[:8], question[:80],
        )

        # 1. Find relevant context inside the knowledge base
        context, kb_match_score = self._search_knowledge_base(question, knowledge_base)

        # 2. Ask Claude for a response + confidence score
        ai_result = self._generate_response(question, context)

        # 3. Decide whether to escalate
        confidence: int = ai_result["confidence"]
        escalated: bool = confidence < self.ESCALATION_THRESHOLD
        escalation_reason: Optional[str] = (
            f"Confidence {confidence}% is below the {self.ESCALATION_THRESHOLD}% threshold — "
            "a human agent should review this."
            if escalated else None
        )

        response_time_ms = int((time.monotonic() - start) * 1000)

        record = {
            "conversation_id": conversation_id,
            "business_id": business_id,
            "question": question,
            "response": ai_result["response"],
            "confidence": confidence,
            "escalated": escalated,
            "escalation_reason": escalation_reason,
            "timestamp": timestamp,
            "response_time_ms": response_time_ms,
            "kb_match_score": round(kb_match_score, 2),
            "tokens_used": ai_result.get("tokens_used", 0),
        }

        # 4. Persist
        store.save(business_id, conversation_id, record)

        logger.info(
            "Done | business=%s | confidence=%d%% | escalated=%s | %dms",
            business_id, confidence, escalated, response_time_ms,
        )
        return record

    def get_conversations(self, business_id: str, limit: int = 100) -> list[dict]:
        """Return the most recent conversations for a business."""
        return store.get_many(business_id, limit)

    def get_conversation(self, business_id: str, conversation_id: str) -> Optional[dict]:
        """Return one conversation by ID, or None if not found."""
        return store.get_one(business_id, conversation_id)

    def get_stats(self, business_id: str) -> dict:
        """Return aggregate statistics for a business."""
        return store.stats(business_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_knowledge_base(
        self, question: str, knowledge_base: str
    ) -> tuple[str, float]:
        """
        Lightweight keyword search over the knowledge base.

        Splits the KB into paragraphs, scores each one by how many
        question keywords it contains, and returns the top-5 paragraphs
        as context alongside a 0-1 match score.

        This is intentionally simple — replace with a vector DB (e.g. pgvector)
        for production-grade semantic search.
        """
        # Trim very large knowledge bases to avoid huge token counts
        kb = knowledge_base[:self.MAX_KB_CHARS]

        paragraphs = [p.strip() for p in kb.splitlines() if p.strip()]
        if not paragraphs:
            return kb, 0.0

        # Extract meaningful keywords from the question
        stopwords = {
            "what", "when", "where", "how", "why", "who", "which",
            "is", "are", "am", "was", "were", "be", "been", "being",
            "do", "does", "did", "have", "has", "had", "will", "would",
            "could", "should", "can", "may", "might",
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "i", "you", "we",
            "they", "it", "my", "your", "our", "its", "me", "us",
        }
        keywords = {
            w for w in question.lower().split() if w not in stopwords and len(w) > 2
        }

        if not keywords:
            # No useful keywords → return entire (trimmed) KB
            return kb, 0.5

        # Score every paragraph
        scored: list[tuple[float, str]] = []
        for para in paragraphs:
            para_lower = para.lower()
            hits = sum(1 for kw in keywords if kw in para_lower)
            if hits:
                scored.append((hits / len(keywords), para))

        if not scored:
            # Nothing matched — still pass the whole KB so Claude can try
            return kb, 0.0

        scored.sort(reverse=True)
        top_context = "\n".join(p for _, p in scored[:5])
        best_score = scored[0][0]

        return top_context, best_score

    def _generate_response(self, question: str, context: str) -> dict:
        """
        Call Claude and parse a structured JSON reply.

        Asks Claude to return ONLY a JSON object with three keys:
          response   — the answer text shown to the customer
          confidence — integer 0-100
          reasoning  — internal note explaining the confidence score

        Falls back to raw text with confidence=50 if JSON parsing fails.
        """
        system_prompt = """You are a helpful, friendly customer service assistant.

Answer the customer's question using ONLY the knowledge base provided.
If the knowledge base does not contain the answer, say so politely and
suggest they contact support directly — do not make up information.

You MUST reply with ONLY valid JSON in this exact format (no other text):
{
  "response": "Your answer to the customer here",
  "confidence": 85,
  "reasoning": "Short internal note explaining your confidence level"
}

Confidence scoring guide:
  90-100  The knowledge base directly and completely answers the question.
  70-89   The knowledge base partially answers the question.
  50-69   You are inferring from related information in the knowledge base.
  0-49    The knowledge base does not contain relevant information."""

        user_message = (
            f"KNOWLEDGE BASE:\n{context}\n\n"
            f"CUSTOMER QUESTION: {question}\n\n"
            "Reply with ONLY the JSON object."
        )

        try:
            message = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except APITimeoutError:
            logger.error("Claude API timed out")
            raise
        except APIError as exc:
            logger.error("Claude API error: %s", exc)
            raise

        raw = message.content[0].text.strip()
        tokens_used = message.usage.output_tokens

        try:
            parsed = json.loads(raw)
            return {
                "response": parsed.get("response", "I'm sorry, I couldn't generate a response."),
                "confidence": max(0, min(100, int(parsed.get("confidence", 50)))),
                "reasoning": parsed.get("reasoning", ""),
                "tokens_used": tokens_used,
            }
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse JSON from Claude — using raw text, confidence=50")
            return {
                "response": raw,
                "confidence": 50,
                "reasoning": "JSON parse failed; defaulting to 50% confidence",
                "tokens_used": tokens_used,
            }


# ---------------------------------------------------------------------------
# Quick demo — run with: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("ERROR: Set CLAUDE_API_KEY in your .env file first.")
        raise SystemExit(1)

    agent = CustomerServiceAgent(api_key=api_key)

    demo_kb = """
    Business Hours: Monday-Friday 9am-6pm, Saturday 10am-4pm, closed Sunday.
    Delivery: Free delivery on orders over $50. Standard delivery takes 3-5 business days.
    Returns: Returns accepted within 30 days with original receipt.
    Contact: Email support@example.com or call 555-1234.
    """

    demo_questions = [
        "What are your hours on Saturday?",
        "Do you offer free delivery?",
        "What is the capital of France?",   # Off-topic → should trigger escalation
    ]

    for q in demo_questions:
        print(f"\n{'─'*60}")
        result = agent.process_question("demo-store", q, demo_kb)
        print(f"Q:          {result['question']}")
        print(f"A:          {result['response']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Escalated:  {result['escalated']}")
        if result["escalation_reason"]:
            print(f"Reason:     {result['escalation_reason']}")
        print(f"Time:       {result['response_time_ms']} ms")
