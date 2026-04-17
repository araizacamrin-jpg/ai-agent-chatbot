# Customer Service Chatbot API

AI-powered 24/7 customer service chatbot for multiple businesses, built with FastAPI and Claude.

## What it does

- Accepts a customer question + a business knowledge base and returns an AI-generated answer
- Rates confidence (0–100%) on every response
- Automatically flags low-confidence answers (< 80%) for human review (escalation)
- Tracks every conversation with metadata (timestamp, response time, tokens used)
- Serves multiple businesses from one deployment — each identified by `business_id`

---

## Quick start (local)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-agent-chatbot.git
cd ai-agent-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a Claude API key

1. Go to <https://console.anthropic.com>
2. Sign up / log in → **API Keys** → **Create Key**
3. Copy the key (starts with `sk-ant-…`)

### 3. Create your `.env` file

```bash
cp .env.example .env
# Open .env and paste your key:
# CLAUDE_API_KEY=sk-ant-your-key-here
```

### 4. Run the server

```bash
python simple_server.py
# Server starts on http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 5. Run the tests (in a second terminal)

```bash
python test_chatbot.py
```

---

## API reference

### `POST /chat`

Answer a customer question.

**Request body:**
```json
{
  "business_id": "acme-pizza",
  "question": "Do you deliver on Sundays?",
  "knowledge_base": "We deliver Mon-Sat 11am-10pm. Closed Sundays."
}
```

**Response:**
```json
{
  "conversation_id": "3fa85f64-...",
  "business_id": "acme-pizza",
  "response": "Unfortunately we don't deliver on Sundays — we're closed that day. We deliver Monday through Saturday, 11am to 10pm.",
  "confidence": 95,
  "escalated": false,
  "escalation_reason": null,
  "timestamp": "2024-01-15T14:32:10.123456",
  "response_time_ms": 842
}
```

### `GET /health`

Liveness probe used by Railway and load balancers.

### `GET /stats/{business_id}`

Aggregate statistics: total conversations, escalation rate, average confidence.

### `GET /conversations/{business_id}`

Last 100 conversations for a business. Add `?limit=10` for fewer.

### `GET /conversations/{business_id}/{conversation_id}`

Full record for a single conversation.

---

## Example curl requests

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "business_id": "marios-pizza",
    "question": "What time do you close on Saturdays?",
    "knowledge_base": "Hours: Mon-Fri 11am-10pm, Sat 11am-11pm, Sun 12pm-9pm."
  }'

# Stats
curl http://localhost:8000/stats/marios-pizza

# Conversation list
curl http://localhost:8000/conversations/marios-pizza
```

---

## Deploy to Railway

### 1. Push code to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-agent-chatbot.git
git push -u origin main
```

### 2. Create a Railway project

1. Go to <https://railway.app> and log in
2. **New Project → Deploy from GitHub repo**
3. Authorise Railway and select your repository
4. Railway detects Python automatically

### 3. Set environment variables

In your Railway project → service → **Variables**, add:

| Variable | Value |
|---|---|
| `CLAUDE_API_KEY` | `sk-ant-your-key-here` |
| `ENVIRONMENT` | `production` |
| `LOG_LEVEL` | `INFO` |

> Railway injects `PORT` automatically — **do not set it manually**.

### 4. Verify deployment

Once the build turns green, open your Railway URL:

```
https://your-app.up.railway.app/health   → {"status":"ok"}
https://your-app.up.railway.app/docs     → Interactive API docs
```

Every push to `main` triggers an automatic re-deploy.

---

## Project structure

```
ai-agent-chatbot/
├── main.py              # CustomerServiceAgent class (core logic)
├── simple_server.py     # FastAPI server with all endpoints
├── test_chatbot.py      # End-to-end test suite
├── requirements.txt     # Python dependencies
├── Procfile             # Railway process definition
├── .env.example         # Environment variable template
└── README.md            # This file
```

## Extending the system

| Goal | How |
|---|---|
| Persistent storage | Replace `ConversationStore` dict in `main.py` with Supabase/Postgres calls |
| Semantic KB search | Replace `_search_knowledge_base()` with pgvector or a vector DB |
| Auth per client | Add an API key middleware in `simple_server.py` |
| Stricter rate limits | Adjust `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW` in `simple_server.py` |
| Different model | Change `MODEL` constant in `CustomerServiceAgent` |
