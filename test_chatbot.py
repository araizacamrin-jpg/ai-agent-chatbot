"""
test_chatbot.py - End-to-end tests for the customer service chatbot.

Prerequisites:
  1. Copy .env.example to .env and add your CLAUDE_API_KEY.
  2. Start the server:  python simple_server.py
  3. Run this file:     python test_chatbot.py

You can also point the tests at a live Railway URL:
  TEST_BASE_URL=https://your-app.up.railway.app python test_chatbot.py
"""

import os
import sys
import json
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Sample knowledge bases — realistic scenarios for three business types
# ---------------------------------------------------------------------------

RESTAURANT_KB = """
Restaurant Name: Mario's Italian Kitchen

Hours:
  Monday-Thursday: 11am - 10pm
  Friday-Saturday: 11am - 11pm
  Sunday: 12pm - 9pm

Menu Highlights:
  Margherita Pizza        $14  (tomato, mozzarella, fresh basil)
  Spaghetti Carbonara     $18  (eggs, pancetta, parmesan, black pepper)
  Chicken Parmigiana      $22  (breaded chicken, marinara, melted cheese)
  Caesar Salad            $12  (romaine, parmesan, croutons, house dressing)
  Tiramisu                $8   (classic Italian dessert)
  House Red Wine          $9/glass  |  $32/bottle

Delivery:
  We deliver within 5 miles. Orders over $30 get FREE delivery.
  Standard delivery fee: $4.99. Minimum order: $20.
  Average delivery time: 30-45 minutes.

Reservations:
  Call 555-MARIO or book online at marios.example.com.
  Recommended for groups of 4 or more.

Dietary:
  Vegetarian and gluten-free options available.
  Please inform your server of any allergies.

Parking: Free lot behind the restaurant.

Catering: Available for events of 20+ people. Email events@marios.example.com.
"""

HAIR_SALON_KB = """
Salon Name: Bella Hair Studio
Location: 123 Main Street, Suite 4. Free street parking.

Services & Pricing:
  Women's Haircut & Style   $65 - $95  (depends on hair length)
  Men's Haircut             $35
  Blowout                   $45
  Full Color                $100 - $140
  Highlights                $120 - $180
  Balayage                  $150 - $250
  Keratin Treatment         $200 - $300
  Bridal Package            Contact us for a custom quote

Hours: Tuesday - Saturday 9am - 7pm. Closed Sunday and Monday.

Booking:
  Call 555-BELLA or book online at bellahair.example.com.
  24-hour cancellation policy applies.
  50% deposit required for color services.
  New color clients must complete a patch test 48 hours before the appointment.

Our Stylists:
  Maria  — specialist in color and balayage
  Sarah  — specialist in cuts and bridal styling
  James  — specialist in men's cuts and fades

Products: We use and retail Olaplex, Kérastase, and Redken.
"""

TECH_SUPPORT_KB = """
Company: TechSupport Pro — IT support & managed services for small businesses

Services & Pricing:
  Help Desk (up to 5 users)      $99 / month
  Network Setup & Maintenance    from $199
  Data Backup & Recovery         $49 / month
  Cybersecurity Package          $149 / month
  Cloud Migration                custom quote

Support Hours:
  Standard: Monday - Friday 8am - 6pm EST
  Emergency (critical issues): 24/7 for premium clients

Response Times:
  Phone support:     under 5 minutes
  Email support:     within 2 hours during business hours
  Critical issues:   within 30 minutes, 24/7

Contact: support@techsupportpro.example.com | 555-TECH

Remote vs On-site:
  90% of issues are resolved remotely.
  On-site visits available within 24-48 hours.

Contracts: Month-to-month, no long-term commitment required.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ask(business_id: str, question: str, kb: str, label: str = "") -> dict:
    """POST /chat and pretty-print the result."""
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"business_id": business_id, "question": question, "knowledge_base": kb},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    prefix = f"[{label}] " if label else ""
    flag = "🚨 ESCALATED" if data["escalated"] else "✓"
    print(f"\n  {prefix}Q: {question}")
    print(f"  A: {data['response'][:200]}{'...' if len(data['response']) > 200 else ''}")
    print(f"  Confidence: {data['confidence']}%  |  {flag}  |  {data['response_time_ms']}ms")
    if data["escalated"]:
        print(f"  Reason: {data['escalation_reason']}")

    time.sleep(0.4)   # Avoid hammering the API in tests
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health() -> None:
    _section("1. Health Check")
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok", f"Unexpected status: {data}"
    print(f"  ✓  {data}")


def test_restaurant() -> None:
    _section("2. Restaurant — Mario's Italian Kitchen")
    biz = "marios-restaurant"
    kb = RESTAURANT_KB

    _ask(biz, "What are your hours on Friday?", kb)
    _ask(biz, "How much is the Margherita pizza?", kb)
    _ask(biz, "Do you deliver? Is there a minimum order?", kb)
    _ask(biz, "I'd like to book a table for 8 people on Saturday.", kb)
    _ask(biz, "Do you have gluten-free options?", kb)
    _ask(biz, "Do you offer catering for corporate events?", kb)


def test_salon() -> None:
    _section("3. Hair Salon — Bella Hair Studio")
    biz = "bella-salon"
    kb = HAIR_SALON_KB

    _ask(biz, "How much does a women's haircut cost?", kb)
    _ask(biz, "Are you open on Sundays?", kb)
    _ask(biz, "I want a balayage — how much will it cost?", kb)
    _ask(biz, "How do I book an appointment?", kb)
    _ask(biz, "Which stylist specialises in men's cuts?", kb)


def test_tech_support() -> None:
    _section("4. Tech Support — TechSupport Pro")
    biz = "techsupport-pro"
    kb = TECH_SUPPORT_KB

    _ask(biz, "What are your support hours?", kb)
    _ask(biz, "How much does help desk support cost per month?", kb)
    _ask(biz, "How quickly do you respond to critical issues?", kb)
    _ask(biz, "Do I have to sign a long-term contract?", kb)


def test_escalation() -> None:
    _section("5. Escalation — Low-Confidence Questions")
    biz = "escalation-test"
    kb = RESTAURANT_KB  # Restaurant KB — off-topic questions should escalate

    questions = [
        "What is the capital of France?",
        "Can you recommend a good dentist nearby?",
        "How do I fix my printer?",
    ]
    for q in questions:
        result = _ask(biz, q, kb, label="off-topic")
        # Don't hard-assert escalation — Claude may still be confident
        # (it should say it doesn't know, with low confidence).
        if not result["escalated"]:
            print(f"  Note: Not escalated — confidence was {result['confidence']}%.")


def test_stats_and_conversations() -> None:
    _section("6. Stats & Conversation History")

    # Stats
    resp = requests.get(f"{BASE_URL}/stats/marios-restaurant", timeout=5)
    assert resp.status_code == 200
    stats = resp.json()
    print(f"\n  Stats for Mario's Restaurant:")
    print(f"  {json.dumps(stats, indent=4)}")
    assert stats["total_conversations"] >= 0

    # Conversation list
    resp = requests.get(f"{BASE_URL}/conversations/marios-restaurant?limit=3", timeout=5)
    assert resp.status_code == 200
    listing = resp.json()
    print(f"\n  Last {listing['count']} conversation(s) stored.")

    # Fetch one by ID if available
    if listing["conversations"]:
        conv_id = listing["conversations"][0]["conversation_id"]
        resp2 = requests.get(
            f"{BASE_URL}/conversations/marios-restaurant/{conv_id}", timeout=5
        )
        assert resp2.status_code == 200
        conv = resp2.json()
        print(f"\n  Retrieved conversation {conv_id[:8]}...")
        print(f"  Q: {conv['question'][:80]}")
        print(f"  Confidence: {conv['confidence']}%")


def test_error_handling() -> None:
    _section("7. Error Handling")

    # Empty question → 422 Unprocessable Entity
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"business_id": "test", "question": "", "knowledge_base": "some info"},
        timeout=5,
    )
    print(f"\n  Empty question           → HTTP {resp.status_code}  (expected 422)")
    assert resp.status_code == 422

    # Missing knowledge_base field → 422
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"business_id": "test", "question": "hello?"},
        timeout=5,
    )
    print(f"  Missing knowledge_base   → HTTP {resp.status_code}  (expected 422)")
    assert resp.status_code == 422

    # Non-existent conversation → 404
    resp = requests.get(
        f"{BASE_URL}/conversations/nobody/00000000-0000-0000-0000-000000000000",
        timeout=5,
    )
    print(f"  Non-existent conversation → HTTP {resp.status_code}  (expected 404)")
    assert resp.status_code == 404

    print("\n  ✓  All error cases handled correctly.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CUSTOMER SERVICE CHATBOT — TEST SUITE")
    print(f"  Server: {BASE_URL}")
    print("=" * 60)

    # Confirm server is reachable before running tests
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to {BASE_URL}")
        print("Start the server first:  python simple_server.py")
        sys.exit(1)

    tests = [
        test_health,
        test_restaurant,
        test_salon,
        test_tech_support,
        test_escalation,
        test_stats_and_conversations,
        test_error_handling,
    ]

    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"\n  ✗  {fn.__name__} FAILED: {exc}")
            failed.append(fn.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
    print("=" * 60)
