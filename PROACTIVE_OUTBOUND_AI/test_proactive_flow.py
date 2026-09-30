#!/usr/bin/env python3
"""
Test the PROACTIVE RAG FLOW end-to-end:
1. Planner analyzes customer input
2. Decides if RAG is needed
3. Generates dynamic RAG query
4. Retrieves from Qdrant
5. LLM refines answer
6. Call notes injected to context
7. Main LLM speaks naturally with bank facts

Run this test to verify the complete architecture works correctly.
"""

import asyncio
import json
from loguru import logger

# Configure detailed logging
logger.remove()
logger.add(
    lambda msg: print(msg.rstrip()),
    format="<level>{level: <8}</level> | {message}",
    level="INFO",
)

from test_rag_simple import SimpleRAG
from outbound_planner import LLMDrivenRAGProcessor
from pipecat.processors.aggregators.llm_context import LLMContext


# ═══════════════════════════════════════════════════════════════════════════
# TEST SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

TEST_SCENARIOS = [
    {
        "stage": "identity",
        "history": [],
        "customer_says": "yes speaking",
        "expected_event": "identity_confirmed",
        "description": "Identity confirmation - customer confirms identity",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "I'm Sathwika calling from KBS Bank regarding our loan options."),
            ("OFFICER", "Are you currently looking for any type of loan?"),
        ],
        "customer_says": "yes",
        "expected_event": "",
        "description": "Customer says yes - should trigger RAG for product overview",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "I'm Sathwika calling from KBS Bank regarding our loan options."),
            ("OFFICER", "Are you currently looking for any type of loan?"),
            ("CUSTOMER", "yes"),
            ("OFFICER", "Great! We offer personal, home, vehicle, education and business loans. Which type are you interested in?"),
        ],
        "customer_says": "I need a car loan",
        "expected_event": "",
        "description": "Customer picks vehicle loan - should trigger dynamic RAG for vehicle loan details",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "I'm Sathwika calling from KBS Bank."),
            ("CUSTOMER", "I am planning for a house"),
        ],
        "customer_says": "I am planning for a house",
        "expected_event": "",
        "description": "Customer mentions house (home loan intent) without explicit product request - should identify product and trigger RAG",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "Our home loans range from ₹10 lakh to ₹50 lakh with rates starting at 7.5%."),
            ("CUSTOMER", "sounds good"),
        ],
        "customer_says": "sounds good, I'm interested",
        "expected_event": "",
        "description": "Customer positive after product explanation - should offer transfer to agent",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "Would you like me to connect you to a loan officer?"),
        ],
        "customer_says": "yes please",
        "expected_event": "transfer_requested",
        "description": "Customer agrees to agent transfer - should trigger transfer_requested event",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "Are you looking for any type of loan?"),
        ],
        "customer_says": "no thanks, not interested",
        "expected_event": "not_interested",
        "description": "Customer declines - should trigger not_interested event",
    },
    {
        "stage": "conversation",
        "history": [
            ("OFFICER", "Thanks for your time today."),
        ],
        "customer_says": "okay bye",
        "expected_event": "end_call",
        "description": "Customer says goodbye - should trigger end_call event",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

async def test_proactive_flow():
    """Test the complete proactive RAG flow"""
    logger.info("🧪 PROACTIVE RAG FLOW TEST")
    logger.info("=" * 80)

    # Initialize RAG system
    logger.info("📚 Initializing RAG system...")
    rag_system = SimpleRAG()
    
    # Initialize context
    context = LLMContext()
    
    # Initialize planner
    planner = LLMDrivenRAGProcessor(rag_system, context)
    
    # Track results
    passed = 0
    failed = 0
    
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        logger.info("=" * 80)
        logger.info(f"TEST {i}/{len(TEST_SCENARIOS)}")
        logger.info(f"Description: {scenario['description']}")
        logger.info("=" * 80)
        
        try:
            # Setup planner state
            planner.stage = scenario["stage"]
            planner.turns = list(scenario["history"])
            
            # Simulate planning turn
            user_text = scenario["customer_says"]
            
            # Mock the planner LLM call to see the decision
            logger.info(f"📝 Customer input: '{user_text}'")
            logger.info(f"📊 Stage: {scenario['stage']}")
            logger.info(f"📋 History: {len(scenario['history'])} turns")
            
            # Call the planner
            plan = await planner._ask_planner(user_text)
            
            logger.info("=" * 80)
            logger.info("[PLANNER OUTPUT]")
            logger.info(json.dumps(plan, indent=2, ensure_ascii=False))
            logger.info("=" * 80)
            
            # Verify event expectation
            if scenario["expected_event"]:
                if plan["event"] == scenario["expected_event"]:
                    logger.info(f"✅ Event correct: {plan['event']}")
                    passed += 1
                else:
                    logger.error(f"❌ Event mismatch: expected {scenario['expected_event']!r}, got {plan['event']!r}")
                    failed += 1
            else:
                logger.info(f"ℹ️  Event: {plan['event'] or '(none)'}")
            
            # Check RAG decision
            if plan["needs_rag"]:
                logger.info(f"🔍 RAG NEEDED: {plan['rag_query']}")
                # Simulate RAG retrieval
                if rag_system.count() > 0:
                    answer = planner._retrieve(plan["rag_query"])
                    if answer:
                        logger.info("✅ RAG retrieval successful")
                        passed += 1
                    else:
                        logger.warning("⚠️  RAG retrieval returned no results")
                else:
                    logger.warning("⚠️  RAG system is empty (no documents loaded)")
            else:
                logger.info("ℹ️  RAG not needed for this turn")
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {type(e).__name__}: {e}")
            failed += 1
            logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(TEST_SCENARIOS)} scenarios")
    logger.info("=" * 80)
    
    # Cleanup
    await planner.cleanup()


# ═══════════════════════════════════════════════════════════════════════════
# MANUAL CONVERSATION TEST
# ═══════════════════════════════════════════════════════════════════════════

async def interactive_test():
    """Interactive test - manually input customer responses and see the flow"""
    logger.info("=" * 80)
    logger.info("🎤 INTERACTIVE PROACTIVE RAG TEST")
    logger.info("Type customer responses and see how the planner handles them.")
    logger.info("Type 'quit' to exit.")
    logger.info("=" * 80)
    
    # Initialize
    rag_system = SimpleRAG()
    context = LLMContext()
    planner = LLMDrivenRAGProcessor(rag_system, context)
    
    turn = 0
    
    while True:
        try:
            customer_input = input(f"\n[Turn {turn}] Customer says: ").strip()
            
            if customer_input.lower() in ("quit", "exit"):
                break
            
            if not customer_input:
                continue
            
            turn += 1
            
            # Run planner
            logger.info("=" * 80)
            await planner._plan_turn(customer_input)
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error: {type(e).__name__}: {e}")
            continue
    
    await planner.cleanup()
    logger.info("\nTest ended.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_test())
    else:
        asyncio.run(test_proactive_flow())
