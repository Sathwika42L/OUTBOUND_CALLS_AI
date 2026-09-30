"""
Outbound KBS Bank loan-officer flow - PLANNER-ONLY architecture.

The main pipeline LLM has NO tools. All intent detection (identity
confirmed/denied, not_interested, end_call, transfer_requested) and all RAG
query generation happen in outbound_planner.py's LLMDrivenRAGProcessor,
which runs BEFORE this LLM in the pipeline (see bot2.py) and:
  1. Analyzes the customer's latest message + real conversation history via
     a dedicated LLM classification call, every turn - genuine intent
     detection, not prompt-conditioned tool-choice.
  2. Decides needs_rag / rag_query itself, retrieves from SimpleRAG, and
     writes a single [CALL NOTES] system message into the shared context.
  3. Fires an event (identity_confirmed / identity_denied / not_interested /
     end_call / transfer_requested) which calls handle_call_event() below to
     switch the active node.
  4. Only THEN does the customer's transcription frame continue downstream
     to this LLM, which speaks naturally using whichever node's
     task_messages and [CALL NOTES] are now in view.

IMPORTANT - do not add FlowsFunctionSchema tools back onto these nodes, and
do not manually queue an extra LLMRunFrame after a node transition in
handle_call_event. Both were tried in earlier versions and both caused
double-firing: tools duplicated event detection the planner already owns,
and a manually queued LLMRunFrame produced a second LLM completion for the
same customer utterance that was already going to reach the LLM naturally
via the pipeline's normal frame flow. handle_call_event's only job is to
pick the right node and (for terminal events) fire the disconnect/transfer
request - the already-in-flight customer frame does the rest.
"""

import os
import re
import requests
from loguru import logger
from dotenv import load_dotenv
from pipecat_flows import FlowManager, NodeConfig

load_dotenv(override=True)

base_url = os.getenv("WEBRTC_URL")

# Optional: an endpoint that reports whether a live loan officer is free to
# take a transfer right now. Leave unset to always treat an agent as available.
AGENT_AVAILABLE_API = os.getenv("AGENT_AVAILABLE_API", "")

# Where the actual transfer hand-off request goes. Defaults to WEBRTC_URL's
# /ai-agent/transfer endpoint - override with your own if different.
AGENT_TRANSFER_URL = os.getenv("AGENT_TRANSFER_URL", base_url)
AGENT_TRANSFER_TARGET = os.getenv("AGENT_TRANSFER_TARGET", "")

CAMPAIGN = {
    "name": "Personal Loan Campaign",
    "company": "KBS Bank",
    "purpose": "Inform customers about personal loan options and assess their interest",
    "goal": "Identify customers who may be interested in applying for a personal loan",
}
CUSTOMER_NAME = os.getenv("TEST_CUSTOMER_NAME", "Mohith")
# Initialize once when the application starts

kbs_rag = SimpleRAG()

# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE RULES (how to use [CALL NOTES] from the planner)
# ═══════════════════════════════════════════════════════════════════════════

RESPONSE_RULES = f"""HOW TO REPLY WHEN CALL NOTES ARE PROVIDED

Call notes arrive as a system message starting with [CALL NOTES]. They are
replaced every turn - only the latest one applies.

1. Address what the customer just said first.
2. Use the bank information in the notes for every fact. Never invent rates,
   amounts, tenure, eligibility, fees or documents. If the notes say nothing
   was found, say "I don't have that specific information right now".
3. Turn facts into natural speech - do not read the text back or dump
   everything in one breath.
4. Be proactive - keep telling the customer more about the loan (rate,
   eligibility, documents, benefits) before asking anything about them
   personally.
5. If a next question is suggested in the notes, ask it naturally.
6. Never ask for something already in the customer profile shown in notes.
7. Never mention notes, RAG, knowledge base, database, retrieval, documents,
   system, planner, or internal processing.

MARKETING STYLE:
When you have loan information, be enthusiastic and confident:
- "Our vehicle loans range from ₹1 lakh to ₹40 lakh with flexible repayment
  periods of 12 to 84 months"
- "Interest rates start at just 8.5% per annum, which is very competitive"
- "We offer fast approval and minimal documentation"

Examples:
Customer: "Yes, I am looking for a loan." (call notes show: personal, home,
vehicle, education and business loans available)
You: "That's great, {CUSTOMER_NAME}. We offer personal, home, vehicle,
education and business loan options. Which type of loan are you currently
considering?"

Customer: "I need a car loan." (call notes show: vehicle loan ₹1 lakh–₹40
lakh, 12–84 months, 8.5% interest)
You: "Excellent choice! Our vehicle loans range from ₹1 lakh to ₹40 lakh,
with repayment periods from 12 to 84 months. Interest rates start at just
8.5% per annum. Are you looking to finance a new vehicle or a used one?"

Customer: "What is the maximum amount?" (call notes show: vehicle loan max
₹40 lakh)
You: "The maximum vehicle loan amount is ₹40 lakh, subject to eligibility.
That can cover premium vehicles too. Approximately how much are you looking
to borrow?"
"""


# ═══════════════════════════════════════════════════════════════════════════
# TTS-safety helper (defensive net only, not a primary control)
# ═══════════════════════════════════════════════════════════════════════════

_LEAK_PATTERNS = [
    r"\{", r"\}", r"\[", r"\]",
    r'"name"\s*:', r'"arguments"\s*:', r'"query"\s*:',
    r"</?tool_call>", r"tool_call",
]


def sanitize_for_tts(text: str) -> str:
    """Strip a reply that looks like leaked JSON/tool syntax rather than
    natural speech. Returns "" if the text looks contaminated."""
    if not text:
        return text
    stripped = text.strip()
    if len(stripped) < 20 and any(
        re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in _LEAK_PATTERNS
    ):
        logger.warning("🚫 Stripped apparent leaked syntax from TTS text: {}", stripped[:120])
        return ""
    for pattern in _LEAK_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            logger.warning("🚫 Stripped apparent leaked syntax from TTS text: {}", text[:120])
            return ""
    return text.strip()


def _disconnect(flow_manager: FlowManager):
    """Hang up via the call backend. Fires synchronously from within
    handle_call_event - the ONLY place that disconnects."""
    call_id = flow_manager.state.get("callId", "")
    if not base_url or not call_id:
        logger.info("Disconnect skipped (WEBRTC_URL or callId missing)")
        return
    try:
        requests.get(
            f"{base_url}/ai-agent/disconnect?callId={call_id}",
            verify=False, timeout=10,
        )
        logger.info("📴 Disconnect request sent")
    except Exception as e:
        logger.warning(f"Disconnect error: {e}")


def _transfer(flow_manager: FlowManager) -> bool:
    """
    Actually hand the call to a live agent. Returns True if a transfer
    request was sent, False if it fell back to a normal disconnect (agent
    unavailable, or the transfer request itself failed).
    """
    if AGENT_AVAILABLE_API:
        try:
            data = requests.get(AGENT_AVAILABLE_API, verify=False, timeout=5).json()
            if not data.get("isAgentAvailable", True):
                logger.info("Agent unavailable - falling back to disconnect")
                _disconnect(flow_manager)
                return False
        except Exception as e:
            logger.warning(f"Agent availability check failed ({e}) - assuming available")

    call_id = flow_manager.state.get("callId", "")
    try:
        if AGENT_TRANSFER_URL and call_id:
            requests.get(
                f"{AGENT_TRANSFER_URL}/ai-agent/transfer?callId={call_id}&target={AGENT_TRANSFER_TARGET}",
                verify=False, timeout=10,
            )
            logger.info("🔀 Transfer-to-agent request sent")
            return True
        logger.info("Transfer skipped (AGENT_TRANSFER_URL/WEBRTC_URL or callId missing)")
    except Exception as e:
        logger.warning(f"Transfer error: {e} - falling back to a normal hangup")
    _disconnect(flow_manager)
    return False


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - whole-call persona (role_messages, set once on the first node)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are Sathwika, a professional loan officer from {CAMPAIGN['company']} making an OUTBOUND call to {CUSTOMER_NAME}.

Your goal: {CAMPAIGN['goal']}

You are the bank representative who called the customer. YOU lead the call.
Never say "How can I help you?" - you called them, so YOU guide the conversation.

CALL BEHAVIOR:
1. Identity confirmation happens first.
2. After identity is confirmed, introduce yourself and briefly explain why
   you're calling.
3. Proactively ask whether the customer is currently looking for a loan.
4. When factual bank information is provided to you via [CALL NOTES], use it
   naturally and conversationally.
5. Keep telling the customer more about the loan (rate, eligibility,
   documents, benefits) proactively - don't wait for them to ask each detail.
6. After providing information, ask ONE useful next question to continue.
7. Remember everything the customer has already told you. Never repeatedly
   ask the same question.
8. Keep responses short and natural for voice calls - normally 1-3 sentences.
9. Be confident and professional, but never pushy.
10. If the customer is not interested, respect the decision and close politely.

CRITICAL RULES ABOUT FACTS:
- When [CALL NOTES] provide bank information, use ONLY that information.
- Never invent: interest rates, loan amounts, tenure, eligibility, income
  requirements, fees, documents, approval conditions.
- If you don't have specific information in the call notes, say "I don't
  have that specific information confirmed right now".
- Speak the information naturally - don't read it back verbatim.
- Never mention: RAG, knowledge base, database, planner, system, retrieval,
  documents, or internal processing.

{RESPONSE_RULES}
"""


# ═══════════════════════════════════════════════════════════════════════════
# CALL STATE
# ═══════════════════════════════════════════════════════════════════════════

async def initialize_call(flow_manager: FlowManager, rag_system):
    """
    rag_system is created once at bot startup and owned entirely by
    outbound_planner.py's LLMDrivenRAGProcessor - it is stored here only so
    the pipeline file has one place to construct it and pass it along; no
    flow node touches it directly.
    """
    flow_manager.state.setdefault("name", CUSTOMER_NAME)
    flow_manager.state["identity_confirmed"] = False
    flow_manager.state["interested"] = None
    flow_manager.state["call_ended"] = False
    flow_manager.state["transfer_requested"] = False
    flow_manager.state["rag_system"] = rag_system


async def handle_call_event(flow_manager: FlowManager, event: str):
    """
    The ONLY thing that switches nodes in this architecture. Called by
    outbound_planner.py's LLMDrivenRAGProcessor after it detects one of:
      identity_confirmed | identity_denied | not_interested | end_call |
      transfer_requested

    Deliberately does NOT queue an extra LLMRunFrame: the customer's
    transcription frame that triggered this event is still flowing
    downstream through the pipeline (the planner runs before the LLM), and
    will reach the LLM immediately after this function returns, using
    whichever node is now active. Forcing an extra LLM run here would
    produce two completions for one customer utterance.
    """
    if event == "identity_confirmed":
        logger.info("✅ Identity confirmed → conversation node")
        flow_manager.state["identity_confirmed"] = True
        await flow_manager.set_node_from_config(create_conversation_node())

    elif event == "identity_denied":
        logger.info("🔚 identity_denied → apologize and end call")
        flow_manager.state["identity_confirmed"] = False
        flow_manager.state["call_ended"] = True
        _disconnect(flow_manager)
        await flow_manager.set_node_from_config(create_identity_denied_node())

    elif event == "transfer_requested":
        logger.info("🔀 transfer_requested → transfer node")
        transferred = _transfer(flow_manager)
        flow_manager.state["transfer_requested"] = transferred
        flow_manager.state["call_ended"] = True
        node = create_transfer_node() if transferred else create_closing_node()
        await flow_manager.set_node_from_config(node)

    elif event in ("not_interested", "end_call"):
        logger.info(f"🔚 {event} → closing node")
        if event == "not_interested":
            flow_manager.state["interested"] = False
        flow_manager.state["call_ended"] = True
        _disconnect(flow_manager)
        await flow_manager.set_node_from_config(create_closing_node())


# ═══════════════════════════════════════════════════════════════════════════
# NODES - no functions/tools on any of these. The planner alone decides when
# to move between them (via handle_call_event above); these NodeConfigs only
# supply the persona/instructions the main LLM speaks from.
# ═══════════════════════════════════════════════════════════════════════════

def create_identity_node() -> NodeConfig:
    """
    First node. The greeting is spoken directly by the pipeline (bypassing
    the LLM - see bot2.py's _initialize_flow_once). This node's task_messages
    only matter if the LLM is invoked again before the planner has switched
    the node (e.g. an unrelated stray frame) - identity itself is decided by
    the planner, not by this LLM.
    """
    greeting_task = (
        f"You just asked if you're speaking with {CUSTOMER_NAME} and are "
        "waiting for their answer. Do not repeat the question or say "
        "anything else yet."
    )

    return NodeConfig(
        name="identity",
        respond_immediately=False,

        role_messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        task_messages=[{"role": "system", "content": greeting_task}],
    )


def create_conversation_node() -> NodeConfig:
    """
    Main conversation node. All RAG intent detection and eligibility/decline/
    transfer detection happen in the planner (outbound_planner.py) - this
    node just tells the LLM how to speak using whatever [CALL NOTES] the
    planner injected.
    """
    conversation_task = (
        f"Identity is confirmed with {CUSTOMER_NAME}.\n\n"
        "FIRST TURN ONLY: introduce yourself naturally and ask if they're "
        "currently looking for a loan, e.g. \"I'm Sathwika calling from KBS "
        "Bank regarding our loan options. Are you currently looking for any "
        "type of loan?\" Say this once.\n\n"
        "EVERY TURN AFTER THAT:\n"
        "1. Listen to what the customer said.\n"
        "2. If [CALL NOTES] provide bank information, use those facts to "
        "answer naturally - proactively, enthusiastically, and only with "
        "facts actually present in the notes.\n"
        "3. Ask the next question suggested in the notes, if any.\n"
        "4. Never repeat a question already answered, and never invent a "
        "fact not present in the notes.\n\n"
        + RESPONSE_RULES
    )

    return NodeConfig(
        name="conversation",
        respond_immediately=False,
        task_messages=[{"role": "system", "content": conversation_task}],
    )


def create_transfer_node() -> NodeConfig:
    """
    Reached only after the planner fired transfer_requested AND the actual
    transfer request already succeeded (see handle_call_event/_transfer
    above) - this node's only job is to speak the confirmation line.
    """
    return NodeConfig(
        name="transfer",
        respond_immediately=False,
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Say one short, warm line confirming you're connecting "
                    "the customer now to a loan officer who will help them "
                    "complete the application, and thank them for their "
                    "time. Do not ask anything else - the handoff is already "
                    "in progress."
                ),
            }
        ],
    )


def create_identity_denied_node() -> NodeConfig:
    """The person on the line is not the customer, or did not clearly
    confirm. Apologize once and end - the planner already fired the
    disconnect, this just supplies the words."""
    return NodeConfig(
        name="identity_denied",
        respond_immediately=False,
        task_messages=[
            {
                "role": "system",
                "content": (
                    "The person on the line has indicated they are not "
                    f"{CUSTOMER_NAME}, or did not clearly confirm it. Say "
                    "one short, polite apology for the inconvenience and "
                    "say goodbye. Do not ask again whether they are the "
                    "customer, and do not mention loans or the bank's "
                    "products."
                ),
            }
        ],
    )


def create_closing_node() -> NodeConfig:
    """Customer declined, wants to end the call, or an agent transfer fell
    back to a normal hangup."""
    return NodeConfig(
        name="closing",
        respond_immediately=False,
        task_messages=[
            {
                "role": "system",
                "content": (
                    "The customer is not interested or wants to end the "
                    "call. Respect their decision and close politely: one "
                    "short, warm line thanking them for their time, then "
                    "say goodbye. Do not ask any question and do not try to "
                    "persuade them further."
                ),
            }
        ],
    )