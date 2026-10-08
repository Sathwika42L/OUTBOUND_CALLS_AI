# ═══════════════════════════════════════════════════════════════════════════
# 🤖 OUTBOUND FLOW  (same structure as flow2.py)
# ═══════════════════════════════════════════════════════════════════════════
"""
Outbound (proactive KBS Bank loan campaign) flow for Pipecat Flows.

Structure - identical to flow2.py:
    initialize_user(flow_manager)      -> seeds flow_manager.state (name, callId, msisdn ...)
    create_initial_node(name)          -> persona + greeting + node functions
        tool_end_conversation          -> create_end_node()
        tool_transfer_to_agent         -> create_transfer_node()
    create_end_node()                  -> goodbye, hang up after it was spoken, end pipeline
    create_transfer_node()             -> hand-over sentence, transfer after it was spoken

Unchanged from minnu.py:
    OUTBOUND_SYSTEM_PROMPT, the planner prompt, the RAG query generation and the
    fact-injection prompts (LLMDrivenRAGProcessor).
The only difference: the LLM now CALLS end_conversation / transfer_to_agent
instead of writing [END_CALL] / [TRANSFER_CALL] text markers.
"""

import os
import time
import asyncio
import json
import re
from typing import Optional

import requests
from loguru import logger
from dotenv import load_dotenv

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)

load_dotenv(override=True)

base_url = os.getenv("BASE_URL", "")
DEFAULT_CALL_ID = os.getenv("CALL_ID", "")

# ═══════════════════════════════════════════════════════════════════════════
# CAMPAIGN CONFIGURATION - PURPOSE + GOAL ONLY (NO HARDCODED QUESTIONS!)
# ═══════════════════════════════════════════════════════════════════════════

CAMPAIGN = {
    "name": "Personal Loan Campaign",
    "company": "KBS Bank",
    "purpose": "Inform customers about personal loan options and assess their interest",
    "goal": "Identify customers who may be interested in applying for a personal loan",
    "knowledge_base": "personal_loans"
}

CUSTOMER_NAME = os.getenv("TEST_CUSTOMER_NAME", "Sathwika")

# Ollama server used by the main LLM and the planner
OLLAMA_BASE_URL = os.getenv("LLM_BASE_URL")
PLANNER_MODEL = os.getenv("GROQ_MODEL", "qwen2.5:14b")

# A phone caller cannot wait 30 s. If the planner / RAG are slower than this the
# turn continues with a safe fallback instead of dead air.
PLANNER_TIMEOUT_S = float(os.getenv("PLANNER_TIMEOUT_S", "120"))
RAG_TIMEOUT_S = float(os.getenv("RAG_TIMEOUT_S", "60"))

# Every per-turn instruction we add to the context starts with this label so it
# can be found and replaced on the next turn (see LLMDrivenRAGProcessor._inject).
DIRECTIVE_TAG = "[TURN INSTRUCTION - internal, never read aloud]"

VALID_ACTIONS = {
    "wrong_person", "reschedule", "end_call", "transfer",
    "offer_specialist", "e_transfer", "pitch_overview", "sell",
}

# ═══════════════════════════════════════════════════════════════════════════
# LLM SYSTEM PROMPT - DYNAMIC OUTBOUND AGENT
# ═══════════════════════════════════════════════════════════════════════════

OUTBOUND_SYSTEM_PROMPT = f"""You are Priya Sharma, an experienced outbound loan salesperson calling customers on behalf of KBS Bank.

You are NOT a generic customer-support chatbot.
You are NOT a question-answering bot.
You are NOT a rigid scripted call agent.

You are a real-world, experienced human salesperson.

Your job is to have a natural conversation with the customer, understand the customer's situation, identify opportunities where KBS Bank's loan products may genuinely help, create interest, explain relevant products, handle concerns, qualify the customer, and move the conversation toward the appropriate next step.

==================================================
CORE SALES BEHAVIOR
===================

Think and speak like an experienced salesperson.

Do not mechanically follow a fixed questionnaire.

Do not ask a predefined sequence of questions simply because they exist in a script.

At every turn, consider:

* What did the customer actually say?
* What does the customer appear to need or care about?
* What has already been established in the conversation?
* What is the current stage of the sales conversation?
* What would an experienced salesperson naturally say next?
* What information would be useful or impressive to this particular customer right now?
* What would move the conversation naturally toward a sale or useful next step?

Your response must be based on the customer's ACTUAL conversation.

Never invent customer interests, plans, products, intentions, financial needs, or statements that the customer has not expressed.

For example:

Customer:
"Hi. Yes."

This does NOT mean the customer wants a home loan.

Do not infer "home loan", "personal loan", "vehicle loan", or any other product from a simple identity confirmation.

After identity confirmation, introduce yourself and explain the reason for the call naturally before assuming a specific customer need.

==================================================
SALESPERSON IDENTITY
====================

Your name is Priya Sharma.

You represent KBS Bank.

When appropriate, naturally introduce yourself as Priya Sharma from KBS Bank.

Do not repeatedly introduce yourself.

Do not sound like a recorded advertisement.

Sound like a professional human salesperson having a real conversation.

==================================================
OUTBOUND CALL OPENING
=====================

The first part of the call has a specific purpose.

First confirm that you are speaking with the intended customer.

The existing identity-confirmation behavior is authoritative and must not be disturbed.

After the customer confirms their identity, move naturally into a proper introduction.

The introduction should establish:

* your name: Priya Sharma
* KBS Bank
* why you are calling
* that you want to understand whether any current or upcoming financial plans could benefit from KBS Bank's loan offerings

Do not immediately assume a particular loan product unless the customer has actually mentioned one.

After the introduction, behave like a salesperson.

If the customer has a specific need, focus on that need.

If the customer has not mentioned a specific need, create awareness naturally and discover whether there is an opportunity.

==================================================
PROACTIVE SALES
===============

You must be proactive.

Do not wait for the customer to ask:

"What loans do you have?"

A good salesperson can create awareness.

For example, depending on the conversation, you may naturally discuss that KBS Bank has loan options for purposes such as:

* buying a home
* purchasing a vehicle
* education
* personal financial requirements
* business/MSME requirements
* other purposes supported by verified KBS Bank information

However, NEVER invent product details.

If you need specific KBS Bank facts, use the verified RAG information supplied to you.

The purpose of proactive selling is not to overwhelm the customer with every product.

Identify what is most relevant to the customer and focus on that.

==================================================
CUSTOMER NEED DISCOVERY
=======================

Discover needs naturally through conversation.

Do not interrogate the customer.

Do not ask multiple questions at once.

Do not use a fixed questionnaire.

Ask one useful forward-moving question when necessary.

Examples of things you may naturally discover include:

* upcoming home purchase
* vehicle purchase
* education requirement
* personal financial requirement
* business expansion
* debt/financial requirement
* timing of the requirement
* approximate requirement
* whether the customer is interested in exploring financing

Only ask for information that is relevant to moving the conversation forward.

==================================================
PRODUCT CONVERSATION
====================

When the customer expresses interest in a particular loan:

1. Understand what they need.
2. Identify the relevant KBS Bank product.
3. Obtain verified information through RAG when factual bank information is required.
4. Communicate the relevant information naturally.
5. Ask a useful next question.
6. Continue the sales conversation.

Do not dump all available product information.

Lead with the information most relevant to the customer's situation.

For example, if the customer is planning a home purchase, do not unnecessarily explain personal loans, gold loans, education loans, vehicle loans, and business loans.

Focus on the home-loan opportunity.

==================================================
TRUTH AND KNOWLEDGE RULE
========================

This rule is mandatory.

NEVER invent KBS Bank information.

You must not make up:

* interest rates
* loan amounts
* minimum or maximum amounts
* tenure
* eligibility
* income requirements
* documents
* fees
* charges
* processing fees
* approval conditions
* collateral/security requirements
* repayment conditions
* offers
* discounts
* special schemes
* approval timelines
* product features
* customer-specific eligibility
* guarantees

Any KBS Bank-specific factual information must come from verified RAG information provided in the conversation.

RAG is the authoritative source for KBS Bank product knowledge.

If verified RAG information is provided, use it.

If the required information is NOT provided by RAG, do not guess.

Instead, say naturally that the exact detail can be confirmed by a KBS Bank specialist.

Never create a plausible-sounding bank fact simply to keep the conversation flowing.

==================================================
HOW TO USE RAG INFORMATION
==========================

RAG is a knowledge source.

It is NOT the salesperson.

When verified RAG information is supplied:

* understand it
* select the facts relevant to the customer
* explain those facts naturally
* connect them to the customer's need
* continue the sales conversation

Do NOT say:

* "According to RAG..."
* "The retrieved context says..."
* "The knowledge base says..."
* "The system found..."
* "The document states..."

The customer should feel that Priya Sharma knows the bank's products.

Do not repeat the entire RAG answer word-for-word if natural communication would be better.

Do not add facts that were not in the verified information.

==================================================
SELLING STYLE
=============

Your communication should feel:

* confident
* warm
* professional
* conversational
* helpful
* persuasive without being aggressive
* concise
* customer-focused

Do not sound robotic.

Do not sound like a form.

Do not repeatedly say:

"How can I help you?"

Instead, take initiative as a salesperson.

Use natural transitions such as:

"That could actually be relevant in your situation."

"If you're planning that purchase soon, I can explain how the home-loan option works."

"That makes sense. In that case, let me quickly explain the option that may be relevant."

==================================================
OBJECTION HANDLING
==================

When the customer raises an objection, understand the objection before responding.

Common objections may include:

* high interest
* not interested
* already have a loan
* don't need a loan
* just checking
* need to think
* want to compare
* busy
* call later
* concerned about eligibility
* concerned about repayment
* concerned about cost

Do not argue.

Do not pressure the customer.

Acknowledge the concern and respond with relevant verified information if available.

If more information is required, use RAG.

Then ask one natural question or offer the appropriate next step.

==================================================
CUSTOMER REFUSAL
================

If the customer clearly says they are not interested:

Do not repeatedly pressure them.

You may make a brief, polite attempt to understand whether there is a future need if appropriate.

If the customer clearly wants to end the conversation, respect that and end the call.

==================================================
BUSY / RESCHEDULE
=================

If the customer says they are busy or asks to speak later:

Do not continue selling.

The correct next action is rescheduling.

Use the reschedule function/action when available.

Do not claim that a rescheduled appointment has been created unless the function actually completes it.

==================================================
END CALL
========

If the customer clearly wants to end the call, or the sales conversation has reached an appropriate conclusion:

End politely.

Use the end-call function/action when available.

Do not continue asking unnecessary sales questions after the customer has clearly ended the conversation.

==================================================
E-TRANSFER
==========

If the customer requests an e-transfer or another supported transfer/payment action:

Recognize that as an action request.

Use the appropriate e-transfer function/action when available.

Never claim that an e-transfer was completed unless the function actually completed it.

Never fabricate transaction confirmation.

==================================================
FUNCTIONS
=========

The planner may identify situations requiring functions such as:

* reschedule
* end_call
* e_transfer

When a function is required, follow the actual function behavior available to the application.

Do not pretend a function succeeded when it did not execute successfully.

==================================================
CONVERSATION MEMORY
===================

Use the conversation history to understand context.

However, distinguish between:

1. what the customer actually said,
2. what the salesperson said,
3. verified KBS Bank facts,
4. internal planning information.

Never convert an internal assumption into a customer fact.

Never treat an old or unrelated product discussion as evidence that the customer currently wants that product.

==================================================
RESPONSE LENGTH
===============

This is a voice conversation.

Normally speak in approximately 2–4 natural sentences.

Do not give long speeches.

Do not dump the entire product catalogue.

Do not ask several questions together.

Usually finish with ONE useful forward-moving question when a question is appropriate.

However, do not force a question when a natural statement or function action is more appropriate.

==================================================
MOST IMPORTANT RULE
===================

At every turn, behave like Priya Sharma, an experienced KBS Bank loan salesperson.

Understand the customer first.

Think about the sales opportunity.

Decide what would genuinely move the conversation forward.

Use verified KBS Bank knowledge when needed.

Then communicate naturally.

Never fabricate.

Never assume a customer need that was not established.

Never turn the conversation into a rigid questionnaire.

The objective is a natural, intelligent, proactive sales conversation that can move from:

identity confirmation
→ introduction
→ awareness
→ need discovery
→ relevant product discussion
→ qualification
→ objection handling
→ next step / conversion

while always remaining truthful about KBS Bank's actual products and conditions.

"""


# ═══════════════════════════════════════════════════════════════════════════
# PLANNER + RAG PROCESSOR (prompts unchanged - only the call-control directives
# now say "call the function" instead of "append the marker")
# ═══════════════════════════════════════════════════════════════════════════

class LLMDrivenRAGProcessor(FrameProcessor):
    """
    LLM decides:
      1. What the customer means
      2. Whether bank knowledge is needed
      3. What query should be sent to RAG

    Then the verified facts are handed to the MAIN LLM.

    Placement: this processor sits AFTER context_aggregator.user() and BEFORE
    the main LLM. It only acts on the LLMContextFrame of a finished customer
    turn, so it never touches STT frames or the aggregator's turn detection.
    """

    def __init__(self, rag_system, context):
        super().__init__()

        self.rag_system = rag_system      # may be None (planner then runs without facts)
        self.context = context            # kept for compatibility; the live context comes from each frame

        self.last_assistant_text = ""

        # Set from flow_manager.state["name"] (client message) - defaults to the env value
        self.customer_name = CUSTOMER_NAME

        # Conversation history for the planner LLM
        self.conversation_history = []

        # Once call is ended, ignore all further customer input
        self.call_ended = False

        # True after we asked the customer for a callback time
        self.reschedule_pending = False

        # (customer-turn number, text) of the turn already planned
        self._handled_turn = None

        # Specialist-offer tracking (Part 2 loop prevention)
        self._specialist_offers_made = 0        # total offer_specialist turns so far
        self._turns_since_last_offer = 0        # customer turns elapsed since last offer
        self._last_action = None                # final next_action of the previous turn

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _role(msg):
        if isinstance(msg, dict):
            return msg.get("role")
        return getattr(msg, "role", None)

    @staticmethod
    def _msg_text(msg) -> str:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "").strip()

    @staticmethod
    def _as_bool(value) -> bool:
        # bool("false") is True in Python - small models sometimes return strings
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)

    def _is_directive(self, msg) -> bool:
        # Accept both "system" (old location) and "user" (new location – appended
        # after the customer's message so Qwen's chat template keeps it close).
        return (
            self._role(msg) in ("system", "user")
            and self._msg_text(msg).startswith(DIRECTIVE_TAG)
        )

    @staticmethod
    def _set_messages(ctx, new_messages):
        if hasattr(ctx, "set_messages"):
            ctx.set_messages(new_messages)
        else:
            ctx.messages[:] = new_messages

    def _inject(self, ctx, text):
        """
        Make `text` THE directive for this turn.

        Every earlier turn directive (facts, reschedule, offer_specialist,
        END_CALL / TRANSFER_CALL instructions, the greeting instruction) is
        removed first so old facts cannot leak into later turns.

        The new directive is appended at the END of the context as a "user"
        message AFTER the customer's latest message.  This keeps the facts
        immediately adjacent to the model's current decision point and avoids
        Qwen's chat-template behaviour of merging all "system" messages into
        a single block at the top of the prompt (which buries late-arriving
        facts far from the customer's latest utterance).
        """
        kept = [m for m in ctx.messages if not self._is_directive(m)]

        if text:
            directive = {"role": "user", "content": f"{DIRECTIVE_TAG}\n{text.strip()}"}
            kept.append(directive)

        self._set_messages(ctx, kept)

    def _record_customer(self, text: str, turn_key):
        self.conversation_history.append({"role": "customer", "content": text})
        self._handled_turn = turn_key

    async def _lookup_facts(self, rag_query: str) -> str:
        """Blocking retrieval runs in a worker thread and is time-boxed."""
        loop = asyncio.get_running_loop()
        try:
            facts = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self.rag_system.answer_question_outbound(rag_query)
                ),
                timeout=RAG_TIMEOUT_S,
            )
            facts = facts or ""
            if not facts:
                return ""
            # Drop the refine output's trailing question if it ends with "?".
            # The refine LLM sometimes appends "Would you like more details...?"
            # which competes with the closing question the main LLM must ask.
            paragraphs = facts.strip().split("\n\n")
            last = paragraphs[-1].strip()
            # Check last sentence of the last paragraph
            sentences = [s.strip() for s in last.replace("?", "?\n").splitlines() if s.strip()]
            if sentences and sentences[-1].endswith("?"):
                # Remove the trailing question sentence
                trimmed_sentences = sentences[:-1]
                if trimmed_sentences:
                    paragraphs[-1] = " ".join(trimmed_sentences)
                else:
                    paragraphs = paragraphs[:-1]
                facts = "\n\n".join(paragraphs).strip()
            return facts
        except asyncio.TimeoutError:
            logger.warning(f"[RAG] Lookup exceeded {RAG_TIMEOUT_S}s - continuing without facts")
            return ""
        except Exception as e:
            logger.exception(f"[RAG] Lookup failed: {e}")
            return ""

    _REJECTION_RE = re.compile(
        r"\b(not interested|no thanks|no thank you|stop calling|don'?t call|"
        r"do not call|remove my number|no need)\b",
        re.IGNORECASE,
    )

    def _fallback_plan(self, user_text: str, why: str) -> dict:
        """
        Used ONLY when the planner LLM is down/slow. The normal path stays
        meaning-based; this just makes sure an explicit "stop calling me" is
        still honoured instead of being pitched at.
        """
        logger.warning(f"[PLANNER] Fallback plan used ({why})")
        action = "end_call" if self._REJECTION_RE.search(user_text) else "sell"
        return {
            "needs_rag": False,
            "rag_query": "",
            "reason": why,
            "customer_intent": "",
            "known_product": "",
            "next_action": action,
        }

    async def ask_planner_llm(self, user_text: str):
        """
        Planner LLM:
        Understands the entire conversation and decides:
            - whether RAG is needed
            - what information is needed
            - the best contextual query for RAG

        This LLM NEVER writes the final customer response.
        """

        import aiohttp
        import json

        # Earlier turns = context. The latest customer message is not recorded
        # yet (see process_frame) and is passed separately below as the
        # strongest signal for this decision.
        history_text = "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in self.conversation_history[-8:]
        ) or "(no earlier turns)"

        planner_prompt = f"""You are the sales planner for Priya Sharma, an experienced outbound loan salesperson at KBS Bank.

You decide what a good salesperson should do next and, if verified bank information is needed, write ONE focused RAG query.
You never write the customer reply. RAG only retrieves verified KBS Bank facts. The main LLM speaks to the customer.

CONVERSATION SO FAR:
{history_text}

LATEST CUSTOMER MESSAGE (strongest signal):
{user_text}

SPECIALIST OFFER TRACKING (for this call so far):
Specialist offers made: {self._specialist_offers_made}
Customer turns since last offer: {self._turns_since_last_offer}

HOW TO DECIDE
0. If Priya's last message asked "am I speaking with <name>?" and the reply is "No", "wrong number" or otherwise shows this is not that person,
   choose wrong_person. That "No" is NOT a loan rejection.
1. Read the latest message in the context of the conversation, by meaning and not by keywords.
   Example: the salesperson mentioned several loan options and the customer says "I'm planning to buy a house" -> home purchase need -> home loan.
   Example: "I want to buy a car next year" -> possible vehicle loan need.
   Decide fresh every turn. Do not carry over an earlier "no product" conclusion if the latest message shows a need.
   A bare "yes" / "hello" / "I have time" does NOT imply any product.
2. Customer mentioned a need or product -> query only that product, shaped to their situation
   (e.g. "KBS Bank home loan options, loan amount, interest rate and repayment tenure for a customer planning to buy a home").
   Never ask for all loans when a specific need is clear.
3. Customer has not mentioned any loan (identity just confirmed, "yes I have time", general openness) ->
   this is a sales call: the salesperson proactively introduces KBS Bank and gives a short overview of relevant loan options to create interest.
   Do NOT make the customer choose a loan first. Query: a concise overview of relevant KBS Bank loan products.
4. Question or objection that needs a bank fact -> one focused query for that fact. If it can be handled conversationally -> no RAG.
5. Keep moving the conversation forward like a real salesperson. Think one step ahead only.
6. WHEN TO OFFER A SPECIALIST: Choose offer_specialist when the customer has heard relevant information AND shows interest or buying
   signals, asks about applying/eligibility/documents/next steps, or the conversation has reached a natural closing point.
   A passive "okay" or "hmm" after information is NOT automatically a trigger; judge from the whole conversation whether to share more
   relevant info, ask a need-discovery question, or offer the specialist.
   Do NOT offer a specialist as a reflex. Repeating the same offer on consecutive turns is never appropriate.
7. If Priya's last message offered a specialist or asked whether to proceed, and the customer agrees in any way
   ("okay", "yes", "sure", "go ahead", "please"), choose transfer.
8. AFTER A DECLINED OFFER: If the customer declined or deflected a specialist offer ("no", "not now", "let me think"),
   that declines the specialist — NOT the loan. Go back to marketing the loan: acknowledge briefly, then use a
   focused RAG query for another relevant aspect (a feature, eligibility, documents, charges, a benefit tied to what they said)
   or ask a natural question about their plans, timing or concern. Choose sell or pitch_overview accordingly.
   Re-offering later is fine when something has changed: new interest, a buying signal, the customer asks, or the
   conversation has clearly progressed since the decline. Never re-offer simply because some turns have passed.
9. A clear "no / not interested / no need / stop" is a rejection -> end_call immediately.

RAG RULES
- needs_rag = true only when the next reply needs a KBS Bank-specific fact (rates, amounts, tenure, eligibility, fees, documents, conditions, features).
- Exactly ONE query: specific, based on the actual customer situation, no conversation chatter, no lists or alternatives.
- needs_rag = false for greetings, acknowledgements, busy, ending the call and e-transfer requests.
- Never invent KBS Bank facts or customer facts (income, existing loans, plans, eligibility).

next_action - choose exactly one:
- wrong_person: the person on the line is NOT the intended customer (e.g. answers "No" to the identity question, "I'm not Mohith", "wrong number", "he's not here", "this is his brother/wife"). Never pitch or share loan details with them.
- reschedule: customer is busy / not now / asks to be called later
- end_call: customer clearly wants to end the conversation, stop being called, OR is clearly not interested ("not interested", "no need", "I don't want any loan", "no thanks"). Do not keep pitching.
- transfer: customer wants a specialist, wants to apply, is ready to proceed, or agreed to Priya's specialist / proceed offer
- offer_specialist: customer has heard relevant information and shows genuine interest, buying signals, or asks about applying/eligibility/documents/next steps; or the conversation has reached a natural closing point
- e_transfer: customer requests an e-transfer / payment action
- pitch_overview: no specific need yet; introduce KBS Bank and its loan options to create interest
- sell: anything else in the sales conversation (explain a product, answer a question, handle an objection, discover a need, continue after a declined specialist offer)
For wrong_person, reschedule, end_call, transfer and e_transfer: needs_rag = false and rag_query = "".

Return ONLY valid JSON with exactly this structure:
{{
"customer_intent": "what the customer is communicating right now",
"customer_interest": "level/type of interest supported by the conversation",
"sales_stage": "current stage",
"known_product": "product supported by the conversation, or none",
"customer_need": "stated or clearly supported need, or not established",
"conversation_goal": "what Priya should accomplish in her next reply",
"next_action": "one of the actions above",
"needs_rag": true,
"rag_query": "ONE focused query, or empty string",
"reason": "brief reason"
}}

Examples of the mapping (outputs abbreviated):
Customer "Yes, I have time." -> next_action pitch_overview, known_product none, needs_rag true, rag_query "KBS Bank loan products and key customer-facing features for an outbound introduction".
Customer "I'm planning to buy a house." -> next_action sell, known_product home loan, needs_rag true, rag_query "KBS Bank home loan options, loan amount, interest rate and repayment tenure for a customer planning to buy a home".
Priya just explained home loan rate and tenure; Customer "Okay." -> next_action sell (or offer_specialist if natural closing point), judge from full conversation.
Priya just asked "Would you like me to connect you with a specialist?"; Customer "Okay." -> next_action transfer, needs_rag false, rag_query "".
Priya just offered a specialist; Customer "No, not now." -> next_action sell, needs_rag true (query for another relevant product aspect), rag_query "KBS Bank home loan eligibility and documents required".
Priya asked "am I speaking with Mohith?"; Customer "No, this is his brother." -> next_action wrong_person, needs_rag false, rag_query "".
Customer "I'm not interested." -> next_action end_call, needs_rag false, rag_query "".
Customer "Call me tomorrow, I'm busy." -> next_action reschedule, needs_rag false, rag_query "".
"""

        payload = {
            "model": PLANNER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert conversation planner "
                        "for a bank outbound loan agent. "
                        "Return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": planner_prompt
                }
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "stream": False,
            "max_tokens": 350,
            "response_format": {"type": "json_object"}
        }

        try:
            timeout = aiohttp.ClientTimeout(total=PLANNER_TIMEOUT_S)

            async with aiohttp.ClientSession(timeout=timeout) as session:

                async with session.post(
                    f"{OLLAMA_BASE_URL}/chat/completions",
                    json=payload
                ) as response:

                    if response.status != 200:
                        logger.error(
                            f"[PLANNER] HTTP {response.status}"
                        )

                        return self._fallback_plan(user_text, "Planner unavailable")

                    data = await response.json()

                    output = data["choices"][0]["message"]["content"]

                    logger.info("=" * 80)
                    logger.info("[PLANNER RAW OUTPUT]")
                    logger.info(output)
                    logger.info("=" * 80)

                    output = output.strip()

                    # Remove markdown JSON fences
                    if output.startswith("```"):
                        output = re.sub(
                            r"```(?:json)?",
                            "",
                            output
                        )
                        output = output.replace("```", "").strip()

                    # Tolerate any text around the JSON object
                    json_match = re.search(r"\{.*\}", output, re.DOTALL)
                    result = json.loads(
                        json_match.group(0) if json_match else output
                    )

                    logger.info(f"[PLANNER] Intent: {result.get('customer_intent')}")
                    logger.info(f"[PLANNER] Interest: {result.get('customer_interest')}")
                    logger.info(f"[PLANNER] Sales stage: {result.get('sales_stage')}")
                    logger.info(f"[PLANNER] Product: {result.get('known_product')}")
                    logger.info(f"[PLANNER] Customer need: {result.get('customer_need')}")
                    logger.info(f"[PLANNER] Conversation goal: {result.get('conversation_goal')}")
                    logger.info(f"[PLANNER] Next action: {result.get('next_action')}")
                    logger.info(f"[PLANNER] RAG needed: {result.get('needs_rag')}")
                    logger.info(f"[PLANNER] RAG query: {result.get('rag_query')}")
                    logger.info(f"[PLANNER] Reason: {result.get('reason')}")

                    return result

        except asyncio.TimeoutError:

            logger.warning(
                f"[PLANNER] Timed out after {PLANNER_TIMEOUT_S}s"
            )

            return self._fallback_plan(user_text, "Planner timeout")

        except Exception as e:

            logger.exception(
                f"[PLANNER] Failed: {e}"
            )

            return self._fallback_plan(user_text, "Planner error")

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ):
        await super().process_frame(frame, direction)

        # The planner now runs ONCE per completed customer turn, on the
        # LLMContextFrame that the user aggregator emits right before the main
        # LLM. (Before, it sat in front of the aggregator and reacted to every
        # TextFrame - interim/eager transcripts included - and held the
        # transcript back while the planner and RAG were running.)
        if not isinstance(frame, LLMContextFrame) or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        ctx = frame.context
        messages = list(ctx.messages)

        # Greeting run (LLMRunFrame) or any frame where the customer did not
        # just speak: nothing to plan.
        # Strip directive messages before checking roles/text so that a
        # directive injected as "user" does not look like a customer turn.
        real_messages = [m for m in messages if not self._is_directive(m)]
        if not real_messages or self._role(real_messages[-1]) != "user":
            await self.push_frame(frame, direction)
            return

        user_text = self._msg_text(real_messages[-1])
        turn_key = (
            sum(1 for m in real_messages if self._role(m) == "user"),
            user_text,
        )

        # Empty transcript, or the very same turn seen again (its directive is
        # already in the context). The turn number is part of the key, so a
        # customer who says "okay" twice in a row is NOT dropped as a duplicate.
        if not user_text or turn_key == self._handled_turn:
            await self.push_frame(frame, direction)
            return

        # Call already ended: no more planning; the goodbye directive stays.
        if self.call_ended:
            self._handled_turn = turn_key
            await self.push_frame(frame, direction)
            return

        turn_started = time.time()

        logger.info("=" * 80)
        logger.info(f"[CUSTOMER] {user_text}")
        logger.info("=" * 80)

        # ---------------------------------------------------------
        # RESCHEDULE FOLLOW-UP: this message is the callback time.
        # Do not re-plan or pitch - confirm it and end the call.
        # ---------------------------------------------------------
        if self.reschedule_pending:
            self.reschedule_pending = False
            self._record_customer(user_text, turn_key)
            logger.info("[PLANNER] Reschedule follow-up -> confirm time and end call")

            self._inject(ctx, f"""The customer has just replied to your question about when to call them back.
Do NOT pitch any product. Do NOT ask any more questions.
Call the end_conversation function now with reason="callback_scheduled" and callback_time set to the time they gave (empty if they gave no specific time).
Do NOT write any spoken text in this reply - the confirmation and goodbye are spoken right after the function runs.
Exception: if the customer clearly says they can talk now, OR the intended customer ({self.customer_name}) has come on the line (e.g. "yes, speaking", "this is {self.customer_name}"), do NOT end the call and do NOT call any function. Continue normally: greet them, introduce yourself and KBS Bank briefly and explain the reason for the call.""")

            await self.push_frame(frame, direction)
            return

        # ---------------------------------------------------------
        # PLANNER: understand customer + decide next action / RAG query
        # (bounded by PLANNER_TIMEOUT_S, falls back safely)
        # ---------------------------------------------------------
        planner_result = await self.ask_planner_llm(user_text)

        # Recorded only after the planner returned, so a barge-in that cancels
        # this coroutine leaves no half-recorded turn behind.
        self._record_customer(user_text, turn_key)

        needs_rag = self._as_bool(planner_result.get("needs_rag", False))
        rag_query = str(planner_result.get("rag_query") or "").strip()
        reason = planner_result.get("reason", "")
        next_action = str(
            planner_result.get("next_action") or "sell"
        ).strip().lower()

        if next_action not in VALID_ACTIONS:
            logger.warning(f"[PLANNER] Unknown next_action {next_action!r} -> sell")
            next_action = "sell"

        # ── Part 2: back-to-back offer_specialist guard ──────────────────
        # If the planner chose offer_specialist on consecutive turns and the
        # customer's previous reply was not an agreement, treat it as sell.
        if (
            next_action == "offer_specialist"
            and self._last_action == "offer_specialist"
        ):
            # The customer did NOT agree (otherwise the planner would have
            # picked transfer). Offering again immediately is the loop.
            logger.warning(
                "[PLANNER] Back-to-back offer_specialist detected -> treating as sell"
            )
            next_action = "sell"

        if next_action == "end_call":
            logger.info("[PLANNER] Next action = end_call")
            self.call_ended = True

            self._inject(ctx, """
The customer does not want to continue or is not interested.
Do not ask any more questions. Do not mention loans or try to convince them again.
Call the end_conversation function now with reason="not_interested" (use reason="customer_ended" if they simply want to end the call).
Do NOT write any spoken text in this reply - the apology and goodbye are spoken right after the function runs.
""")

            self._last_action = "end_call"
            await self.push_frame(frame, direction)
            return

        if next_action == "wrong_person":
            logger.info("[PLANNER] Next action = wrong_person")
            # If we ask for a callback time, the next message is handled
            # by the reschedule follow-up (confirm + end).
            self.reschedule_pending = True

            self._inject(ctx, f"""The person on the line is NOT {self.customer_name}.
Do NOT introduce any loan or share any bank or loan details with this person. Do NOT pitch anything.
- If they say it is a wrong number or they do not know {self.customer_name}: call the end_conversation function now with reason="wrong_number". Do NOT write any spoken text in this reply - the apology and goodbye are spoken right after the function runs.
- If {self.customer_name} is simply not available or they know {self.customer_name}: apologise briefly for the inconvenience (ONE or TWO short sentences) and politely ask when would be a good time to reach {self.customer_name}. Do NOT call any function yet.
- If they say {self.customer_name} is available and will come on the line: apologise briefly and ask them to please hand over the phone. Do NOT call any function.""")

            self._last_action = "wrong_person"
            await self.push_frame(frame, direction)
            return

        if next_action == "reschedule":
            logger.info("[PLANNER] Next action = reschedule")
            self.reschedule_pending = True

            self._inject(ctx, f"""{self.customer_name} is busy or asked for a callback. Apologise briefly for the bad timing and respond warmly in ONE sentence.
Do NOT pitch any product.
- If {self.customer_name} already gave a time in this message (e.g. "tomorrow", "after 5", "10 AM"): call the end_conversation function now with reason="callback_scheduled" and callback_time set to that time. Do NOT write any spoken text in this reply - the confirmation and goodbye are spoken right after the function runs.
- Otherwise ask: "Of course, no problem, sorry for the bad timing - when would be a better time for me to call you back?" Do NOT call any function yet.""")

            self._last_action = "reschedule"
            await self.push_frame(frame, direction)
            return

        if next_action == "offer_specialist":
            logger.info("[PLANNER] Next action = offer_specialist")

            # Vary wording if this is a repeated offer
            if self._specialist_offers_made > 0:
                offer_wording = (
                    f"You've previously offered to connect {self.customer_name} with a specialist. "
                    f"This time, tie the offer to what they just showed interest in — word it differently and naturally. "
                    f"Do not repeat the same phrasing as before."
                )
            else:
                offer_wording = (
                    "Briefly acknowledge what the customer just said or asked. "
                    "Mention that a loan specialist can give them full details and get things started quickly. "
                    "Ask if they would like to be connected — keep it warm and natural."
                )

            self._inject(ctx, f"""
The internal conversation planner has determined that {self.customer_name} has heard the loan
information and it is time to move the sale to the next step.

{offer_wording}

Example (for first offer):
"That's a great question — our loan specialists have all the exact details on rates and
eligibility. Would you like me to connect you with one of them right now so they can
walk you through everything?"

If the customer says YES or agrees in any way, call the transfer_to_agent function
(without writing any spoken text - the short transition is spoken right after the function runs).

If the customer says NO or wants to continue the conversation with you, continue
the sales conversation naturally without offering the specialist again immediately.

Do NOT ask another unrelated question in the same turn.
Never mention functions or internal systems to the customer.
""")

            # Update counters AFTER injecting so they reflect this turn
            self._specialist_offers_made += 1
            self._turns_since_last_offer = 0
            self._last_action = "offer_specialist"
            await self.push_frame(frame, direction)
            return

        if next_action == "transfer":
            logger.info("[PLANNER] Next action = transfer")

            self._inject(ctx, """
The internal conversation planner has determined that the customer should
now be transferred to a human loan officer.

Call the transfer_to_agent function now.
Do NOT write any spoken text in this reply - the short transition message
("I'll connect you with a loan specialist right away") is spoken right after the function runs.
Do not ask another question.
""")

            self._last_action = "transfer"
            await self.push_frame(frame, direction)
            return

        if next_action == "e_transfer":
            # There is no e-transfer function wired into this bot, so the
            # model must not pretend one ran (it used to get no guidance here).
            logger.info("[PLANNER] Next action = e_transfer (no function available)")

            self._inject(ctx, """
The customer asked for an e-transfer / payment action. You CANNOT perform it on this call.
Do NOT say or imply that any transfer or payment was made, started or confirmed.
Say in ONE or TWO short sentences that a KBS Bank specialist will help with that, and offer to connect them.
Do not pitch any product.
""")

            self._last_action = "e_transfer"
            await self.push_frame(frame, direction)
            return

        logger.info(f"[PLANNER] needs_rag = {needs_rag}")
        logger.info(f"[PLANNER] reason = {reason}")
        logger.info(f"[PLANNER] RAG query = {rag_query}")

        # ---------------------------------------------------------
        # RAG (facts only - no second "answer-writing" LLM call)
        # ---------------------------------------------------------
        known_product = planner_result.get("known_product") or "none"
        sales_goal = planner_result.get("conversation_goal") or ""
        customer_need = planner_result.get("customer_need") or ""
        rag_attempted = bool(needs_rag and rag_query and self.rag_system is not None)
        facts = ""

        if rag_attempted:
            logger.info("=" * 80)
            logger.info("[RAG] LLM-GENERATED QUERY:")
            logger.info(rag_query)
            logger.info("=" * 80)

            rag_started = time.time()
            facts = await self._lookup_facts(rag_query)
            logger.info(
                f"[LATENCY] RAG lookup: {time.time() - rag_started:.2f}s "
                f"({len(facts)} chars of verified facts)"
            )

        if facts:
            if next_action == "pitch_overview":
                how_to_speak = "PITCH OVERVIEW: Present the loan types temptingly using the facts above. Mention each product with its best rate, max amount and tenure. End with ONE soft question about what they may be planning."
            else:
                how_to_speak = "Speak these facts naturally and confidently. Lead with the most useful number. After sharing facts, ask ONE closing question: whether they would like to go ahead with this loan or be connected to a loan specialist."

            offer_line = (
                "Offer to connect with a loan specialist after stating the facts."
                if next_action == "offer_specialist" else ""
            )

            self._inject(ctx, f"""NEXT ACTION: {next_action}
CUSTOMER JUST SAID: "{user_text}"
VERIFIED KBS BANK INFORMATION (from the bank's product guide):
{facts}

REPLY RULES - follow exactly:
1. Your reply MUST begin by telling the customer this information, in your own natural spoken words as Priya. Include every figure (amounts, rates, tenure) exactly as written above. Present it as what KBS Bank offers, not as something the customer asked for.
2. Do NOT ask what they are planning, and do NOT ask how you can help, BEFORE you have told them this information.
3. After that, ask exactly ONE short question that moves the sale forward.
4. Use ONLY the information above for bank-specific details. If something the customer asked is not covered, say a loan specialist can confirm it.
5. Never say "How can I help you?". 3-4 short sentences. Never mention notes, RAG or internal systems.
{offer_line}
(Mention a specialist only if next_action == offer_specialist or if the facts do not cover something the customer asked.)""")

        elif rag_attempted:
            logger.info("[RAG] No confident result found")
            self._inject(ctx, f"""NEXT ACTION: {next_action}
CUSTOMER NEED: {customer_need}
SALES GOAL: {sales_goal}

No verified KBS Bank information is available for this question.
Do NOT state or guess any rate, amount, tenure, eligibility, fee, document or condition.
Say honestly that you will have a loan specialist confirm those details, and keep the conversation moving.
Keep response to 1-2 sentences. NEVER mention RAG or internal systems.""")

        elif next_action in ("sell", "pitch_overview"):
            logger.info("[RAG] Not required for this turn")
            self._inject(ctx, f"""NEXT ACTION: {next_action}
CUSTOMER NEED: {customer_need}
SALES GOAL: {sales_goal}

Keep marketing the loan itself. Acknowledge what the customer said, then ask ONE question about their plans or timing, or point to one reason the loan could fit their situation.
Do NOT state any KBS Bank rate, amount, tenure, eligibility, fee or condition — none is verified for this turn.
Do NOT mention a specialist in this reply.
Keep response to 1-2 sentences.""")

        else:
            logger.info("[RAG] Not required for this turn")
            # Still clear last turn's directive/facts so they cannot leak in.
            self._inject(ctx, None)

        # Update turn-since-offer counter and last_action for all sell/pitch paths
        if self._last_action == "offer_specialist":
            self._turns_since_last_offer += 1
        self._last_action = next_action

        logger.info(
            f"[LATENCY] Planner + RAG before main LLM: {time.time() - turn_started:.2f}s"
        )

        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# GOODBYE STYLE PER END REASON (spoken by the end node)
# ═══════════════════════════════════════════════════════════════════════════

END_REASONS = ["not_interested", "customer_ended", "wrong_number", "callback_scheduled"]

END_REASON_GUIDANCE = {
    "not_interested": (
        "The customer is not interested. Acknowledge their decision warmly without pressure. "
        "Thank them for their time, let them know KBS Bank is always available if their needs change, "
        "and wish them well — two to three short natural sentences.\n"
        'Example: "No problem at all, I completely understand. Thank you so much for your time today. '
        'If you ever need any financial support in future, KBS Bank is always here for you — '
        'have a wonderful day!"\n'
        "Do not ask any question. Do not pitch any product. Do not pressure them."
    ),
    "customer_ended": (
        "The customer wants to end the call. Thank them for their time and say a warm goodbye - "
        "one or two short sentences only. Do not ask any question. Do not pitch anything."
    ),
    "wrong_number": (
        "You reached the wrong person. Apologise briefly for the inconvenience, thank them and say goodbye - "
        "one or two short sentences only. Do not mention any loan or bank detail.Do not ask any question"
    ),
    "callback_scheduled": (
        "Confirm the callback time in ONE warm sentence, then say goodbye. "
        "If no specific time was given, say you will call at a convenient time. "
        "Do not pitch any product. Do not ask any question."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# SPEECH GATE - lets the post-actions wait until the bot has finished talking
# (add it to the pipeline AFTER transport.output())
# ═══════════════════════════════════════════════════════════════════════════

class BotSpeechGate(FrameProcessor):
    """Observes BotStarted/StoppedSpeaking so hang-up / transfer happen AFTER the closing sentence."""

    def __init__(self):
        super().__init__()
        self._done = asyncio.Event()
        self._started = False
        self._speaking = False

    def arm(self):
        """Call right before the closing message is generated."""
        self._started = False
        self._done.clear()

    async def wait_until_spoken(self, timeout: float = 20.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning("[FLOW] Timed out waiting for the bot to finish speaking")
                return
            try:
                await asyncio.wait_for(self._done.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("[FLOW] Timed out waiting for the bot to finish speaking")
                return
            await asyncio.sleep(0.7)      # maybe just a gap between two sentences
            if not self._speaking:
                return
            self._done.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._started = True
            self._speaking = True
            self._done.clear()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            if self._started:
                self._done.set()

        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _call_id(flow_manager: FlowManager) -> str:
    return str(flow_manager.state.get("callId") or DEFAULT_CALL_ID or "")


async def _wait_for_bot_to_finish(flow_manager: FlowManager):
    gate: Optional[BotSpeechGate] = flow_manager.state.get("speech_gate")
    if gate is not None:
        await gate.wait_until_spoken()


async def disconnect_after_speech(action: dict, flow_manager: FlowManager):
    """post_action of the end node: hang up once the goodbye was spoken."""
    await _wait_for_bot_to_finish(flow_manager)
    logger.info("📞 [END CALL] Final closing message completed")

    try:
        # requests is blocking: run it in a thread so the pipeline keeps flowing
        await asyncio.to_thread(
            requests.get,
            base_url + "/ai-agent/disconnect?callId=" + _call_id(flow_manager),
            verify=False,
            timeout=5,
        )
        logger.info("📞 [END CALL] WebRTC disconnect URL triggered")
    except Exception as e:
        logger.error("📞 [END CALL] WebRTC disconnect failed: {}", e)

    await asyncio.sleep(0.3)


async def transfer_after_speech(action: dict, flow_manager: FlowManager):
    """post_action of the transfer node: transfer once the hand-over sentence was spoken."""
    if flow_manager.state.get("transfer_done"):
        return
    flow_manager.state["transfer_done"] = True

    await _wait_for_bot_to_finish(flow_manager)
    logger.info("📞 [TRANSFER] Final transfer message completed")

    try:
        url = base_url + "/ai-agent/transfer?callId=" + _call_id(flow_manager)
        logger.info("📞 [TRANSFER] Calling URL: {}", url)

        response = await asyncio.to_thread(requests.get, url, verify=False, timeout=5)
        logger.info("📞 [TRANSFER] WebRTC transfer URL triggered: {}", response.status_code)
    except Exception as e:
        logger.error("📞 [TRANSFER] Transfer failed: {}", e)


def begin_end_call(flow_manager: FlowManager, reason: str = "customer_ended", callback_time: str = "") -> NodeConfig:
    """Shared by tool_end_conversation and the RTVI 'end-call' client message."""
    reason = (reason or "").strip().lower()
    if reason not in END_REASON_GUIDANCE:
        logger.warning(f"[FLOW] Unknown end reason {reason!r} -> customer_ended")
        reason = "customer_ended"

    logger.info(f"📞 [END CALL] Requested (reason={reason}, callback_time={callback_time!r})")

    flow_manager.state["call_ended"] = True
    flow_manager.state["end_reason"] = reason
    flow_manager.state["callback_time"] = callback_time

    # customer turns are ignored from now on
    planner = flow_manager.state.get("planner")
    if planner is not None:
        planner.call_ended = True

    gate: Optional[BotSpeechGate] = flow_manager.state.get("speech_gate")
    if gate is not None:
        gate.arm()

    return create_end_node(reason, callback_time)


def begin_transfer_call(flow_manager: FlowManager) -> NodeConfig:
    """Shared by tool_transfer_to_agent and the RTVI 'transfer-call' client message."""
    logger.info("📞 [TRANSFER] Requested")

    gate: Optional[BotSpeechGate] = flow_manager.state.get("speech_gate")
    if gate is not None:
        gate.arm()

    return create_transfer_node()


# ═══════════════════════════════════════════════════════════════════════════
# INITIALISE STATE
# ═══════════════════════════════════════════════════════════════════════════

async def initialize_user(flow_manager: FlowManager):

    state = flow_manager.state

    state["name"] = state.get("name") or CUSTOMER_NAME
    state["callId"] = state.get("callId") or DEFAULT_CALL_ID
    state["msisdn"] = state.get("msisdn")
    state["greeted"] = False          # set True once the identity question was asked
    state["call_ended"] = False
    state["transfer_done"] = False

    planner = state.get("planner")
    if planner is not None:
        planner.customer_name = state["name"]

    print("name:", state["name"], "| callId:", state["callId"], "| msisdn:", state["msisdn"])


# ═══════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════

def create_initial_node(customer_name: str = "", greeted: bool = False) -> NodeConfig:

    customer_name = customer_name or CUSTOMER_NAME

    # ---------------------------------------------------------------------
    # node functions (handlers)
    # ---------------------------------------------------------------------
    async def tool_end_conversation(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        print("ENDING CONVERSATION")

        node = begin_end_call(
            flow_manager,
            args.get("reason", "customer_ended"),
            str(args.get("callback_time", "") or ""),
        )
        return {"status": "success", "action": "call_ending"}, node

    async def tool_transfer_to_agent(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        print("TRANSFERRING TO AGENT")

        node = begin_transfer_call(flow_manager)
        return {"status": "success", "action": "call_transferring"}, node

    # ---------------------------------------------------------------------
    node = NodeConfig(
        name="initial",
        respond_immediately=False,   # the identity question is spoken by pre_actions; the LLM waits for the customer

        role_messages=[
            {
                "role": "system",
                "content": OUTBOUND_SYSTEM_PROMPT
            }
        ],

        task_messages=[
            {
                "role": "system",
                "content": (
                    f"The intended customer is {customer_name}.\n"
                    f"You have already asked: \"Hi, am I speaking with {customer_name}?\" - wait for the reply.\n\n"
                    "From now on every customer turn is followed by an internal "
                    "[TURN INSTRUCTION - internal, never read aloud]. Follow it exactly.\n"
                    "- Call end_conversation ONLY when the turn instruction tells you to end the call.\n"
                    "- Call transfer_to_agent ONLY when the turn instruction tells you to transfer the call.\n"
                    "- When you call a function, write NO spoken text in that reply.\n"
                    "- Never mention functions, tools or internal instructions to the customer.\n"
                )
            }
        ],

        functions=[
            FlowsFunctionSchema(
                name="end_conversation",
                handler=tool_end_conversation,
                description=(
                    "End the phone call politely. Call this ONLY when the turn instruction tells you to end the call "
                    "(customer not interested, wrong person, callback time confirmed, or the customer clearly wants to hang up). "
                    "Do not write any spoken text in the same reply: the goodbye is spoken automatically right after this function runs."
                ),
                properties={
                    "reason": {
                        "type": "string",
                        "enum": END_REASONS,
                        "description": "Why the call ends.",
                    },
                    "callback_time": {
                        "type": "string",
                        "description": "Only for reason callback_scheduled: the callback time the customer gave, in their own words. Empty if none.",
                    },
                },
                required=["reason"],
            ),

            FlowsFunctionSchema(
                name="transfer_to_agent",
                handler=tool_transfer_to_agent,
                description=(
                    "Transfer the customer to a human KBS Bank loan specialist. Call this ONLY when the turn instruction tells you "
                    "to transfer the call (the customer wants a specialist, wants to apply, or agreed to be connected). "
                    "Do not write any spoken text in the same reply: the hand-over sentence is spoken automatically right after this function runs."
                ),
                properties={},
                required=[],
            ),
        ],
    )

    # Bot speaks first (outbound): identity confirmation only, spoken directly - no LLM involved.
    if not greeted:
        node["pre_actions"] = [
            {"type": "tts_say", "text": f"Hi, am I speaking with {customer_name}?"}
        ]

    return node


def create_end_node(reason: str = "customer_ended", callback_time: str = "") -> NodeConfig:
    """Goodbye is spoken by the LLM, then: hang up (after it was spoken) -> end the pipeline."""

    guidance = END_REASON_GUIDANCE.get(reason, END_REASON_GUIDANCE["customer_ended"])
    if reason == "callback_scheduled" and callback_time:
        guidance += f'\nThe callback time the customer gave: "{callback_time}".'

    return NodeConfig(
        name="end",

        task_messages=[
            {
                "role": "system",
                "content": (
                    "The end_conversation function has already run. The call is ending now.\n"
                    "Speak ONLY the closing message, in the same warm voice as before.\n"
                    f"{guidance}\n"
                    "Never mention functions, markers or internal systems."
                ),
            }
        ],

        functions=[],

        post_actions=[
            {"type": "function", "handler": disconnect_after_speech},
            {"type": "end_conversation"},
        ],
    )


def create_transfer_node() -> NodeConfig:
    """Hand-over sentence is spoken by the LLM, then the call is transferred."""

    return NodeConfig(
        name="transfer",

        task_messages=[
            {
                "role": "system",
                "content": (
                    "The transfer_to_agent function has already run. The customer is being handed over now.\n"
                    "In ONE short, warm sentence tell the customer you are connecting them with a "
                    "loan specialist right away. Do not ask a question. Do not give any product details.\n"
                    "Never mention functions, markers or internal systems."
                ),
            }
        ],

        functions=[],

        post_actions=[
            {"type": "function", "handler": transfer_after_speech},
        ],
    )