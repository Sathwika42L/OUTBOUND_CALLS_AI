# ═══════════════════════════════════════════════════════════════════════════
# 🧠 OUTBOUND PLANNER + CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════
"""
Planner LLM + RAG processor for the outbound loan bot.

What changed compared with the first node-flow version
------------------------------------------------------
1. PROACTIVE PLANNER. Besides "RAG or not", the planner now also
     - tracks what the customer has told us (facts + focus product),
     - proposes the single best NEXT QUESTION,
   and both are handed to the main LLM. The bot therefore drives the call
   toward an application instead of only reacting, and never re-asks things.

2. NO PROMPT GROWTH.
     - ONE "[CALL NOTES]" system message lives in the context. It is REPLACED
       every turn (never appended), so it costs the same on turn 2 and turn 40.
     - The long "how to reply" rules + examples are sent ONCE (conversation
       node) instead of being re-injected on every RAG turn.
     - Old chat turns are trimmed to a sliding window; the facts the planner
       tracks survive the trimming, so "don't ask again" still works.

3. SMALLER / FASTER PLANNER CALL.
     - Static instructions are in the system message (identical every call →
       Ollama can reuse its prompt cache); only a few lines change per turn.
     - Planner sees BOTH sides of the conversation (before it only saw the
       customer, so its 'BANK:' examples never matched anything).
     - JSON mode, small max_tokens, short timeout, reused HTTP session, and it
       fails open (no RAG) instead of stalling the call for 30 s.

4. STAGE EVENTS. The planner also reports identity_confirmed / not_interested /
   end_call, and this processor moves the node flow (outbound_flow.py). The main
   LLM has no tools any more - qwen2.5 via Ollama leaked its instructions and
   printed tool calls as text.

5. Duplicate-utterance guard now only ignores an identical transcript that
   repeats within ~1.5 s (STT double emit). Before, a second genuine "yes"
   later in the call was silently dropped.
"""

import asyncio
import json
import os
import re
import time

import aiohttp
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from flow_bot2 import CAMPAIGN, CUSTOMER_NAME

# ─── tunables (override with env vars, no code change needed) ───────────────
PLANNER_URL = os.getenv("PLANNER_URL", "http://202.164.134.176:11434/v1/chat/completions")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen2.5:14b")   # a 7B is usually enough for this job and faster
PLANNER_TIMEOUT_S = float(os.getenv("PLANNER_TIMEOUT_S", "8"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))
RAG_MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "2200"))          # cap on bank info injected per turn
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "16"))  # chat messages kept (trim runs before each turn → up to +2 during a turn)
PLANNER_HISTORY_TURNS = 6
DUPLICATE_WINDOW_S = 1.5
MAX_FACTS = 12
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))   # only used for the size warning below
VALID_EVENTS = (
    "identity_confirmed",
    "identity_denied",
    "not_interested",
    "end_call",
    "transfer_requested",
)

VALID_ACTIONS = (
    "search_kbs",
    "end_conversation",
    "transfer_to_agent",
    "none",
)
NOTES_MARKER = "[CALL NOTES]"
_EMPTY_PLAN = {
    "action": "none",
    "needs_rag": False,
    "rag_query": "",
    "product": "",
    "facts": {},
    "next_question": "",
    "event": "",
}


# ═══════════════════════════════════════════════════════════════════════════
# PLANNER PROMPT  (static → cache friendly; dynamic part is built per turn)
# ═══════════════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM_TEMPLATE = """You are the planning brain of a __COMPANY__ loan officer on an OUTBOUND call to __CUSTOMER__. You never speak to the customer; you only prepare the officer's next reply.

Each turn you get: the call STAGE and what is already known, the recent conversation (OFFICER / CUSTOMER) and the customer's latest message.

Your job is to keep the officer PROACTIVE and MARKETING-MINDED - always moving the customer toward taking a loan - while staying 100% factual. You never invent bank facts yourself; you only decide WHICH facts to fetch (rag_query) and WHEN to push toward the application (next_question, transfer_requested). The actual numbers always come from the bank knowledge base, never from you.

Decide six things:

1. needs_rag - true whenever the officer can proactively deliver ONE more piece of bank information about the current (or newly-identified) product: overview, amount, interest rate, tenure, eligibility, documents, fees, benefits. Default to true whenever a product is known and not everything about it has been said yet - do not wait for the customer to ask. Being proactive here is how the officer markets the loan with real facts instead of vague sales talk.
   CRITICAL: the moment the customer says what they want to buy or finance - a vehicle (car, truck, bike, bus), a house/property, a business need, education, or anything similar - that IS picking a product, even if phrased as their own plan ("I am planning for a truck", "I want to buy a house"). Do NOT file this only under facts/purpose and move on. Set product to the matching loan type in the SAME turn and set needs_rag=true so its details can be fetched immediately, before any follow-up question is asked.
   false only for: identity confirmation, greetings, thanks, a plain decline, or when the product has already been fully explained and nothing new is left to fetch.
2. rag_query - only if needs_rag. ONE complete, standalone, factual search query for the bank knowledge base. Never copy the customer's words; resolve "it", "the rate", "documents" using the known product. When the customer has just picked a product, ask for the full picture (amount, tenure, rate, eligibility, benefits) - the more the officer can proactively offer, the more persuasive the call.
3. product - the loan product the customer is focused on now ("" if none yet).
4. facts - ONLY new or changed facts the customer just stated about themselves.Keep values short (max 6 words). {} if none.
5. next_question - prefer a question that surfaces MORE bank/loan information (rate, eligibility, documents, "want to hear about X too?") or moves toward the application. Only suggest a question about the customer's own situation (income, employment, amount they want) once the product has been fully explained. If the product is being identified for the first time this turn, leave next_question "" - the officer must state loan information about it before asking anything else, including "new or used". "" if the customer wants to end the call or has declined.
   ONCE the product has been explained with real numbers AND the customer reacts positively (e.g. "sounds good", "that works", "I want to apply", "yes I'm interested", asks how to proceed) - stop asking more discovery questions and instead set next_question to an offer to connect them to a human loan officer, e.g. "Would you like me to connect you now to one of our loan officers who can help you complete the application?". This is the natural close of a successful proactive call.
6. event - "" unless one of these clearly happened:
   identity_confirmed: STAGE is identity and the customer confirms they are __CUSTOMER__ (yes, speaking, this is me).
   identity_denied: STAGE is identity and the customer's answer is anything other than a clear yes - a denial ("no", "wrong number", "there's no one here by that name"), a deflection, or silence/confusion about who they are. Do NOT wait for a second try and do NOT ask again - the officer must apologize and end the call the FIRST time identity is not clearly confirmed.
   not_interested: STAGE is conversation and the customer clearly declines taking any loan. A "no" to a follow-up question (for example new or used?) is NOT a decline.
   end_call: the customer clearly wants to end the call (bye, that's all, do not call again) at any stage other than identity (a "wrong number"/"not me" during identity is identity_denied, not end_call).
   transfer_requested: STAGE is conversation, the officer's last message OFFERED to connect the customer to a human loan officer/agent (matches the next_question offer described in point 5), AND the customer just agreed ("yes", "sure", "okay", "please connect me", "go ahead"). Only fire this after that specific offer was made, not on a generic "yes" to something else.

Return ONLY this JSON, nothing else:
{"needs_rag": false, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": ""}

Examples (OFFICER asked / CUSTOMER said -> output)
- STAGE identity: "Hello, am I speaking with __CUSTOMER__?" / "yes speaking"
  -> {"needs_rag": false, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": "identity_confirmed"}
- STAGE identity: "Hello, am I speaking with __CUSTOMER__?" / "no, wrong number"
  -> {"needs_rag": false, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": "identity_denied"}
  (WRONG: asking again or trying "is __CUSTOMER__ available?" - the officer must apologize and end the call on this first negative answer, not retry.)
- STAGE identity: "Hello, am I speaking with __CUSTOMER__?" / "who's calling?"
  -> {"needs_rag": false, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": "identity_denied"}
  (An unclear, non-confirming answer is treated the same as a denial - never keep probing for identity.)
- STAGE conversation: "Are you currently looking for a loan?" / "yes"
  -> {"needs_rag": true, "rag_query": "What loan products and categories does __COMPANY__ offer?", "product": "", "facts": {}, "next_question": "Which type of loan are you considering?", "event": ""}
- STAGE conversation: "Which type are you looking for?" / "a car loan"
  -> {"needs_rag": true, "rag_query": "What vehicle loan options does __COMPANY__ offer, including loan amount, tenure, interest rate, fees and eligibility?", "product": "vehicle loan", "facts": {"loan_type": "vehicle loan"}, "next_question": "Are you financing a new vehicle or a used one?", "event": ""}
- STAGE conversation, no product yet: "Are you currently looking for a loan?" / "I am planning for a truck"
  -> {"needs_rag": true, "rag_query": "What vehicle loan options does __COMPANY__ offer for trucks, including loan amount, tenure, interest rate, fees and eligibility?", "product": "vehicle loan", "facts": {}, "next_question": "", "event": ""}
  (WRONG: {"needs_rag": false, "facts": {"purpose": "truck"}, "next_question": "Are you looking to finance a new truck or a used one?"} - this files the truck as a bio-fact and asks before ever giving loan information; do not do this.)
- STAGE conversation, product = vehicle loan, full details already given: "The rate sounds good, I'm interested." / (customer just reacted positively)
  -> {"needs_rag": false, "rag_query": "", "product": "vehicle loan", "facts": {}, "next_question": "Would you like me to connect you now to one of our loan officers who can help you complete the application?", "event": ""}
- STAGE conversation, officer JUST offered to connect to a loan officer: "Would you like me to connect you now to one of our loan officers?" / "yes please"
  -> {"needs_rag": false, "rag_query": "", "product": "vehicle loan", "facts": {}, "next_question": "", "event": "transfer_requested"}
- STAGE conversation: "What is your monthly income?" / "around thirty thousand"
  -> {"needs_rag": false, "rag_query": "", "product": "vehicle loan", "facts": {"monthly_income": "about thirty thousand"}, "next_question": "Would you like to go ahead with an application?", "event": ""}
- STAGE conversation: anything / "no thanks, I'm not interested"
  -> {"needs_rag": false, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": "not_interested"}
- any stage other than identity: anything / "okay bye"
  -> {"needs_rag": false, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": "end_call"}"""

PLANNER_SYSTEM_PROMPT = (
    _PLANNER_SYSTEM_TEMPLATE
    .replace("__COMPANY__", CAMPAIGN["company"])
    .replace("__CUSTOMER__", CUSTOMER_NAME)
)

_EMPTY_PLAN = {"needs_rag": False, "rag_query": "", "product": "", "facts": {}, "next_question": "", "event": ""}


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT HELPERS  (single replaceable notes slot + sliding window)
# ═══════════════════════════════════════════════════════════════════════════

def _is_notes(msg) -> bool:
    return (
        isinstance(msg, dict)
        and msg.get("role") == "system"
        and isinstance(msg.get("content"), str)
        and msg["content"].startswith(NOTES_MARKER)
    )


def set_call_notes(context, notes):
    """Replace the ONE notes message in the context (or remove it when notes is None)."""
    messages = [m for m in context.get_messages() if not _is_notes(m)]
    if notes:
        messages.append({"role": "system", "content": notes})
    context.set_messages(messages)


def trim_history(context, keep_last: int = MAX_HISTORY_MESSAGES):
    """
    Keep every system message (role prompt, node instructions, notes) and only the
    last `keep_last` user/assistant/tool messages. The cut always lands on a user
    message so we never leave an orphaned tool result behind.
    """
    messages = list(context.get_messages())
    convo = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
    ]
    if len(convo) <= keep_last:
        return

    cut = convo[-keep_last]
    while cut < len(messages) and not (
        isinstance(messages[cut], dict) and messages[cut].get("role") == "user"
    ):
        cut += 1
    if cut >= len(messages):
        return

    kept = [
        m for i, m in enumerate(messages)
        if i >= cut or not (isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool"))
    ]
    dropped = len(messages) - len(kept)
    if dropped:
        context.set_messages(kept)
        logger.info(f"[CONTEXT] trimmed {dropped} old messages (window={keep_last})")


# ═══════════════════════════════════════════════════════════════════════════
# PLANNER + RAG PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class LLMDrivenRAGProcessor(FrameProcessor):
    """
    Runs on every customer transcription, BEFORE the user aggregator:
      1. planner LLM → needs_rag / rag_query / product / new facts / next_question
      2. Qdrant lookup when needed
      3. writes ONE compact [CALL NOTES] message into the context (replacing the last one)
      4. trims old chat history
    The main LLM then answers with the notes in view.
    """

    def __init__(self, rag_system, context):
        super().__init__()
        self.rag_system = rag_system
        self.context = context

        # Call memory (survives history trimming)
        self.product = ""
        self.facts = {}
        self.bank_info = None      # (topic, text) - latest verified bank info
        self.turns = []            # [(role, text)] for the planner prompt

        # Call stage: identity → conversation → closing. Advanced by planner events.
        self.stage = "identity"
        self.on_call_event = None   # async callback(event), set by the bot (drives the node flow)

        self._last_text = ""
        self._last_text_at = 0.0
        self._session = None

    # ── memory ──────────────────────────────────────────────────────────────
    def add_bot_turn(self, text: str):
        """Call from on_assistant_turn_stopped so the planner sees the officer's side too."""
        text = (text or "").strip()
        if text:
            self.turns.append(("OFFICER", text[:300]))
            self.turns = self.turns[-20:]

    def _merge_plan(self, plan: dict):
        product = plan.get("product")
        if isinstance(product, str) and product.strip():
            self.product = product.strip()[:60]

        facts = plan.get("facts")
        if isinstance(facts, dict):
            for key, value in facts.items():
                if isinstance(key, str) and value not in (None, "", [], {}):
                    self.facts.pop(key, None)                 # re-insert → keeps newest last
                    self.facts[key.strip()[:30]] = str(value).strip()[:60]
            while len(self.facts) > MAX_FACTS:
                self.facts.pop(next(iter(self.facts)))

    # ── planner call ────────────────────────────────────────────────────────
    def _build_user_message(self, user_text: str) -> str:
        known = {"stage": self.stage, "product": self.product or "none yet", "facts": self.facts}
        lines = ["KNOWN: " + json.dumps(known, ensure_ascii=False), "", "RECENT CONVERSATION:"]
        for role, text in self.turns[-PLANNER_HISTORY_TURNS:]:
            lines.append(f"{role}: {text}")
        lines += ["", f"LATEST CUSTOMER MESSAGE: {user_text}"]
        return "\n".join(lines)

    @staticmethod
    def _parse_plan(raw: str) -> dict:
        raw = re.sub(r"```(?:json)?", "", raw or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in planner output")
        data = json.loads(raw[start:end + 1])

        needs_rag = data.get("needs_rag", False)
        if isinstance(needs_rag, str):
            needs_rag = needs_rag.strip().lower() == "true"
        plan = dict(_EMPTY_PLAN)
        plan["needs_rag"] = bool(needs_rag)
        plan["rag_query"] = str(data.get("rag_query") or "").strip()
        plan["product"] = str(data.get("product") or "").strip()
        plan["facts"] = data.get("facts") if isinstance(data.get("facts"), dict) else {}
        plan["next_question"] = str(data.get("next_question") or "").strip()
        event = str(data.get("event") or "").strip().lower()
        plan["event"] = event if event in VALID_EVENTS else ""
        return plan

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=PLANNER_TIMEOUT_S)
            )
        return self._session

    async def _ask_planner(self, user_text: str) -> dict:
        payload = {
            "model": PLANNER_MODEL,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_message(user_text)},
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 220,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            session = await self._get_session()
            async with session.post(PLANNER_URL, json=payload) as response:
                if response.status != 200:
                    logger.error(f"[PLANNER] HTTP {response.status}")
                    return dict(_EMPTY_PLAN)
                data = await response.json()
            raw = data["choices"][0]["message"]["content"]
            logger.info(f"[PLANNER RAW] {raw.strip()}")
            return self._parse_plan(raw)
        except Exception as e:
            # Fail open: the call continues without RAG rather than stalling.
            logger.warning(f"[PLANNER] failed ({type(e).__name__}: {e}) - continuing without RAG")
            return dict(_EMPTY_PLAN)

    # ── RAG ─────────────────────────────────────────────────────────────────
    def _retrieve(self, rag_query: str):
        """Complete RAG flow: retrieve context → refine with LLM → return answer"""
        logger.info("=" * 80)
        logger.info(f"[RAG FLOW START] Query: {rag_query}")
        logger.info("=" * 80)
        
        # Step 1: Semantic search in Qdrant
        # NOTE: SimpleRAG.retrieve_answer()'s parameter is `rag_query`, not
        # `question` - the previous `question=rag_query` keyword would raise
        # TypeError: retrieve_answer() got an unexpected keyword argument.
        context_text, found, results = self.rag_system.retrieve_answer(
            rag_query=rag_query, min_score=RAG_MIN_SCORE
        )
        
        if not (found and results):
            logger.warning("[RAG FLOW] ❌ No relevant context found")
            logger.info("=" * 80)
            return None
        
        # Log search results
        for idx, result in enumerate(results[:5], 1):
            logger.info(f"[RAG RESULT {idx}] Score: {result['score']:.4f} | {result['text'][:80]}...")
        
        # Step 2: Refine with internal RAG LLM
        text = (context_text or "").strip()
        if len(text) > RAG_MAX_CHARS:
            text = text[:RAG_MAX_CHARS].rsplit(" ", 1)[0] + " ..."
            logger.info(f"[RAG CONTEXT] Truncated to {len(text)} chars")
        
        # Use internal LLM to refine the answer.
        # refine_answer(question, context) does ONLY the final "turn context
        # into a natural spoken answer" step - it does NOT call
        # generate_rag_query() or retrieve_answer() again, because this
        # planner already did both (rag_query above, context_text just now).
        # answer_question(question, min_score) would be the wrong call here:
        # its second positional argument is a similarity threshold, not a
        # context string, and it would redundantly re-generate the query.
        refined_answer = self.rag_system.refine_answer(rag_query, text)
        
        logger.info("=" * 80)
        logger.info(f"[RAG FLOW COMPLETE] Final refined answer:")
        logger.info(f"{refined_answer}")
        logger.info("=" * 80)
        
        return refined_answer

    # ── notes ───────────────────────────────────────────────────────────────
    def _build_notes(self, next_question: str, turn_notice):
        parts = [
            f"{NOTES_MARKER} Internal guidance for your next reply. Never mention or read "
            "this aloud. Only the latest notes apply."
        ]

        profile = "; ".join(f"{k}: {v}" for k, v in self.facts.items())
        if self.product:
            profile = f"focus product: {self.product}" + (f"; {profile}" if profile else "")
        if profile:
            parts.append(
                f"Customer profile so far: {profile}. Never ask again for anything already listed."
            )

        if turn_notice:
            parts.append(turn_notice)
            logger.info(f"[CALL NOTES] turn_notice: {turn_notice[:100]}")
        elif self.bank_info:
            topic, text = self.bank_info
            parts.append(
                f"BANK INFORMATION (topic: {topic}) - the ONLY source for rates, amounts, "
                f"tenure, eligibility, fees and documents:\n{text}\n\n"
                "STEP 1 of your reply: put the relevant numbers/details above into natural "
                "speech. This is mandatory and comes first, before anything else in your reply."
            )
            logger.info(f"[CALL NOTES] bank_info topic: {topic}")
            logger.info(f"[CALL NOTES] bank_info text: {text[:150]}...")

        if next_question:
            step = "STEP 2" if (self.bank_info and not turn_notice) else "Then"
            parts.append(f'{step} of your reply: ask "{next_question}"')
            logger.info(f"[CALL NOTES] next_question: {next_question}")

        notes = "\n\n".join(parts) if len(parts) > 1 else None
        if notes:
            logger.info(f"[CALL NOTES] FULL NOTES INJECTED TO CONTEXT ({len(notes)} chars)")
        else:
            logger.info("[CALL NOTES] NO NOTES (closing stage or no bank info needed)")
        
        return notes

    # ── per-turn pipeline ───────────────────────────────────────────────────
    def _is_duplicate(self, text: str) -> bool:
        now = time.monotonic()
        duplicate = text == self._last_text and (now - self._last_text_at) < DUPLICATE_WINDOW_S
        self._last_text, self._last_text_at = text, now
        return duplicate

    async def _dispatch_event(self, event: str):
        """Apply a stage event to the node flow (only if valid for the current stage)."""
        allowed = {
            "identity_confirmed": self.stage == "identity",
            "identity_denied": self.stage == "identity",
            "not_interested": self.stage == "conversation",
            "end_call": self.stage in ("identity", "conversation"),
            "transfer_requested": self.stage == "conversation",
        }
        if not event or not allowed.get(event) or self.on_call_event is None:
            return
        try:
            await self.on_call_event(event)
        except Exception as e:
            logger.error(f"[FLOW] transition for {event!r} failed: {type(e).__name__}: {e}")
            return
        self.stage = "conversation" if event == "identity_confirmed" else "closing"
        logger.info(f"[FLOW] event={event} → stage={self.stage}")

    def _log_context_size(self):
        chars = sum(len(m.get("content") or "") for m in self.context.get_messages() if isinstance(m, dict))
        approx = int(chars / 3.5)
        line = f"[CONTEXT] ~{approx} tokens in prompt (LLM_NUM_CTX={LLM_NUM_CTX})"
        if approx > LLM_NUM_CTX * 0.85:
            logger.warning(line + " - close to the limit, Ollama will silently truncate")
        else:
            logger.info(line)

    async def _plan_turn(self, user_text: str):
        started = time.monotonic()
        logger.info("=" * 80)
        logger.info(f"[CUSTOMER INPUT] {user_text}")
        logger.info("=" * 80)

        # Call is wrapping up: no planning / RAG needed, just let the closing node speak.
        if self.stage == "closing":
            logger.info("[STAGE] closing - no planner/RAG needed")
            set_call_notes(self.context, None)
            return

        # Step 1: Ask planner LLM
        logger.info("[PLANNER] Analyzing customer input...")
        plan = await self._ask_planner(user_text)
        
        logger.info("=" * 80)
        logger.info("[PLANNER DECISION]")
        logger.info(f"  needs_rag: {plan['needs_rag']}")
        logger.info(f"  rag_query: {plan['rag_query']}")
        logger.info(f"  product: {plan['product']}")
        logger.info(f"  facts: {plan['facts']}")
        logger.info(f"  next_question: {plan['next_question']}")
        logger.info(f"  event: {plan['event']}")
        logger.info("=" * 80)
        
        self._merge_plan(plan)
        await self._dispatch_event(plan["event"])

        if self.stage == "closing":
            logger.info("[STAGE] transitioned to closing")
            set_call_notes(self.context, None)      # goodbye needs no bank info / questions
            self.turns.append(("CUSTOMER", user_text[:300]))
            return

        # Step 2: RAG retrieval if needed
        turn_notice = None
        if plan["needs_rag"] and plan["rag_query"]:
            logger.info("[RAG DECISION] needs_rag=true, executing RAG flow...")
            refined_answer = self._retrieve(plan["rag_query"])
            if refined_answer:
                self.bank_info = (plan["rag_query"], refined_answer)
                logger.info("[RAG RESULT] ✅ Refined answer ready for call notes")
            else:
                logger.warning("[RAG RESULT] ❌ No relevant context found, continuing without bank info")
                turn_notice = (
                    f"No verified bank information was found for: {plan['rag_query']}. "
                    "Do not guess. Say you don't have that specific information right now, "
                    "then continue with your next question."
                )
        else:
            if plan["needs_rag"]:
                logger.info("[RAG DECISION] needs_rag=true but rag_query is empty - no RAG needed")
            else:
                logger.info("[RAG DECISION] needs_rag=false - using cached bank info or asking discovery questions")

        # Step 3: Inject call notes into LLM context
        logger.info("[CALL NOTES] Building and injecting into context...")
        set_call_notes(self.context, self._build_notes(plan["next_question"], turn_notice))
        trim_history(self.context)
        self.turns.append(("CUSTOMER", user_text[:300]))

        elapsed = (time.monotonic() - started) * 1000
        logger.info("=" * 80)
        logger.info(f"[TURN COMPLETE] {elapsed:.0f} ms | stage={self.stage} | product={self.product!r}")
        logger.info("=" * 80)
        self._log_context_size()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Support both voice (TranscriptionFrame) and text chat (TextFrame)
        user_text = None
        
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if user_text:
                logger.info("[FRAME SOURCE] TranscriptionFrame (voice)")
        elif isinstance(frame, TextFrame):
            user_text = frame.text.strip()
            if user_text:
                logger.info("[FRAME SOURCE] TextFrame (text chat)")

        if user_text and not self._is_duplicate(user_text):
            await self._plan_turn(user_text)

        await self.push_frame(frame, direction)

    async def cleanup(self):
        await super().cleanup()
        if self._session and not self._session.closed:
            await self._session.close()