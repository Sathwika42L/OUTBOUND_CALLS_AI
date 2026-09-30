# ═══════════════════════════════════════════════════════════════════════════
# 🤖 PROACTIVE OUTBOUND BOT - CLEAN LLM-DRIVEN IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════
"""
TRULY DYNAMIC PROACTIVE OUTBOUND BOT

This bot is NOT scripted. The LLM dynamically decides every response based on:
- Campaign purpose and goal
- Customer's latest response
- Conversation history
- Retrieved knowledge (RAG)

NO hardcoded questions. NO fixed state machine. PURE LLM intelligence.
"""

import os
import asyncio
import json
import re
from loguru import logger
from dotenv import load_dotenv

from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
    AssistantTurnStoppedMessage
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMTextFrame,
    EndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    LLMFullResponseEndFrame,
    UserStoppedSpeakingFrame,
    AudioRawFrame
)
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.base_transport import TransportParams
from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments
from pipecat.services.tts_service import TextAggregationMode
import requests

# RAG — import directly from test_rag_simple (single source of truth)
from test_rag_simple import SimpleRAG

load_dotenv(override=True)
base_url = os.getenv("BASE_URL", "")
call_id = os.getenv("CALL_ID", "")
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

CUSTOMER_NAME = os.getenv("TEST_CUSTOMER_NAME", "Chanakya")

# ═══════════════════════════════════════════════════════════════════════════
# LLM SYSTEM PROMPT - DYNAMIC OUTBOUND AGENT
# ═══════════════════════════════════════════════════════════════════════════

OUTBOUND_SYSTEM_PROMPT = f"""You are Sathwika, a loan officer from {CAMPAIGN['company']}.
You have called {CUSTOMER_NAME}. You are making a professional bank sales call.

YOUR PERSONA:
- Warm, confident, and conversational — like a knowledgeable friend at a bank, not a robot reading a script.
- You lead the conversation but you never feel pushy or transactional.
- You explain things naturally. You share useful information before asking questions.

CALL STRUCTURE:

TURN 1 — IDENTITY ONLY (first message, already sent before customer speaks)
The very first message asks ONE thing only: confirm you are speaking with {CUSTOMER_NAME}.
"Hi, am I speaking with {CUSTOMER_NAME}?"
Nothing else. No introduction. No product. No offer. Just identity.

TURN 2 — AFTER IDENTITY CONFIRMED
Once {CUSTOMER_NAME} confirms (says "yes", "speaking", "that's me", or similar):
- Introduce yourself and the bank: "This is Sathwika calling from KBS Bank."
- Ask if now is a good time: "Is this a good time for a quick minute?"
Do NOT mention any product or offer yet. Just introduce and check availability.

IF NOT A GOOD TIME:
Ask: "Of course — when would be a good time to call back?"
If they are clearly not interested, close warmly and add [END_CALL].

TURN 3 — AFTER THEY SAY IT'S A GOOD TIME
Now present KBS Bank's loan portfolio as a compelling offer — all product types, best rates, max amounts, tenure.
Use the verified facts provided. Make it sound attractive and worth listening to.
Then ask ONE soft question: what type of loan they might be thinking about.

Example (using real numbers from context):
"Great! So at KBS Bank right now, we have home loans starting from 8.25% up to ₹1.5 crore, vehicle loans from 8.75% up to ₹40 lakhs, and personal loans from 10.5% up to ₹20 lakhs — all with very flexible tenures. Is there any particular type of loan you've been thinking about, or any big purchase or expense coming up?"

The goal of Turn 3 is to TEMPT the customer with the full picture — not push one product.

TURN 4+ — FOLLOW THE CUSTOMER'S INTEREST
Once the customer shows interest in any product or expense:
- Dive into that product's details (rate, tenure, amount, eligibility)
- Share facts first, then ask one natural follow-up
- Progress: rate → tenure → amount → eligibility → documents → specialist offer

Example:
- Customer says vehicle → "Our vehicle loans go up to ₹40 lakhs with rates from 8.75% and tenures up to 84 months. Are you looking at a new vehicle or used?"
- Customer says new → "For a new vehicle we need income proof, bank statements, and the vehicle invoice. What's the approximate price of the vehicle?"
- Customer says amount → "That fits well within our range. Are you salaried or self-employed — just so I can give you a rough EMI picture?"

HANDLING "NOT INTERESTED IN THIS PRODUCT":
If the customer says they're not interested in a specific loan type you mentioned:
DO NOT end the call. Pivot to another product.
Example: "No problem — apart from vehicle loans, we also have home loans and personal loans. Is there anything else you might be planning for?"
Only add [END_CALL] if the customer explicitly says they do not want any loan at all and want to end the call.

WHEN TO OFFER A SPECIALIST:
Once the customer has engaged on 2+ product details (rate, tenure, amount, eligibility, documents), say:
"You know {CUSTOMER_NAME}, based on what you've shared I think you'd qualify quite well. Would you like me to connect you with one of our loan specialists who can give you a more precise picture and walk you through the next steps?"
If they agree, give a warm handover line and add [TRANSFER_CALL].
Example: "Excellent — I'll connect you right away. [TRANSFER_CALL]"

Also add [TRANSFER_CALL] immediately if the customer says:
"I want to apply", "connect me to someone", "I want to proceed", "speak to an agent", "let's do it"

HANDLING SHORT RESPONSES:
- "wait" / "hold on" → say only: "Of course, take your time."
- "okay" / "alright" / "sure" → do NOT summarize. Advance to the next natural point.
- Unclear or off-topic → gently redirect: "Sorry, I think we got a bit cut off — you were asking about the vehicle loan, right?"

HANDLING OBJECTIONS:
If customer asks "why KBS Bank?" or "what's different?" → give 2 specific facts from the verified information, then ask a qualifying question.

STRICT RULES:
- Use ONLY verified bank information given to you in context. Never invent rates, amounts, tenure, fees, or eligibility criteria.
- When verified facts are given to you — speak them naturally and confidently. Do not hold them back.
- Keep responses to 2–3 sentences. Voice calls must be brief.
- Never repeat a question already asked.
- Never re-introduce yourself after the first turn.
- Never mention RAG, knowledge base, database, retrieval, or internal systems.

END CALL:
If the customer is not {CUSTOMER_NAME}: apologise briefly and end. Add [END_CALL].
    Example:"I apologize for the inconvenience , {CUSTOMER_NAME}. It seems I’ve reached the wrong person. Thank you for your time, and have a great day. [END_CALL]"
If the customer explicitly says they do not want any loan and want to stop: close warmly. Add [END_CALL].
    Example:"No problem {CUSTOMER_NAME}, I completely understand. If you’re interested in a loan in the future, please feel free to contact us. Thank you for your time, and have a great day. [END_CALL]"

NEVER add [END_CALL] just because the customer is not interested in one specific product.
Always try to pivot to another product type before ending.

[END_CALL] and [TRANSFER_CALL] are internal signals. Never explain them to the customer.
"""

# ═══════════════════════════════════════════════════════════════════════════
# PROCESSORS
# ═══════════════════════════════════════════════════════════════════════════

class MuteSTTDuringTTS(FrameProcessor):
    """Mute microphone input while bot is speaking"""
    def __init__(self):
        super().__init__()
        self.tts_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self.tts_active = True
        elif isinstance(frame, TTSStoppedFrame):
            self.tts_active = False

        if self.tts_active and isinstance(frame, AudioRawFrame):
            return  # Mute

        await self.push_frame(frame, direction)

class LLMEndCallProcessor(FrameProcessor):
    """
    Detects [END_CALL] even when the LLM streams the marker
    across multiple LLMTextFrame chunks.

    The marker is removed BEFORE Piper receives the text.
    The actual disconnect is still handled AFTER TTS finishes.
    """

    END_MARKER = "[END_CALL]"

    def __init__(self):
        super().__init__()
        self.end_call_requested = False
        self.llm_response = ""
        self.pending_text = ""

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ):
        await super().process_frame(frame, direction)

        # ---------------------------------------------------------
        # LLM TEXT
        # ---------------------------------------------------------
        if isinstance(frame, (LLMTextFrame, TextFrame)):

            text = frame.text

            # Add any previously held partial marker text
            combined = self.pending_text + text
            self.pending_text = ""

            # -----------------------------------------------------
            # Full marker already present
            # -----------------------------------------------------
            if self.END_MARKER in combined:

                self.end_call_requested = True

                logger.info(
                    "📞 [END CALL] Main LLM requested call termination"
                )

                combined = combined.replace(
                    self.END_MARKER,
                    ""
                )

            # -----------------------------------------------------
            # Check if the END marker is split across LLM chunks
            # -----------------------------------------------------
            marker_prefix_length = 0

            for i in range(
                1,
                min(len(self.END_MARKER), len(combined)) + 1
            ):
                if combined.endswith(
                    self.END_MARKER[:i]
                ):
                    marker_prefix_length = i

            if marker_prefix_length > 0:

                # Hold the possible partial marker.
                self.pending_text = combined[
                    -marker_prefix_length:
                ]

                combined = combined[
                    :-marker_prefix_length
                ]

            # -----------------------------------------------------
            # Send only clean customer-facing text to Piper
            # -----------------------------------------------------
            if combined:
                frame.text = combined
                self.llm_response += combined

                await self.push_frame(
                    frame,
                    direction
                )

            return

        # ---------------------------------------------------------
        # LLM RESPONSE COMPLETED
        # ---------------------------------------------------------
        elif isinstance(
            frame,
            LLMFullResponseEndFrame
        ):

            # If the remaining buffered text is the END marker,
            # consume it instead of sending it to Piper.
            if self.pending_text:

                if self.END_MARKER in self.pending_text:

                    self.end_call_requested = True

                    logger.info(
                        "📞 [END CALL] Main LLM requested call termination"
                    )

                    self.pending_text = ""

                else:
                    # It was normal text, so release it.
                    pending = self.pending_text
                    self.pending_text = ""

                    self.llm_response += pending

                    await self.push_frame(
                        TextFrame(pending),
                        direction
                    )

            logger.info(
                "📞 [END CALL] LLM response completed"
            )

            await self.push_frame(
                frame,
                direction
            )

            return

        # ---------------------------------------------------------
        # ALL OTHER FRAMES
        # ---------------------------------------------------------
        await self.push_frame(
            frame,
            direction
        )

class LLMTransferCallProcessor(FrameProcessor):
    """
    Detects [TRANSFER_CALL] from the LLM and removes it
    before Piper receives the text.

    The actual transfer happens only after TTS finishes.
    """

    TRANSFER_MARKER = "[TRANSFER_CALL]"

    def __init__(self):
        super().__init__()
        self.transfer_requested = False
        self.llm_response = ""
        self.pending_text = ""

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMTextFrame, TextFrame)):

            text = frame.text

            combined = self.pending_text + text
            self.pending_text = ""

            if self.TRANSFER_MARKER in combined:

                self.transfer_requested = True

                logger.info(
                    "📞 [TRANSFER] Main LLM requested call transfer"
                )

                combined = combined.replace(
                    self.TRANSFER_MARKER,
                    ""
                )

            marker_prefix_length = 0

            for i in range(
                1,
                min(len(self.TRANSFER_MARKER), len(combined)) + 1
            ):
                if combined.endswith(
                    self.TRANSFER_MARKER[:i]
                ):
                    marker_prefix_length = i

            if marker_prefix_length > 0:

                self.pending_text = combined[
                    -marker_prefix_length:
                ]

                combined = combined[
                    :-marker_prefix_length
                ]

            if combined:

                frame.text = combined

                self.llm_response += combined

                await self.push_frame(
                    frame,
                    direction
                )

            return

        elif isinstance(
            frame,
            LLMFullResponseEndFrame
        ):

            if self.pending_text:

                if self.TRANSFER_MARKER in self.pending_text:

                    self.transfer_requested = True

                    logger.info(
                        "📞 [TRANSFER] Main LLM requested call transfer"
                    )

                    self.pending_text = ""

                else:

                    pending = self.pending_text

                    self.pending_text = ""

                    self.llm_response += pending

                    await self.push_frame(
                        TextFrame(pending),
                        direction
                    )

            logger.info(
                "📞 [TRANSFER] LLM response completed"
            )

            await self.push_frame(
                frame,
                direction
            )

            return

        await self.push_frame(
            frame,
            direction
        )
class EndCallAfterTTSProcessor(FrameProcessor):
    """
    Actually disconnects the call only AFTER Piper finishes
    speaking the final closing message.
    """

    def __init__(self, end_call_processor):
        super().__init__()

        self.end_call_processor = end_call_processor
        self.call_id = call_id

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ):
        await super().process_frame(frame, direction)

        # ---------------------------------------------------------
        # PIPER FINISHED SPEAKING
        # ---------------------------------------------------------
        if isinstance(frame, TTSStoppedFrame):

            if self.end_call_processor.end_call_requested:

                logger.info(
                    "📞 [END CALL] Final closing message completed"
                )

                try:

                    requests.get(
                        base_url +
                        "/ai-agent/disconnect?callId=" +
                        str(self.call_id),
                        verify=False
                    )

                    logger.info(
                        "📞 [END CALL] WebRTC disconnect URL triggered"
                    )

                except Exception as e:

                    logger.error(
                        "📞 [END CALL] WebRTC disconnect failed: {}",
                        e
                    )

                self.end_call_processor.end_call_requested = False
                self.end_call_processor.llm_response = ""

                await asyncio.sleep(0.3)

                logger.info(
                    "📞 [END CALL] Sending EndFrame"
                )

                await self.push_frame(
                    EndFrame(),
                    FrameDirection.DOWNSTREAM
                )

                return

        await self.push_frame(frame, direction)
class TransferAfterTTSProcessor(FrameProcessor):
    """
    Actually transfers the call only AFTER Piper finishes
    speaking the transfer message.
    """

    def __init__(self, transfer_processor):
        super().__init__()

        self.transfer_processor = transfer_processor
        self.call_id = call_id

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ):
        await super().process_frame(frame, direction)

        # ---------------------------------------------------------
        # PIPER FINISHED SPEAKING
        # ---------------------------------------------------------
        if isinstance(frame, TTSStoppedFrame):

            if self.transfer_processor.transfer_requested:

                logger.info(
                    "📞 [TRANSFER] Final transfer message completed"
                )

                try:

                    url = (
                        base_url +
                        "/ai-agent/transfer?callId=" +
                        str(self.call_id)
                    )

                    logger.info(
                        "📞 [TRANSFER] Calling URL: {}",
                        url
                    )

                    response = requests.get(
                        url,
                        verify=False
                    )

                    logger.info(
                        "📞 [TRANSFER] WebRTC transfer URL triggered: {}",
                        response.status_code
                    )

                except Exception as e:

                    logger.error(
                        "📞 [TRANSFER] Transfer failed: {}",
                        e
                    )

                self.transfer_processor.transfer_requested = False
                self.transfer_processor.llm_response = ""

                return

        await self.push_frame(frame, direction)
class LLMDrivenRAGProcessor(FrameProcessor):
    """
    LLM decides:
      1. What the customer means
      2. Whether bank knowledge is needed
      3. What query should be sent to RAG

    Then the RAG result is injected into the MAIN LLM.
    """

    def __init__(self, rag_system, context):
        super().__init__()

        self.rag_system = rag_system
        self.context = context

        self.last_user_text = ""
        self.last_assistant_text = ""

        # Conversation history for the planner LLM
        self.conversation_history = []

        # Once call is ended, ignore all further frames
        self.call_ended = False

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

        history_text = ""

        for item in self.conversation_history[-6:]:
            history_text += (
                f"{item['role'].upper()}: "
                f"{item['content']}\n"
            )

        planner_prompt = f"""You are the sales planner for Sathwika, a KBS Bank outbound loan officer.
You are NOT speaking to the customer. Return valid JSON only.

CONVERSATION HISTORY:
{history_text}

LATEST CUSTOMER MESSAGE:
{user_text}

DECIDE:
1. What is the best next sales action?
2. Is a verified KBS Bank fact needed right now?
3. If yes — write the exact RAG query for the SPECIFIC product the customer mentioned.

NEXT ACTION (pick one):
- PITCH_OVERVIEW   : customer confirmed it is a good time to talk AND no product pitched yet
- PRESENT_PRODUCTS : customer named a specific product → fetch details for THAT product
- ANSWER           : customer asked a factual question about a specific product
- DISCOVER         : customer hasn't named a product yet — ask what they need
- QUALIFY          : product known — gather amount, income, employment, timeline
- OFFER_SPECIALIST : product + amount + one qualifying detail known → offer specialist
- HANDLE_OBJECTION : customer asks "why your bank?" or similar
- PIVOT_PRODUCT    : customer not interested in current product — offer another
- BUSY             : customer says they are busy or asks to call back later
- CONTINUE_SALES   : advance naturally
- END_CALL         : customer clearly ending the conversation (says "bye", "goodbye", "no thanks" after callback scheduled, or refuses all loans)
- TRANSFER_CALL    : customer explicitly wants to apply or speak to someone

CRITICAL RULES:

1. PITCH_OVERVIEW: customer just said "yes" / "good time" AND no loan products pitched yet →
   next_action = "PITCH_OVERVIEW", needs_rag = true
   rag_query = "What are all the loan products KBS Bank offers — home loan, vehicle loan, personal loan, education loan — with interest rates, maximum loan amounts, and maximum tenure for each?"

2. TRANSFER_CALL — SPECIALIST AGREEMENT RULE (HIGHEST PRIORITY):
   Look at the conversation history. If the LAST assistant message contained words like
   "connect you with a loan specialist", "would you like me to connect", "connect you with one of our",
   "loan specialist for more details" — AND the customer's current message is any of:
   "yes", "please", "yes please", "connect", "sure", "okay", "go ahead", "yes please connect"
   → next_action = "TRANSFER_CALL", needs_rag = false, immediately.
   Do NOT keep presenting product facts. The customer already agreed to connect.

3. PRESENT_PRODUCTS — PRODUCT-SPECIFIC QUERY: When customer names a product (e.g. "home loan", "vehicle loan", "personal loan"):
   next_action = "PRESENT_PRODUCTS", needs_rag = true
   Write the rag_query for THAT specific product only.
   Example: customer says "home loan" → rag_query = "What are the KBS Bank home loan details — interest rate, loan amount range, tenure, eligibility criteria?"
   NEVER query a different product than what the customer asked about.

4. END_CALL: Use when customer says "bye", "goodbye", "take care", "thanks bye" OR after a callback
   is scheduled and customer says goodbye. Also use if customer explicitly refuses all loan types.

5. BUSY: Customer says "busy", "call later", "not a good time" → next_action = "BUSY", needs_rag = false.

6. PIVOT_PRODUCT: Customer not interested in one product → offer another. needs_rag = true, query the next product.

RAG QUERY — CRITICAL:
- ALWAYS write the query for the product the customer ACTUALLY mentioned.
- NEVER write a query for a different product.
- For overview: ask for ALL products together.
- For specific product: ask only for that product's details.
GOOD: customer said "home loan" → "KBS Bank home loan interest rate, amount range, tenure, eligibility"
GOOD: customer said "vehicle" → "KBS Bank vehicle loan interest rate, amount range, tenure"
BAD: customer said "home loan" but query asks about "personal loan"

Return ONLY this JSON:
{{
  "needs_rag": true,
  "rag_query": "...",
  "reason": "...",
  "customer_intent": "...",
  "customer_interest": "none|low|medium|high|very_high",
  "sales_stage": "identity|opening|discovery|product_match|qualification|objection_handling|conversion|closing",
  "known_product": "...",
  "customer_need": "...",
  "conversation_goal": "...",
  "next_action": "PITCH_OVERVIEW|PRESENT_PRODUCTS|ANSWER|DISCOVER|QUALIFY|OFFER_SPECIALIST|HANDLE_OBJECTION|PIVOT_PRODUCT|BUSY|CONTINUE_SALES|END_CALL|TRANSFER_CALL"
}}"""

        payload = {
            "model": "qwen2.5:14b",
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
            "stream": False
        }

        try:
            timeout = aiohttp.ClientTimeout(total=30)

            async with aiohttp.ClientSession(timeout=timeout) as session:

                async with session.post(
                    "http://202.164.134.176:11434/v1/chat/completions",
                    json=payload
                ) as response:

                    if response.status != 200:
                        logger.error(
                            f"[PLANNER] HTTP {response.status}"
                        )

                        return {
                            "needs_rag": False,
                            "rag_query": "",
                            "reason": "Planner unavailable",
                            "customer_intent": "",
                            "known_product": ""
                        }

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

                    result = json.loads(output)

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

        except Exception as e:

            logger.exception(
                f"[PLANNER] Failed: {e}"
            )

            return {
                "needs_rag": False,
                "rag_query": "",
                "reason": "Planner error",
                "customer_intent": "",
                "known_product": ""
            }

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ):

        await super().process_frame(frame, direction)

        user_text = None

        if isinstance(frame, TranscriptionFrame):

            # If call already ended, ignore all further customer speech
            if self.call_ended:
                await self.push_frame(frame, direction)
                return

            user_text = frame.text.strip()

            if not user_text:
                await self.push_frame(frame, direction)
                return

            # Prevent duplicate transcription
            if user_text == self.last_user_text:
                await self.push_frame(frame, direction)
                return

            self.last_user_text = user_text

            logger.info("=" * 80)
            logger.info(f"[CUSTOMER] {user_text}")
            logger.info("=" * 80)

            # Store customer message
            self.conversation_history.append({
                "role": "customer",
                "content": user_text
            })

            # ---------------------------------------------------------
            # FIRST LLM: UNDERSTAND CUSTOMER + CREATE RAG QUERY
            # ---------------------------------------------------------

            planner_result = await self.ask_planner_llm(
                user_text
            )

            needs_rag = planner_result.get(
                "needs_rag",
                False
            )

            rag_query = planner_result.get(
                "rag_query",
                ""
            )

            reason = planner_result.get(
                "reason",
                ""
            )
            next_action = planner_result.get("next_action", "CONTINUE_SALES")

            if next_action == "END_CALL":
                logger.info("[PLANNER] Next action = END_CALL")
                self.call_ended = True

                self.context.add_message({
                    "role": "system",
                    "content": f"""
The customer does not want to continue. Give a warm, brief closing — one sentence only.
Do not ask any more questions. Do not mention loans again.
Append exactly: [END_CALL]
The marker must not be explained to the customer.
"""
                })

                await self.push_frame(frame, direction)
                return

            if next_action == "BUSY":
                logger.info("[PLANNER] Next action = BUSY")

                self.context.add_message({
                    "role": "system",
                    "content": f"""
{CUSTOMER_NAME} is busy right now. Respond warmly and offer to call back at a convenient time.
Say: "Of course, no problem — when would be a better time for me to call you back?"
Do NOT pitch any product. Keep it to one sentence.
Once {CUSTOMER_NAME} gives a time (e.g. "tomorrow", "after 5", "10 AM"), confirm it warmly in one sentence and add [END_CALL].
Example: "Perfect — I'll call you tomorrow at 10 AM then. Talk soon! [END_CALL]"
"""
                })

                await self.push_frame(frame, direction)
                return


            if next_action == "OFFER_SPECIALIST":
                logger.info("[PLANNER] Next action = OFFER_SPECIALIST")

                self.context.add_message({
                    "role": "system",
                    "content": f"""
The internal conversation planner has determined that {CUSTOMER_NAME} is showing
strong interest and is engaging with detailed product questions.

This is the right moment to proactively offer to connect them with a loan specialist.

Do the following:
1. Briefly acknowledge what the customer just said or asked.
2. Mention that a loan specialist can give them full details and get things started quickly.
3. Ask if they would like to be connected — keep it warm and natural.

Example:
"That's a great question — our loan specialists have all the exact details on rates and
eligibility. Would you like me to connect you with one of them right now so they can
walk you through everything?"

If the customer says YES or agrees in any way, respond with a short transition and
append exactly: [TRANSFER_CALL]

If the customer says NO or wants to continue the conversation with you, continue
the sales conversation naturally without offering the specialist again immediately.

Do NOT ask another unrelated question in the same turn.
The [TRANSFER_CALL] marker must not be explained to the customer.
"""
                })

                await self.push_frame(frame, direction)
                return

            if next_action == "TRANSFER_CALL":
                logger.info("[PLANNER] Next action = TRANSFER_CALL")

                self.context.add_message({
                    "role": "system",
                    "content": """
The internal conversation planner has determined that the customer should
now be transferred to a human loan officer.

Respond with a short, natural transition message telling the customer that
you will connect them with a loan specialist right away.
Do not ask another question.
After the spoken transition message, append exactly:
[TRANSFER_CALL]

The marker must not be explained to the customer.
"""
                })

                await self.push_frame(frame, direction)
                return

            logger.info(
                f"[PLANNER] needs_rag = {needs_rag}"
            )

            logger.info(
                f"[PLANNER] reason = {reason}"
            )

            logger.info(
                f"[PLANNER] RAG query = {rag_query}"
            )

            # ---------------------------------------------------------
            # RAG
            # ---------------------------------------------------------

            if needs_rag and rag_query:

                logger.info("=" * 80)
                logger.info("[RAG] LLM-GENERATED QUERY:")
                logger.info(rag_query)
                logger.info("=" * 80)

                # Single call — answer_question_outbound does retrieve + refine
                refined = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.rag_system.answer_question_outbound(rag_query)
                )

                logger.info(f"[RAG] Answer: {refined[:200]}")

                if refined and "don't have confident" not in refined:
                    self.context.add_message({
                        "role": "system",
                        "content": f"""NEXT ACTION: {next_action}
KNOWN PRODUCT: {planner_result.get("known_product", "unknown")}

VERIFIED KBS BANK FACTS:
{refined}

{"PITCH OVERVIEW: Present ALL loan types temptingly using the facts above. Mention each product with its best rate, max amount and tenure. End with ONE soft question about what loan they are thinking about." if next_action == "PITCH_OVERVIEW" else "Speak these facts naturally and confidently. Lead with the most useful number. After sharing facts, ask ONE forward-moving question."}
{"Offer to connect with a loan specialist after stating the facts." if next_action == "OFFER_SPECIALIST" else ""}
Keep response to 2–3 sentences. NEVER invent facts. NEVER mention RAG or internal systems."""
                    })
                else:
                    logger.info("[RAG] No confident result found")

            else:
                logger.info("[RAG] Not required for this turn")

        await self.push_frame(frame, direction)



# MAIN BOT LOGIC
# ═══════════════════════════════════════════════════════════════════════════

async def run_bot(transport):
    """Main bot logic"""
    logger.info("Starting DYNAMIC PROACTIVE OUTBOUND BOT")
    
    # Initialize RAG using SimpleRAG from test_rag_simple.py
    rag_system = SimpleRAG()
    
    if rag_system.count() == 0:
        logger.error("=" * 80)
        logger.error("❌ NO DOCUMENTS IN RAG DATABASE!")
        logger.error("=" * 80)
        logger.error("Upload documents first:")
        logger.error("  python test_rag_simple.py --upload document.pdf")
        logger.error("=" * 80)
        return
    
    # STT
    stt = DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        interim_results=True,
        params=DeepgramFluxSTTService.InputParams(
            eager_eot_threshold=0.3,
            eot_threshold=0.5,
            eot_timeout_ms=1500
        )
    )
    
    # TTS
    tts = PiperTTSService(
        use_cuda=True,
        settings=PiperTTSService.Settings(
            voice="en_US-hfc_female-medium"
        ),
        text_aggregation_mode=TextAggregationMode.SENTENCE
    )
    
    # LLM
    llm = OLLamaLLMService(
        base_url="http://202.164.134.176:11434/v1",
        settings=OLLamaLLMService.Settings(
            model="qwen2.5:14b",
            temperature=0.3,  # More creative for natural conversation
            top_p=0.8,
        )
    )
    
    # Context - NO TOOLS (pure conversation, no function calling)
    context = LLMContext(
        messages=[{
            "role": "system",
            "content": OUTBOUND_SYSTEM_PROMPT
        }]
    )
    
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,
                    start_secs=0.1,
                    min_volume=0.75,
                )
            ),
        ),
    )
    
    # Processors
    mute_stt = MuteSTTDuringTTS()
    llm_rag_planner = LLMDrivenRAGProcessor(
        rag_system,
        context
    ) # RAG for ALL messages!   
    end_call_processor = LLMEndCallProcessor()
    transfer_processor = LLMTransferCallProcessor()

    end_call_after_tts = EndCallAfterTTSProcessor(
        end_call_processor
    )

    transfer_after_tts = TransferAfterTTSProcessor(
        transfer_processor
    )
    # Pipeline
    pipeline = Pipeline([
        transport.input(),
        mute_stt,
        stt,
        llm_rag_planner,  # ← RAG INTERCEPTS HERE (voice + text)
        context_aggregator.user(),
        llm,
        end_call_processor,
        transfer_processor,
        tts,
        end_call_after_tts,
        transfer_after_tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════
    
    @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"[BOT] {message.content[:100]}...")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROACTIVE GREETING - BOT SPEAKS FIRST!
    # ═══════════════════════════════════════════════════════════════════════
    
    greeting_sent = [False]
    
    async def send_proactive_greeting():
        """Bot initiates the call - IDENTITY CONFIRMATION FIRST"""
        if greeting_sent[0]:
            return
        greeting_sent[0] = True
        
        logger.info("📞 PROACTIVE OUTBOUND CALL - Bot initiating")
        await asyncio.sleep(1.5)  # Wait for connection
        
        # Proactive opening - confirm identity and pitch in one breath
        greeting_prompt = f"""This is the very first message of your outbound call.

ONLY ask one thing — confirm the identity of the person you are speaking with.
Nothing else. No introduction. No offer. No pitch.

Say exactly this (or a close natural variant):
"Hi, am I speaking with {CUSTOMER_NAME}?"

Wait for the customer to respond. That is all for this first message."""
        
        context.add_message({
            "role": "system",
            "content": greeting_prompt
        })
        
        # Trigger LLM to generate opening
        from pipecat.frames.frames import LLMRunFrame
        await task.queue_frames([LLMRunFrame()])
        
        logger.info("🤖 LLM generating proactive greeting")
    
    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client connected - sending proactive greeting")
        await send_proactive_greeting()
    
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("WebRTC client connected")
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()
    
    # Run
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Bot entry point"""
    transport = None
    
    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_in_filter=RNNoiseFilter(resampler_quality="VHQ"),
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments: {type(runner_args)}")
            return
    
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main
    logger.info("=" * 80)
    logger.info("🤖 DYNAMIC PROACTIVE OUTBOUND BOT")
    logger.info("=" * 80)
    logger.info(f"Campaign: {CAMPAIGN['name']}")
    logger.info(f"Purpose: {CAMPAIGN['purpose']}")
    logger.info(f"Goal: {CAMPAIGN['goal']}")
    logger.info(f"Customer: {CUSTOMER_NAME}")
    logger.info("=" * 80)
    main()
