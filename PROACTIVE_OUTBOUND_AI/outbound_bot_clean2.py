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

# Qdrant + BGE-large RAG
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

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

OUTBOUND_SYSTEM_PROMPT = f"""
You are Sathwika, a professional loan officer from
{CAMPAIGN['company']} making an OUTBOUND call to
{CUSTOMER_NAME}.

Your goal:
{CAMPAIGN['goal']}

You are the bank representative who called the customer.
YOU lead the conversation.

IMPORTANT:

The customer should feel that they are speaking with
a knowledgeable human loan officer.

You must understand what the customer says and decide
the most useful next step.

Do NOT behave like a customer-support chatbot.

Never say:
"How can I help you?"

You called the customer, so YOU guide the conversation.

CALL BEHAVIOR:

1. First confirm identity.

Example:
"Hello, am I speaking with {CUSTOMER_NAME}?"

2. After identity confirmation, introduce yourself and
briefly explain why you are calling.

3. Proactively ask whether the customer is currently
looking for a loan.

4. If the customer says YES:

The conversation planner may provide bank information
about the loan products available.

Use that information naturally.

For example, if the bank information says that the bank
offers home, vehicle and personal loans, say:

"Great, {CUSTOMER_NAME}. We offer home, vehicle and personal
loan options. Is there a particular type you're
looking for?"

Do not invent products.

5. Once the customer identifies a requirement,
proactively ask the next useful question.

Example:

Customer:
"I need a car loan."

You:
"Certainly. Are you looking to finance a new vehicle
or a used one?"

Then continue based on the customer's answer.

6. When factual bank information is provided to you,
use ONLY that information.

Never invent:
- interest rates
- loan amounts
- tenure
- eligibility
- income requirements
- fees
- documents
- approval conditions

7. If the customer asks a factual bank question,
use the supplied bank knowledge naturally.

8. After answering, continue the conversation with
ONE useful next question.

9. Remember everything the customer has already told you.
Never repeatedly ask the same question.

10. Keep responses short and natural for voice calls.
Normally 1-3 sentences.

11. Be confident and professional, but never pushy.
12. END CALL CONDITIONS

If the customer denies the identity or says they are not the requested person:
- Politely acknowledge.
- Do not continue the loan conversation.
- End the call politely.
- Add [END_CALL].

Example:
"I understand. Sorry for the inconvenience. Thank you for your time. Have a great day. [END_CALL]"

If the customer clearly says they are not interested or wants to end the conversation:
- Politely acknowledge.
- Give a short closing.
- Do not ask another question.
- Do not continue selling.
- Add [END_CALL].

[END_CALL]

The [END_CALL] marker is an internal control signal.
Never explain the marker to the customer.

Example:

Customer:
"I'm not interested in taking a loan."

Response:
"I understand. Thank you for your time, {CUSTOMER_NAME}. Have a great day. [END_CALL]"

Another example:

Customer:
"No, I don't need a loan."

Response:
"No problem, {CUSTOMER_NAME}. Thank you for your time. Have a great day. [END_CALL]"

Only use [END_CALL] when the customer has clearly decided
not to continue or wants to end the conversation.
13. If the customer is clearly very interested in taking a loan
and wants to proceed with the application, speak to a human loan
agent, or continue with a bank representative, transfer the call
to a human agent.

Also transfer the call if the customer explicitly asks:

- "Can I speak to someone?"
- "Can you connect me to an agent?"
- "I want to talk to a person."
- "I want to proceed with the loan."
- "I'd like to apply."
- "Please connect me to someone who can help me apply."

In this situation:

- Do not continue asking unnecessary questions.
- Give a short natural transition message.
- Mark the conversation as ready for transfer by adding:

[TRANSFER_CALL]

The [TRANSFER_CALL] marker is an internal control signal.
Never explain the marker to the customer.

Example:

Customer:
"I want to apply for the loan."

Response:
"Absolutely. I'll connect you with one of our loan specialists who can help you with the application. [TRANSFER_CALL]"

Another example:

Customer:
"Can I speak to a person?"

Response:
"Of course. I'll connect you with a loan specialist now. [TRANSFER_CALL]"

Only use [TRANSFER_CALL] when the customer clearly wants
human assistance or wants to proceed with the loan.

IMPORTANT:
You are proactive.

Do not wait for the customer to decide the entire
conversation.

You should continuously understand:

- What does the customer want?
- What loan might be relevant?
- What information do we already know?
- What information is still needed?
- What should I ask next?

The conversation planner and bank knowledge will provide
relevant factual information when necessary.

Never mention:
RAG, knowledge base, database, planner, system,
retrieval, documents, or internal processing.

Speak directly to the customer as a KBS Bank loan officer.
"""
# ═══════════════════════════════════════════════════════════════════════════
# RAG QUERY PROMPT - FOR ANSWERING CUSTOMER QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

RAG_ANSWER_PROMPT = """Answer the customer's question using ONLY the provided knowledge-base context below.

CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

INSTRUCTIONS:
- Give a concise, natural answer suitable for voice conversation
- Do NOT mention "the document", "according to records", or "based on information"
- Just answer directly and naturally
- If the context doesn't contain the answer, say "I don't have that specific information right now"
- Keep it to 1-3 sentences unless more detail is needed
- After answering, you can naturally continue the conversation

ANSWER:"""

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

        # Keep a small conversation history for the planner LLM
        self.conversation_history = []

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

        for item in self.conversation_history[-12:]:
            history_text += (
                f"{item['role'].upper()}: "
                f"{item['content']}\n"
            )

        planner_prompt = f"""
        You are the internal conversation intelligence of a
        professional KBS Bank loan officer.

        You are NOT speaking to the customer.

        Your job is to analyze the conversation and determine
        whether bank knowledge is required.

        If bank knowledge is required, create the BEST possible
        semantic search query for the bank knowledge base.

        IMPORTANT:
        The RAG query must NOT simply copy the customer's words.

        The query must express the actual information that the
        conversation needs.

        CONVERSATION:

        {history_text}

        LATEST CUSTOMER MESSAGE:

        {user_text}


        ========================================================
        HOW TO THINK
        ========================================================

        First understand:

        1. What does the customer mean?
        2. What does the customer want?
        3. What loan/product is relevant, if known?
        4. What information has already been given?
        5. What information is missing?
        6. What information from the bank knowledge base would
        allow the main LLM to give the customer a useful
        response?
        7. What should the RAG query ask for?


        ========================================================
        IMPORTANT EXAMPLES
        ========================================================

        EXAMPLE 1
        ---------

        BANK:
        "{CUSTOMER_NAME}, are you currently looking for a loan?"

        CUSTOMER:
        "Yes."

        Interpretation:
        Customer is generally interested in obtaining a loan.
        No loan type has been selected yet.

        The next useful information is the list of loan products
        available from the bank.

        GOOD RAG QUERY:

        "What loan products and loan categories does KBS Bank offer?"

        BAD RAG QUERY:

        "Yes"

        BAD RAG QUERY:

        "Is {CUSTOMER_NAME} looking for a loan?"


        ========================================================
        EXAMPLE 2
        ========================================================

        BANK:
        "We offer several loan options. Is there a particular
        type you are interested in?"

        CUSTOMER:
        "I want something for my house."

        Interpretation:
        Customer likely needs home financing.

        GOOD RAG QUERY:

        "What home loan options does KBS Bank offer, including
        loan amount, repayment tenure, interest rate and
        basic eligibility requirements?"


        ========================================================
        EXAMPLE 3
        ========================================================

        CUSTOMER:
        "I need a car loan."

        Interpretation:
        Customer needs vehicle financing.

        GOOD RAG QUERY:

        "What vehicle loan options does KBS Bank offer,
        including available loan amount, repayment tenure,
        interest rate, fees and eligibility requirements?"


        ========================================================
        EXAMPLE 4
        ========================================================

        CUSTOMER:
        "What is the maximum amount for a vehicle loan?"

        Interpretation:
        Customer explicitly wants a factual answer.

        GOOD RAG QUERY:

        "What is the maximum loan amount available under
        the KBS Bank vehicle loan?"


        ========================================================
        EXAMPLE 5
        ========================================================

        CUSTOMER:
        "What documents do I need?"

        Previous conversation:
        Customer selected home loan.

        GOOD RAG QUERY:

        "What documents are required for a KBS Bank home loan?"


        Do NOT query:

        "What documents do I need?"

        The query must include the relevant product when the
        product is known from the conversation.


        ========================================================
        EXAMPLE 6
        ========================================================

        CUSTOMER:
        "How long can I take to repay it?"

        Previous conversation:
        Customer selected vehicle loan.

        GOOD RAG QUERY:

        "What is the maximum repayment tenure for the
        KBS Bank vehicle loan?"


        ========================================================
        EXAMPLE 7
        ========================================================

        CUSTOMER:
        "Thirty thousand."

        Previous question:
        "What is your monthly income?"

        Interpretation:
        This is customer information.

        NO RAG.

        ========================================================
        EXAMPLE 8
        ========================================================

        CUSTOMER:
        "Yes, I am interested."

        Previous conversation:
        Bank has just asked whether customer wants to know
        about available loan options.

        Interpretation:
        Customer wants more information.

        GOOD RAG QUERY:

        "What loan products does KBS Bank currently offer?"

        ========================================================
        EXAMPLE 9
        ========================================================

        CUSTOMER:
        "I need around ten lakh."

        Previous conversation:
        Customer has already selected vehicle loan.

        Interpretation:
        Customer is providing the desired loan amount.

        Usually NO RAG unless factual information is needed.

        ========================================================
        EXAMPLE 10
        ========================================================

        CUSTOMER:
        "What is the interest rate?"

        Previous conversation:
        Customer selected vehicle loan.

        GOOD RAG QUERY:

        "What is the interest rate range for the KBS Bank
        vehicle loan?"

        ========================================================


        ========================================================
        WHEN RAG IS REQUIRED
        ========================================================

        Use RAG when the main LLM needs factual bank information,
        such as:

        - available loan products
        - loan amount
        - interest rate
        - tenure
        - eligibility
        - minimum income
        - documents
        - processing fee
        - security/collateral
        - loan purpose
        - repayment information
        - product-specific conditions


        ========================================================
        WHEN RAG IS NOT REQUIRED
        ========================================================

        Do NOT use RAG merely because the customer spoke.

        Do not use RAG for:

        - yes
        - no
        - okay
        - thank you
        - greetings
        - customer's name
        - customer's income
        - employment information
        - loan purpose
        - desired amount
        - simple confirmations
        - normal conversation

        UNLESS the conversation requires bank knowledge to
        continue naturally.


        ========================================================
        QUERY QUALITY RULE
        ========================================================

        The RAG query should be:

        - specific
        - contextual
        - product-aware
        - complete
        - factual
        - based on the entire conversation

        It should contain the relevant loan product whenever
        the product is already known.

        Do not make vague queries.

        BAD:
        "What is the rate?"

        GOOD:
        "What is the interest rate range for the KBS Bank
        vehicle loan?"

        BAD:
        "What about documents?"

        GOOD:
        "What documents are required for a KBS Bank home loan?"

        BAD:
        "Tell me about loans."

        GOOD:
        "What loan products and categories does KBS Bank offer?"


        ========================================================
        OUTPUT
        ========================================================

        Return ONLY valid JSON.

        If RAG is required:

        {{
        "needs_rag": true,
        "rag_query": "...",
        "reason": "...",
        "customer_intent": "...",
        "known_product": "..."
        }}

        If RAG is NOT required:

        {{
        "needs_rag": false,
        "rag_query": "",
        "reason": "...",
        "customer_intent": "...",
        "known_product": "..."
        }}
        """

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

                    logger.info(
                        f"[PLANNER] Intent: "
                        f"{result.get('customer_intent')}"
                    )

                    logger.info(
                        f"[PLANNER] Product: "
                        f"{result.get('known_product')}"
                    )

                    logger.info(
                        f"[PLANNER] RAG needed: "
                        f"{result.get('needs_rag')}"
                    )

                    logger.info(
                        f"[PLANNER] RAG query: "
                        f"{result.get('rag_query')}"
                    )

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
                logger.info(
                    "[RAG] LLM-GENERATED QUERY:"
                )
                logger.info(rag_query)
                logger.info("=" * 80)

                context_text, found, results = (
                    self.rag_system.retrieve_answer(
                        question=rag_query,
                        min_score=0.5
                    )
                )

                if found and results:

                    score = results[0]["score"]

                    logger.info(
                        f"[RAG] HIT score={score:.4f}"
                    )

                    logger.info("=" * 80)
                    logger.info("[RAG] KNOWLEDGE RETURNED:")
                    logger.info(context_text)
                    logger.info("=" * 80)

                    # -------------------------------------------------
                    # GIVE RAG RESULT TO MAIN LLM
                    # -------------------------------------------------

                    self.context.add_message({
                        "role": "system",
                        "content": f"""
                    You now have VERIFIED BANK INFORMATION that can be used
                    for the customer's current conversation.

                    IMPORTANT:
                    This information is internal knowledge for you.
                    Do not mention how you obtained it.

                    CUSTOMER'S LATEST MESSAGE:
                    {user_text}

                    INTERNAL RAG QUERY:
                    {rag_query}

                    BANK INFORMATION:
                    {context_text}


                    ========================================================
                    HOW TO RESPOND
                    ========================================================

                    You are a professional KBS Bank loan officer.

                    IMPORTANT SALES PRIORITY:

                    1.The VERIFIED BANK INFORMATION above is the primary
                    source of bank information for this turn.

                    2.When verified bank information is available and relevant,
                    your FIRST PRIORITY is to CONVEY the useful verified
                    information to the customer.

                    3.Do NOT treat the bank information as optional background
                    context.

                    4.Even if the customer's latest message is short, vague,
                    casual, or does not directly ask a factual question,
                    proactively convey the most relevant verified bank
                    information when it helps the conversation.

                    Do NOT ignore useful verified bank information.

                    5.Do NOT simply repeat the retrieved text.

                    Convert the verified information into natural spoken
                    conversation.

                    Do NOT dump the entire knowledge base.

                    Only convey the information that is relevant to the
                    customer's current situation and sales stage.

                    Never invent bank facts that are not present in the
                    VERIFIED BANK INFORMATION.

                    After conveying the useful verified information,
                    continue the sales conversation naturally.

                    6.Ask ONLY ONE useful next question when appropriate.

                    7. The next question should help understand the customer's
                    loan requirement or move the conversation toward the
                    next appropriate step.

                    8. Never ask a question that the customer has already
                    answered.

                    9. Never invent information that is not in the bank
                    knowledge.

                    10. Never mention:
                        - RAG
                        - knowledge base
                        - document
                        - database
                        - retrieval
                        - system
                        - internal query


                    ========================================================
                    EXAMPLE
                    ========================================================

                    Customer:
                    "Yes, I am looking for a loan."

                    Bank information:
                    Personal Loan
                    Home Loan
                    Vehicle Loan
                    Education Loan
                    Business Loan

                    GOOD RESPONSE:

                    "That's great, {CUSTOMER_NAME}. We offer personal, home, vehicle,
                    education and business loan options. Which type of loan
                    are you currently considering?"


                    ========================================================
                    ANOTHER EXAMPLE
                    ========================================================

                    Customer:
                    "I need a car loan."

                    Bank information:
                    Vehicle Loan:
                    ₹1 lakh–₹40 lakh
                    12–84 months
                    8.75–14.5%

                    GOOD RESPONSE:

                    "Certainly, {CUSTOMER_NAME}. Our vehicle loans range from ₹1 lakh
                    to ₹40 lakh, with repayment periods of 12 to 84 months.
                    Are you looking to finance a new vehicle or a used one?"


                    ========================================================
                    ANOTHER EXAMPLE
                    ========================================================

                    Customer:
                    "What is the maximum amount?"

                    Known product:
                    Vehicle Loan

                    Bank information:
                    ₹1 lakh–₹40 lakh

                    GOOD RESPONSE:

                    "The maximum vehicle loan amount is ₹40 lakh, subject
                    to eligibility and applicable loan terms. Approximately
                    how much are you looking to borrow?"

                    Do NOT answer with unrelated product information.


                    ========================================================

                    Now generate ONLY the response that should be spoken
                    to the customer.
                    """
                    })

            else:

                logger.info(
                    "[RAG] Not required for this customer message"
                )

        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# QDRANT + BGE-LARGE RAG SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

QDRANT_PATH = "./knowledge_base/qdrant_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # BGE-large for best accuracy
EMBEDDING_DIM = 1024

class QdrantBGERAG:
    """Qdrant + BGE-large RAG for perfect semantic search"""
    
    def __init__(self):
        logger.info("🔍 Initializing Qdrant + BGE-large RAG...")
        
        # Load BGE-large
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
            logger.info("✅ BGE-large loaded on GPU")
        except:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            logger.info("✅ BGE-large loaded on CPU")
        
        # Initialize Qdrant
        os.makedirs(QDRANT_PATH, exist_ok=True)
        self.client = QdrantClient(path=QDRANT_PATH)
        
        # Check collection
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            logger.warning(f"⚠️  Collection '{COLLECTION_NAME}' not found!")
            logger.info("   Upload documents first:")
            logger.info("   python test_rag_simple.py --upload document.pdf")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
        
        count = self.client.count(collection_name=COLLECTION_NAME).count
        logger.info(f"📊 RAG ready: {count} chunks")
    
    def search(self, query: str, top_k: int = 3):
        """Semantic search with BGE-large"""
        logger.info("[RAG] Semantic search query: {}", query)

        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        try:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )

            results = response.points

        except Exception as exc:
            logger.error("[RAG] Qdrant query_points failed: {}", exc)
            raise

        formatted_results = []

        for r in results:
            payload = r.payload or {}

            formatted_results.append({
                "text": str(payload.get("text", "")).strip(),
                "score": float(r.score),
                "metadata": payload.get("metadata", {})
            })

        logger.info(
            "[RAG] Qdrant returned {} candidate results",
            len(formatted_results)
        )

        return formatted_results
    
    def retrieve_answer(self, question: str, min_score: float = 0.5):
        """Retrieve best answer with score"""
        logger.info(f"[RAG] Querying: {question}")
        
        results = self.search(question, top_k=3)
        
        if not results:
            logger.warning("[RAG] No results")
            return "I don't have specific information about that.", False, []
        
        top_score = results[0]["score"]
        logger.info(f"[RAG] Top score: {top_score:.4f}")
        
        if top_score < min_score:
            logger.warning(f"[RAG] Score {top_score:.4f} < threshold {min_score}")
            return "I don't have confident information about that.", False, results
        
        # Combine top results
        context = "\n\n".join([r["text"] for r in results[:2]])
        logger.info(f"[RAG] ✅ Retrieved {len(results)} chunks")
        
        return context, True, results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BOT LOGIC
# ═══════════════════════════════════════════════════════════════════════════

async def run_bot(transport):
    """Main bot logic"""
    logger.info("Starting DYNAMIC PROACTIVE OUTBOUND BOT")
    
    # Initialize Qdrant + BGE-large RAG
    rag_system = QdrantBGERAG()
    
    if rag_system.client.count(collection_name=COLLECTION_NAME).count == 0:
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
        
        # Proactive opening - ALWAYS start with identity confirmation
        greeting_prompt = f"""This is the FIRST message of your outbound call. Follow the CALL FLOW step 1:

CONFIRM IDENTITY: Ask if you're speaking with {CUSTOMER_NAME}. 

Example: "Hello, am I speaking with {CUSTOMER_NAME}?"

Keep it simple and natural. Just confirm identity first - nothing else."""
        
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
