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

CUSTOMER_NAME = os.getenv("TEST_CUSTOMER_NAME", "Sathwika")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")   # the model that already works for you

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

        # True after we asked the customer for a callback time
        self.reschedule_pending = False

        # number of customer messages already planned for
        self._seen_user_count = 0

    GREETING_MARKER = "This is the very first message of your outbound call"

    @staticmethod
    def _latest_user_message(context):
        """Returns (text of the latest user message, number of user messages) from the LLM context."""
        msgs = context.get_messages() if hasattr(context, "get_messages") else context.messages
        count, text = 0, ""
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                count += 1
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                text = str(c).strip()
        return text, count

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

        # Earlier turns = context. The latest customer message is passed
        # separately below and is the strongest signal for this decision.
        history_text = "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in self.conversation_history[-9:-1]
        ) or "(no earlier turns)"

        planner_prompt = f"""You are the sales planner for Priya Sharma, an experienced outbound loan salesperson at KBS Bank.

You decide what a good salesperson should do next and, if verified bank information is needed, write ONE focused RAG query.
You never write the customer reply. RAG only retrieves verified KBS Bank facts. The main LLM speaks to the customer.

CONVERSATION SO FAR:
{history_text}

LATEST CUSTOMER MESSAGE (strongest signal):
{user_text}

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
6. CLOSING PUSH: look at what the salesperson already said. If Priya has already given loan information and the customer only
   acknowledges passively ("okay", "hmm", "fine", "I see", "ok ok", "acha") or shows mild interest, do NOT just repeat information and
   do NOT let the call drift. A passive reply is not a "no". A real salesperson now asks for the next step:
   choose offer_specialist so Priya asks if they want to go ahead / take this loan and offers to connect them with a loan specialist.
   Only stay in sell if the customer has not yet heard any product information, or is still asking questions that need answers.
7. If Priya's last message offered a specialist or asked whether to proceed, and the customer agrees in any way
   ("okay", "yes", "sure", "go ahead", "please"), choose transfer.
8. Persist politely: if the customer declined a specialist offer but did not reject loans altogether, continue selling
   and offer again after the next useful piece of information. A clear "no / not interested / no need / stop" is a rejection -> end_call immediately. A passive "okay" or "hmm" is NOT a rejection.

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
- offer_specialist: information has been shared and the customer is interested OR only acknowledging passively; time to ask if they want to go ahead and offer a specialist
- e_transfer: customer requests an e-transfer / payment action
- pitch_overview: no specific need yet; introduce KBS Bank and its loan options to create interest
- sell: anything else in the sales conversation (explain a product, answer a question, handle an objection, discover a need)
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
Priya just explained home loan rate and tenure; Customer "Okay." -> next_action offer_specialist, needs_rag false, rag_query "".
Priya just asked "Would you like me to connect you with a specialist?"; Customer "Okay." -> next_action transfer, needs_rag false, rag_query "".
Priya asked "am I speaking with Mohith?"; Customer "No, this is his brother." -> next_action wrong_person, needs_rag false, rag_query "".
Customer "I'm not interested." -> next_action end_call, needs_rag false, rag_query "".
Customer "Call me tomorrow, I'm busy." -> next_action reschedule, needs_rag false, rag_query "".
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
            "stream": False,
            "response_format": {"type": "json_object"}
        }

        try:
            timeout = aiohttp.ClientTimeout(total=30)

            async with aiohttp.ClientSession(timeout=timeout) as session:

                async with session.post(
                    "http://202.164.134.176:11434/v1/chat/completions",
                    json=payload
                ) as response:
                # async with session.post(
                #     "https://api.groq.com/openai/v1/chat/completions",   # was the 202.164... Ollama URL
                #     json=payload,
                #     headers={"Authorization": f"Bearer {os.getenv('GROQ_KEY')}"},
                # ) as response:

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

        # The planner now sits AFTER the user aggregator: it reacts to the finished user turn
        # (the LLM context frame) instead of holding the raw transcription frame.
        if (
            frame.__class__.__name__ in ("LLMContextFrame", "OpenAILLMContextFrame")
            and direction == FrameDirection.DOWNSTREAM
        ):

            # If call already ended, ignore all further customer input
            if self.call_ended:
                await self.push_frame(frame, direction)
                return

            user_text, user_count = self._latest_user_message(frame.context)

            # Nothing new from the customer (e.g. the opening greeting): straight to the LLM
            if (
                not user_text
                or user_count == self._seen_user_count
                or user_text.startswith(self.GREETING_MARKER)
            ):
                await self.push_frame(frame, direction)
                return

            self._seen_user_count = user_count
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
            # RESCHEDULE FOLLOW-UP: this message is the callback time.
            # Do not re-plan or pitch - confirm it and end the call.
            # ---------------------------------------------------------
            if self.reschedule_pending:
                self.reschedule_pending = False
                logger.info("[PLANNER] Reschedule follow-up -> confirm time and end call")

                self.context.add_message({
                    "role": "system",
                    "content": f"""The customer has just replied to your question about when to call them back.
Confirm the callback time they gave in ONE warm sentence (if they gave no specific time, say you will call at a convenient time).
Do NOT pitch any product. Do NOT ask any more questions.
Append exactly: [END_CALL]
The marker must not be explained to the customer.
Exception: if the customer clearly says they can talk now, OR the intended customer ({CUSTOMER_NAME}) has come on the line (e.g. "yes, speaking", "this is {CUSTOMER_NAME}"), do NOT end the call and do NOT add the marker. Continue normally: greet them, introduce yourself and KBS Bank briefly and explain the reason for the call."""
                })

                await self.push_frame(frame, direction)
                return

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
            next_action = str(
                planner_result.get("next_action") or "sell"
            ).strip().lower()

            if next_action == "end_call":
                logger.info("[PLANNER] Next action = end_call")
                self.call_ended = True

                self.context.add_message({
                    "role": "system",
                    "content": f"""
The customer does not want to continue or is not interested. Apologise briefly for taking their time, thank them, and say a warm goodbye — one or two short sentences only.
Example: "I'm sorry to have disturbed you, thank you for your time. Have a great day!"
Do not ask any more questions. Do not mention loans or try to convince them again.
Append exactly: [END_CALL]
The marker must not be explained to the customer.
"""
                })

                await self.push_frame(frame, direction)
                return

            if next_action == "wrong_person":
                logger.info("[PLANNER] Next action = wrong_person")
                # If we ask for a callback time, the next message is handled
                # by the reschedule follow-up (confirm + end).
                self.reschedule_pending = True

                self.context.add_message({
                    "role": "system",
                    "content": f"""The person on the line is NOT {CUSTOMER_NAME}.
Do NOT introduce any loan or share any bank or loan details with this person. Do NOT pitch anything.
Apologise briefly for the inconvenience, in ONE or TWO short sentences.
- If they say it is a wrong number or they do not know {CUSTOMER_NAME}: thank them, say goodbye and append exactly: [END_CALL]
- If {CUSTOMER_NAME} is simply not available or they know {CUSTOMER_NAME}: politely ask when would be a good time to reach {CUSTOMER_NAME}. Do NOT add the marker yet.
- If they say {CUSTOMER_NAME} is available and will come on the line, ask them to please hand over the phone. Do NOT add the marker.
The marker must not be explained to the customer."""
                })

                await self.push_frame(frame, direction)
                return

            if next_action == "reschedule":
                logger.info("[PLANNER] Next action = reschedule")
                self.reschedule_pending = True

                self.context.add_message({
                    "role": "system",
                    "content": f"""{CUSTOMER_NAME} is busy or asked for a callback. Apologise briefly for the bad timing and respond warmly in ONE sentence.
Do NOT pitch any product.
- If {CUSTOMER_NAME} already gave a time in this message (e.g. "tomorrow", "after 5", "10 AM"), confirm it warmly and append exactly: [END_CALL]
- Otherwise ask: "Of course, no problem, sorry for the bad timing - when would be a better time for me to call you back?" Do NOT add the marker yet.
The marker must not be explained to the customer."""
                })

                await self.push_frame(frame, direction)
                return


            if next_action == "offer_specialist":
                logger.info("[PLANNER] Next action = offer_specialist")

                self.context.add_message({
                    "role": "system",
                    "content": f"""
The internal conversation planner has determined that {CUSTOMER_NAME} has heard the loan
information and it is time to move the sale to the next step.

This is the right moment to ask if they would like to go ahead with the loan and to
proactively offer to connect them with a loan specialist.

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

            if next_action == "transfer":
                logger.info("[PLANNER] Next action = transfer")

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

            known_product = planner_result.get("known_product", "none")
            sales_goal = planner_result.get("conversation_goal", "")
            customer_need = planner_result.get("customer_need", "")
            rag_attempted = bool(needs_rag and rag_query)
            refined = ""

            if rag_attempted:

                logger.info("=" * 80)
                logger.info("[RAG] LLM-GENERATED QUERY:")
                logger.info(rag_query)
                logger.info("=" * 80)

                # Single call - answer_question_outbound does retrieve + refine
                refined = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.rag_system.answer_question_outbound(rag_query)
                ) or ""

                logger.info(f"[RAG] Answer: {refined[:200]}")

            if refined and "don't have confident" not in refined:
                self.context.add_message({
                    "role": "system",
                    "content": f"""NEXT ACTION: {next_action}
KNOWN PRODUCT: {known_product}
CUSTOMER NEED: {customer_need}
SALES GOAL: {sales_goal}

VERIFIED KBS BANK FACTS:
{refined}

{"PITCH OVERVIEW: Present the loan types temptingly using the facts above. Mention each product with its best rate, max amount and tenure. End with ONE soft question about what they may be planning." if next_action == "pitch_overview" else "Speak these facts naturally and confidently. Lead with the most useful number. After sharing facts, ask ONE closing question: whether they would like to go ahead with this loan or be connected to a loan specialist."}
{"Offer to connect with a loan specialist after stating the facts." if next_action == "offer_specialist" else ""}
Use ONLY the facts above for bank-specific details. Keep response to 2-3 sentences. NEVER invent facts. NEVER mention RAG or internal systems."""
                })

            elif rag_attempted:
                logger.info("[RAG] No confident result found")
                self.context.add_message({
                    "role": "system",
                    "content": f"""NEXT ACTION: {next_action}
CUSTOMER NEED: {customer_need}
SALES GOAL: {sales_goal}

No verified KBS Bank information is available for this question.
Do NOT state or guess any rate, amount, tenure, eligibility, fee, document or condition.
Say honestly that you will have a loan specialist confirm those details, and keep the conversation moving.
Keep response to 1-2 sentences. NEVER mention RAG or internal systems."""
                })

            elif next_action in ("sell", "pitch_overview"):
                logger.info("[RAG] Not required for this turn")
                self.context.add_message({
                    "role": "system",
                    "content": f"""NEXT ACTION: {next_action}
CUSTOMER NEED: {customer_need}
SALES GOAL: {sales_goal}

Respond naturally as a salesperson to advance this goal. If the customer has already heard loan information, ask if they would like to proceed.
Do NOT state any KBS Bank rate, amount, tenure, eligibility, fee or condition - none is verified for this turn.
Keep response to 1-2 sentences."""
                })

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
    # tts = PiperTTSService(
    #     use_cuda=True,
    #     settings=PiperTTSService.Settings(
    #         voice="en_US-hfc_female-medium"
    #     ),
    #     text_aggregation_mode=TextAggregationMode.SENTENCE
    # )

    # from smart_tts_service import VoiceCloneTTSService

    # tts = VoiceCloneTTSService(
    #     ref_audio="sathwika_voice.mp3",
    #     ref_text="Hi.Hello how are you all,I am sathwika,Today i am calling you to make happy, once again congratulations,have a nice day",
    #     backend="qwen",
    #     text_aggregation_mode=TextAggregationMode.SENTENCE
    # )
    from qwen_ws_tts_service import QwenWSTTSService

    tts = QwenWSTTSService(
        url="http://127.0.0.1:8888",      # your server address, see below
        api_key="choose-a-secret",        # the same TTS_API_KEY you gave the server
        text_aggregation_mode=TextAggregationMode.SENTENCE,
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
    # from pipecat.services.openai.llm import OpenAILLMService

    # llm = OpenAILLMService(
    #     api_key=os.getenv("GROQ_KEY"),
    #     base_url="https://api.groq.com/openai/v1",
    #     model="qwen/qwen3.8-27b",
    # )
        
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
        context_aggregator.user(),
        llm_rag_planner,  # runs on the COMPLETED user turn, right before the LLM
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
        # Planner needs the salesperson's side of the conversation too
        llm_rag_planner.conversation_history.append({
            "role": "assistant",
            "content": message.content
        })
    
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
        # context.add_message({
        #     "role": "user",          # was "system"
        #     "content": greeting_prompt
        # })
        
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