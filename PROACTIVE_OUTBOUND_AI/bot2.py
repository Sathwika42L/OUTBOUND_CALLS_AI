#
# Outbound KBS Bank loan-officer bot - built on the same pipeline structure
# as the inbound reference bot (STT -> LLM -> TTS cascade, FlowManager,
# RTVI wiring), trimmed of inbound-only extras (background music, ack
# sender, live-transcription spies, tracing/latency observer) that aren't
# part of what was asked for here.
#
# The one architectural change from the inbound reference: RAG retrieval is
# done LOCALLY via SimpleRAG (test_rag_simple.py) instead of an HTTP call to
# an external search API - see outbound_flow.py's tool_search_kbs.
#

import os
import re
import asyncio
import logging

from loguru import logger
from dotenv import load_dotenv

logging.getLogger("pipecat.services.openai.base_llm").setLevel(logging.INFO)
logging.getLogger("pipecat").setLevel(logging.INFO)

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
    AssistantTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMTextFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    AudioRawFrame,
    UserStoppedSpeakingFrame,
    EndFrame,
    TTSSpeakFrame,
)
from pipecat_flows import FlowManager

from flow_bot2 import (
    CUSTOMER_NAME,
    create_identity_node,
    initialize_call,
    sanitize_for_tts,
    handle_call_event,
)
from test_rag_simple import SimpleRAG
from outbound_planner import LLMDrivenRAGProcessor

load_dotenv(override=True)


# ═══════════════════════════════════════════════════════════════════════════
# FRAME PROCESSORS
# ═══════════════════════════════════════════════════════════════════════════

class MuteSTTDuringTTS(FrameProcessor):
    """Drops incoming mic audio while the bot is speaking, so the bot doesn't
    hear (and try to transcribe) its own voice."""

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
            return

        await self.push_frame(frame, direction)


class GoodbyeDetector(FrameProcessor):
    """
    Safety net only - the flow's tool handlers (identity_denied,
    not_interested, end_conversation, transfer_call in outbound_flow.py)
    already disconnect/transfer directly and synchronously. This just
    catches the rare case where a farewell gets spoken without the matching
    tool call firing (e.g. a leaked/garbled tool call), so the call still
    ends instead of hanging open.
    """

    def __init__(self):
        super().__init__()
        self._last_bot_response = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            self._last_bot_response += frame.text.lower()

        if isinstance(frame, TTSStoppedFrame):
            goodbye_phrases = ["goodbye", "bye", "have a great day", "have a good day", "take care"]
            if any(p in self._last_bot_response for p in goodbye_phrases):
                logger.info("🔚 Bot said goodbye - ending call as a fallback")
                await asyncio.sleep(0.5)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
            self._last_bot_response = ""

        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BOT LOGIC
# ═══════════════════════════════════════════════════════════════════════════

async def run_bot(transport: BaseTransport):
    logger.info("Starting OUTBOUND KBS Bank loan-officer bot (flow + tool-calling)")

    # SimpleRAG loads the BGE embedding model - do this ONCE at startup, not
    # per call, and thread the instance through outbound_flow.initialize_call.
    rag_system = SimpleRAG()

    stt = DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        interim_results=True,
        params=DeepgramFluxSTTService.InputParams(
            eager_eot_threshold=0.3,
            eot_threshold=0.5,
            eot_timeout_ms=1500,
            interim_results=True,
        ),
    )

    class FilteredPiperTTSService(PiperTTSService):
        """Strips apparent tool-call/JSON leakage from assembled sentences
        before synthesis - qwen2.5 via Ollama occasionally echoes raw
        tool-call syntax instead of natural speech."""

        async def run_tts(self, text: str, context_id: str):
            cleaned = sanitize_for_tts(text)
            if not cleaned:
                logger.warning(f"🚫 TTS blocked sentence: {text[:80]}")
                return
            async for frame in super().run_tts(cleaned, context_id):
                yield frame

    tts = FilteredPiperTTSService(
        use_cuda=True,
        settings=PiperTTSService.Settings(
            voice="en_US-hfc_female-medium",
        ),
        text_aggregation_mode=TextAggregationMode.SENTENCE,
        pause_frame_processing=True,
        stop_frame_timeout_s=6.0,
    )

    llm = OLLamaLLMService(
        base_url="http://202.164.134.176:11434/v1",
        settings=OLLamaLLMService.Settings(
            model="qwen2.5:14b",
            temperature=0.0,   # zero temperature for maximum tool-calling reliability
            top_p=0.5,
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.5, start_secs=0.1, min_volume=0.75),
            ),
        ),
    )

    mute_stt = MuteSTTDuringTTS()
    proactive_rag_planner = LLMDrivenRAGProcessor(rag_system, context)  # Intent detection + RAG
    goodbye_detector = GoodbyeDetector()

    pipeline = Pipeline([
        transport.input(),
        mute_stt,
        stt,
        proactive_rag_planner,  # ← PROACTIVE INTENT DETECTION + RAG HERE
        context_aggregator.user(),
        llm,
        goodbye_detector,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # Connect planner to flow events
    async def on_call_event(event: str):
        await handle_call_event(flow_manager, event)

    proactive_rag_planner.on_call_event = on_call_event

    @context_aggregator.user().event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.info(f"[USER DONE] {message.content}")
        flow_manager.state["last_user_text"] = message.content
        if hasattr(task.rtvi, "send_message"):
            try:
                await task.rtvi.send_message({
                    "type": "user-transcript",
                    "data": {"text": message.content, "is_final": True},
                })
            except Exception:
                pass

    @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        bot_text = message.content
        logger.info("=" * 80)
        logger.info(f"[BOT  DONE] {bot_text}")
        logger.info("=" * 80)
        
        # Planner needs bot turns for conversation history
        proactive_rag_planner.add_bot_turn(bot_text)
        
        if hasattr(task.rtvi, "send_message"):
            try:
                await task.rtvi.send_message({
                    "type": "bot-transcript",
                    "data": {"text": bot_text, "is_final": True},
                })
            except Exception:
                pass

    # Guard: initialize the flow exactly once, whichever event fires first.
    _flow_initialized = [False]

    async def _initialize_flow_once():
        if _flow_initialized[0]:
            return
        _flow_initialized[0] = True
        logger.info("Initializing outbound flow (once)")
        await initialize_call(flow_manager, rag_system)
        await flow_manager.initialize(create_identity_node())

        # Speak the identity-confirmation greeting directly - no LLM call,
        # no chance of a stray tool call before the customer has even
        # answered the phone.
        customer_name = flow_manager.state.get("name", CUSTOMER_NAME)
        greeting = f"Hello, am I speaking with {customer_name}?"
        logger.info(f"🎤 Speaking greeting: {greeting}")
        await tts.queue_frame(TTSSpeakFrame(greeting))

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await _initialize_flow_once()

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi_instance, msg):
        logger.info(f"RTVI client message: {msg.type} {msg.data}")
        if msg.data:
            if msg.data.get("msisdn"):
                flow_manager.state["msisdn"] = msg.data["msisdn"]
            if msg.data.get("name"):
                flow_manager.state["name"] = msg.data["name"]
            if msg.data.get("callId"):
                flow_manager.state["callId"] = msg.data["callId"]
                logger.info(f"Stored callId: {flow_manager.state['callId']}")
            flow_manager.state["last_user_text"] = msg.data.get("text", "")
            await _initialize_flow_once()
        else:
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
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
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()