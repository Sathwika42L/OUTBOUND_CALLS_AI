# ═══════════════════════════════════════════════════════════════════════════
# 🤖 OUTBOUND BOT  (same layout as bot2.py: pipeline + FlowManager + RTVI handlers)
# Run:  uv run bot_outbound.py
# ═══════════════════════════════════════════════════════════════════════════

import os
import asyncio

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
    AssistantTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
    AudioRawFrame,
)
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.piper.tts import PiperTTSService
# from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments
from pipecat.services.tts_service import TextAggregationMode
from pipecat_flows import FlowManager

# RAG - single source of truth
from test_rag_simple import SimpleRAG

# Flow (nodes, functions, planner + RAG processor)
from flow_perfect import (
    CAMPAIGN,
    CUSTOMER_NAME,
    OLLAMA_BASE_URL,
    LLMDrivenRAGProcessor,
    BotSpeechGate,
    create_initial_node,
    initialize_user,
    begin_end_call,
    begin_transfer_call,
)

load_dotenv(override=True)


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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BOT LOGIC
# ═══════════════════════════════════════════════════════════════════════════


# Module-level RAG instance — loaded once at startup before any call arrives.
# run_bot() reuses it so the cold-start delay never blocks a live WebRTC connection.
_shared_rag: "SimpleRAG | None" = None


async def run_bot(transport: BaseTransport):
    """Main bot logic"""
    logger.info("Starting DYNAMIC PROACTIVE OUTBOUND BOT (Pipecat Flows)")

    # Reuse the module-level instance loaded at startup; fall back to a fresh
    # load only if the pre-load somehow didn't run (e.g. direct import).
    rag_system = _shared_rag
    if rag_system is None:
        logger.warning("RAG not pre-loaded; loading now (may delay first call)")
        try:
            rag_system = await asyncio.to_thread(SimpleRAG)
        except Exception as e:
            logger.exception(f"❌ RAG failed to load - the call will run WITHOUT verified facts: {e}")
            rag_system = None

    if rag_system is not None and rag_system.count() == 0:
        logger.error("=" * 80)
        logger.error("❌ NO DOCUMENTS IN RAG DATABASE - the call will run WITHOUT verified facts")
        logger.error("Upload documents first:")
        logger.error("  python test_rag_simple.py --upload document.pdf")
        logger.error("=" * 80)
        rag_system = None

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
    # from nemo_stt_client import NemotronWebSocketSTTService

    # stt = NemotronWebSocketSTTService(
    #     url="ws://127.0.0.1:9092",
    #     language="en-US",
    #     sample_rate=16000,
    #     ttfs_p99_latency=0.15,
    # )
    
    # TTS
    # tts = PiperTTSService(
    #     use_cuda=True,
    #     settings=PiperTTSService.Settings(
    #         voice="en_US-hfc_female-medium"
    #     ),
    #     text_aggregation_mode=TextAggregationMode.SENTENCE
    # )
    from qwen_ws_tts_service import QwenWSTTSService
    
    tts = QwenWSTTSService(
        url="http://127.0.0.1:8888",      # your server address, see below
        api_key="choose-a-secret",        # the same TTS_API_KEY you gave the server
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )
    
    llm = OLLamaLLMService(
            base_url="http://16.192.104.155:11434/v1",
            settings=OLLamaLLMService.Settings(
                model="qwen2.5:14b",
                temperature=0.3,  # More creative for natural conversation
                top_p=0.8,
            )
        )

    # Context starts EMPTY: the persona (role_messages) comes from the initial Flow node
    context = LLMContext()

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
    llm_rag_planner = LLMDrivenRAGProcessor(rag_system, context)   # plans + RAG once per finished customer turn
    speech_gate = BotSpeechGate()                                  # end / transfer wait until the bot finished speaking

    pipeline = Pipeline([
        transport.input(),

        mute_stt,

        stt,

        context_aggregator.user(),

        llm_rag_planner,      # ← planner + RAG, right before the LLM (prompts unchanged)

        llm,

        tts,

        transport.output(),

        speech_gate,          # ← observes BotStarted/StoppedSpeaking for the end / transfer nodes

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

    # objects the flow needs (used by the end / transfer nodes)
    flow_manager.state["speech_gate"] = speech_gate
    flow_manager.state["planner"] = llm_rag_planner

    # ── TURN LOGS + UI TRANSCRIPTS (same as bot2.py) ─────────────────────────
    @context_aggregator.user().event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.info(f"[USER DONE ] {message.content}")

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
        logger.info(f"[BOT  DONE ] {message.content}")

        text = (message.content or "").strip()
        # a function-call-only turn has no spoken text
        if not text:
            return

        # the planner needs the salesperson's side of the conversation too
        history = llm_rag_planner.conversation_history
        if not (history and history[-1]["role"] == "assistant" and history[-1]["content"].strip() == text):
            history.append({"role": "assistant", "content": text})

        if hasattr(task.rtvi, "send_message"):
            try:
                await task.rtvi.send_message({
                    "type": "bot-transcript",
                    "data": {"text": text, "is_final": True},
                })
            except Exception:
                pass

    # ── FLOW START (exactly once) ────────────────────────────────────────────
    _flow_initialized = [False]
    _client_params_ready = asyncio.Event()   # set when the client sent name / callId

    async def _initialize_flow_once():
        if _flow_initialized[0]:
            return
        _flow_initialized[0] = True
        logger.info("Initializing flow (once)")

        # give the platform a moment to deliver name / callId (else env defaults are used)
        try:
            await asyncio.wait_for(_client_params_ready.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            pass

        await initialize_user(flow_manager)
        name = flow_manager.state.get("name", "")

        # the planner's rule 0 needs to know the identity question was asked
        llm_rag_planner.conversation_history.append(
            {"role": "assistant", "content": f"Hi, am I speaking with {name}?"}
        )
        flow_manager.state["greeted"] = True

        logger.info("📞 PROACTIVE OUTBOUND CALL - Bot initiating (identity confirmation first)")
        await flow_manager.initialize(create_initial_node(name, greeted=False))

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await _initialize_flow_once()

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi_instance, msg):
        """
        client -> bot messages:
          {type: "end-call",      data: {reason, callback_time}}   hang up (goodbye is spoken first)
          {type: "transfer-call"}                                   transfer to a human specialist
          {type: "get-call-state"}                                  request/response
          anything else with data {name, callId, msisdn}            call parameters, then start the flow
        """
        logger.info(f"RTVI client message: {msg.type} {msg.data}")
        data = msg.data if isinstance(msg.data, dict) else {}

        try:
            if msg.type == "end-call":
                node = begin_end_call(
                    flow_manager,
                    data.get("reason", "customer_ended"),
                    str(data.get("callback_time", "") or ""),
                )
                await flow_manager.set_node_from_config(node)
                await rtvi_instance.send_server_response(msg, {"status": "ending"})
                return

            if msg.type == "transfer-call":
                node = begin_transfer_call(flow_manager)
                await flow_manager.set_node_from_config(node)
                await rtvi_instance.send_server_response(msg, {"status": "transferring"})
                return

            if msg.type == "get-call-state":
                await rtvi_instance.send_server_response(msg, {
                    "name": flow_manager.state.get("name"),
                    "callId": flow_manager.state.get("callId"),
                    "call_ended": bool(flow_manager.state.get("call_ended")),
                    "end_reason": flow_manager.state.get("end_reason"),
                    "transfer_done": bool(flow_manager.state.get("transfer_done")),
                })
                return

            # ── call parameters (same keys as bot2.py) ──
            if data:
                if data.get("msisdn"):
                    flow_manager.state["msisdn"] = data["msisdn"]
                if data.get("name"):
                    flow_manager.state["name"] = data["name"]
                    llm_rag_planner.customer_name = data["name"]
                if data.get("callId"):
                    flow_manager.state["callId"] = data["callId"]
                    print("Stored callId:", flow_manager.state["callId"])

                _client_params_ready.set()
                # Initialize exactly once - calling it again would re-speak the greeting
                await _initialize_flow_once()

        except Exception as e:
            logger.exception(f"[RTVI] client message failed: {e}")
            try:
                await rtvi_instance.send_error_response(msg, str(e))
            except Exception:
                pass

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - scheduling flow init fallback")
        # on_client_ready fires via RTVI after the data channel handshake.
        # If that message was queued before the pipeline was ready it may be
        # replayed, or it may be lost. Either way, we start a small-delay
        # fallback here so the greeting always fires.
        async def _fallback_init():
            await asyncio.sleep(2.0)   # give RTVI / on_client_ready a chance to fire first
            await _initialize_flow_once()
        asyncio.ensure_future(_fallback_init())

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

    logger.info("=" * 80)
    logger.info("🤖 DYNAMIC PROACTIVE OUTBOUND BOT (Pipecat Flows)")
    logger.info("=" * 80)
    logger.info(f"Campaign: {CAMPAIGN['name']}")
    logger.info(f"Purpose: {CAMPAIGN['purpose']}")
    logger.info(f"Goal: {CAMPAIGN['goal']}")
    logger.info(f"Default customer: {CUSTOMER_NAME}")
    logger.info("=" * 80)

    # Load BGE-large + Qdrant ONCE at startup and store it in the module-level
    # variable so run_bot() can reuse it without blocking the WebRTC connection.
    try:
        _shared_rag = SimpleRAG()
        logger.info(f"✅ RAG pre-loaded ({_shared_rag.count()} chunks)")
    except Exception as e:
        logger.error(f"RAG pre-load failed (will retry per call): {e}")

    main()