import asyncio
import logging
import os

from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import (
    PipelineParams,
    PipelineTask,
)

from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)

from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.services.deepgram.flux.stt import (
    DeepgramFluxSTTService,
)

from pipecat.services.openai.llm import (
    OpenAILLMService,
)

from pipecat.services.piper.tts import (
    PiperTTSService,
)

from pipecat.services.tts_service import (
    TextAggregationMode,
)

from pipecat.audio.vad.silero import (
    SileroVADAnalyzer,
)

from pipecat.audio.vad.vad_analyzer import (
    VADParams,
)

from pipecat.audio.filters.rnnoise_filter import (
    RNNoiseFilter,
)

from pipecat.transports.base_transport import (
    BaseTransport,
    TransportParams,
)

from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCTransport,
)

from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
)

from pipecat_flows import FlowManager

from chatgpt_flow import (
    initialize_user,
    create_initial_node,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override=True)


DEEPGRAM_API_KEY = os.getenv(
    "DEEPGRAM_API_KEY"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017",
)

MONGO_DB_NAME = os.getenv(
    "MONGO_DB_NAME",
    "outbound_calls",
)

MONGO_COLLECTION = os.getenv(
    "MONGO_COLLECTION",
    "call_queue",
)


# ============================================================
# LOGGING
# ============================================================

logging.getLogger("pipecat").setLevel(
    logging.INFO
)


# ============================================================
# MONGODB
# ============================================================

mongo_client = MongoClient(
    MONGODB_URI
)

mongo_db = mongo_client[
    MONGO_DB_NAME
]

call_collection = mongo_db[
    MONGO_COLLECTION
]


# ============================================================
# GET PENDING OUTBOUND CALL
# ============================================================

def get_pending_call():
    """
    Get one pending call from MongoDB.
    """

    call = call_collection.find_one(
        {
            "status": "pending"
        }
    )

    if not call:
        logger.warning(
            "No pending outbound call found."
        )
        return None

    logger.info(
        "=========================================="
    )

    logger.info(
        f"CALL ID       : {call.get('_id')}"
    )

    logger.info(
        f"CUSTOMER      : {call.get('customer_name', '')}"
    )

    logger.info(
        f"PHONE         : {call.get('phone_number', '')}"
    )

    logger.info(
        f"MESSAGE       : {call.get('message', '')}"
    )

    logger.info(
        "=========================================="
    )

    return call


# ============================================================
# UPDATE CALL STATUS
# ============================================================

def update_call_status(
    call_id,
    status,
):
    try:
        call_collection.update_one(
            {
                "_id": call_id,
            },
            {
                "$set": {
                    "status": status,
                },
            },
        )

        logger.info(
            f"Call {call_id} -> {status}"
        )

    except Exception as exc:
        logger.exception(
            f"Failed to update call status: {exc}"
        )


# ============================================================
# MARK CALL AS CALLING
# ============================================================

def mark_call_calling(
    call_id,
):
    try:
        call_collection.update_one(
            {
                "_id": call_id,
            },
            {
                "$set": {
                    "status": "calling",
                },
                "$inc": {
                    "call_attempts": 1,
                },
            },
        )

    except Exception as exc:
        logger.exception(
            f"Failed to mark call as calling: {exc}"
        )


# ============================================================
# MAIN BOT
# ============================================================

async def run_bot(
    transport: BaseTransport,
):
    """
    Main Pipecat voice pipeline.

    AUDIO
      ↓
    SmallWebRTC
      ↓
    Deepgram Flux STT
      ↓
    User Context
      ↓
    Qwen LLM
      ↓
    Piper TTS
      ↓
    SmallWebRTC
      ↓
    AUDIO
    """

    logger.info(
        "=========================================="
    )

    logger.info(
        "STARTING OUTBOUND AI VOICE BOT"
    )

    logger.info(
        "=========================================="
    )

    # ========================================================
    # GET CUSTOMER
    # ========================================================

    call_record = get_pending_call()

    if not call_record:
        return

    call_id = call_record.get(
        "_id"
    )

    customer_name = call_record.get(
        "customer_name",
        "",
    )

    phone_number = call_record.get(
        "phone_number",
        "",
    )

    outbound_message = call_record.get(
        "message",
        "",
    )

    # ========================================================
    # MARK CALLING
    # ========================================================

    mark_call_calling(
        call_id
    )

    # ========================================================
    # DEEPGRAM FLUX
    # ========================================================

    logger.info(
        "Initializing Deepgram Flux..."
    )

    stt = DeepgramFluxSTTService(
        api_key=DEEPGRAM_API_KEY,

        params=DeepgramFluxSTTService.InputParams(
            eager_eot_threshold=0.3,
            eot_threshold=0.5,
            eot_timeout_ms=1500,
            interim_results=True,
        ),
    )

    # ========================================================
    # QWEN
    # ========================================================
    #
    # Qwen is the ONLY LLM.
    #
    # search_ics does NOT call another LLM.
    # ========================================================

    logger.info(
        "Initializing Qwen..."
    )

    llm = OpenAILLMService(
        api_key=HF_TOKEN,

        base_url=(
            "https://router.huggingface.co/v1"
        ),

        model=(
            "Qwen/Qwen2.5-72B-Instruct"
        ),

        temperature=0.2,

        top_p=0.8,
    )

    # ========================================================
    # PIPER TTS
    # ========================================================

    logger.info(
        "Initializing Piper..."
    )

    tts = PiperTTSService(
        use_cuda=True,

        settings=PiperTTSService.Settings(
            voice="en_US-hfc_female-medium",
        ),

        text_aggregation_mode=(
            TextAggregationMode.SENTENCE
        ),

        pause_frame_processing=True,

        stop_frame_timeout_s=6.0,
    )

    # ========================================================
    # LLM CONTEXT
    # ========================================================

    context = LLMContext()

    # ========================================================
    # CONTEXT AGGREGATOR
    # ========================================================

    context_aggregator = (
        LLMContextAggregatorPair(
            context,

            user_params=(
                LLMUserAggregatorParams(
                    vad_analyzer=(
                        SileroVADAnalyzer(
                            params=VADParams(
                                stop_secs=0.5,
                                start_secs=0.1,
                                min_volume=0.6,
                            )
                        )
                    )
                )
            ),
        )
    )

    # ========================================================
    # FLOW MANAGER
    # ========================================================

    flow_manager = FlowManager(
        task=None,
        llm=llm,
        context=context,
    )

    # ========================================================
    # STORE CALL INFORMATION
    # ========================================================

    flow_manager.state[
        "call_id"
    ] = call_id

    flow_manager.state[
        "customer_name"
    ] = customer_name

    flow_manager.state[
        "phone_number"
    ] = phone_number

    flow_manager.state[
        "outbound_message"
    ] = outbound_message

    # ========================================================
    # INITIALIZE USER STATE
    # ========================================================

    await initialize_user(
        flow_manager
    )

    # ========================================================
    # PIPELINE
    # ========================================================

    pipeline = Pipeline(
        [
            transport.input(),

            stt,

            context_aggregator.user(),

            llm,

            tts,

            transport.output(),

            context_aggregator.assistant(),
        ]
    )

    # ========================================================
    # PIPELINE TASK
    # ========================================================

    task = PipelineTask(
        pipeline,

        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),

        enable_tracing=True,
    )

    # ========================================================
    # CONNECT FLOW TO TASK
    # ========================================================

    flow_manager.task = task

    flow_initialized = False

    # ========================================================
    # CLIENT CONNECTED
    # ========================================================

    @transport.event_handler(
        "on_client_connected"
    )
    async def on_client_connected(
        transport,
        client,
    ):
        nonlocal flow_initialized

        logger.info(
            "=========================================="
        )

        logger.info(
            "CUSTOMER CONNECTED"
        )

        logger.info(
            f"Customer : {customer_name}"
        )

        logger.info(
            f"Phone    : {phone_number}"
        )

        logger.info(
            "=========================================="
        )

        if flow_initialized:
            return

        flow_initialized = True

        # ====================================================
        # CREATE INITIAL NODE
        # ====================================================

        initial_node = create_initial_node(
            customer_name=customer_name,
            outbound_message=outbound_message,
        )

        # ====================================================
        # START FLOW
        # ====================================================
        #
        # Qwen generates the greeting.
        #
        # There is NO TextFrame.
        # There is NO direct TTS.
        # There is NO LLM bypass.
        # ====================================================

        await flow_manager.initialize(
            initial_node
        )

        logger.info(
            "=========================================="
        )

        logger.info(
            "FLOW STARTED"
        )

        logger.info(
            "FIRST RESPONSE WILL COME FROM QWEN"
        )

        logger.info(
            "LLM BYPASS = FALSE"
        )

        logger.info(
            "=========================================="
        )

    # ========================================================
    # CLIENT DISCONNECTED
    # ========================================================

    @transport.event_handler(
        "on_client_disconnected"
    )
    async def on_client_disconnected(
        transport,
        client,
    ):
        logger.info(
            "Customer disconnected."
        )

        update_call_status(
            call_id,
            "completed",
        )

        try:
            await task.cancel()
        except Exception:
            pass

    # ========================================================
    # RUN
    # ========================================================

    runner = PipelineRunner(
        handle_sigint=False
    )

    try:
        await runner.run(
            task
        )

    except asyncio.CancelledError:
        logger.info(
            "Pipeline cancelled."
        )

    except Exception as exc:
        logger.exception(
            f"Outbound bot error: {exc}"
        )

        update_call_status(
            call_id,
            "failed",
        )

        raise


# ============================================================
# PIPECAT BOT ENTRY POINT
# ============================================================

async def bot(
    runner_args: RunnerArguments,
):
    """
    Pipecat entry point.

    Real telephony will be connected later.

    For now we use SmallWebRTC to test the complete
    voice-agent application.
    """

    if isinstance(
        runner_args,
        SmallWebRTCRunnerArguments,
    ):

        logger.info(
            "Creating SmallWebRTC transport..."
        )

        transport = SmallWebRTCTransport(
            webrtc_connection=(
                runner_args.webrtc_connection
            ),

            params=TransportParams(
                audio_in_enabled=True,

                audio_in_filter=RNNoiseFilter(
                    resampler_quality="VHQ"
                ),

                audio_out_enabled=True,
            ),
        )

        await run_bot(
            transport
        )

        return

    logger.error(
        "Unsupported runner arguments: "
        f"{type(runner_args).__name__}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not DEEPGRAM_API_KEY:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is missing."
        )

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing."
        )

    from pipecat.runner.run import main

    main()