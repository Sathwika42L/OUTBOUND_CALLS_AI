import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import aiohttp
from aiohttp import web
import numpy as np
import torch

import nemo.collections.asr as nemo_asr


# ============================================================
# CONFIG
# ============================================================

# MODEL_PATH = (
#     "/home/ai-server/AI-Support_Sathwika/AI-Voice/"
#     "models/nemotron-3.5-asr-streaming-0.6b.nemo"
# )
MODEL_PATH = (
    "/home/ai-server/OUTBOUND_CALLS_SATHWIKA/OUTBOUND_CALLS_AI/PROACTIVE_OUTBOUND_AI/models/nemotron-3.5-asr-streaming-0.6b.nemo"
)
# MODEL_PATH = "./models/nemotron-3.5-asr-streaming-0.6b.nemo"

HOST = "127.0.0.1"
PORT = 9092
SAMPLE_RATE = 16000

# Nemotron 3.5 trained streaming operating points:
# [56, 0]  -> ~80 ms
# [56, 1]  -> ~160 ms
# [56, 3]  -> ~320 ms  (recommended balance)
# [56, 6]  -> ~560 ms
# [56,13]  -> ~1.12 s (best accuracy / highest latency)
ATT_CONTEXT_SIZE = [56, 13]

TARGET_LANG = "en-US"
DEVICE = "cuda:0"

# The live frontend sends small PCM packets. We do NOT run the
# mel preprocessor over the whole utterance for every packet.
# This is only a small fixed audio context used to make the
# current mel chunk. It keeps CPU/GPU work O(1) per streaming step.
RAW_FEATURE_CONTEXT_FRAMES = 10

# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("Loading Nemotron 3.5")
print("=" * 70)

torch.set_grad_enabled(False)
device = torch.device(DEVICE)

model = nemo_asr.models.ASRModel.restore_from(
    MODEL_PATH,
    map_location="cpu",
)
model = model.to(device)
model.eval()

# Prompt once. Language is fixed for this server.
model.set_inference_prompt(TARGET_LANG)
model.decoding.set_strip_lang_tags(True)

# Select the actual trained lookahead profile.
model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)

# Deterministic inference preprocessing.
if hasattr(model.preprocessor, "featurizer"):
    model.preprocessor.featurizer.dither = 0.0

# Avoid accidentally using training mode in the frontend.
model.preprocessor.eval()

print("=" * 70)
print("Nemotron 3.5 loaded")
print(f"Device: {DEVICE}")
print(f"Language: {TARGET_LANG}")
print(f"Attention context: {ATT_CONTEXT_SIZE}")
print(f"Streaming config: {model.encoder.streaming_cfg}")
print("=" * 70)


# ============================================================
# STREAMING GEOMETRY
# ============================================================

def _value_for_step(value, first_step: bool):
    if isinstance(value, (list, tuple)):
        return int(value[0] if first_step else value[1])
    return int(value)


def _get_streaming_geometry(first_step: bool):
    cfg = model.encoder.streaming_cfg

    chunk_frames = _value_for_step(cfg.chunk_size, first_step)
    shift_frames = _value_for_step(cfg.shift_size, first_step)
    pre_cache_frames = _value_for_step(cfg.pre_encode_cache_size, first_step)

    drop_extra = 0 if first_step else int(
        getattr(cfg, "drop_extra_pre_encoded", 0)
    )

    return chunk_frames, shift_frames, pre_cache_frames, drop_extra


# Nemotron 3.5 uses 25 ms / 10 ms / 16 kHz frontend.
# Read these from the checkpoint rather than hard-coding them.
PREPROCESSOR_CFG = model.cfg.preprocessor
WINDOW_STRIDE_SAMPLES = int(
    round(float(PREPROCESSOR_CFG.window_stride) * SAMPLE_RATE)
)

# NeMo's config uses 0.025 s for this checkpoint.
WINDOW_SAMPLES = int(
    round(float(PREPROCESSOR_CFG.window_size) * SAMPLE_RATE)
)

print(
    "Frontend:",
    f"window={WINDOW_SAMPLES} samples, "
    f"hop={WINDOW_STRIDE_SAMPLES} samples, "
    f"normalize={getattr(PREPROCESSOR_CFG, 'normalize', None)}",
)


# ============================================================
# SESSION
# ============================================================

@dataclass
class StreamingSession:
    session_id: str

    # Only a bounded recent raw-audio window is retained.
    raw_audio: Optional[np.ndarray] = None
    total_samples: int = 0

    # This is the feature-frame cursor (the same conceptual
    # buffer_idx used by NVIDIA's CacheAwareStreamingAudioBuffer).
    buffer_idx: int = 0

    # RNNT streaming state.
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None
    previous_hypotheses: Optional[list] = None
    previous_pred_out: Optional[torch.Tensor] = None

    # Transcript state.
    current_text: str = ""
    last_sent_text: str = ""

    step: int = 0

    first_audio_time: Optional[float] = None
    last_inference_time: Optional[float] = None

    def initialize(self):
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)

        self.raw_audio = np.empty(0, dtype=np.float32)
        self.total_samples = 0
        self.buffer_idx = 0

        self.previous_hypotheses = None
        self.previous_pred_out = None

        self.current_text = ""
        self.last_sent_text = ""
        self.step = 0

        self.first_audio_time = None
        self.last_inference_time = None


# ============================================================
# FRONTEND
# ============================================================

def _append_raw_audio(session: StreamingSession, audio: np.ndarray):
    """
    Keep only enough raw audio to reproduce the current feature chunk.

    We intentionally do not keep the complete utterance and do not
    repeatedly preprocess it. This removes the O(n^2) behavior in the
    original server.
    """
    if session.raw_audio is None:
        session.raw_audio = np.empty(0, dtype=np.float32)

    session.raw_audio = np.concatenate((session.raw_audio, audio))
    session.total_samples += int(audio.size)

    # Keep ~1.5 seconds. The largest normal Nemotron chunk is well
    # below this, including pre-encoder cache and a few frontend frames.
    max_keep = int(1.5 * SAMPLE_RATE)
    if session.raw_audio.size > max_keep:
        session.raw_audio = session.raw_audio[-max_keep:]


def _make_feature_chunk(
    session: StreamingSession,
    chunk_frames: int,
    pre_cache_frames: int,
):
    """
    Build exactly the feature view expected by conformer_stream_step.

    NVIDIA's cache-aware path expects:
        [pre-encode cache] + [current chunk]

    The original server incorrectly used `shift_size` as the chunk
    width. For Nemotron 3.5 these are not necessarily the same on the
    first step. We use streaming_cfg.chunk_size and shift_size exactly.

    A few extra frontend frames are processed before the requested
    region so pre-emphasis/STFT boundary effects do not contaminate the
    first requested frame.
    """
    current_start_frame = session.buffer_idx
    feature_start_frame = max(
        0,
        current_start_frame - pre_cache_frames,
    )
    feature_end_frame = (
        current_start_frame + chunk_frames
    )

    requested_frames = (
        feature_end_frame - feature_start_frame
    )

    # Give the waveform frontend a small history before the first
    # requested frame. This is intentionally independent of the
    # encoder's 56-frame attention cache.
    context_frames = min(
        RAW_FEATURE_CONTEXT_FRAMES,
        feature_start_frame,
    )

    raw_start_frame = feature_start_frame - context_frames

    # NeMo's frontend produces approximately one feature frame per
    # window_stride. Use enough samples for the requested end frame.
    required_end_samples = (
        feature_end_frame * WINDOW_STRIDE_SAMPLES
    )

    # Current raw_audio is a bounded suffix of the utterance.
    raw_global_start = (
        session.total_samples - len(session.raw_audio)
    )

    if raw_global_start > raw_start_frame * WINDOW_STRIDE_SAMPLES:
        raise RuntimeError(
            "Raw audio history is too short for the requested "
            "streaming feature chunk."
        )

    required_samples = (
        required_end_samples
        - raw_start_frame * WINDOW_STRIDE_SAMPLES
    )

    local_start = (
        raw_start_frame * WINDOW_STRIDE_SAMPLES
        - raw_global_start
    )

    local_end = local_start + required_samples

    if local_end > len(session.raw_audio):
        return None, None

    waveform = session.raw_audio[
        local_start:local_end
    ]

    if waveform.size < required_samples:
        return None, None

    audio_tensor = torch.from_numpy(
        waveform.copy()
    ).unsqueeze(0).to(device)

    audio_len = torch.tensor(
        [waveform.size],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        features, feature_len = model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )

    valid_features = int(feature_len[0].item())

    # The frontend should have produced enough frames for the
    # requested region. If it produced a few extra frames because of
    # frontend implementation details, take the tail that corresponds
    # to the requested [feature_start_frame, feature_end_frame).
    if valid_features < requested_frames:
        return None, None

    chunk_mel = features[
        :,
        :,
        valid_features - requested_frames:valid_features,
    ].contiguous()

    chunk_len = torch.tensor(
        [requested_frames],
        dtype=torch.long,
        device=device,
    )

    return chunk_mel, chunk_len


# ============================================================
# RNNT STREAMING STEP
# ============================================================

def process_streaming_step(
    session: StreamingSession,
    chunk_mel: torch.Tensor,
    chunk_len: torch.Tensor,
    keep_all_outputs: bool,
    drop_extra_pre_encoded: int,
):
    with torch.inference_mode():
        (
            pred_out,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            previous_hypotheses=session.previous_hypotheses,
            previous_pred_out=session.previous_pred_out,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=True,
        )

    session.cache_last_channel = cache_last_channel
    session.cache_last_time = cache_last_time
    session.cache_last_channel_len = cache_last_channel_len
    session.previous_hypotheses = previous_hypotheses
    session.previous_pred_out = pred_out
    session.step += 1

    return extract_text(transcribed_texts)


def extract_text(transcribed_texts):
    if not transcribed_texts:
        return ""

    hyp = transcribed_texts[0]

    if hasattr(hyp, "text"):
        return hyp.text.strip()

    if isinstance(hyp, str):
        return hyp.strip()

    return str(hyp).strip()


# ============================================================
# STREAM AUDIO
# ============================================================

def process_audio(
    session: StreamingSession,
    pcm_bytes: bytes,
):
    if not pcm_bytes:
        return None

    pcm = np.frombuffer(
        pcm_bytes,
        dtype=np.int16,
    )

    if pcm.size == 0:
        return None

    audio = (
        pcm.astype(np.float32) / 32768.0
    )

    if session.first_audio_time is None:
        session.first_audio_time = time.perf_counter()

    _append_raw_audio(session, audio)

    latest_text = None

    # A single websocket message can contain enough audio for more
    # than one streaming step. Drain all ready steps.
    while True:
        first_step = session.step == 0

        (
            chunk_frames,
            shift_frames,
            pre_cache_frames,
            drop_extra,
        ) = _get_streaming_geometry(first_step)

        # The model cannot emit this step until the complete main
        # chunk has arrived.
        feature_end_frame = (
            session.buffer_idx + chunk_frames
        )
        required_samples = (
            feature_end_frame * WINDOW_STRIDE_SAMPLES
        )

        if session.total_samples < required_samples:
            break

        chunk_mel, chunk_len = _make_feature_chunk(
            session,
            chunk_frames,
            pre_cache_frames,
        )

        if chunk_mel is None:
            break

        step_start = time.perf_counter()

        text = process_streaming_step(
            session=session,
            chunk_mel=chunk_mel,
            chunk_len=chunk_len,
            # Do not drop the final encoder outputs here. Finalization
            # will run the final ready chunk with keep_all_outputs=True.
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra,
        )

        session.last_inference_time = time.perf_counter()

        if text:
            session.current_text = text
            latest_text = text

        # Exactly mirror NVIDIA's CacheAwareStreamingAudioBuffer:
        # advance by shift_size, not chunk_size.
        session.buffer_idx += shift_frames

        processing_ms = (
            time.perf_counter() - step_start
        ) * 1000.0

        print(
            f"[{session.session_id}] "
            f"step={session.step} "
            f"chunk={chunk_frames} "
            f"shift={shift_frames} "
            f"precache={pre_cache_frames} "
            f"infer={processing_ms:.1f} ms "
            f"text={session.current_text!r}"
        )

    return latest_text


# ============================================================
# FINALIZATION
# ============================================================

def finalize_session(session: StreamingSession):
    """
    Finish only the not-yet-emitted tail.

    The old implementation re-preprocessed the complete utterance
    from sample zero and replayed every streaming step. That caused
    large final latency. Here we only add enough zero audio to make
    the next trained streaming chunk complete.
    """
    if session.total_samples <= 0:
        return session.current_text.strip()

    # We need at most the next chunk to become available. Adding
    # silence also supplies the model's trained right-lookahead.
    first_step = session.step == 0
    chunk_frames, _, _, _ = _get_streaming_geometry(first_step)

    target_end_frame = (
        session.buffer_idx + chunk_frames
    )
    target_samples = (
        target_end_frame * WINDOW_STRIDE_SAMPLES
    )

    if session.total_samples < target_samples:
        pad_samples = (
            target_samples - session.total_samples
        )
        _append_raw_audio(
            session,
            np.zeros(
                pad_samples,
                dtype=np.float32,
            ),
        )

    # Drain ready chunks. The last one must keep all outputs so that
    # the final encoder frames are not discarded.
    last_text = session.current_text

    while True:
        first_step = session.step == 0

        (
            chunk_frames,
            shift_frames,
            pre_cache_frames,
            drop_extra,
        ) = _get_streaming_geometry(first_step)

        required_samples = (
            (session.buffer_idx + chunk_frames)
            * WINDOW_STRIDE_SAMPLES
        )

        if session.total_samples < required_samples:
            break

        chunk_mel, chunk_len = _make_feature_chunk(
            session,
            chunk_frames,
            pre_cache_frames,
        )

        if chunk_mel is None:
            break

        text = process_streaming_step(
            session=session,
            chunk_mel=chunk_mel,
            chunk_len=chunk_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=drop_extra,
        )

        if text:
            last_text = text
            session.current_text = text

        session.buffer_idx += shift_frames

        # Normally one padded tail step is sufficient. If the caller
        # sent a very short utterance, the loop naturally drains the
        # first step and stops.
        if session.step > 0:
            # Do not manufacture another full speech turn from silence.
            break

    return last_text.strip()


# ============================================================
# WEBSOCKET
# ============================================================

async def websocket_handler(request: web.Request):
    ws = web.WebSocketResponse(
        max_msg_size=10 * 1024 * 1024
    )
    await ws.prepare(request)

    session = StreamingSession(
        session_id=str(uuid.uuid4())[:8]
    )
    session.initialize()

    print(
        f"[{session.session_id}] client connected"
    )

    await ws.send_json(
        {
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "language": TARGET_LANG,
            "att_context_size": ATT_CONTEXT_SIZE,
            "streaming_config": str(
                model.encoder.streaming_cfg
            ),
        }
    )

    try:
        async for message in ws:

            if message.type == aiohttp.WSMsgType.BINARY:
                start = time.perf_counter()

                text = await asyncio.to_thread(
                    process_audio,
                    session,
                    message.data,
                )

                processing_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                if text and text != session.last_sent_text:
                    session.last_sent_text = text

                    await ws.send_json(
                        {
                            "type": "transcript",
                            "text": text,
                            "is_final": False,
                            "processing_ms": round(
                                processing_ms,
                                2,
                            ),
                        }
                    )

            elif message.type == aiohttp.WSMsgType.TEXT:
                try:
                    command = json.loads(
                        message.data
                    )
                except json.JSONDecodeError:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Invalid JSON",
                        }
                    )
                    continue

                command_type = command.get("type")

                if command_type == "reset":
                    start = time.perf_counter()

                    final_text = await asyncio.to_thread(
                        finalize_session,
                        session,
                    )

                    final_ms = (
                        time.perf_counter() - start
                    ) * 1000.0

                    if final_text:
                        await ws.send_json(
                            {
                                "type": "transcript",
                                "text": final_text,
                                "is_final": True,
                                "finalize": True,
                                "processing_ms": round(
                                    final_ms,
                                    2,
                                ),
                            }
                        )

                    # New utterance; websocket stays alive.
                    session = StreamingSession(
                        session_id=session.session_id
                    )
                    session.initialize()

                    await ws.send_json(
                        {
                            "type": "reset_complete"
                        }
                    )

                elif command_type == "language":
                    language = command.get(
                        "language",
                        TARGET_LANG,
                    )

                    model.set_inference_prompt(
                        language
                    )
                    model.decoding.set_strip_lang_tags(
                        True
                    )

                    await ws.send_json(
                        {
                            "type": "language_changed",
                            "language": language,
                        }
                    )

                elif command_type == "ping":
                    await ws.send_json(
                        {
                            "type": "pong"
                        }
                    )

            elif message.type == aiohttp.WSMsgType.ERROR:
                print(
                    f"[{session.session_id}] "
                    f"WebSocket error: "
                    f"{ws.exception()}"
                )

    except Exception as e:
        print(
            f"[{session.session_id}] ERROR: {e}"
        )

        try:
            await ws.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except Exception:
            pass

    finally:
        print(
            f"[{session.session_id}] "
            "client disconnected"
        )

    return ws


# ============================================================
# HEALTH
# ============================================================

async def health(request):
    return web.json_response(
        {
            "status": "ok",
            "model": "nemotron-3.5-asr-streaming-0.6b",
            "device": DEVICE,
            "sample_rate": SAMPLE_RATE,
            "att_context_size": ATT_CONTEXT_SIZE,
            "streaming_config": str(
                model.encoder.streaming_cfg
            ),
        }
    )


# ============================================================
# MAIN
# ============================================================

async def main():
    app = web.Application()

    app.router.add_get(
        "/health",
        health,
    )

    app.router.add_get(
        "/",
        websocket_handler,
    )

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(
        runner,
        HOST,
        PORT,
    )

    await site.start()

    print("=" * 70)
    print(
        f"Nemotron WebSocket server: "
        f"ws://127.0.0.1:{PORT}"
    )
    print(
        f"Health: "
        f"http://127.0.0.1:{PORT}/health"
    )
    print("=" * 70)

    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
