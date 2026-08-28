"""
whisper_server.py — Standalone Whisper WebSocket STT Server
============================================================
Runs Whisper (distil-large-v3) in its OWN process so it never competes
with Piper TTS, Ollama LLM, or Silero VAD for CUDA memory.

Start once, keep running:
    python whisper_server.py

Bot connects via: ws://127.0.0.1:8081
Multiple bot calls reuse the same loaded model — no reloads, no OOM.

Protocol (matches pipecat's WhisperWebSocketSTTService expected format):
  Client → Server:  raw PCM int16 bytes  (16 kHz, mono)
  Client → Server:  JSON {"type": "reset"}   # end of utterance / flush
  Server → Client:  JSON {"type": "ready"}   # on connect
  Server → Client:  JSON {"type": "transcript", "text": "...", "is_final": false}
  Server → Client:  JSON {"type": "transcript", "text": "...", "is_final": true}
"""

import asyncio
import json
import logging
import numpy as np
import torch
from faster_whisper import WhisperModel
from websockets.asyncio.server import serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [whisper_server] %(levelname)s %(message)s",
)
log = logging.getLogger("whisper_server")

# ─────────────────────────────────────────────────────────────────────────────
# Model config — change here to switch model/precision
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME    = "distil-large-v3"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE  = "int8_float16" if DEVICE == "cuda" else "int8"
LANGUAGE      = "en"
HOST          = "127.0.0.1"
PORT          = 8080

# Minimum audio before we attempt transcription (0.5 s @ 16 kHz)
# Shorter = faster interim for short phrases like "hi", "OCI", "passport"
MIN_SAMPLES   = 8000
# Overlap kept between chunks to avoid cutting words (0.3 s)
OVERLAP_SAMPLES = 4800

# Domain prompt — primes Whisper with vocabulary it will hear in this app.
# Dramatically improves recognition of proper nouns and domain terms.
INITIAL_PROMPT = (
    "Indian Consulate Services, Johannesburg. "
    "Passport, OCI, PCC, police clearance certificate, "
    "visa, application, renewal, reissue, documents, "
    "office hours, working hours, appointment, status, track."
)

# ─────────────────────────────────────────────────────────────────────────────
# Load model ONCE at startup
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE}) …")
whisper_model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    num_workers=1,
)
log.info("Model loaded and ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-connection handler
# ─────────────────────────────────────────────────────────────────────────────
async def handler(websocket):
    remote = websocket.remote_address
    log.info(f"Client connected: {remote}")

    # Tell the client the server is ready
    await websocket.send(json.dumps({"type": "ready"}))

    # utterance_buffer holds ALL audio for the current utterance (for final)
    utterance_buffer: np.ndarray = np.array([], dtype=np.float32)
    # interim_buffer is the sliding window used for rolling interim transcripts
    interim_buffer: np.ndarray = np.array([], dtype=np.float32)
    last_interim_text = ""
    last_sent_final   = ""

    try:
        async for message in websocket:

            # ── Text frame: control message ─────────────────────────────────
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                if data.get("type") == "reset":
                    log.debug("Reset received — transcribing full utterance")

                    # Transcribe the FULL utterance audio for an accurate final
                    if len(utterance_buffer) > 0:
                        text = _transcribe(utterance_buffer)
                        if not text:
                            # Fall back to last interim if Whisper returns empty
                            text = last_interim_text.strip()
                        if text and text != last_sent_final:
                            await websocket.send(json.dumps({
                                "type":     "transcript",
                                "text":     text,
                                "is_final": True,
                            }))
                            last_sent_final = text

                    # Reset all state for next utterance
                    utterance_buffer  = np.array([], dtype=np.float32)
                    interim_buffer    = np.array([], dtype=np.float32)
                    last_interim_text = ""
                continue

            # ── Binary frame: raw PCM int16 audio ──────────────────────────
            if isinstance(message, bytes):
                # PCM int16 → float32 normalised to [-1.0, 1.0]
                samples = (
                    np.frombuffer(message, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )

                # Always accumulate the FULL utterance (never trimmed)
                utterance_buffer = np.concatenate((utterance_buffer, samples))

                # Sliding interim buffer — grow then keep overlap tail
                interim_buffer = np.concatenate((interim_buffer, samples))

                # Don't transcribe interim until we have at least 1 s of audio
                if len(interim_buffer) < MIN_SAMPLES:
                    continue

                # Transcribe the current sliding window for interim results
                text = _transcribe(interim_buffer)

                # Slide the window: keep overlap tail for next interim
                interim_buffer = interim_buffer[-OVERLAP_SAMPLES:]

                if not text:
                    continue

                # Only send if something new was recognised
                if text.lower() == last_interim_text.lower():
                    continue

                last_interim_text = text

                await websocket.send(json.dumps({
                    "type":     "transcript",
                    "text":     text,
                    "is_final": False,
                }))

    except asyncio.CancelledError:
        log.info(f"Client cancelled: {remote}")
    except Exception as exc:
        log.error(f"Handler error for {remote}: {exc}", exc_info=True)
    finally:
        log.info(f"Client disconnected: {remote}")


def _transcribe(audio: np.ndarray, is_final: bool = False) -> str:
    """Run Whisper on a numpy float32 array. Returns stripped text or ''."""
    try:
        segments, info = whisper_model.transcribe(
            audio,
            language=LANGUAGE,
            # beam_size=5 considers 5 candidate paths — much more accurate than
            # greedy beam_size=1, especially for short or accented speech.
            # Safe here since Whisper runs in its own process.
            beam_size=5,
            # Use higher beam for final transcript — we have the full utterance
            # and accuracy matters more than speed at this point.
            # best_of only applies when temperature > 0, keep at default.
            vad_filter=True,
            # Domain prompt: tells Whisper the vocabulary context upfront.
            # Massively improves recognition of "OCI", "PCC", "consulate" etc.
            initial_prompt=INITIAL_PROMPT,
            # Use previous text as context for multi-sentence utterances.
            condition_on_previous_text=True,
            temperature=0,   # deterministic — no random sampling
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob < 0.8:
                parts.append(seg.text)
            else:
                log.debug(f"Dropping segment no_speech_prob={seg.no_speech_prob:.2f}: {seg.text!r}")
        return " ".join(parts).strip()
    except Exception as exc:
        log.error(f"Transcription error: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    log.info(f"Whisper WebSocket STT server listening on ws://{HOST}:{PORT}")
    async with serve(handler, HOST, PORT, ping_interval=None, max_size=10 * 1024 * 1024):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
