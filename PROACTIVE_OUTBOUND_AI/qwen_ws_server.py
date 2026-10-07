"""
Qwen3-TTS WebSocket server (run on the GPU machine).

  pip install faster-qwen3-tts fastapi "uvicorn[standard]" librosa soundfile
  REF_AUDIO=sathwika_voice.mp3 REF_TEXT="exact words in the first 8 seconds" \
  TTS_API_KEY=secret python qwen_ws_server.py

Protocol on  ws://HOST:8000/ws?key=<TTS_API_KEY>
  server -> {"type":"ready","sample_rate":24000}
  client -> {"type":"speak","id":"<8 chars>","text":"...","emotion":"happy"}   (or "instruct": "free text")
  server -> {"type":"start","id":...}
  server -> BINARY: 8 ASCII bytes of request id + raw int16 mono PCM @24 kHz   (many frames, streamed)
  server -> {"type":"end","id":...,"first_audio_s":..,"audio_s":..,"took_s":..,"cancelled":false}
  server -> {"type":"error","id":...,"message":"..."}
  client -> {"type":"cancel","id":...}     stop the current sentence (user interrupted)
"""
import asyncio
import os
import threading
import time

import numpy as np
import torch

# transformers 5.x MimiConfig has no `rope_theta`, but qwen_tts reads it
from transformers import MimiConfig

if not hasattr(MimiConfig, "rope_theta"):
    MimiConfig.rope_theta = property(
        lambda self: (getattr(self, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
    )

import librosa
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from faster_qwen3_tts import FasterQwen3TTS
# ref_voice=r"/home/ai-server/Downloads/english_voice.mp3"
ref_voice = r"/home/ai-server/OUTBOUND_CALLS_SATHWIKA/OUTBOUND_CALLS_AI/PROACTIVE_OUTBOUND_AI/ref_clean.wav"
REF_AUDIO = os.getenv("REF_AUDIO", ref_voice)
# REF_TEXT = os.getenv("REF_TEXT", "").strip()
# REF_TEXT="Hi.Hello how are you all,I am sathwika,Today i am calling you to make happy, once again congratulations."
REF_TEXT= "The irresistible tingling taste of cornito's nacho crisp.that will sizzle your taste buts and leave you wanting more."
API_KEY = os.getenv("TTS_API_KEY", "")
MODEL = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
LANGUAGE = os.getenv("TTS_LANGUAGE", "English")
CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", "8"))  # codec steps per chunk: smaller = earlier first audio
MAX_SEQ_LEN = int(os.getenv("TTS_MAX_SEQ_LEN", "1024"))
SAMPLE_RATE = 24000

EMOTIONS = {
    "neutral": "",
    "support": "Speak warmly, patiently, and professionally, with genuine empathy. Sound helpful and reassuring, not overly emotional.",
    "happy": "Speak in a warm, cheerful and happy tone.",
    "excited": "Speak with high energy and excitement.",
    "sad": "Speak softly in a sad, subdued tone.",
    "calm": "Speak slowly in a calm, soothing tone.",
    "empathetic": "Speak gently with warmth and empathy.",
    "angry": "Speak in a firm, angry tone.",
}

if len(REF_TEXT) < 5:
    raise SystemExit("Set REF_TEXT to the exact words spoken in your reference audio.")
if not torch.cuda.is_available():
    raise SystemExit("This server needs an NVIDIA GPU (CUDA).")

# ---- reference clip: mono 24 kHz, silence trimmed, max 8 s (REF_TEXT must match this clip)
audio, _ = librosa.load(REF_AUDIO, sr=SAMPLE_RATE, mono=True)
audio, _ = librosa.effects.trim(audio, top_db=35)
REF_MAX_SECONDS = float(os.getenv("REF_MAX_SECONDS", "8"))
if len(audio) / SAMPLE_RATE > REF_MAX_SECONDS:
    print(f"WARNING: reference is {len(audio) / SAMPLE_RATE:.1f}s, only the first {REF_MAX_SECONDS:.0f}s is used. "
          "REF_TEXT must match ONLY those seconds (use a shorter clip, or raise REF_MAX_SECONDS).")
audio = audio[: int(SAMPLE_RATE * REF_MAX_SECONDS)]
if len(audio) / SAMPLE_RATE < 3:
    raise SystemExit("Reference has less than 3 s of speech. Use 5-8 s of clear speech.")
audio = audio / max(float(np.abs(audio).max()), 1e-6) * 0.9
REF_WAV = "ref_clean.wav"
sf.write(REF_WAV, audio, SAMPLE_RATE)
print(f"Reference ready: {len(audio) / SAMPLE_RATE:.1f}s")

# ---- model: bf16 on Ampere+ GPUs; float32 on older GPUs (float16 gave NaN crashes on T4)
_dt = os.getenv("TTS_DTYPE", "").lower()
if _dt in ("bf16", "fp16", "fp32"):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[_dt]
else:
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype}")
model = FasterQwen3TTS.from_pretrained(MODEL, device="cuda", dtype=dtype, max_seq_len=MAX_SEQ_LEN)

GPU_LOCK = threading.Lock()  # the CUDA-graph model serves one generation at a time


def stream_pcm(text: str, instruct: str):
    # cap the length so a bad generation cannot run on for minutes
    max_tokens = int(min(1200, max(60, len(text.split()) * 12 + 30)))
    for chunk, _sr, _timing in model.generate_voice_clone_streaming(
        text=text,
        language=LANGUAGE,
        ref_audio=REF_WAV,
        ref_text=REF_TEXT,
        chunk_size=CHUNK_SIZE,
        xvec_only=False,
        instruct=instruct or None,
        max_new_tokens=max_tokens,
    ):
        if hasattr(chunk, "cpu"):
            chunk = chunk.detach().float().cpu().numpy()
        pcm = np.clip(np.asarray(chunk, dtype=np.float32).flatten(), -1.0, 1.0)
        yield (pcm * 32767).astype(np.int16).tobytes()


print("Warming up (captures CUDA graphs, caches the voice)...")
for _ in stream_pcm("Hello, this is a warm up sentence.", ""):
    pass
print("Ready")

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype)}


async def generate(ws: WebSocket, req_id: str, text: str, instruct: str, stop: threading.Event):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            with GPU_LOCK:
                if stop.is_set():
                    return
                for pcm in stream_pcm(text, instruct):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, pcm)
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()
    prefix = req_id.encode("ascii", "replace")[:8]
    t0, first, nbytes = time.time(), None, 0
    try:
        await ws.send_json({"type": "start", "id": req_id})
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                await ws.send_json({"type": "error", "id": req_id, "message": str(item)})
                return
            if first is None:
                first = time.time() - t0
            nbytes += len(item)
            if not stop.is_set():
                await ws.send_bytes(prefix + item)
        dur, took = nbytes / 2 / SAMPLE_RATE, time.time() - t0
        print(f"[{req_id}] first {first or 0:.2f}s | {dur:.1f}s audio in {took:.1f}s "
              f"(RTF {took / max(dur, 1e-6):.2f}) | cancelled={stop.is_set()} | {text[:50]!r}")
        await ws.send_json({
            "type": "end", "id": req_id, "first_audio_s": first, "audio_s": dur,
            "took_s": took, "cancelled": stop.is_set(),
        })
    except Exception:  # noqa: BLE001  (connection went away)
        stop.set()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if API_KEY and ws.query_params.get("key") != API_KEY:
        await ws.close(code=4401)
        return
    await ws.accept()
    await ws.send_json({"type": "ready", "sample_rate": SAMPLE_RATE})
    cur = {"task": None, "stop": None, "id": None}

    async def stop_current():
        if cur["task"] and not cur["task"].done():
            cur["stop"].set()
            try:
                await asyncio.wait_for(cur["task"], 10)
            except Exception:  # noqa: BLE001
                pass

    try:
        while True:
            try:
                msg = await ws.receive_json()
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "speak":
                await stop_current()
                req_id = (str(msg.get("id", "")) + "00000000")[:8]
                text = str(msg.get("text", "")).strip()
                if not text:
                    await ws.send_json({"type": "end", "id": req_id, "audio_s": 0, "cancelled": False})
                    continue
                instruct = msg.get("instruct")
                if instruct is None:
                    emo = str(msg.get("emotion", "support"))
                    instruct = EMOTIONS.get(emo, emo)
                stop = threading.Event()
                cur.update(task=asyncio.create_task(generate(ws, req_id, text, instruct, stop)),
                           stop=stop, id=req_id)
            elif kind == "cancel":
                if cur["stop"] is not None and msg.get("id") in (None, cur["id"]):
                    cur["stop"].set()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if cur["stop"] is not None:
            cur["stop"].set()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8888")), ws_ping_interval=20, ws_ping_timeout=20)