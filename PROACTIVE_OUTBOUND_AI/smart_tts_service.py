"""
VoiceCloneTTSService: one Pipecat TTS service for laptop (CPU) and server (GPU).

  backend="auto" (default)
      GPU + faster-qwen3-tts installed -> Qwen3-TTS  (fast, text-instruction emotion)
      otherwise                        -> Chatterbox (works on CPU and GPU)
  If Qwen fails to load, it falls back to Chatterbox automatically.

Force one with backend="qwen" / "chatterbox" or env TTS_BACKEND.

Install per machine, in SEPARATE environments (their dependencies conflict):
  Laptop (CPU) :  pip install chatterbox-tts pipecat-ai soundfile librosa
  Server (GPU) :  pip install faster-qwen3-tts pipecat-ai soundfile librosa   (CUDA build of torch)
                  optional fallback env: chatterbox-tts

Usage in your agent (replace the Piper line):
    from smart_tts_service import VoiceCloneTTSService
    tts = VoiceCloneTTSService(ref_audio="my_voice.mp3", ref_text="words said in the clip")

Quick test without a pipeline:   python smart_tts_service.py my_voice.mp3 "words said in the clip"
"""
import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Iterator, Optional

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

SAMPLE_RATE = 24000
REF_MAX_SECONDS = 8  # keep the reference short; ref_text must match this clip

# Same emotion names work for both engines.
QWEN_EMOTIONS = {
    "neutral": "",
    "happy": "Speak in a warm, cheerful and happy tone.",
    "excited": "Speak with high energy and excitement.",
    "sad": "Speak softly in a sad, subdued tone.",
    "calm": "Speak slowly in a calm, soothing tone.",
    "empathetic": "Speak gently with warmth and empathy.",
    "angry": "Speak in a firm, angry tone.",
}
CHATTERBOX_EMOTIONS = {  # (exaggeration, cfg_weight) - tune by ear
    "neutral": (0.5, 0.5),
    "calm": (0.3, 0.6),
    "happy": (0.7, 0.4),
    "excited": (0.9, 0.3),
    "sad": (0.4, 0.5),
    "empathetic": (0.45, 0.5),
    "angry": (0.85, 0.35),
}


def prepare_reference(path: str) -> str:
    """mp3/wav -> mono 24 kHz wav, silence trimmed, max 8 s, normalized."""
    import librosa
    import soundfile as sf

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")
    audio, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=35)
    if len(audio) / SAMPLE_RATE < 3:
        raise ValueError("Reference is shorter than 3 s of speech. Use 5-8 s of clear speech.")
    if len(audio) / SAMPLE_RATE > REF_MAX_SECONDS:
        logger.warning(f"Reference longer than {REF_MAX_SECONDS}s: using the first {REF_MAX_SECONDS}s only")
        audio = audio[: SAMPLE_RATE * REF_MAX_SECONDS]
    audio = audio / max(float(np.abs(audio).max()), 1e-6) * 0.9
    out = p.with_name(p.stem + "_ref24k.wav")
    sf.write(str(out), audio, SAMPLE_RATE)
    return str(out)


# ----------------------------------------------------------------- backends
class QwenBackend:
    name = "qwen"

    def __init__(self, ref_wav: str, ref_text: str, language: str):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen backend needs a GPU")
        if len((ref_text or "").strip()) < 5:
            raise ValueError("ref_text is required for Qwen (exact words spoken in the reference clip)")
        from transformers import MimiConfig

        if not hasattr(MimiConfig, "rope_theta"):  # transformers 5.x compatibility
            MimiConfig.rope_theta = property(
                lambda self: (getattr(self, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
            )
        from faster_qwen3_tts import FasterQwen3TTS

        self.ref_wav, self.ref_text, self.language = ref_wav, ref_text.strip(), language
        # bf16 on Ampere+ GPUs; float32 on older ones (float16 produced NaN crashes on T4)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        self.model = FasterQwen3TTS.from_pretrained(
            os.getenv("TTS_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            device="cuda",
            dtype=dtype,
            max_seq_len=2048,
        )

    def stream(self, text: str, emotion: str) -> Iterator[np.ndarray]:
        instruct = QWEN_EMOTIONS.get(emotion, emotion)  # free text also allowed
        for chunk, _sr, _t in self.model.generate_voice_clone_streaming(
            text=text,
            language=self.language,
            ref_audio=self.ref_wav,
            ref_text=self.ref_text,
            chunk_size=8,
            xvec_only=False,
            instruct=instruct or None,
        ):
            if hasattr(chunk, "cpu"):
                chunk = chunk.detach().float().cpu().numpy()
            yield np.asarray(chunk, dtype=np.float32).flatten()


class ChatterboxBackend:
    name = "chatterbox"

    def __init__(self, ref_wav: str):
        self._ensure_watermarker()
        from chatterbox.tts import ChatterboxTTS

        self.ref_wav = ref_wav
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChatterboxTTS.from_pretrained(device=self.device)

    @staticmethod
    def _ensure_watermarker():
        """Chatterbox crashes with "'NoneType' object is not callable" when the `perth`
        watermark library failed to import (usually: pip install "setuptools<81").
        Fix the cause if you can. Otherwise this uses a pass-through so the service
        still runs, and logs a warning. Set TTS_REQUIRE_WATERMARK=1 to fail instead."""
        import perth

        if getattr(perth, "PerthImplicitWatermarker", None) is not None:
            return
        try:
            from perth.perth_net.perth_net_implicit.perth_watermarker import (  # noqa: F401
                PerthImplicitWatermarker,
            )

            perth.PerthImplicitWatermarker = PerthImplicitWatermarker
            return
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}"
        msg = (
            f"Chatterbox watermarker (perth) failed to load: {reason}. "
            'Proper fix: pip install "setuptools<81". '
        )
        if os.getenv("TTS_REQUIRE_WATERMARK") == "1":
            raise RuntimeError(msg)
        logger.warning(msg + "Continuing WITHOUT the AI watermark (ok for testing, fix before deploying).")

        class _PassThroughWatermarker:
            def apply_watermark(self, wav, sample_rate=None, **kwargs):
                return wav

            def get_watermark(self, *args, **kwargs):
                return 0.0

        perth.PerthImplicitWatermarker = _PassThroughWatermarker

    def stream(self, text: str, emotion: str) -> Iterator[np.ndarray]:
        exaggeration, cfg = CHATTERBOX_EMOTIONS.get(emotion, CHATTERBOX_EMOTIONS["neutral"])
        with torch.inference_mode():
            wav = self.model.generate(
                text, audio_prompt_path=self.ref_wav, exaggeration=exaggeration, cfg_weight=cfg
            )
        audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        step = SAMPLE_RATE // 2  # 0.5 s pieces
        for i in range(0, len(audio), step):
            yield audio[i : i + step]


def build_backend(choice: str, ref_wav: str, ref_text: str, language: str):
    choice = (choice or "auto").lower()
    use_qwen_first = choice == "qwen" or (choice == "auto" and torch.cuda.is_available())
    if use_qwen_first:
        try:
            b = QwenBackend(ref_wav, ref_text, language)
            logger.info("TTS backend: Qwen3-TTS (GPU)")
            return b
        except Exception as e:  # noqa: BLE001
            if choice == "qwen":
                raise
            logger.warning(f"Qwen unavailable ({e}). Falling back to Chatterbox.")
    b = ChatterboxBackend(ref_wav)
    logger.info(f"TTS backend: Chatterbox ({b.device})")
    return b


# ------------------------------------------------------------------ service
class VoiceCloneTTSService(TTSService):
    def __init__(
        self,
        *,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        backend: Optional[str] = None,
        language: str = "English",
        emotion: str = "neutral",
        **kwargs,
    ):
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
        ref_audio = ref_audio or os.getenv("TTS_REF_AUDIO")
        ref_text = ref_text or os.getenv("TTS_REF_TEXT", "")
        if not ref_audio:
            raise ValueError("ref_audio (or env TTS_REF_AUDIO) is required")
        self._emotion = emotion
        self._lock = asyncio.Lock()  # one generation at a time
        self._backend = build_backend(
            backend or os.getenv("TTS_BACKEND", "auto"), prepare_reference(ref_audio), ref_text, language
        )
        for _ in self._backend.stream("Warm up.", "neutral"):  # warm-up (CUDA graphs, caches)
            pass
        logger.info("TTS ready")

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def can_generate_metrics(self) -> bool:
        return True

    def set_emotion(self, emotion: str):
        """neutral, happy, excited, sad, calm, empathetic, angry (Qwen also accepts free text)."""
        self._emotion = emotion

    @staticmethod
    def _pcm16(chunk: np.ndarray) -> bytes:
        return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    async def run_tts(self, text: str, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        text = text.strip()
        if not text:
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        emotion = self._emotion

        def worker():  # generation is blocking, so it runs in a thread
            try:
                for chunk in self._backend.stream(text, emotion):
                    if stop.is_set():  # user interrupted
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, self._pcm16(chunk))
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        async with self._lock:
            try:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                threading.Thread(target=worker, daemon=True).start()
                first = True
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        logger.error(f"TTS error: {item}")
                        yield ErrorFrame(f"TTS error: {item}")
                        break
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    yield TTSAudioRawFrame(item, SAMPLE_RATE, 1)
            except asyncio.CancelledError:
                stop.set()
                raise
            finally:
                stop.set()
                yield TTSStoppedFrame()


# ------------------------------------------------------------- quick self-test
if __name__ == "__main__":
    import soundfile as sf

    ref = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TTS_REF_AUDIO", "sathwika_voice.mp3")
    txt = sys.argv[2] if len(sys.argv) > 2 else os.getenv("TTS_REF_TEXT", "")
    svc = VoiceCloneTTSService(ref_audio=ref, ref_text=txt)
    print("Backend:", svc.backend_name)
    for emo in ("neutral", "happy"):
        t0, first, parts = time.time(), None, []
        for c in svc._backend.stream("Hello, how are you today? I hope you are doing well.", emo):
            first = first or time.time() - t0
            parts.append(c)
        a = np.concatenate(parts)
        dur, tot = len(a) / SAMPLE_RATE, time.time() - t0
        print(f"[{emo}] first audio {first:.2f}s | audio {dur:.1f}s | took {tot:.1f}s | RTF {tot / dur:.2f}")
        sf.write(f"test_{emo}.wav", a, SAMPLE_RATE)