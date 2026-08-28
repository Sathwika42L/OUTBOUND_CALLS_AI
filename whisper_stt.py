"""
whisper_stt.py
Pipecat STT service that delegates to whisper_server.py over WebSocket.

KEY BEHAVIOUR: Audio is ONLY forwarded to the Whisper server while
Silero VAD has confirmed the user is speaking. A pre-roll buffer keeps
the last 300ms of audio so the first syllable is never lost when the
VAD gate opens slightly after speech begins.
"""
import os
import json
from collections import deque
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language

# Pre-roll: how many audio bytes to keep before VAD opens the gate.
# 16000 Hz * 2 bytes/sample * 0.6 s = 19200 bytes  (~600 ms)
# Longer pre-roll captures more of the opening speech, but uses more RAM.
PREROLL_BYTES = 19200


class WhisperWebSocketSTTService(WebsocketSTTService):
    """
    Streams audio to a remote Whisper WebSocket server (whisper_server.py).

    Audio is gated by Silero VAD — only audio frames while the user is
    speaking are forwarded. A 300 ms pre-roll buffer is flushed when
    speech starts so the first syllables are never missed.
    """

    def __init__(
        self,
        url: str = os.getenv("STT_URL", "ws://stt:8080"),
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = url
        self._language = language
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None

        # VAD gate state
        self._user_speaking: bool = False

        # Rolling pre-roll buffer — stores recent audio chunks as raw bytes.
        # When the VAD gate opens we flush this first so the opening syllable
        # is included. We accumulate bytes and trim to PREROLL_BYTES.
        self._preroll: deque = deque()          # deque of bytes chunks
        self._preroll_total: int = 0            # total bytes currently held

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preroll_add(self, audio: bytes) -> None:
        """Add a chunk to the pre-roll buffer, trimming the oldest bytes."""
        self._preroll.append(audio)
        self._preroll_total += len(audio)
        # Trim front until we're within the limit
        while self._preroll_total > PREROLL_BYTES and self._preroll:
            oldest = self._preroll.popleft()
            self._preroll_total -= len(oldest)

    def _preroll_flush(self) -> bytes:
        """Return all buffered pre-roll bytes and clear the buffer."""
        data = b"".join(self._preroll)
        self._preroll.clear()
        self._preroll_total = 0
        return data

    # ------------------------------------------------------------------
    # WebsocketService abstract methods
    # ------------------------------------------------------------------

    async def _connect_websocket(self):
        logger.info(f"[WhisperWS] Connecting to {self._url} ...")
        self._websocket = await websockets.connect(
            self._url,
            ping_interval=None,
            max_size=10 * 1024 * 1024,
            open_timeout=10,
        )
        logger.info("[WhisperWS] Connected.")

    async def _disconnect_websocket(self):
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None
        logger.info("[WhisperWS] Disconnected.")

    async def _receive_messages(self):
        async for raw in self._websocket:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type")
            if msg_type == "ready":
                logger.info("[WhisperWS] Server ready.")
                continue

            if msg_type == "transcript":
                text = msg.get("text", "").strip()
                is_final = msg.get("is_final", False)
                if not text:
                    continue

                logger.debug(f"[WhisperWS] {'FINAL' if is_final else 'interim'}: {text}")
                lang = Language(self._language) if self._language else Language.EN

                if is_final:
                    frame = TranscriptionFrame(
                        text=text, user_id="", timestamp="", language=lang,
                    )
                else:
                    frame = InterimTranscriptionFrame(
                        text=text, user_id="", timestamp="", language=lang,
                    )
                await self.push_frame(frame)

    # ------------------------------------------------------------------
    # STTService abstract method — called by base for every AudioRawFrame
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Gate audio by VAD state.
        - Gate closed: keep audio in pre-roll buffer only (not sent to server)
        - Gate open:   send audio to Whisper server
        """
        if not self._user_speaking:
            # Gate closed — just update pre-roll, don't send to server
            self._preroll_add(audio)
            # logger.debug(f"[WhisperWS] 🔒 Gate closed — buffered {len(audio)} bytes (total preroll: {self._preroll_total})")
        elif self._websocket:
            # Gate open — send to server
            try:
                await self._websocket.send(audio)
                # logger.debug(f"[WhisperWS] 🔓 Gate open — sent {len(audio)} bytes to server")
            except Exception as exc:
                logger.warning(f"[WhisperWS] Audio send error: {exc}")
        return
        yield  # make this an async generator

    # ------------------------------------------------------------------
    # process_frame — tracks VAD state, flushes pre-roll on speech start
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            logger.info("[WhisperWS] ✅ VAD gate OPEN — user started speaking")

            # Flush pre-roll so the first syllable reaches Whisper
            preroll_data = self._preroll_flush()
            if preroll_data and self._websocket:
                try:
                    await self._websocket.send(preroll_data)
                    logger.info(f"[WhisperWS] 🔊 Flushed {len(preroll_data)} bytes of pre-roll")
                except Exception as exc:
                    logger.warning(f"[WhisperWS] Pre-roll flush error: {exc}")

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            logger.info("[WhisperWS] ❌ VAD gate CLOSED — user stopped speaking")

            # Tell server to finalise and flush the utterance
            if self._websocket:
                try:
                    await self._websocket.send(json.dumps({"type": "reset"}))
                    logger.info("[WhisperWS] 🔄 Sent reset (end of utterance)")
                except Exception as exc:
                    logger.warning(f"[WhisperWS] Reset send error: {exc}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()
        self._receive_task = self.create_task(
            self._receive_task_handler(self._report_error),
            name="whisper_ws_recv",
        )
