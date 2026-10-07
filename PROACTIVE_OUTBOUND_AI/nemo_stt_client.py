import asyncio
import json
from typing import AsyncGenerator, Optional

from loguru import logger
import websockets

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601


class NemotronWebSocketSTTService(WebsocketSTTService):
    """
    Pipecat 0.0.108 client for custom Nemotron 3.5
    cache-aware streaming WebSocket server.
    """

    def __init__(
        self,
        *,
        # url: str = "ws://127.0.0.1:8012/",
        url: str = "ws://stt_nemo:9092/",
        sample_rate: int = 16000,
        **kwargs,
    ):
        # Give Pipecat a realistic STT P99 safety value.
        #
        # Normal Nemotron reset -> final is ~25 ms according to
        # your server logs, so this should almost never be reached.
        #
        # 150 ms is deliberately conservative.
        kwargs.setdefault(
            "ttfs_p99_latency",
            0.15,
        )

        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        self._url = url

        self._websocket = None
        self._receive_task: Optional[asyncio.Task] = None
        self._ready = False

        self._audio_send_lock = asyncio.Lock()

        # ---------------------------------------------------------
        # USER STOP FRAME STATE
        # ---------------------------------------------------------

        # Smart Turn can potentially produce UserStoppedSpeakingFrame
        # before Nemotron's final transcript reaches Pipecat.
        #
        # In that case we temporarily hold the frame.
        self._pending_user_stopped_frame = None

        self._pending_frame_direction = (
            FrameDirection.DOWNSTREAM
        )

        self._pending_frame_timeout_task: Optional[
            asyncio.Task
        ] = None

        # Safety only.
        #
        # Normal path should release the stop frame as soon as the
        # finalized Nemotron transcript arrives.
        self._pending_frame_timeout_s = 0.50

        # ---------------------------------------------------------
        # DIAGNOSTICS
        # ---------------------------------------------------------

        self._vad_stop_time: Optional[float] = None
        self._reset_sent_time: Optional[float] = None

    # =============================================================
    # START
    # =============================================================

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    # =============================================================
    # STOP
    # =============================================================

    async def stop(self, frame: EndFrame):
        await self._cancel_pending_frame_timeout()

        if self._pending_user_stopped_frame is not None:
            await self.push_frame(
                self._pending_user_stopped_frame,
                self._pending_frame_direction,
            )

            self._pending_user_stopped_frame = None

        await self._send_reset()

        await super().stop(frame)

        await self._disconnect()

    # =============================================================
    # CANCEL
    # =============================================================

    async def cancel(self, frame: CancelFrame):
        await self._cancel_pending_frame_timeout()

        self._pending_user_stopped_frame = None

        try:
            await self._send_reset()
        except Exception:
            pass

        await super().cancel(frame)

        await self._disconnect()

    # =============================================================
    # CONNECT
    # =============================================================

    async def _connect(self):
        logger.info(
            f"[NEMOTRON] Connecting to {self._url}"
        )

        try:
            await self._connect_websocket()

            self._ready = True

            logger.info(
                "[NEMOTRON] WebSocket connected and ready"
            )

            self._receive_task = asyncio.create_task(
                self._receive_task_handler(
                    self._report_error
                )
            )

            await self._call_event_handler(
                "on_connected",
                self,
            )

        except Exception as e:
            self._ready = False

            logger.error(
                f"[NEMOTRON] Connection failed: {e}"
            )

            raise

    # =============================================================
    # DISCONNECT
    # =============================================================

    async def _disconnect(self):
        self._ready = False

        if self._receive_task:
            self._receive_task.cancel()

            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

            self._receive_task = None

        try:
            await self._disconnect_websocket()

        except Exception as e:
            logger.debug(
                f"[NEMOTRON] WebSocket close error: {e}"
            )

        try:
            await self._call_event_handler(
                "on_disconnected",
                self,
            )
        except Exception:
            pass

    # =============================================================
    # CONNECT WEBSOCKET
    # =============================================================

    async def _connect_websocket(self):
        self._websocket = await websockets.connect(
            self._url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

        try:
            ready_message = await asyncio.wait_for(
                self._websocket.recv(),
                timeout=5.0,
            )

            if isinstance(
                ready_message,
                bytes,
            ):
                raise RuntimeError(
                    "Nemotron server sent binary data "
                    "instead of ready JSON"
                )

            data = json.loads(
                ready_message
            )

            if data.get("type") != "ready":
                raise RuntimeError(
                    f"Unexpected Nemotron first message: "
                    f"{data}"
                )

            logger.info(
                "[NEMOTRON] Server ready: "
                f"sample_rate={data.get('sample_rate')}, "
                f"language={data.get('language')}, "
                f"context={data.get('att_context_size')}, "
                f"streaming_config={data.get('streaming_config')}"
            )

        except Exception:
            await self._websocket.close()
            self._websocket = None
            raise

    # =============================================================
    # DISCONNECT WEBSOCKET
    # =============================================================

    async def _disconnect_websocket(self):
        if self._websocket:
            try:
                await self._websocket.close()
            finally:
                self._websocket = None

    # =============================================================
    # STREAMING AUDIO -> NEMOTRON
    # =============================================================

    async def run_stt(
        self,
        audio: bytes,
    ) -> AsyncGenerator[Frame, None]:

        if not self._websocket or not self._ready:
            return

        try:
            async with self._audio_send_lock:

                await self._websocket.send(
                    audio
                )

        except Exception as e:
            logger.error(
                f"[NEMOTRON] Audio send failed: {e}"
            )

            await self._report_error(
                ErrorFrame(
                    f"Nemotron audio send failed: {e}"
                )
            )

        # Pipecat 0.0.108 expects this method to be an async
        # generator. Actual transcripts arrive through the
        # WebSocket receive task.
        if False:
            yield None

    # =============================================================
    # FRAME PROCESSING
    # =============================================================

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):

        # =========================================================
        # CRITICAL CHANGE
        #
        # VAD STOP -> RESET NEMOTRON IMMEDIATELY
        # =========================================================

        if isinstance(
            frame,
            VADUserStoppedSpeakingFrame,
        ):

            self._vad_stop_time = (
                asyncio.get_running_loop().time()
            )

            logger.info(
                "[NEMOTRON] VAD STOP -> "
                "immediately finalizing Nemotron"
            )

            # IMPORTANT:
            #
            # Do NOT wait for UserStoppedSpeakingFrame.
            #
            # VAD stop is the earliest reliable signal that the
            # user's speech has ended.
            await self._send_reset()

            # Let Pipecat continue processing the VAD stop.
            #
            # Smart Turn can now run while Nemotron is producing
            # the final transcript.
            await super().process_frame(
                frame,
                direction,
            )

            return

        # =========================================================
        # USER STOP
        #
        # Smart Turn may reach this before Nemotron's final
        # transcript. Hold it only in that race condition.
        # =========================================================

        if isinstance(
            frame,
            UserStoppedSpeakingFrame,
        ):

            if self._pending_user_stopped_frame is not None:
                logger.debug(
                    "[NEMOTRON] Replacing stale pending "
                    "UserStoppedSpeakingFrame"
                )

            # If the final transcript has already arrived,
            # there is no reason to hold this frame.
            #
            # We detect that situation by checking whether the
            # pending reset has already been cleared.
            if self._reset_sent_time is not None:

                logger.debug(
                    "[NEMOTRON] User stopped speaking -> "
                    "holding until final transcript"
                )

                self._pending_user_stopped_frame = frame

                self._pending_frame_direction = (
                    direction
                )

                self._start_pending_frame_timeout()

                return

            # No pending Nemotron finalization.
            await super().process_frame(
                frame,
                direction,
            )

            return

        # =========================================================
        # EVERYTHING ELSE
        # =========================================================

        await super().process_frame(
            frame,
            direction,
        )

    # =============================================================
    # SEND RESET
    # =============================================================

    async def _send_reset(self):

        if not self._websocket or not self._ready:
            return

        try:

            async with self._audio_send_lock:

                self._reset_sent_time = (
                    asyncio.get_running_loop().time()
                )

                await self._websocket.send(
                    json.dumps(
                        {
                            "type": "reset"
                        }
                    )
                )

            logger.debug(
                "[NEMOTRON] reset sent"
            )

        except Exception as e:

            logger.error(
                f"[NEMOTRON] Reset failed: {e}"
            )

    # =============================================================
    # WEBSOCKET RECEIVE LOOP
    # =============================================================

    async def _receive_messages(self):

        if not self._websocket:
            return

        async for message in self._websocket:

            try:

                if isinstance(
                    message,
                    bytes,
                ):
                    continue

                data = json.loads(
                    message
                )

                message_type = data.get(
                    "type"
                )

                # -------------------------------------------------
                # READY
                # -------------------------------------------------

                if message_type == "ready":

                    self._ready = True

                # -------------------------------------------------
                # TRANSCRIPT
                # -------------------------------------------------

                elif message_type == "transcript":

                    await self._handle_transcript(
                        data
                    )

                # -------------------------------------------------
                # RESET COMPLETE
                # -------------------------------------------------

                elif message_type == "reset_complete":

                    logger.debug(
                        "[NEMOTRON] reset_complete"
                    )

                # -------------------------------------------------
                # ERROR
                # -------------------------------------------------

                elif message_type == "error":

                    error_message = data.get(
                        "message",
                        "Unknown Nemotron server error",
                    )

                    logger.error(
                        "[NEMOTRON] Server error: "
                        f"{error_message}"
                    )

                    await self._report_error(
                        ErrorFrame(
                            f"Nemotron server error: "
                            f"{error_message}"
                        )
                    )

                # -------------------------------------------------
                # PONG
                # -------------------------------------------------

                elif message_type == "pong":

                    pass

                else:

                    logger.debug(
                        "[NEMOTRON] Unknown message: "
                        f"{data}"
                    )

            except json.JSONDecodeError as e:

                logger.error(
                    f"[NEMOTRON] Invalid JSON: {e}"
                )

            except Exception as e:

                logger.error(
                    f"[NEMOTRON] Message handling error: "
                    f"{e}"
                )

    # =============================================================
    # TRANSCRIPT HANDLER
    # =============================================================

    async def _handle_transcript(
        self,
        data: dict,
    ):

        text = (
            data.get("text") or ""
        ).strip()

        is_final = bool(
            data.get(
                "is_final",
                False,
            )
        )

        if not text:
            return

        timestamp = time_now_iso8601()

        # =========================================================
        # FINAL
        # =========================================================

        if is_final:

            server_processing_ms = data.get(
                "processing_ms"
            )
            
            # Get total STT time from server (if available)
            total_stt_ms = data.get("total_stt_ms")

            reset_to_final_ms = None

            if self._reset_sent_time is not None:

                reset_to_final_ms = (
                    asyncio.get_running_loop().time()
                    - self._reset_sent_time
                ) * 1000.0

            vad_to_final_ms = None

            if self._vad_stop_time is not None:

                vad_to_final_ms = (
                    asyncio.get_running_loop().time()
                    - self._vad_stop_time
                ) * 1000.0

            # Build log message
            log_parts = [f"[NEMOTRON FINAL] {text}"]
            log_parts.append(f"server={server_processing_ms} ms")
            
            if reset_to_final_ms is not None:
                log_parts.append(f"reset_to_final={reset_to_final_ms:.1f} ms")
            
            if total_stt_ms is not None:
                log_parts.append(f"⏱️  TOTAL_STT={total_stt_ms:.1f} ms")
            
            logger.info(" | ".join(log_parts))

            if vad_to_final_ms is not None:

                logger.info(
                    "[NEMOTRON] "
                    f"VAD_STOP -> FINAL = "
                    f"{vad_to_final_ms:.1f} ms"
                )

            # =====================================================
            # CRITICAL:
            #
            # finalized=True
            #
            # This allows TurnAnalyzerUserTurnStopStrategy to take
            # the fast path instead of waiting for its STT timeout.
            # =====================================================

            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=timestamp,
                    language=None,
                    result=data,
                    finalized=True,
                )
            )

            # =====================================================
            # FINAL TRANSCRIPT HAS NOW ARRIVED.
            #
            # Release any UserStoppedSpeakingFrame that was waiting.
            # =====================================================

            await self._release_pending_user_stopped()

            # Clear turn timing state.
            self._reset_sent_time = None
            self._vad_stop_time = None

        # =========================================================
        # INTERIM
        # =========================================================

        else:

            logger.debug(
                "[NEMOTRON INTERIM] "
                f"{text}"
            )

            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    timestamp,
                    language=None,
                    result=data,
                )
            )

    # =============================================================
    # PENDING USER STOP TIMEOUT
    # =============================================================

    def _start_pending_frame_timeout(self):

        if self._pending_frame_timeout_task:

            self._pending_frame_timeout_task.cancel()

        self._pending_frame_timeout_task = (
            asyncio.create_task(
                self._pending_frame_timeout_handler()
            )
        )

    async def _pending_frame_timeout_handler(
        self,
    ):

        try:

            await asyncio.sleep(
                self._pending_frame_timeout_s
            )

            if (
                self._pending_user_stopped_frame
                is not None
            ):

                logger.warning(
                    "[NEMOTRON] Final transcript timeout "
                    f"after "
                    f"{self._pending_frame_timeout_s:.2f}s; "
                    "releasing stop frame"
                )

                await self.push_frame(
                    self._pending_user_stopped_frame,
                    self._pending_frame_direction,
                )

                self._pending_user_stopped_frame = None

        except asyncio.CancelledError:

            pass

        finally:

            self._pending_frame_timeout_task = None

    # =============================================================
    # CANCEL PENDING TIMEOUT
    # =============================================================

    async def _cancel_pending_frame_timeout(self):

        if self._pending_frame_timeout_task:

            self._pending_frame_timeout_task.cancel()

            try:

                await self._pending_frame_timeout_task

            except asyncio.CancelledError:

                pass

            self._pending_frame_timeout_task = None

    # =============================================================
    # RELEASE PENDING USER STOP
    # =============================================================

    async def _release_pending_user_stopped(self):

        if (
            self._pending_user_stopped_frame
            is None
        ):
            return

        await self._cancel_pending_frame_timeout()

        frame = (
            self._pending_user_stopped_frame
        )

        direction = (
            self._pending_frame_direction
        )

        self._pending_user_stopped_frame = None

        logger.debug(
            "[NEMOTRON] Final transcript received -> "
            "releasing UserStoppedSpeakingFrame"
        )

        await self.push_frame(
            frame,
            direction,
        )


NemotronSTTService = NemotronWebSocketSTTService