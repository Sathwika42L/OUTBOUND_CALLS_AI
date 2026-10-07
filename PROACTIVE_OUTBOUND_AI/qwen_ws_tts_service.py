"""
Pipecat TTS service for the Qwen3-TTS WebSocket server (qwen_ws_server.py).
Needs only:  pip install pipecat-ai aiohttp      (no torch, no GPU on the agent machine)

    from qwen_ws_tts_service import QwenWSTTSService
    tts = QwenWSTTSService(
        url="ws://GPU_SERVER:8000",          # or wss://...
        api_key="secret",
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )
    tts.set_emotion("happy")                 # neutral, happy, excited, sad, calm, empathetic, angry, or free text
"""
import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional
from urllib.parse import quote

import aiohttp
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


class QwenWSTTSService(TTSService):
    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        emotion: str = "support",
        timeout_s: float = 20.0,
        **kwargs,
    ):
        if "settings" not in kwargs:
            try:  # Pipecat 0.0.108 wants every settings field initialised (None = unsupported)
                from pipecat.services.settings import TTSSettings

                kwargs["settings"] = TTSSettings(model=None, voice=None, language=None)
            except Exception:  # noqa: BLE001  (older/newer Pipecat: just skip)
                pass
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
        u = url.strip().rstrip("/")
        u = u.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        if not u.endswith("/ws"):
            u += "/ws"
        self._url = u + (f"?key={quote(api_key)}" if api_key else "")
        self._emotion = emotion
        self._timeout = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    def set_emotion(self, emotion: str):
        self._emotion = emotion

    # ------------------------------------------------------------ connection
    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        if self._ws is not None and not self._ws.closed:
            return self._ws
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                ws = await asyncio.wait_for(
                    self._session.ws_connect(self._url, heartbeat=15, max_msg_size=0), self._timeout
                )
                hello = await asyncio.wait_for(ws.receive(), self._timeout)  # {"type":"ready",...}
                if hello.type != aiohttp.WSMsgType.TEXT:
                    await ws.close()
                    raise ConnectionError("TTS server refused the connection (wrong api_key?)")
                self._ws = ws
                logger.info("Connected to Qwen TTS server")
                return ws
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(0.5 * (attempt + 1))
        raise ConnectionError(f"Cannot reach the TTS server: {last}")

    async def _reset(self):
        ws, self._ws = self._ws, None
        if ws is not None and not ws.closed:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    async def _close(self):
        await self._reset()
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def start(self, frame):
        await super().start(frame)
        try:
            await self._connect()  # connect early so the first sentence has no setup delay
        except Exception as e:  # noqa: BLE001
            logger.warning(f"TTS server not reachable yet, will retry on first sentence: {e}")

    async def stop(self, frame):
        await super().stop(frame)
        await self._close()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._close()

    # -------------------------------------------------------------- synthesis
    async def run_tts(self, text: str, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        text = text.strip()
        if not text:
            return
        async with self._lock:
            req_id = uuid.uuid4().hex[:8]
            prefix = req_id.encode("ascii")
            t0 = time.time()
            ended = False
            try:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()

                # send the request (one reconnect attempt if the connection was dropped)
                for attempt in range(2):
                    try:
                        ws = await self._connect()
                        await ws.send_json({"type": "speak", "id": req_id, "text": text, "emotion": self._emotion})
                        break
                    except Exception:  # noqa: BLE001
                        await self._reset()
                        if attempt == 1:
                            raise

                first = True
                while True:
                    msg = await asyncio.wait_for(ws.receive(), self._timeout)
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        data = msg.data
                        if data[:8] != prefix:  # leftover audio from an earlier, cancelled sentence
                            continue
                        pcm = data[8:]
                        if not pcm:
                            continue
                        if first:
                            await self.stop_ttfb_metrics()
                            logger.info(f"TTS first audio after {time.time() - t0:.2f}s")
                            first = False
                        yield TTSAudioRawFrame(pcm, SAMPLE_RATE, 1)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        info = json.loads(msg.data)
                        if info.get("id") != req_id:
                            continue
                        if info.get("type") == "end":
                            ended = True
                            logger.info(
                                f"TTS done: {info.get('audio_s', 0):.1f}s audio, server {info.get('took_s', 0):.1f}s, "
                                f"total {time.time() - t0:.1f}s"
                            )
                            break
                        if info.get("type") == "error":
                            ended = True
                            yield ErrorFrame(f"TTS server error: {info.get('message')}")
                            break
                    else:  # CLOSE / CLOSING / CLOSED / ERROR
                        raise ConnectionError("TTS connection closed")
            except asyncio.CancelledError:
                # user interrupted: tell the server to stop generating this sentence
                try:
                    if self._ws is not None and not self._ws.closed:
                        await self._ws.send_json({"type": "cancel", "id": req_id})
                except Exception:  # noqa: BLE001
                    pass
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"TTS failed: {e}")
                await self._reset()  # next sentence reconnects
                yield ErrorFrame(f"TTS failed: {e}")
            finally:
                if not ended and self._ws is not None and not self._ws.closed:
                    try:
                        await self._ws.send_json({"type": "cancel", "id": req_id})
                    except Exception:  # noqa: BLE001
                        pass
                yield TTSStoppedFrame()

    async def cleanup(self):
        await self._close()
        await super().cleanup()