"""
Test the Qwen TTS WebSocket server without Pipecat.

    set TTS_URL=ws://GPU_SERVER:8000      (PowerShell: $env:TTS_URL="ws://...")
    set TTS_API_KEY=secret
    python test_ws_client.py

Prints time-to-first-audio and real-time factor for each sentence and saves ws_test_*.wav
"""
import asyncio
import json
import os
import time
import uuid
import wave
from urllib.parse import quote

import aiohttp

URL = os.getenv("TTS_URL", "ws://127.0.0.1:8888").rstrip("/")
KEY = os.getenv("TTS_API_KEY", "")
SENTENCES = [
    ("neutral", "Hello, how are you today?"),
    ("happy", "Great news, your loan has been pre approved."),
    ("empathetic", "I'm sorry to hear that. Let me see how I can help."),
    ("neutral", "Our personal loans start with flexible tenures and simple documentation."),
]


async def main():
    url = URL.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not url.endswith("/ws"):
        url += "/ws"
    if KEY:
        url += f"?key={quote(KEY)}"

    async with aiohttp.ClientSession() as session:
        t = time.time()
        async with session.ws_connect(url, heartbeat=15, max_msg_size=0) as ws:
            hello = await ws.receive()
            print(f"connected in {time.time() - t:.2f}s: {hello.data}\n")

            for i, (emotion, text) in enumerate(SENTENCES, 1):
                rid = uuid.uuid4().hex[:8]
                prefix = rid.encode()
                t0, first, pcm, chunks = time.time(), None, bytearray(), 0
                await ws.send_json({"type": "speak", "id": rid, "text": text, "emotion": emotion})
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY and msg.data[:8] == prefix:
                        if first is None:
                            first = time.time() - t0
                        pcm += msg.data[8:]
                        chunks += 1
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        info = json.loads(msg.data)
                        if info.get("id") == rid and info["type"] in ("end", "error"):
                            if info["type"] == "error":
                                print("ERROR:", info.get("message"))
                            break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        print("connection closed")
                        return
                took, dur = time.time() - t0, len(pcm) / 2 / 24000
                if dur == 0:
                    print(f"[{i}] no audio received")
                    continue
                print(f"[{i}] {emotion:10s} first audio {first:.2f}s | {dur:.1f}s audio in {took:.1f}s "
                      f"(RTF {took / dur:.2f}) | {chunks} chunks")
                with wave.open(f"ws_test_{i}.wav", "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(bytes(pcm))
    print("\nSaved ws_test_1.wav ... Listen to them. RTF below 1 = faster than real time.")


if __name__ == "__main__":
    asyncio.run(main())