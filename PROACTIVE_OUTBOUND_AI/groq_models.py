"""
Picks a Groq chat model that your API key can actually use, then builds the Pipecat LLM service.
Avoids "404 model_not_found" when Groq renames/retires models or your org lacks access.

    from groq_model import make_groq_llm
    llm = make_groq_llm()                       # picks automatically
    llm = make_groq_llm(["llama-3.3-70b-versatile"])   # your own preference order

Override from the environment:  GROQ_MODEL=<exact id>     (must be in the list printed at startup)
Needs env var GROQ_API_KEY.
"""
import json
import os
import urllib.error
import urllib.request
from dotenv import load_dotenv
load_dotenv(override=True)

GROQ_BASE = "https://api.groq.com/openai/v1"

# preference order: fast and cheap first
PREFERRED = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
NOT_CHAT = ("whisper", "tts", "orpheus", "playai", "guard", "embed", "compound", "distil-whisper")


def list_groq_models(api_key: str) -> list:
    req = urllib.request.Request(
        f"{GROQ_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "pipecat-agent/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY.") from e
        raise RuntimeError(f"Groq model list failed: HTTP {e.code}") from e
    return sorted(m["id"] for m in data.get("data", []))


def pick_groq_model(preferred=None) -> str:
    key = os.getenv("GROQ_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    available = list_groq_models(key)
    chat = [m for m in available if not any(x in m.lower() for x in NOT_CHAT)]
    print("Groq models available to this key:", ", ".join(available))

    forced = os.environ.get("GROQ_MODEL")
    if forced:
        if forced not in available:
            raise RuntimeError(f"GROQ_MODEL={forced} is not available. Choose one of: {chat}")
        return forced
    for name in (preferred or []) + PREFERRED:
        if name in available:
            print("Using Groq model:", name)
            return name
    if chat:
        print("Using Groq model:", chat[0])
        return chat[0]
    raise RuntimeError("No chat models are available for this Groq key.")


def make_groq_llm(preferred=None, **kwargs):
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key=os.getenv("GROQ_KEY"),
        base_url=GROQ_BASE,
        model=pick_groq_model(preferred),
        **kwargs,
    )


if __name__ == "__main__":
    pick_groq_model()