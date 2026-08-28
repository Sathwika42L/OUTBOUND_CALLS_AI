#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Outbound Calling Bot - Pipecat Voice Agent

This bot makes PROACTIVE outbound calls where the bot speaks FIRST.
Based on bot.py but with proactive greeting and conversation flow.

Key Differences from Inbound (bot.py):
- Bot speaks FIRST: "Hello! This is Indian Consulate Services calling..."
- Proactive conversation flow (bot drives the conversation)
- Uses outbound_flow.py with respond_immediately=True

Run the bot using::

    python outbound_bot.py
"""

import os,requests
from loguru import logger
import sys
import logging

# ═══════════════════════════════════════════════════════════════════════════
# 🔇 SUPPRESS DEBUG LOGS - Remove pipecat verbose logging
# ═══════════════════════════════════════════════════════════════════════════
logging.getLogger("pipecat.services.openai.base_llm").setLevel(logging.INFO)
logging.getLogger("pipecat").setLevel(logging.INFO)

from pipecat.audio.vad.vad_analyzer import VADParams
from dotenv import load_dotenv
from pipecat.pipeline.runner import PipelineRunner
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.runner.types import RunnerArguments
from pipecat.frames.frames import LLMRunFrame, TextFrame, LLMTextFrame, Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams, UserTurnStoppedMessage, AssistantTurnStoppedMessage
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.base_transport import TransportParams
from pipecat_flows import FlowManager
from pipecat.utils.tracing.setup import setup_tracing
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# 🚨 IMPORTANT: Import from outbound_flow.py (not flow.py)
from oubound_pipecatflow import create_initial_node,initialize_user
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.services.tts_service import TextAggregationMode
from pipecat.services.piper.tts import PiperTTSService
from pipecat.frames.frames import (
    LLMRunFrame,
    TextFrame,
    LLMTextFrame,
    Frame,
    TranscriptionFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    EndFrame,
)
import asyncio,re
from pipecat.frames.frames import (
    AudioRawFrame, TextFrame, TTSStartedFrame, TTSStoppedFrame, Frame, LLMTextFrame, 
    MixerEnableFrame, OutputAudioRawFrame
)

def normalize_time_for_speech(text: str) -> str:
    """
    🕐 DEDICATED TIME CONVERTER - Converts ALL time formats to words for TTS
    NOW HANDLES ANY MINUTE VALUE (00-59)!
    Examples:
      8:23 AM → eight twenty three AM
      8:30 AM → eight thirty AM
      12:47 PM → twelve forty seven PM
      5:05 AM → five zero five AM
    """
    
    def number_to_words(num):
        """Convert any number 0-59 to words"""
        ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 
                 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty']
        
        if num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        else:
            ten_digit = num // 10
            one_digit = num % 10
            if one_digit == 0:
                return tens[ten_digit]
            else:
                return f"{tens[ten_digit]} {ones[one_digit]}"
    
    def time_with_period(match):
        """Handle times WITH AM/PM"""
        time_str = match.group(1)
        period = match.group(2)
        
        # Remove separators (: - .) to normalize
        time_str = time_str.replace(':', '').replace('-', '').replace('.', '')
        
        if len(time_str) <= 2:  # Just hour: "8 AM" or "9 PM"
            hour = int(time_str)
            return f"{number_to_words(hour)} {period}"
            
        elif len(time_str) == 3:  # 823 → 8:23
            hour = int(time_str[0])
            minute = int(time_str[1:])
        else:  # 1247 → 12:47
            hour = int(time_str[:-2])
            minute = int(time_str[-2:])
        
        hour_word = number_to_words(hour)
        if minute == 0:
            return f"{hour_word} {period}"
        else:
            minute_word = number_to_words(minute)
            return f"{hour_word} {minute_word} {period}"
    
    def time_without_period(match):
        """Handle times WITHOUT AM/PM - standalone times like 8:23"""
        time_str = match.group(1)
        
        # Remove separators (: - .) to normalize
        time_str = time_str.replace(':', '').replace('-', '').replace('.', '')
        
        if len(time_str) <= 2:  # Just hour: "8" or "9"
            hour = int(time_str)
            return number_to_words(hour)
            
        elif len(time_str) == 3:  # 823 → 8:23
            hour = int(time_str[0])
            minute = int(time_str[1:])
        else:  # 1247 → 12:47
            hour = int(time_str[:-2])
            minute = int(time_str[-2:])
        
        hour_word = number_to_words(hour)
        if minute == 0:
            return hour_word
        else:
            minute_word = number_to_words(minute)
            return f"{hour_word} {minute_word}"
    
    # 🎯 PATTERN 1: Times WITH AM/PM (e.g., 8:23 AM, 8-30PM, 823AM)
    text = re.sub(
        r'\b(\d{1,2}[\:\-\.]?\d{0,2})\s*(AM|PM|am|pm)\b',
        time_with_period,
        text,
        flags=re.IGNORECASE
    )
    
    # 🎯 PATTERN 2: Standalone times WITHOUT AM/PM (e.g., 8:23, 8-30, 12.47)
    # CRITICAL: This prevents TTS from reading "8:23" as "eight colon twenty three"
    text = re.sub(
        r'\b(\d{1,2}[\:\-\.]\d{2})\b',
        time_without_period,
        text,
        flags=re.IGNORECASE
    )
    
    return text


def sanitize_for_tts(text: str) -> str:
    """
    AGGRESSIVELY remove all tool-call artifacts and NON-ENGLISH text from LLM output.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # STEP -1: STRIP PREFIX JUNK AND CODE LEAKS
    # ══════════════════════════════════════════════════════════════════════════
    
    # 1. Answer Recovery: Strip everything before 'assistant' if leaked
    if "assistant" in text.lower():
        # Look for the last occurrence of common role markers
        recovery_markers = [r'assistant\s*:', r'assistant\s+', r'assistant']
        for marker in recovery_markers:
            matches = list(re.finditer(marker, text, flags=re.IGNORECASE))
            if matches:
                text = text[matches[-1].end():].lstrip(':').strip()
                break

    # 2. Block Code Fragments: Detect patterns like iNdEx++, strconv, jsonQuery, etc.
    code_patterns = [
        r'\bstrconv\b', r'\bItoa\b', r'\bjsonQuery\b', r'\bstringWithLead\b',
        r'\bpItemNdEx\b', r'\biNdEx\+\+\b', r'\bfunc\b', r'\bvar\b', r'\breturn\s+\w+',
        # NEW: Block JavaScript/Python reasoning patterns
        r'\bif\s*\(', r'\)\s*\{', r'\}\s*if\b', r'\breturn\s*\{', r'\};\s*',
        r'_icall_', r'tool_filterwhere', r'tools_json_list', r'CallCheckEndConvo',
        r'shouldTransferToAgent', r'tool_result', r'tool_response', r'isinstance',
        r'json\.loads', r'pprint\.pprint', r'for\s+tool', r'customerSaysGoodbye',
        r'arguments\s*=\s*', r'tool_response\s*=\s*', r'name\s*=\s*',
    ]
    for pattern in code_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            logger.warning(f"🚫 CODE LEAK DETECTED ({pattern}) - blocking chunk")
            return ""

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 0: CRITICAL - DETECT AND BLOCK NON-ENGLISH IMMEDIATELY
    # ══════════════════════════════════════════════════════════════════════════
    
    # Count Latin vs non-Latin characters
    latin_count = 0
    non_latin_count = 0
    
    for char in text:
        if char.isalpha():
            if ord(char) < 256:  # ASCII + Latin-1
                latin_count += 1
            else:
                non_latin_count += 1
    
    total_alpha = latin_count + non_latin_count
    
    # If we have text and it's more than 10% non-English, BLOCK IT
    if total_alpha > 5:
        non_english_ratio = non_latin_count / total_alpha
        if non_english_ratio > 0.10:  # More than 10% non-English
            logger.error(f"🚫 NON-ENGLISH TEXT BLOCKED ({non_english_ratio:.0%} non-Latin)")
            logger.error(f"🚫 Blocked: {text[:150]}")
            return ""
    
    # Also check for specific Unicode ranges (Thai, Hindi, Chinese, Arabic, etc.)
    for char in text:
        code = ord(char)
        # Thai: 0x0E00-0x0E7F
        # Hindi/Devanagari: 0x0900-0x097F  
        # Chinese: 0x4E00-0x9FFF
        # Arabic: 0x0600-0x06FF
        if (0x0E00 <= code <= 0x0E7F or    # Thai
            0x0900 <= code <= 0x097F or    # Hindi
            0x4E00 <= code <= 0x9FFF or    # Chinese
            0x0600 <= code <= 0x06FF):     # Arabic
            logger.error(f"🚫 NON-LATIN SCRIPT DETECTED AND BLOCKED (Unicode: {hex(code)})")
            return ""

    # Rest of sanitization code from bot.py...
    # (Keeping all the same tool-call artifact removal logic)
    
    if 'searchics' in text.lower() or 'search_ics' in text.lower() or 'callcheck' in text.lower():
        if any(marker in text for marker in ['"name"', '"arguments"', '"query"', '":"', '{', '}']):
            return ""
        logger.warning(f"🚫 Tool name in output - blocking: {text[:80]}")
        return ""
    
    text = ''.join(char for char in text if ord(char) < 256 or char.isspace())
    text = re.sub(r'^.*?</tool_call>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'</tool_call>', '', text)
    text = re.sub(r'\{[^}]*"name"\s*:\s*"search_ics"[^}]*\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{[^}]*"name"\s*:\s*"[^"]*"[^}]*\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'HeaderCode:\d+', '', text)
    text = re.sub(r'ToolCall(?:Check)?', '', text)
    text = re.sub(r'(?:Tool)?Call(?:Check)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\w*_CALL\b', '', text)
    text = re.sub(r'\b\w*_TOOL\b', '', text)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'\[TOOL_[^\]]*\]', '', text)
    
    tool_call_pattern = re.compile(
        r'^.*?"name"\s*[":]\s*"[^"]+"\s*,?\s*"arguments"\s*[":]\s*\{[^}]*\}\s*',
        re.IGNORECASE | re.DOTALL
    )
    m = tool_call_pattern.match(text)
    if m:
        text = text[m.end():]
    
    text = re.sub(r'[^\n]*"name"\s*:[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\n]*"arguments"[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\n]*"query"[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFilterWhere\s*:.*?\}', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\bCanBeConvertedToEnglish\s*:.*?spNet', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    technical_markers = [
        r'.*search[_\s]?ics.*', r'.*searchics.*',
        r'.*FilterWhere.*', r'.*CanBeConvertedToEnglish.*',
        r'.*sPid.*', r'.*spNet.*'
    ]
    for pattern in technical_markers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'""\s*:\s*"searchics"', '', text, flags=re.IGNORECASE)
    text = re.sub(r'""\s*:\s*"search_ics"', '', text, flags=re.IGNORECASE)
    text = re.sub(r'""\s*:\s*"query"', '', text, flags=re.IGNORECASE)
    text = re.sub(r'""\s*:\s*["\'][^"\']*["\']', '', text)
    text = re.sub(r'":\s*"[^"]*"', '', text)
    text = re.sub(r'\b[a-z]\s*"\s*:\s*"', '', text)
    text = re.sub(r'""\s*:\s*"?\w+"?\s*:', '', text)
    text = re.sub(r'^\s*[:\-,;]+\s*', '', text)
    text = re.sub(r'\n\s*[:\-,;]+\s*', '\n', text)
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'"name"\s*:\s*"[^"]*"', '', text, flags=re.IGNORECASE)
    text = re.sub(r'"arguments"\s*:.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'"query"\s*:\s*"[^"]*"', '', text, flags=re.IGNORECASE)
    text = re.sub(r'"\w*"\s*:\s*"[^"]*",?\s*', '', text)
    text = re.sub(r'""\s*:\s*"[^"]*",?\s*', '', text)
    text = re.sub(r'\bFilterWhere\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCanBeConvertedToEnglish\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsPid\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bspNet\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b_icall_\b\s*', '', text)
    text = text.replace("*", "").replace("#", "").replace("`", "")
    text = re.sub(r'[{}\[\]]', ' ', text)
    text = ''.join(char for char in text if ord(char) < 128 or char in ' \n\t')
    text = normalize_time_for_speech(text)
    text = re.sub(r'\b(AM|am)\b', 'A M', text)
    text = re.sub(r'\b(PM|pm)\b', 'P M', text)
    
    if text:
        final_non_latin = sum(1 for c in text if c.isalpha() and ord(c) >= 128)
        if final_non_latin > 0:
            logger.error(f"🚫 FINAL CHECK: Non-ASCII characters still present. BLOCKED.")
            return ""
    
    return text


load_dotenv(override=True)
exporter = OTLPSpanExporter()

def patch_trace_input_output():
    from pipecat.utils.tracing import service_decorators
    original = service_decorators.add_llm_span_attributes
    first_call = [True]

    def patched(span, *args, **kwargs):
        original(span, *args, **kwargs)

        if first_call[0] and kwargs.get("messages"):
            span.set_attribute("langfuse.trace.input", kwargs["messages"])
            first_call[0] = False

        orig_set = span.set_attribute
        def new_set(key, value):
            orig_set(key, value)
            if key == "output":
                orig_set("langfuse.trace.output", value)
        span.set_attribute = new_set

    service_decorators.add_llm_span_attributes = patched

patch_trace_input_output()

setup_tracing(
    service_name="pipecat-outbound-demo",
    exporter=None,
    console_export=bool(os.getenv("OTEL_CONSOLE_EXPORT")),
)

latency_observer = UserBotLatencyObserver()

class LiveTranscriptionSpy(FrameProcessor):
    def __init__(self, label: str, rtvi_task=None):
        super().__init__()
        self._label = label
        self._rtvi_task = rtvi_task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        text = None
        is_final = False
        msg_type = None

        if self._label == "USER" and isinstance(frame, TextFrame):
            text = frame.text
            print("text:",text)
            msg_type = "user-transcript"
        
        elif self._label == "BOT" and isinstance(frame, LLMTextFrame):
            text = frame.text
            msg_type = "bot-transcript"

        if text:
            logger.info(f"[{self._label} LIVE] {text}")
            
            if self._rtvi_task and hasattr(self._rtvi_task, "rtvi"):
                try:
                    if hasattr(self._rtvi_task.rtvi, "send_message"):
                        await self._rtvi_task.rtvi.send_message({
                            "type": msg_type,
                            "data": {"text": text, "is_final": False}
                        })
                except Exception:
                    pass
        
        await self.push_frame(frame, direction)

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")

        await self.push_frame(frame, direction)

class TTSCleaner(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            cleaned = sanitize_for_tts(frame.text)
            if not cleaned:
                return
            frame.text = cleaned
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

class TTSTextFilter(FrameProcessor):
    """
    AGGRESSIVE text filter to block technical metadata and tool call fragments 
    from reaching the TTS engine.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMTextFrame):
            await self.push_frame(frame, direction)
            return

        text = frame.text
        lower_text = text.lower()

        if lower_text in {"able", "call", "choose", "confirm", "colon", "color"}:
            return

        if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)+$', text):
            return

        if any('\u0E00' <= char <= '\u0E7F' for char in text):
            logger.warning(f"🚫 BLOCKING Thai text: {text[:50]}")
            return
        
        if any('\u0900' <= char <= '\u097F' for char in text):
            logger.warning(f"🚫 BLOCKING Hindi text: {text[:50]}")
            return
        if any('\u4E00' <= char <= '\u9FFF' for char in text):
            logger.warning(f"🚫 BLOCKING Chinese text: {text[:50]}")
            return
        if any('\u0600' <= char <= '\u06FF' for char in text):
            logger.warning(f"🚫 BLOCKING Arabic text: {text[:50]}")
            return

        technical_keywords = [
            "index", "argument", "arguments", "callcheck", "criteria", "applyresources", 
            "headercode", "imagerelation", "searchics", "search_ics", "search ics",
            "tool_call", "toolcall", "function", "name", "query", "storyboardsegue",
            "filterwhere", "json", "python", "script", "canbeconvertedtoenglish",
            "spid", "spnet", "forcell", "for cell",
            "callcheckendconvo", "shouldtransfertoagent", "_icall_", "tools_json_list",
            "tool_filterwhere", "return {", "if (", "} if", "for tool", "tool_result",
            "tool_response", "arguments", "tool_call", "isinstance", "json.loads"
        ]
        if any(kw in lower_text for kw in technical_keywords):
            logger.warning(f"🚫 BLOCKING technical keyword: {text[:50]}")
            return

        block_immediately = [
            '{', '}', '[', ']',
            '::invoke', ':invoke', 'invoke_',
            '"name"', '"arguments"', '"query"',
            'search_ics', 'searchics', 'tool_call',
            '</tool_call>', '<tool_call>',
            '":"', '"',
            'forcell', 'for cell',
            'if (', ') {', '} if', 'return {', '};',
            '_icall_', 'tool_filterwhere', 'tools_json_list',
            'callcheckendconvo', 'shouldtransfertoagent',
            'tool_result', 'tool_response', 'isinstance',
            'json.loads', 'for tool', 'pprint.pprint',
            '= {', '};', '==', '!=', '<=', '>=',
        ]
        
        for pattern in block_immediately:
            if pattern in text.lower():
                logger.warning(f"🚫 BLOCKING pattern '{pattern}': {text[:50]}")
                return
        
        if text.strip().startswith((':', '::')):
            return
        
        if (
            "choose_" in text
            or text.startswith("choose")
            or text.startswith("confirm")
            or text.startswith("cancel")
            or "_" in text
            or "tool_call" in text
            or "function" in text
            or "</tool_call>" in text
            or "name" in text
            or "StoryboardSegue" in text
            or "{" in text
            or "}" in text
            or "(" in text
            or ")" in text
            or "[" in text
            or "]" in text
            or ":" in text
            or "\"" in text
        ):
            logger.warning(f"🚫 BLOCKING structured text fragment: {text}")
            return

        if not text:
            return

        if len(text) > 300:
            logger.warning(f"🚫 BLOCKING overly long frame (>300 chars)")
            return

        text = re.sub(r'([,.!?])(?=\S)', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text)

        frame.text = text
        await self.push_frame(frame, direction)


class MuteSTTDuringTTS(FrameProcessor):

    def __init__(self):
        super().__init__()
        self.tts_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):

        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self.tts_active = True

        elif isinstance(frame, TTSStoppedFrame):
            self.tts_active = False

        if self.tts_active and isinstance(frame, AudioRawFrame):
            return

        await self.push_frame(frame, direction)


class GoodbyeDetector(FrameProcessor):
    """
    🔚 GOODBYE DETECTOR - Ends call when user OR bot says bye/goodbye
    """
    
    def __init__(self):
        super().__init__()
        self._last_bot_response = ""
        self._last_user_input = ""
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMTextFrame) and frame.text:
            self._last_bot_response += frame.text.lower()
        
        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._last_user_input = frame.text.lower()
            logger.info(f"[GOODBYE DETECTOR] User said: {frame.text}")
        
        if isinstance(frame, UserStoppedSpeakingFrame):
            goodbye_phrases = [
                'goodbye', 'bye', 'bye bye', 'see you', 'take care',
                'thank you bye', 'thanks bye', "that's all", 
                'nothing else', 'end call', 'disconnect'
            ]
            
            user_said_goodbye = any(phrase in self._last_user_input for phrase in goodbye_phrases)
            
            if user_said_goodbye:
                logger.info(f"🔚 User said goodbye: '{self._last_user_input}' - ending call")
                await asyncio.sleep(1.0)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                logger.info("🔚 EndFrame sent - call will disconnect")
            
            self._last_user_input = ""
        
        if isinstance(frame, TTSStoppedFrame):
            goodbye_phrases = [
                'goodbye', 'bye', 'bye bye', 'see you', 'take care',
                'thank you bye', 'thanks bye', "that's all", 
                'nothing else', 'end call', 'disconnect', 'have a great day',
                'have a good day'
            ]
            
            bot_said_goodbye = any(phrase in self._last_bot_response for phrase in goodbye_phrases)
            
            if bot_said_goodbye:
                logger.info(f"🔚 Bot said goodbye: '{self._last_bot_response}' - ending call")
                await asyncio.sleep(0.5)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                logger.info("🔚 EndFrame sent - call will disconnect")
            
            self._last_bot_response = ""
        
        await self.push_frame(frame, direction)


class BackgroundMusicPlayer:
    """Plays background music by sending OutputAudioRawFrame directly to transport output."""
    
    def __init__(self):
        self._enabled = False
        self._music_task = None
        self._audio_data = None
        self._transport = None
        self._load_music()
    
    def _load_music(self):
        try:
            from pydub import AudioSegment
            import os
            mp3_path = os.path.join(os.path.dirname(__file__), "Clear_Call_Tone.mp3")
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio + 5
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            self._audio_data = audio.raw_data
            logger.info(f"🎵 Background music loaded: {len(self._audio_data)} bytes")
        except Exception as e:
            logger.warning(f"Could not load background music: {e}")
            self._audio_data = None

    async def enable(self):
        logger.info(f"🎵 enable() called — already_enabled={self._enabled}, has_audio={self._audio_data is not None}, has_transport={self._transport is not None}")
        if self._enabled or not self._audio_data or not self._transport:
            return
        self._enabled = True
        self._music_task = asyncio.create_task(self._play_loop())

    async def disable(self):
        if not self._enabled:
            return
        logger.info("🎵 Stopping music")
        self._enabled = False
        if self._music_task:
            self._music_task.cancel()
            try:
                await self._music_task
            except asyncio.CancelledError:
                pass
            self._music_task = None
        await asyncio.sleep(0.15)
        logger.info("🎵 Music stopped, buffer drained")
    
    async def _play_loop(self):
        logger.info("🎵 Background music playing")
        chunk_size = 1600
        chunk_duration = chunk_size / (16000 * 2)
        while self._enabled:
            for i in range(0, len(self._audio_data), chunk_size):
                if not self._enabled:
                    break
                frame = OutputAudioRawFrame(
                    audio=self._audio_data[i:i+chunk_size],
                    sample_rate=16000,
                    num_channels=1
                )
                await self._transport.send_audio(frame)
                await asyncio.sleep(chunk_duration)
        logger.info("🎵 Background music stopped")


async def run_bot(transport: BaseTransport):
    """Main bot logic for OUTBOUND calls."""
    logger.info("Starting OUTBOUND bot")

    stt = DeepgramFluxSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            interim_results=True,
            params=DeepgramFluxSTTService.InputParams(
                eager_eot_threshold=0.3,
                eot_threshold=0.5,
                eot_timeout_ms=1500,
                interim_results=True
            ),
        )

    _WELCOME_RE = re.compile(
        r'(?:hello\s*[,.]?\s*)?'
        r'welcome to indian consulate\s*(?:services\s*)?'
        r'(?:johannesburg|south africa|joburg|jhb)?\s*'
        r'[,.]?\s*',
        re.IGNORECASE
    )

    _greeting_spoken = [False]

    class FilteredPiperTTSService(PiperTTSService):
        """
        Subclass of PiperTTSService that filters assembled sentences BEFORE synthesis.
        """
        async def run_tts(self, text: str, context_id: str):
            if _greeting_spoken[0]:
                text = _WELCOME_RE.sub('', text).strip()
                if not text:
                    logger.warning("🚫 Suppressed pure welcome-repeat sentence")
                    return

            cleaned = sanitize_for_tts(text)
            if not cleaned or not cleaned.strip():
                logger.warning(f"🚫 TTS blocked sentence: {text[:80]}")
                return
            async for frame in super().run_tts(cleaned, context_id):
                yield frame

    tts = FilteredPiperTTSService(
        use_cuda=True,
        settings=PiperTTSService.Settings(
            voice="en_US-hfc_female-medium"
        ),
        text_aggregation_mode=TextAggregationMode.SENTENCE,
        pause_frame_processing=True,
        stop_frame_timeout_s=6.0,
    )
    
    llm = OpenAILLMService(
        api_key=os.getenv("HF_TOKEN"),
        base_url="https://router.huggingface.co/v1",
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.0,
        top_p=0.5,
    )

    from pipecat.frames.frames import TTSSpeakFrame

    import random
    _ack_messages = [
        "Let me check that for you.",
        "One moment please.",
        "Just a second please.",
        "Let me look that up for you.",
        "Give me a moment.",
        "Checking that now.",
        "Please hold on.",
        "Let me find that information for you.",
        "Sure, let me check.",
        "Right away, let me look into that.",
    ]
    _ack_index = 0

    _music_state = "IDLE"

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        nonlocal _ack_index, _music_state
        msg = _ack_messages[_ack_index % len(_ack_messages)]
        _ack_index += 1
        _music_state = "WAIT_ACK_END"
        await tts.queue_frame(TTSSpeakFrame(msg))

    @llm.event_handler("on_function_calls_finished")
    async def on_function_calls_finished(service, function_calls):
        nonlocal _music_state
        if _music_state == "MUSIC_ON":
            _music_state = "IDLE"
            await music_player.disable()
        else:
            _music_state = "IDLE"


    class AckMusicBridge(FrameProcessor):
        """
        Sits after `tts` in the pipeline and watches TTS lifecycle frames to
        control music with exact timing.
        """

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            nonlocal _music_state
            await super().process_frame(frame, direction)

            if isinstance(frame, TTSStoppedFrame) and _music_state == "WAIT_ACK_END":
                _music_state = "MUSIC_ON"
                logger.info("🎵 Ack done → music ON")
                await music_player.enable()

            elif isinstance(frame, TTSStartedFrame) and _music_state == "MUSIC_ON":
                _music_state = "IDLE"
                logger.info("🎵 Answer starting → music OFF")
                await music_player.disable()

            await self.push_frame(frame, direction)


    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,
                    start_secs=0.1,
                    min_volume=0.75,
                )
            ),
        ),
    )

    user_spy = LiveTranscriptionSpy("USER")
    bot_spy = LiveTranscriptionSpy("BOT")
    goodbye_detector = GoodbyeDetector()

    tl = TranscriptionLogger()
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer()
    )
    tts_filter = TTSTextFilter()
    mute_stt = MuteSTTDuringTTS()

    music_player = BackgroundMusicPlayer()
    ack_music_bridge = AckMusicBridge()

    pipeline = Pipeline([
        transport.input(),
 
        mute_stt,
 
        stt,
 
        context_aggregator.user(),
 
        llm,
        goodbye_detector,
        tts_filter,
        TTSCleaner(),
        tts,
        ack_music_bridge,
 
        transport.output(),
 
        context_aggregator.assistant(),
    ])


    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_tracing=True,
        observers=[latency_observer]
    )

    user_spy._rtvi_task = task
    bot_spy._rtvi_task = task

    music_player._transport = transport.output()

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @context_aggregator.user().event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.info(f"[USER DONE ] {message.content}")
        
        flow_manager.state["last_user_text"] = message.content
        logger.info(f"[STORED USER TEXT] {message.content}")
        
        if hasattr(task.rtvi, "send_message"):
            try:
                await task.rtvi.send_message({
                    "type": "user-transcript",
                    "data": {
                        "text": message.content,
                        "is_final": True,
                    }
                })
            except Exception:
                pass

    @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"[BOT  DONE ] {message.content}")

        if not _greeting_spoken[0]:
            _greeting_spoken[0] = True
            logger.info("🎤 Greeting turn done — welcome phrase will be stripped from now on")
        
        clean_text = sanitize_for_tts(message.content).strip()
        if not clean_text:
            return
        if hasattr(task.rtvi, "send_message"):
            try:
                await task.rtvi.send_message({
                    "type": "bot-transcript",
                    "data": {
                        "text": clean_text,
                        "is_final": True,
                    }
                })
            except Exception:
                pass

    _flow_initialized = [False]

    async def _initialize_flow_once():
        if _flow_initialized[0]:
            return
        _flow_initialized[0] = True
        logger.info("Initializing OUTBOUND flow (once)")
        
        # 🎯 SET OUTBOUND MESSAGE BEFORE INITIALIZING
        # This is the specific message the bot will deliver
        if not flow_manager.state.get("outbound_message"):
            flow_manager.state["outbound_message"] = (
                "your passport application is ready for collection at our Johannesburg office. "
                "You can collect it Monday to Friday between nine A M and three P M. "
                "Please bring your original ID and the application receipt."
            )
        
        await initialize_user(flow_manager)
        
        # 🚨 PASS outbound_message to create_initial_node
        outbound_message = flow_manager.state.get("outbound_message", "")
        customer_name = flow_manager.state.get("name", "")
        
        # Build the initial greeting that will be in the system prompt
        if customer_name:
            initial_greeting = f"Hello {customer_name}! This is Indian Consulate Services calling. I'm calling to inform you that {outbound_message} Do you have any questions about this?"
        else:
            initial_greeting = f"Hello! This is Indian Consulate Services calling. I'm calling to inform you that {outbound_message} Do you have any questions about this?"
        
        # Store the greeting in state so the flow can use it
        flow_manager.state["initial_greeting"] = initial_greeting
        
        await flow_manager.initialize(
            create_initial_node(
                customer_name=customer_name,
                greeted=True,
                outbound_message=outbound_message
            )
        )

        logger.info(f"📞 OUTBOUND flow initialized with message: {initial_greeting[:150]}...")

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("🎯 Client ready - initializing outbound flow")
        await _initialize_flow_once()
        
        # 🚨 PROACTIVE OUTBOUND: Bot speaks FIRST
        # Wait a moment for pipeline to be ready, then speak
        await asyncio.sleep(1.0)
        
        outbound_message = flow_manager.state.get("outbound_message", "")
        customer_name = flow_manager.state.get("name", "")
        
        if customer_name:
            greeting = f"Hello {customer_name}! This is Indian Consulate Services calling. I'm calling to inform you that {outbound_message} Do you have any questions about this?"
        else:
            greeting = f"Hello! This is Indian Consulate Services calling. I'm calling to inform you that {outbound_message} Do you have any questions about this?"
        
        logger.info(f"📞 SPEAKING FIRST (OUTBOUND): {greeting[:100]}...")
        
        # Mark greeting as delivered
        flow_manager.state["greeting_delivered"] = True
        _greeting_spoken[0] = True
        
        # 🚨 CRITICAL: Send greeting directly through task pipeline
        # This bypasses LLM and goes straight to TTS
        await task.queue_frames([TextFrame(greeting)])

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi_instance, msg):
        logger.info(f"RTVI client message: {msg.type} {msg.data}")
        if msg.data:
            # 🎯 ACCEPT OUTBOUND MESSAGE FROM CLIENT
            if msg.data.get("msisdn"):
                flow_manager.state["msisdn"] = msg.data["msisdn"]
            if msg.data.get("name"):
                flow_manager.state["name"] = msg.data["name"]
            if msg.data.get("callId"):
                flow_manager.state["callId"] = msg.data["callId"]
                print("Stored callId:", flow_manager.state["callId"])
            # 🚨 CRITICAL: Accept the outbound_message from the calling system
            if msg.data.get("outbound_message"):
                flow_manager.state["outbound_message"] = msg.data["outbound_message"]
                logger.info(f"📞 Received outbound message: {msg.data['outbound_message'][:100]}...")
            
            flow_manager.state["last_user_text"] = msg.data.get("text", "")
            await _initialize_flow_once()
        else:
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected to OUTBOUND bot")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected from OUTBOUND bot")
        await task.cancel()

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency):
        logger.info(f"Response latency: {latency:.3f}s")

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for OUTBOUND calls."""

    transport = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection

            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_in_filter=RNNoiseFilter(resampler_quality="VHQ"),
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
