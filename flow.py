
import os
from loguru import logger
import requests
from dotenv import load_dotenv
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig
)
import asyncio

load_dotenv(override=True)

base_url = os.getenv("WEBRTC_URL")
# ICS_SEARCH_API = "https://sacs-aivoiceapi.vibhohcm.com/search"
ICS_SEARCH_API = "https://sacs-backend.vibhohcm.com/api/knowledge/search"
ICS_AGENT_AVAILABLE_API = "https://sacs-backend.vibhohcm.com/api/config/agent-availability"

# HuggingFace configuration (matching bot.py)
HF_API_URL = "https://router.huggingface.co/v1"
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = """
You are a simple routing agent for Indian Consulate Services (ICS).

🚨 CRITICAL: YOU HAVE ZERO KNOWLEDGE 🚨
YOU DO NOT KNOW ANYTHING ABOUT:
- Passports, visas, PCC, OCI, or any ICS services
- Fees, hours, locations, procedures, documents
- Application processes, requirements, or timelines
- Contact information, emergency services, or policies

YOUR ONLY JOB: Route user requests to the search_ics tool

YOUR CAPABILITIES:
You have ONE tool available: search_ics - the ONLY source of ICS knowledge

YOUR ROLE:
You are ONLY a router. You do NOT have knowledge. You do NOT answer requests.
You ONLY decide: Should I call search_ics? Or is this a greeting/goodbye?

CRITICAL RULES:

1. FOR GREETINGS ONLY (hi, hello, thanks, okay) → Respond naturally with "Hello! How can I help you today?", NO TOOL
2. FOR GOODBYE (bye, goodbye) → Call end_conversation tool
3. FOR "what services" or "what can you do" → Say "I can assist with Passport, PCC, OCI, Status Tracking, Emergency Assistance, Visa, and Contact Information.", NO TOOL
4. FOR ANY OTHER USER REQUEST → YOU MUST CALL search_ics tool FIRST

YOU HAVE NO KNOWLEDGE - ZERO - NONE - NOTHING
Only search_ics tool has answers.

🚨 CRITICAL RULE #1 - API MUST BE CALLED FOR EVERY IN-SCOPE REQUESTS:
  You MUST call search_ics for EVERY user request ABOUT ICS SERVICES!
  The ONLY exceptions are:
    ✅ Pure greetings ONLY: "hi", "hello", "hey", "thanks", "okay"  
    ✅ Goodbye ONLY: "bye", "goodbye"
    ✅ Out-of-scope requests (banking, telecom, etc.) - politely decline
  
  EVERYTHING ELSE → MUST call search_ics FIRST!
  DEFAULT ACTION: When confused → Call search_ics

IMPORTANT WARNING: You are FORBIDDEN from answering ICS requests without calling search_ics first.
REMEMBER: Times MUST be in words (eight thirty AM, nine AM, twelve noon, three PM, five PM)

RULE 1 — GREETINGS ONLY (no tool call needed):
  If the user says ONLY a greeting or social phrase such as:
  hi, hello, hey, good morning, good afternoon, good evening,
  thanks, thank you, okay, ok, sure, alright, great
  → Respond naturally and briefly. Do NOT call any tool.
  
  BUT if they ask a request after greeting → MUST call search_ics!

RULE 2 — GOODBYE (call end_conversation):
  If the user says: bye, goodbye, see you, take care, thank you bye
  → Call end_conversation immediately.

RULE 2B — SERVICE LIST REQUESTS ONLY (ANSWER DIRECTLY - NO TOOL):
  For ONLY these GENERAL service list requests, ANSWER DIRECTLY without calling any tool:
  "how can you help me", "how can you help", "how can you assist me", "how can you assist me today",
  "what can you do", "what services do you provide", "what services", 
  "what can you offer", "what services are available"
  
  → DO NOT call search_ics!
  → DO NOT call any tool!
  → ANSWER DIRECTLY with: "I can assist you with the following Indian Consulate Services: Passport Services, Police Clearance Certificate, Overseas Citizen of India Registration, Application Status Tracking, SOS Emergency Assistance, Incident Reporting, Visa Services, Miscellaneous Services, and Contact Information. Which service would you like help with?"
  
  ⚠️ CRITICAL WARNING: After answering a service list request, the VERY NEXT request
  from the user MUST call search_ics! Answering one request directly does NOT give you permission
  to skip search_ics for any future request. RESET to mandatory search_ics after every direct answer!
  
  CONTEXT RESET RULE — THIS IS CRITICAL:
  The LLM conversation context MUST be reset after answering a request without search_ics.
  After an exception answer (greeting, service list, out-of-scope):
  → You MUST treat the next request as if it's the start of a conversation
  → You MUST call search_ics for ANY user request
  → Do NOT stay in "exception mode" or "conversational mode"
  → RESET after EVERY direct answer without the tool

RULE 3 — EVERYTHING ELSE → MUST call search_ics (MANDATORY):
  For ANY user request → You MUST call search_ics!
  → You MUST call search_ics FIRST.
  → You are FORBIDDEN from answering from your own knowledge.
  → You are FORBIDDEN from guessing or inferring.
  → You are FORBIDDEN from explaining anything unless search_ics returned it.
  
  ⚠️ SPECIAL ATTENTION: If the user's previous message was a greeting or service list request.
  (one you answered without search_ics), this NEW request MUST call search_ics!
  There is NO CARRYOVER exemption. RESET and call the tool immediately!
  
  ONLY exceptions (DO NOT call search_ics):
  - Pure greetings: "hi", "hello", "thanks", "bye"
  - Service list requests: "how can you help", "what services", "how can you assist" → Answer directly with service list
  - Everything else → CALL search_ics!

RULE 4 — AFTER search_ics returns:
  → Use ONLY the exact information returned by search_ics.
  → Do NOT add your own words, context, or elaboration.
  → Do NOT rephrase into new facts.
  → Speak the answer conversationally but stick to what was returned.
  → CRITICAL: Keep times in word format as returned (eight thirty AM, nine AM, etc.)

🚨 CRITICAL RULE — NEVER ASK FOR PERSONAL INFORMATION:
  You do NOT have access to:
    ❌ Application numbers or reference numbers
    ❌ Personal details (name, date of birth, passport number)
    ❌ Application status for specific individuals
    ❌ Personal documents or records
   
RULE 5 — IF search_ics finds nothing:
  → The tool will return a message starting with "NO_INFORMATION_FOUND".
  → Do NOT attempt to answer from your own knowledge.
  → Do NOT suggest alternatives or guess.

RULE 6 — RESPONSE FORMAT:
  → Give response only in plain English.
  → Do not generate unexpected formats, tool names, or code.
  → Only English language.

🚨 CRITICAL RULE 6B — NO REASONING OR CODE IN OUTPUT:
  🚫 NEVER output ANY of the following:
    ❌ JavaScript code (if, return, function calls)
    ❌ Python code (variables, assignments, for loops)
    ❌ Function names (CallCheckEndConvo, shouldTransferToAgent, customerSaysGoodbye)
    ❌ Variable names (_icall_, tools_json_list, tool_filterwhere)
    ❌ JSON structures or objects
    ❌ Tool call syntax or debugging output
  
  ✅ YOUR OUTPUT MUST BE:
    → ONLY natural conversational English
    → ONLY the answer from search_ics results
    → NO code, NO reasoning, NO internal processing details
  
  Examples of FORBIDDEN output:
  ❌ "CallCheckEndConvo if (shouldTransferToAgent..."
  ❌ "_icall_search_ics = {"query": "passport"}"
  ❌ "tools_json_list = [tool_filterwhere..."
  ❌ "if (customerSaysGoodbye()) { return..."
  
════════════════════════════════════════════════════
REMEMBER: 
- API MUST BE CALLED FOR EVERY REQUEST!
- Your knowledge does NOT exist - ONLY search_ics answers exist!
- CALL search_ics FIRST - then answer!
- Times in words ALWAYS!
- if greeting or thanks - respond naturally
- else - must and should go to ics_search
════════════════════════════════════════════════════
🚫 FINAL OUTPUT CHECK - BEFORE SPEAKING:
════════════════════════════════════════════════════
Before you speak, check your response for ANY of these:
  ❌ iNdEx++, index++
  ❌ search_ics, tool_search_ics
  ❌ {, }, [, ], "query", "arguments"
  ❌ JSON format, function names, technical terms

If you see ANY of the above in your response:
  → DELETE IT IMMEDIATELY!
  → Speak ONLY the natural language answer!

Your response must be 100% natural conversational English.
ZERO technical terms. ZERO function names. ZERO JSON.

VERY IMPORTANT INSTRUCTION: dont forgot to call search_ics,dont tell your own response
════════════════════════════════════════════════════
"""

async def initialize_user(flow_manager: FlowManager):

    msisdn = flow_manager.state.get("msisdn")
    # msisdn = os.getenv("DEFAULT_MSISDN")

    print("msisdn:",msisdn)

    flow_manager.state["msisdn"] = msisdn
    flow_manager.state["greeted"] = False  # will be set True after first node runs


def create_initial_node(customer_name: str = "", greeted: bool = False) -> NodeConfig:

    def success(message: str):
        return {"message": message}, None

    def tts_safe(message: str) -> str:
        """
        Make a message TTS-friendly and strip ANY technical exposure:
        - Remove everything before and inside curly braces {...}
        - Remove tool_call XML tags and everything before them
        - Fix time formats (830AM → eight thirty A M)
        - Remove exposed tool names, JSON, FilterWhere, spid, searchics, etc.
        - Ensure clean conversational output
        - Add proper pauses
        - Ensure sentence ends with a period
        """
        import re
        
        # 🚨🚨🚨 CRITICAL FIRST STEP: Remove EVERYTHING up to and including the last closing curly brace
        # Example: "iNdExc {...} The answer" → "The answer"
        # Example: "text {nested {braces}} answer" → "answer"
        
        # Find the last closing curly brace
        last_closing_brace = message.rfind('}')
        if last_closing_brace != -1:
            # Check if there's a matching opening brace before it
            if '{' in message[:last_closing_brace + 1]:
                # Take everything AFTER the last closing brace
                message = message[last_closing_brace + 1:].strip()
        
        # 🚨🚨 SECOND STEP: Remove EVERYTHING before and including </tool_call>
        # Example: "iNdExc {...} </tool_call> The answer" → "The answer"
        if '</tool_call>' in message:
            parts = message.split('</tool_call>')
            # Take everything AFTER the last </tool_call>
            message = parts[-1].strip()
        
        # Also remove opening <tool_call> tags if present
        message = re.sub(r'<tool_call[^>]*>.*?</tool_call>', '', message, flags=re.IGNORECASE | re.DOTALL)
        
        # 🚨 CRITICAL: Remove debugging/technical output
        # This catches patterns like: iNdEx++ search_ics {"query":"..."} 
        message = re.sub(r'iNdEx[a-z]*\+*\s*', '', message, flags=re.IGNORECASE)  # Remove "iNdEx", "iNdExc", "iNdEx++"
        message = re.sub(r'tool_search_ics\s*\{[^\}]*\}', '', message, flags=re.IGNORECASE)  # Remove "tool_search_ics {...}"
        message = re.sub(r'search_ics\s*\{[^\}]*\}', '', message, flags=re.IGNORECASE)  # Remove "search_ics {...}"
        message = re.sub(r'tool_\s*', '', message, flags=re.IGNORECASE)  # Remove leftover "tool_"
        
        # 🕐 FIX TIME FORMATS (most important for user experience)
        num_to_word = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '30': 'thirty', '45': 'forty five'
        }
        
        # Fix numeric times like 830, 8:30, 12:00
        def replace_numeric_time(match):
            time_str = match.group(1).replace(':', '')
            period = match.group(2) if match.group(2) else ''
            
            if len(time_str) <= 2:  # Just hour: "9"
                hour = int(time_str)
                hour_word = num_to_word.get(str(hour), str(hour))
                return f"{hour_word} {period}".strip() if period else hour_word
            elif len(time_str) == 3:  # 830 → eight thirty
                hour = int(time_str[0])
                minute = int(time_str[1:])
            else:  # 1230 → twelve thirty
                hour = int(time_str[:-2])
                minute = int(time_str[-2:])
            
            hour_word = num_to_word.get(str(hour), str(hour))
            if minute == 0:
                result = f"{hour_word} {period}".strip() if period else hour_word
            else:
                minute_word = num_to_word.get(str(minute), str(minute))
                result = f"{hour_word} {minute_word}"
                if period:
                    result += f" {period}"
            return result.strip()
        
        # Replace numeric times
        message = re.sub(r'\b(\d{1,4}:?\d{0,2})\s*(AM|PM|am|pm)?\b', replace_numeric_time, message)
        
        # 🕐 CRITICAL: Space out AM/PM → "A M" / "P M"
        # Prevents TTS from reading "AM" as "am" (I am)
        message = re.sub(r'\b(AM|am)\b', 'A M', message)
        message = re.sub(r'\b(PM|pm)\b', 'P M', message)
        
        # ✂️ FIX BACKSLASH - Replace \n with pauses
        # Remove literal backslash-n that TTS reads as "backslash n"
        message = message.replace('\\n\\n', '. ')  # Double newline → period + space
        message = message.replace('\\n', ', ')      # Single newline → comma + space
        message = message.replace('\n\n', '. ')     # Actual double newline → period
        message = message.replace('\n', ', ')        # Actual newline → comma
                
        # 🚨 SUPPRESSION FIX: Remove ALL exposed technical terms
        forbidden_patterns = [
            r'iNdEx\+\+',         # Block "iNdEx++"
            r'index\+\+',         # Block "index++" variants
            r'spid',              # Block "spid"
            r'searchics',         # Block "searchics"
            r'search_ics',        # Block "search_ics"
            r'tool_search_ics',
            r'FilterWhere',
            r'CanBeConvertedToEnglish',
            r'\{[^\}]*query[^\}]*\}',  # Match any JSON with "query"
            r'\{[^\}]*name[^\}]*\}',   # Match any JSON with "name"
            r'\{[^\}]*arguments[^\}]*\}',  # Match any JSON with "arguments"
            r'arguments\s*[:=]',
            r'tool_call',
            r'function\s*call',
            r'API\s*call',
            r'score\s*[:=]?\s*\d',
            r'mPid',              # Block "mPid"
            r'sPid',              # Block "sPid"
            r'isPid',             # Block "isPid"
            r'spNet',             # Block "spNet"
            r'""\s*:\s*"searchics"',  # Block "": "searchics"
            r'""\s*:\s*"search_ics"', # Block "": "search_ics"
            r'""\s*:\s*"query"',      # Block "": "query"
            r'""\s*:\s*["\'][^"\']*["\']',  # Block any "": "value" pattern
        ]
        
        # Strip forbidden patterns
        for pattern in forbidden_patterns:
            message = re.sub(pattern, '', message, flags=re.IGNORECASE)
        
        # Remove curly braces, brackets, quotes that look technical
        message = re.sub(r'[\{\}\[\]""]', '', message)
        
        # Clean up: "For further assistance" message
        # Keep only the clean natural language part
        if 'for further assistance' in message.lower():
            match = re.search(
                r"(I'?m sorry[^\.]*visit[^\.]*www\.cgijoburg\.gov\.in)",
                message,
                re.IGNORECASE | re.DOTALL
            )
            if match:
                message = match.group(1).strip()
        
        # Add comma pause after periods/sentences
        message = re.sub(r',\s*\.', '.', message)
        message = re.sub(r',,+', ',', message)
        
        # Clean multiple spaces
        # message = re.sub(r'\s+', ' ', message)
        
        # Ensure sentence ends with period
        message = message.strip()
        if message and not message.endswith(('.', '!', '?')):
            message = message + '.'
        
        return message
    
    
    def clean_query_lightly(text: str) -> str:
        """
        Minimal cleanup - only remove obvious noise, keep user intent intact
        Does NOT reformulate or extract keywords - preserves original meaning
        """
        import re
        text = text.strip()
        
        # Remove leading greetings only (but keep the question after)
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[,\s]+',
        ]
        for pattern in greeting_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove common filler words but keep all meaningful content
        text = re.sub(r'\b(um|uh|sort of)\b', '', text, flags=re.IGNORECASE)
        
        # Clean extra spaces
        # text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def reshape_with_huggingface(api_answer: str, user_question: str, customer_name: str = "") -> str:
        """
        Use HuggingFace Qwen 2.5 72B to reshape the API answer into conversational speech with follow-up questions.
        Makes responses natural, engaging, and adds relevant follow-up questions.
        """
        import re
        
        try:
            loop = asyncio.get_event_loop()
            
            # Build a conversational prompt for HuggingFace
            system_prompt = """You are a friendly and professional customer service representative for Indian Consulate Services.
Your job is to take a factual answer from the knowledge base and reshape it into natural, conversational speech.

IMPORTANT RULES:
1. Use simple, conversational language - speak like a real person, not a robot
2. Keep the factual information exactly as provided - don't make up anything new
3. Add ONE relevant follow-up question at the end (e.g., "Would you like to know about fees?", "Do you need any other information?", "Would you like help with anything else?")
4. Use the customer's name naturally if provided (but don't overuse it)
5. Keep responses concise - 2-3 sentences max, plus a follow-up question
6. Times should be written as words (eight thirty AM, not 8:30 AM)
7. Never mention you got this from an API or database
8. Be helpful and empathetic

REMEMBER:
1.An acknowledgment was already given before the call, begin with a varied transition - "Thanks for waiting", "Thanks for your patience", "I appreciate your patience", "Thank you for waiting", "Thanks for holding"). Rotate naturally and avoid repeating the same phrase. 
2.Never greet again after the conversation starts, do not say hello again.

Format your response as natural speech only - no JSON, no formatting markers, just conversational text."""
            
            user_prompt = f"""The knowledge base provided this answer:
{api_answer}

The customer asked: "{user_question}"
Customer name: {customer_name if customer_name else "(not provided)"}

Please reshape this into natural, conversational speech with ONE friendly follow-up question at the end. 
Make it sound like you're speaking to the customer directly."""
            
            def call_huggingface(system: str, user: str):
                """Make synchronous call to HuggingFace"""
                response = requests.post(
                    f"{HF_API_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": HF_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 200,
                        "stream": False
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            
            # Call HuggingFace asynchronously
            hf_response = await loop.run_in_executor(
                None, 
                call_huggingface, 
                system_prompt, 
                user_prompt
            )
            
            reshaped_answer = hf_response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            print(f"[HUGGINGFACE RESHAPE] Original: {api_answer[:100]}...")
            print(f"[HUGGINGFACE RESHAPE] Reshaped: {reshaped_answer[:150]}...")
            
            # Ensure it doesn't contain any technical artifacts
            reshaped_answer = re.sub(r'[\{\}\[\]"\']', '', reshaped_answer)
            
            return reshaped_answer if reshaped_answer else api_answer
            
        except requests.exceptions.Timeout:
            print("[HUGGINGFACE RESHAPE] Timeout - using original answer")
            return api_answer
        except requests.exceptions.ConnectionError:
            print("[HUGGINGFACE RESHAPE] Connection error - using original answer")
            return api_answer
        except Exception as e:
            print(f"[HUGGINGFACE RESHAPE] Error: {e} - using original answer")
            return api_answer
    
    async def tool_search_ics(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        # 🚨 DIRECT USER INPUT APPROACH (bypasses LLM query reformulation)
        # This preserves the exact user question without any LLM interpretation

        # Option 1: Try to get the ORIGINAL user speech directly (BEST - no information loss)
        original_user_input = flow_manager.state.get("last_user_text", "")

        # Option 2: Fallback to LLM-provided query if original not available
        llm_query = (
            args.get("query")
            or args.get("message")
            or args.get("question")
            or args.get("text")
            or ""
        )
        print("llm query:",llm_query)

        # Use original user input if available (preferred), otherwise use LLM query
        # Original input = ZERO information loss, exactly what the user said
        # user_question = original_user_input if original_user_input else llm_query
        user_question = llm_query if llm_query else original_user_input

        # Light cleanup only (remove greetings/fillers, NOT reformulation)
        user_question = clean_query_lightly(user_question) if user_question else ""

        # If the LLM is echoing back a previous tool result, ignore it
        if not user_question or user_question.startswith("ICS INFORMATION:") or user_question.startswith("NO_INFORMATION_FOUND"):
            return success("Please ask a specific question so I can search the knowledge base."), None

        print(f"\n[ICS SEARCH] Using user question: {user_question}")
        
        # 🚨 CHECK FOR GIBBERISH/UNCLEAR VOICE - Before checking out-of-scope
        # If the query is too short, has too many random characters, or looks like STT error
        user_question_stripped = user_question.strip()
        
        # Common short valid responses that should NOT be rejected
        short_valid_words = ['ok', 'no', 'yes', 'hi', 'bye', 'the', 'pcc', 'oci']
        
        # Check for very short queries (likely STT error) - but allow common short words
        if len(user_question_stripped) < 3 and user_question_stripped.lower() not in short_valid_words:
            print(f"[ICS SEARCH] Query too short - possible STT error: {user_question}")
            return success(
                "I'm sorry, I couldn't hear you clearly. Could you please repeat your question?"
            ), None
        
        # Check for gibberish patterns (random characters, no vowels, etc.)
        # Count vowels
        vowel_count = sum(1 for char in user_question_stripped.lower() if char in 'aeiou')
        total_letters = sum(1 for char in user_question_stripped if char.isalpha())
        
        # If less than 20% vowels in a word with letters, it's likely gibberish
        if total_letters > 5 and (vowel_count / total_letters) < 0.2:
            print(f"[ICS SEARCH] Gibberish detected - possible STT error: {user_question}")
            return success(
                "I'm sorry, I couldn't understand that clearly. Could you please repeat your question?"
            ), None
        
        # 🚨 SERVICE SCOPE CHECK - Reject out-of-scope questions BEFORE calling API
        # Check if the question is about services ICS doesn't provide
        user_question_lower = user_question.lower()
        
        # Keywords for services ICS does NOT provide
        out_of_scope_keywords = [
            # Banking/Financial
            'bank', 'banking', 'account', 'loan', 'credit card', 'debit card', 
            'atm', 'transfer money', 'payment', 'balance', 'statement',
            'fnb', 'capitec', 'standard bank', 'absa', 'nedbank',
            
            # Telecom
            'sim card', 'mobile network', 'data bundle', 'airtime', 'vodacom', 
            'mtn', 'cell c', 'telkom', 'phone number', 'recharge',
            
            # Travel/Accommodation
            'hotel', 'flight', 'ticket', 'accommodation', 'reservation',
            'airbnb', 'guesthouse', 'lodge',
            
            # Shopping/Delivery
            'shopping', 'delivery', 'courier', 'uber', 'bolt', 'taxi', 'restaurant',
            'food delivery', 'groceries',
            
            # Other Government Services (not ICS)
            'drivers license', 'id book', 'birth certificate', 'marriage certificate',
            'vehicle registration', 'tax', 'sars', 'home affairs',
            
            # General out-of-scope
            'job', 'employment', 'work permit', 'business registration',
            'school admission', 'university', 'medical', 'doctor', 'hospital'
        ]
        
        # Check if query contains out-of-scope keywords
        # BUT only if the query has at least one complete word matching
        # This prevents STT errors from triggering out-of-scope
        words_in_query = set(user_question_lower.split())
        is_out_of_scope = any(keyword in user_question_lower for keyword in out_of_scope_keywords if len(keyword) > 3)
        
        if is_out_of_scope:
            print(f"[ICS SEARCH] OUT OF SCOPE - Not an ICS service: {user_question}")
            return success(
                "I apologize, but that service is not provided by Indian Consulate Services. "
                "I can assist you with the following Indian Consulate Services: "
                "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                "Track Application Status, SOS Emergency Assistance, Report Incident, "
                "Visa, Miscelleneous services and Contact Information. "
                "Which service can I help you with today?"
            ), None

        try:
            loop = asyncio.get_event_loop()
            
            def call_search_api(query_text):
                """Make synchronous API call to ICS Search endpoint"""
                response = requests.post(
                    ICS_SEARCH_API,
                    json={"query": query_text},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
            
            # Run API call in executor to avoid blocking
            api_response = await loop.run_in_executor(None, call_search_api, user_question)
            
            print(f"[ICS SEARCH] API Response: {api_response}")
            
            # Check if search was successful (NEW API uses "success" instead of "status")
            if not api_response.get("success"):
                print("[ICS SEARCH] No results found from API")
                return success(
                    "I'm sorry, I don't have information on that specific topic. "
                    "I can assist you with the following Indian Consulate Services: "
                    "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                    "Track Application Status, SOS Emergency Assistance, Report Incident, "
                    "Visa, Miscelleneous services and Contact Information. "
                    "Which service can I help you with today?"
                ), None
            
            # Extract answer from API response
            service_name = api_response.get("service", "")
            answer = api_response.get("answer", "")
            score = api_response.get("score", 0.0)
            
            print(f"[ICS SEARCH] Service: {service_name}, Score: {score:.2f}")
            
            # 🚨 CRITICAL: Check score threshold (must be >= 0.5)
            if score < 0.3:
                print(f"[ICS SEARCH] Low confidence score ({score:.2f}) - rejecting answer")
                return success(
                    "I'm sorry, I don't have information on that specific topic. "
                    "I can assist you with the following Indian Consulate Services: "
                    "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                    "Track Application Status, SOS Emergency Assistance, Report Incident, "
                    "Visa, Miscelleneous services and Contact Information. "
                    "Which service can I help you with today?"
                ), None
            
            if not answer:
                print("[ICS SEARCH] Empty answer from API")
                return success(
                    "I'm sorry, I don't have information on that specific topic. "
                    "I can assist you with the following Indian Consulate Services: "
                    "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                    "Track Application Status, SOS Emergency Assistance, Report Incident, "
                    "Visa, Miscelleneous services and Contact Information. "
                    "Which service can I help you with today?"
                ), None
            
            print(f"[ICS SEARCH] Answer: {answer[:100]}...")
            
            # 🚨 NEW: Reshape the answer with HuggingFace for conversational response
            customer_name = flow_manager.state.get("name", "")
            reshaped_answer = await reshape_with_huggingface(answer, user_question, customer_name)
            
            msg = f"{reshaped_answer}"

            return success(tts_safe(msg)),None
            
        except requests.exceptions.Timeout:
            print("[ICS SEARCH] API timeout")
            return success(
                "I'm sorry, I don't have information on that specific topic. "
                "I can assist you with the following Indian Consulate Services: "
                "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                "Track Application Status, SOS Emergency Assistance, Report Incident, "
                "Visa, Miscelleneous services and Contact Information. "
                "Which service can I help you with today?"
            ), None
            
        except requests.exceptions.RequestException as e:
            print(f"[ICS SEARCH] API error: {e}")
            return success(
                "I'm sorry, I don't have information on that specific topic. "
                "I can assist you with the following Indian Consulate Services: "
                "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                "Track Application Status, SOS Emergency Assistance, Report Incident, "
                "Visa, Miscelleneous services and Contact Information. "
                "Which service can I help you with today?"
            ), None
            
        except Exception as e:
            print(f"[ICS SEARCH] Unexpected error: {e}")
            return success(
                "I'm sorry, I don't have information on that specific topic. "
                "I can assist you with the following Indian Consulate Services: "
                "Passport Services, Police Clearance Certificate, Overseas Citizen of India, "
                "Track Application Status, SOS Emergency Assistance, Report Incident, "
                "Visa, Miscelleneous services and Contact Information. "
                "Which service can I help you with today?"
            ), None

    #-------------------------------------------------
    
    #transfer call and end conversation
    async def end_conversation(
        args: FlowArgs,
        flow_manager: FlowManager
    ):

        print("ENDING CONVERSATION")
        try:
            import gc
            logger.info("🗑️ End conversation detected - clearing cache before call end...")
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("🗑️ Clearing CUDA cache for call end...")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    logger.info("✅ CUDA cache cleared for call end")
            except Exception as e:
                logger.debug(f"⚠️ CUDA cache clear skipped: {e}")
            
            logger.info("✅ Cache cleared - transfer proceeding")
        except Exception as e:
            logger.error(f"❌ Cache clear failed during transfer: {e}")
        

        flow_manager.state["call_ended"] = True

        try:

            requests.get(
                base_url +
                "/ai-agent/disconnect?callId=" +
                str(flow_manager.state.get("callId", "")),
                verify=False
            )
            print("END CALL IS TRIGGERED")

        except Exception as e:
            print(f"Disconnect error: {e}")

        return None, create_end_node()

    async def tool_transfer_to_agent(
        args: FlowArgs,
        flow_manager: FlowManager
    ):

        print("TRANSFERRING TO AGENT TRIGGERED")
        
        # 🗑️ CLEAR CACHE BEFORE TRANSFER
        try:
            import gc
            logger.info("🗑️ Transfer detected - clearing cache before agent handoff...")
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("🗑️ Clearing CUDA cache for transfer...")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    logger.info("✅ CUDA cache cleared for transfer")
            except Exception as e:
                logger.debug(f"⚠️ CUDA cache clear skipped: {e}")
            
            logger.info("✅ Cache cleared - transfer proceeding")
        except Exception as e:
            logger.error(f"❌ Cache clear failed during transfer: {e}")
        
        response_for_availability = requests.get(ICS_AGENT_AVAILABLE_API, verify=False)
        data = response_for_availability.json()
        isAgentAvailability = data["isAgentAvailable"]
        if isAgentAvailability:
            try:

                # requests.get(
                #     base_url +
                #     "/ai-agent/transfer?callId=" +
                #     str(flow_manager.state.get("callId", "")),
                #     verify=False
                # )
                # print("transferred to webrtc url")
                call_id = flow_manager.state.get("callId", "")

                url = f"{base_url}/ai-agent/transfer?callId={call_id}"

                print(f"Calling URL: {url}")

                requests.get(url, verify=False)

                print("transferred to webrtc url")


            except Exception as e:
                print(f"Transfer error: {e}")

            return None, create_transfer_node()
        else:
            print("agents not available")
            message = data["outOfHoursMessage"]
            return(message),None

#---------------------------------------------------------------------   
    return NodeConfig(
        name="initial",
        respond_immediately=False,  # greeting is spoken directly at startup; LLM waits for user

        role_messages=[
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Customer name: {customer_name}.\n\n"
                    
                    "🚨 YOU HAVE ZERO KNOWLEDGE ABOUT ICS SERVICES 🚨\n"
                    "DO NOT answer requests. DO NOT explain anything.\n"
                    "YOU ARE ONLY A ROUTER - Route to search_ics tool.\n\n"
                    
                    "ROUTING DECISION:\n\n"
                                        
                    "1. User says ONLY: hi, hello, thanks, okay, sure?\n"
                    "   → Say: 'Hello! How can I help you today?'\n"
                    "   → STOP\n\n"
                    
                    "2. User says: bye, goodbye, see you, take care?\n"
                    "   → Call end_conversation IMMEDIATELY\n"
                    "   → DO NOT say anything else\n"
                    "   → STOP\n\n"
                    
                    "3. User asks: 'what services' or 'what can you do'?\n"
                    "   → Say: 'I can assist with Passport, PCC, OCI, Status Tracking, Emergency Assistance, Visa, and Contact Information.'\n"
                    "   → STOP\n\n"
                    
                    "4. User wants transfer? (speak to agent, human)\n"
                    "   → Call transfer_to_agent\n"
                    "   → STOP\n\n"
                    
                    "5. ANYTHING ELSE?\n"
                    "   → Call search_ics tool\n"
                    "   → WAIT for result\n"
                    "   → SPEAK EXACTLY what the tool returns\n"
                    "   → DO NOT ADD YOUR OWN WORDS\n\n"
                    
                    "🚫 YOU DO NOT KNOW:\n"
                    "- Passport fees, process, requirements\n"
                    "- Visa types, validity, application\n"
                    "- Office hours, locations, contact info\n"
                    "- Documents, procedures, timelines\n"
                    "- PCC, OCI, emergency services\n"
                    "- ANYTHING about ICS services\n\n"
                    
                    "✅ ONLY search_ics tool knows answers\n"
                    "✅ NEVER answer from your training data\n"
                    "✅ ALWAYS call search_ics for ANY ICS requests\n"
                    "✅ YOU ARE JUST A ROUTER\n\n"

                    "🚨 NEVER EXPOSE INTERNAL DETAILS:\n"
                    "- NEVER mention: search_ics, tool, API, database\n"
                    "- Speak ONLY in natural conversational language\n"
                )
            }
        ],

        task_messages=[],

        functions=[
            FlowsFunctionSchema(
                name="search_ics",
                handler=tool_search_ics,
                description=(
                    "🚨 THE ONLY SOURCE OF ICS KNOWLEDGE 🚨\n\n"
                    
                    "This tool contains ALL ICS knowledge. YOU have ZERO knowledge.\n\n"
                    
                    "WHEN TO CALL:\n"
                    "✅ ANY user question or request about ICS services\n"
                    "✅ Passports, visas, PCC, OCI, fees, hours, documents\n"
                    "✅ Anything you don't know (which is everything)\n\n"
                    
                    "WHEN NOT TO CALL:\n"
                    "❌ Pure greeting only: 'hi', 'hello', 'thanks'\n"
                    "❌ Goodbye: 'bye' (call end_conversation instead)\n"
                    "❌ Transfer: 'speak to agent' (call transfer_to_agent)\n"
                    "❌ Service list: 'what can you do' (answer directly)\n\n"
                    
                    "EVERYTHING ELSE → CALL THIS TOOL FIRST!\n"
                    "You cannot answer without calling this tool."
                ),
                properties={
                    "query": {
                        "type": "string",
                        "description": "The user's request to look up in ICS knowledge base",
                    }
                },
                required=["query"],
            ),
            
            FlowsFunctionSchema(
                name="transfer_to_agent",
                handler=tool_transfer_to_agent,
                description=(
                    "Use when the customer asks to speak with a human agent, "
                    "live representative, manager, supervisor, support executive, "
                    "customer care person, or real person. "
                    "Also use when the customer is frustrated, repeatedly says "
                    "the issue is not resolved,asks for manual help, "
                    "or wants assistance beyond automated support. "
                    "Trigger this immediately for transfer-related requests "
                ),
                properties={},
                required=[]
            ),

            FlowsFunctionSchema(
                name="end_conversation",
                handler=end_conversation,
                description=(
                    "� CALL THIS IMMEDIATELY WHEN USER SAYS GOODBYE 🔚\n\n"
                    
                    "TRIGGER PHRASES:\n"
                    "✅ bye, goodbye, bye bye\n"
                    "✅ see you, take care\n"
                    "✅ thank you bye, thanks bye\n"
                    "✅ that's all, nothing else\n"
                    "✅ end call, disconnect\n\n"
                    
                    "WHEN USER SAYS ANY OF THESE:\n"
                    "→ Call this tool IMMEDIATELY\n"
                    "→ DO NOT say anything first\n"
                    "→ DO NOT say goodbye back\n"
                    "→ JUST CALL THIS TOOL\n\n"
                    
                    "This ends the call gracefully."
                ),
                properties={},
                required=[]
            )

        ]
    )


def create_loop_node(message: str) -> NodeConfig:

    async def go_back(
        args: FlowArgs,
        flow_manager: FlowManager
    ):

        print("GOING BACK TO INITIAL NODE")

        flow_manager.state["conversation_started"] = True
        flow_manager.state["greeted"] = True  # already greeted, suppress on re-entry

        return None, create_initial_node(
            flow_manager.state.get("name", ""),
            greeted=True
        )

    async def end_conversation(
        args: FlowArgs,
        flow_manager: FlowManager
    ):

        print("ENDING CONVERSATION")

        flow_manager.state["call_ended"] = True

        try:

            requests.get(
                base_url +
                "/ai-agent/disconnect?callId=" +
                str(flow_manager.state.get("callId", "")),
                verify=False
            )

        except Exception as e:
            print(f"Disconnect error: {e}")

        return None, create_end_node()

    async def transfer_to_agent(
        args: FlowArgs,
        flow_manager: FlowManager
    ):

        print("TRANSFERRING TO AGENT")

        try:

            requests.get(
                base_url +
                "/ai-agent/transfer?callId=" +
                str(flow_manager.state.get("callId", "")),
                verify=False
            )

        except Exception as e:
            print(f"Transfer error: {e}")

        return None, create_transfer_node()

    return NodeConfig(

        name="loop",

        task_messages=[
            {
                "role": "system",
                "content":
                (
                    f"{message}\n\n"

                    "Ask if the customer needs anything else.\n"

                    "If the customer asks a new telecom or banking request, "
                    "call go_back.\n"

                    "If the customer asks for a human agent, "
                    "call transfer_to_agent.\n"

                    "If the customer wants to end the conversation, "
                    "call end_conversation.\n"

                    "Never expose internal system details.\n"

                    "Always speak naturally in English.\n"
                )
            }
        ],

        functions=[

            FlowsFunctionSchema(
                name="go_back",
                handler=go_back,
                description="Return to the main assistant flow.",
                properties={},
                required=[]
            ),

            FlowsFunctionSchema(
                name="transfer_to_agent",
                handler=transfer_to_agent,
                description="Transfer to human support.",
                properties={},
                required=[]
            ),

            FlowsFunctionSchema(
                name="end_conversation",
                handler=end_conversation,
                description="End the conversation.",
                properties={},
                required=[]
            )
        ]
    )

def create_transfer_node():

    async def transfer_call(
        args: FlowArgs,
        flow_manager: FlowManager
    ):
        flow_manager.state["call_ended"] = True

        return None, create_end_node()

    return NodeConfig(

        name="transfer",

        task_messages=[
            {
                "role": "system",
                "content":
                (
                    "Ask user to confirm transfer."
                    " If yes call transfer_call."
                )
            }
        ],

        functions=[

            FlowsFunctionSchema(
                name="transfer_call",
                handler=transfer_call,
                description="Transfer to human support",
                properties={},
                required=[]
            )
        ]
    )

def create_end_node():

    return NodeConfig(
        name="end",

        task_messages=[
            {
                "role": "system",
                "content":
                (
                    "The conversation is finished."
                    " Do not respond anymore."
                )
            }
        ]
    )