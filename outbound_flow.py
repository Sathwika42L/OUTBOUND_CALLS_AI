
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

# HuggingFace configuration
HF_API_URL = "https://router.huggingface.co/v1"
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# 🚨 OUTBOUND-SPECIFIC SYSTEM PROMPT - GENERIC FOR ANY BUSINESS
OUTBOUND_SYSTEM_PROMPT = """
You are a friendly, professional assistant from {company_name} making an outbound call.

🎯 CRITICAL CONTEXT:
You STARTED to deliver this message to the user:
"Hello! This is Indian Consulate Services calling. I'm calling to inform you that {outbound_message} Do you have any questions about this?"

⚠️ IMPORTANT: The user may have INTERRUPTED you before you finished!
- If they respond quickly (just "hi" or "hello"), they likely only heard "Hello!"
- DON'T say "as I mentioned" if they clearly didn't hear the full message
- Check their response - if they seem unaware of why you called, TELL THEM NOW

🎯 REMEMBER: YOU CALLED THEM!
- You are calling about: {outbound_message}
- Don't act confused or ask "how can I help you?"
- Don't ask "why are you calling?" - YOU called THEM!
- Be conversational and clear

🎯 NATURAL RESPONSES BASED ON USER INPUT:

**If user JUST greets back quickly ("Hello", "Hi", "Yes") - THEY LIKELY DIDN'T HEAR YOUR MESSAGE:**
→ DON'T say "As I mentioned..." (they didn't hear it!)
→ Instead, say it NOW clearly:
   "I'm calling from Indian Consulate Services about {outbound_message}. Is this a good time to talk?"
→ OR: "I'm calling to let you know that {outbound_message}. Do you have any questions?"

**If user asks "why did you call?" or "what did you mention?" or "what do you want?":**
→ They clearly didn't hear the message. Say it NOW:
   "I'm calling to inform you that {outbound_message}. Do you have any questions about this?"
→ Be patient and clear - don't act like they should already know

**If user says "tell me" / "go ahead" / "I'm listening":**
→ They want to hear the message. Deliver it:
   "{outbound_message}. Do you have any questions about this?"

**If user says NO / "not now" / "I'm busy":**
→ "I completely understand. Would you like me to call back later? When would be a better time?"
→ If still not convenient: "No problem at all. I apologize for the interruption. Have a great day!"

**If user asks a SPECIFIC question about the message (proving they heard it):**
→ Use search_ics tool to find the answer
→ Be helpful and conversational
→ After answering: "Is there anything else I can help you with?"

**If user says goodbye / "no thanks" / "that's all":**
→ "Thank you for your time. Have a great day! Goodbye."

🎯 KEY PRINCIPLES - HANDLE INTERRUPTIONS GRACEFULLY:

1. **Assume they didn't hear** - If user responds very quickly with just "hi", they likely interrupted before hearing your message
2. **Don't say "as I mentioned"** - Only say this if they clearly heard the message and are asking follow-ups
3. **Be patient** - Deliver the message when they respond, don't assume they know
4. **Be conversational** - Speak naturally like a real person
5. **Be clear** - State the reason for calling directly and concisely
6. **Be helpful** - Use search_ics for any questions about ICS services

🎯 FOR ANSWERING QUESTIONS:
- Use search_ics tool for ALL ICS service questions (passport, visa, PCC, OCI, etc.)
- You have NO knowledge about ICS services - the tool has everything
- Use exact information from tool responses
- Keep times in WORDS (eight thirty A M, not 8:30 AM)
- Be conversational and natural

🎯 CRITICAL DECISION TREE:

User responds → Analyze their response:
├─ Just "hi/hello" quickly? → They INTERRUPTED, didn't hear message
│  └─ Say: "I'm calling about {outbound_message}. Is this a good time?"
│
├─ "Why are you calling?" → They MISSED the message
│  └─ Say: "I'm calling to inform you that {outbound_message}."
│
├─ "Tell me more" → They heard SOMETHING, want details
│  └─ Say: "{outbound_message}. Any questions?"
│
├─ Specific question about passport/visa/etc → They HEARD and understood
│  └─ Use search_ics tool to answer
│
└─ "I'm busy" → They want to end call
   └─ Offer to call back later

🎯 CRITICAL REMINDERS:
- You are the CALLER, not the recipient
- The message you're calling about is: {outbound_message}
- If user seems unaware, deliver the message NOW - don't assume they heard it!
- Be warm, professional, and genuinely helpful

You're a REAL PERSON calling to help, not a scripted robot! 😊
"""

async def initialize_user(flow_manager: FlowManager):
    """Initialize user state for outbound call with specific message"""
    msisdn = flow_manager.state.get("msisdn")
    print("msisdn:", msisdn)
    
    # 🎯 OUTBOUND MESSAGE - What the bot will proactively tell the user
    # This can be passed from the calling system or set here
    outbound_message = flow_manager.state.get("outbound_message", "")
    
    # Example outbound messages:
    if not outbound_message:
        # Default message if none provided
        outbound_message = (
            "your passport application is ready for collection at our Johannesburg office. "
            "You can collect it Monday to Friday between nine A M and three P M. "
            "Please bring your original ID and the application receipt."
        )
    
    flow_manager.state["msisdn"] = msisdn
    flow_manager.state["greeted"] = True  # Bot speaks first in outbound
    flow_manager.state["call_initiated"] = True
    flow_manager.state["outbound_message"] = outbound_message
    flow_manager.state["message_delivered"] = False  # Track if message delivered


def create_initial_node(customer_name: str = "", greeted: bool = True, outbound_message: str = "") -> NodeConfig:
    """
    Create the initial node for OUTBOUND flow.
    Key difference: Bot proactively delivers a specific message first
    """
    
    def success(message: str):
        return {"message": message}, None

    async def deliver_outbound_message(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        🎯 PROACTIVELY DELIVER THE OUTBOUND MESSAGE
        This is called FIRST when the call connects
        """
        # Get the outbound message from state
        outbound_message = flow_manager.state.get("outbound_message", "")
        customer_name = flow_manager.state.get("customer_name", "")
        
        # Build the complete initial message
        greeting = "Hello! This is Indian Consulate Services calling."
        
        # Add customer name if available
        if customer_name:
            greeting = f"Hello {customer_name}! This is Indian Consulate Services calling."
        
        # Combine greeting + purpose + message
        full_message = f"{greeting} I'm calling to inform you that {outbound_message} Do you have any questions about this?"
        
        # Mark message as delivered
        flow_manager.state["message_delivered"] = True
        
        print(f"[OUTBOUND] Delivering message: {full_message[:150]}...")
        
        # Make TTS-safe
        final_message = tts_safe(full_message)
        
        return success(final_message), None

    def tts_safe(message: str) -> str:
        """
        Make a message TTS-friendly and strip ANY technical exposure
        """
        import re
        
        # Remove everything before and including last closing curly brace
        last_closing_brace = message.rfind('}')
        if last_closing_brace != -1:
            if '{' in message[:last_closing_brace + 1]:
                message = message[last_closing_brace + 1:].strip()
        
        # Remove everything before and including </tool_call>
        if '</tool_call>' in message:
            parts = message.split('</tool_call>')
            message = parts[-1].strip()
        
        message = re.sub(r'<tool_call[^>]*>.*?</tool_call>', '', message, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove debugging/technical output
        message = re.sub(r'iNdEx[a-z]*\+*\s*', '', message, flags=re.IGNORECASE)
        message = re.sub(r'tool_search_ics\s*\{[^\}]*\}', '', message, flags=re.IGNORECASE)
        message = re.sub(r'search_ics\s*\{[^\}]*\}', '', message, flags=re.IGNORECASE)
        message = re.sub(r'tool_\s*', '', message, flags=re.IGNORECASE)
        
        # Fix time formats
        num_to_word = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '30': 'thirty', '45': 'forty five'
        }
        
        def replace_numeric_time(match):
            time_str = match.group(1).replace(':', '')
            period = match.group(2) if match.group(2) else ''
            
            if len(time_str) <= 2:
                hour = int(time_str)
                hour_word = num_to_word.get(str(hour), str(hour))
                return f"{hour_word} {period}".strip() if period else hour_word
            elif len(time_str) == 3:
                hour = int(time_str[0])
                minute = int(time_str[1:])
            else:
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
        
        message = re.sub(r'\b(\d{1,4}:?\d{0,2})\s*(AM|PM|am|pm)?\b', replace_numeric_time, message)
        
        # Space out AM/PM
        message = re.sub(r'\b(AM|am)\b', 'A M', message)
        message = re.sub(r'\b(PM|pm)\b', 'P M', message)
        
        # Fix backslash-n
        message = message.replace('\\n\\n', '. ')
        message = message.replace('\\n', ', ')
        message = message.replace('\n\n', '. ')
        message = message.replace('\n', ', ')
        
        # Remove forbidden technical patterns
        forbidden_patterns = [
            r'iNdEx\+\+', r'index\+\+', r'spid', r'searchics', r'search_ics',
            r'tool_search_ics', r'FilterWhere', r'CanBeConvertedToEnglish',
            r'\{[^\}]*query[^\}]*\}', r'\{[^\}]*name[^\}]*\}',
            r'\{[^\}]*arguments[^\}]*\}', r'arguments\s*[:=]',
            r'tool_call', r'function\s*call', r'API\s*call',
            r'score\s*[:=]?\s*\d', r'mPid', r'sPid', r'isPid', r'spNet',
            r'""\s*:\s*"searchics"', r'""\s*:\s*"search_ics"',
            r'""\s*:\s*"query"', r'""\s*:\s*["\'][^"\']*["\']',
        ]
        
        for pattern in forbidden_patterns:
            message = re.sub(pattern, '', message, flags=re.IGNORECASE)
        
        # Remove curly braces, brackets, quotes
        message = re.sub(r'[\{\}\[\]""]', '', message)
        
        # Ensure sentence ends with period
        message = message.strip()
        if message and not message.endswith(('.', '!', '?')):
            message = message + '.'
        
        return message
    
    def clean_query_lightly(text: str) -> str:
        """Minimal cleanup - preserve user intent"""
        import re
        text = text.strip()
        
        # Remove leading greetings only
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[,\s]+',
        ]
        for pattern in greeting_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove filler words
        text = re.sub(r'\b(um|uh|sort of)\b', '', text, flags=re.IGNORECASE)
        
        return text
    
    async def reshape_with_huggingface(api_answer: str, user_question: str, customer_name: str = "") -> str:
        """
        Use HuggingFace to reshape API answer into conversational speech with follow-up questions
        """
        import re
        
        try:
            loop = asyncio.get_event_loop()
            
            system_prompt = """You are a friendly and professional customer service representative for Indian Consulate Services.
Your job is to take a factual answer from the knowledge base and reshape it into natural, conversational speech.

IMPORTANT RULES:
1. Use simple, conversational language - speak like a real person
2. Keep the factual information exactly as provided - don't make up anything
3. Add ONE relevant follow-up question at the end
4. Use the customer's name naturally if provided (don't overuse it)
5. Keep responses concise - 2-3 sentences max, plus a follow-up
6. Times should be written as words (eight thirty AM, not 8:30 AM)
7. Never mention API or database
8. Be helpful and empathetic

REMEMBER:
1. Start with a varied transition - "Thanks for waiting", "Thanks for your patience", "I appreciate your patience"
2. Never greet again after conversation starts

Format as natural speech only - no JSON, no formatting, just conversational text."""
            
            user_prompt = f"""The knowledge base provided this answer:
{api_answer}

The customer asked: "{user_question}"
Customer name: {customer_name if customer_name else "(not provided)"}

Please reshape this into natural, conversational speech with ONE friendly follow-up question."""
            
            def call_huggingface(system: str, user: str):
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
            
            hf_response = await loop.run_in_executor(
                None, 
                call_huggingface, 
                system_prompt, 
                user_prompt
            )
            
            reshaped_answer = hf_response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            print(f"[HUGGINGFACE RESHAPE] Original: {api_answer[:100]}...")
            print(f"[HUGGINGFACE RESHAPE] Reshaped: {reshaped_answer[:150]}...")
            
            reshaped_answer = re.sub(r'[\{\}\[\]"\']', '', reshaped_answer)
            
            return reshaped_answer if reshaped_answer else api_answer
            
        except Exception as e:
            print(f"[HUGGINGFACE RESHAPE] Error: {e} - using original answer")
            return api_answer
    
    async def tool_search_ics(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """Search ICS knowledge base"""
        
        original_user_input = flow_manager.state.get("last_user_text", "")
        
        llm_query = (
            args.get("query")
            or args.get("message")
            or args.get("question")
            or args.get("text")
            or ""
        )
        print("llm query:", llm_query)
        
        user_question = llm_query if llm_query else original_user_input
        user_question = clean_query_lightly(user_question) if user_question else ""
        
        if not user_question or user_question.startswith("ICS INFORMATION:") or user_question.startswith("NO_INFORMATION_FOUND"):
            return success("Please ask a specific question so I can search the knowledge base."), None
        
        print(f"\n[ICS SEARCH] Using user question: {user_question}")
        
        # Check for gibberish/unclear voice
        user_question_stripped = user_question.strip()
        short_valid_words = ['ok', 'no', 'yes', 'hi', 'bye', 'the', 'pcc', 'oci']
        
        if len(user_question_stripped) < 3 and user_question_stripped.lower() not in short_valid_words:
            print(f"[ICS SEARCH] Query too short - possible STT error: {user_question}")
            return success(
                "I'm sorry, I couldn't hear you clearly. Could you please repeat your question?"
            ), None
        
        # Check for gibberish patterns
        vowel_count = sum(1 for char in user_question_stripped.lower() if char in 'aeiou')
        total_letters = sum(1 for char in user_question_stripped if char.isalpha())
        
        if total_letters > 5 and (vowel_count / total_letters) < 0.2:
            print(f"[ICS SEARCH] Gibberish detected - possible STT error: {user_question}")
            return success(
                "I'm sorry, I couldn't understand that clearly. Could you please repeat your question?"
            ), None
        
        # Out-of-scope check
        user_question_lower = user_question.lower()
        
        out_of_scope_keywords = [
            'bank', 'banking', 'account', 'loan', 'credit card', 'debit card',
            'atm', 'transfer money', 'payment', 'balance', 'statement',
            'fnb', 'capitec', 'standard bank', 'absa', 'nedbank',
            'sim card', 'mobile network', 'data bundle', 'airtime', 'vodacom',
            'mtn', 'cell c', 'telkom', 'phone number', 'recharge',
            'hotel', 'flight', 'ticket', 'accommodation', 'reservation',
            'airbnb', 'guesthouse', 'lodge',
            'shopping', 'delivery', 'courier', 'uber', 'bolt', 'taxi', 'restaurant',
            'food delivery', 'groceries',
            'drivers license', 'id book', 'birth certificate', 'marriage certificate',
            'vehicle registration', 'tax', 'sars', 'home affairs',
            'job', 'employment', 'work permit', 'business registration',
            'school admission', 'university', 'medical', 'doctor', 'hospital'
        ]
        
        is_out_of_scope = any(keyword in user_question_lower for keyword in out_of_scope_keywords if len(keyword) > 3)
        
        if is_out_of_scope:
            print(f"[ICS SEARCH] OUT OF SCOPE - Not an ICS service: {user_question}")
            return success(
                "I apologize, but that service is not provided by Indian Consulate Services. "
                "I can assist you with Passport Services, Police Clearance Certificate, "
                "Overseas Citizen of India, Track Application Status, SOS Emergency Assistance, "
                "Report Incident, Visa, Miscellaneous services and Contact Information. "
                "Which service can I help you with today?"
            ), None
        
        try:
            loop = asyncio.get_event_loop()
            
            def call_search_api(query_text):
                response = requests.post(
                    ICS_SEARCH_API,
                    json={"query": query_text},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
            
            api_response = await loop.run_in_executor(None, call_search_api, user_question)
            
            print(f"[ICS SEARCH] API Response: {api_response}")
            
            if not api_response.get("success"):
                print("[ICS SEARCH] No results found from API")
                return success(
                    "I'm sorry, I don't have information on that specific topic. "
                    "I can assist you with Passport Services, Police Clearance Certificate, "
                    "Overseas Citizen of India, Track Application Status, SOS Emergency Assistance, "
                    "Report Incident, Visa, Miscellaneous services and Contact Information. "
                    "Which service can I help you with today?"
                ), None
            
            service_name = api_response.get("service", "")
            answer = api_response.get("answer", "")
            score = api_response.get("score", 0.0)
            
            print(f"[ICS SEARCH] Service: {service_name}, Score: {score:.2f}")
            
            if score < 0.3:
                print(f"[ICS SEARCH] Low confidence score ({score:.2f}) - rejecting answer")
                return success(
                    "I'm sorry, I don't have information on that specific topic. "
                    "I can assist you with Passport Services, Police Clearance Certificate, "
                    "Overseas Citizen of India, Track Application Status, SOS Emergency Assistance, "
                    "Report Incident, Visa, Miscellaneous services and Contact Information. "
                    "Which service can I help you with today?"
                ), None
            
            if not answer:
                return success(
                    "I'm sorry, I couldn't find specific information on that. "
                    "Would you like to know about other services?"
                ), None
            
            # Get customer name for personalization
            customer_name = flow_manager.state.get("customer_name", "")
            
            # Reshape answer with HuggingFace
            reshaped_answer = await reshape_with_huggingface(answer, user_question, customer_name)
            
            # Make TTS-safe
            final_answer = tts_safe(reshaped_answer)
            
            print(f"[ICS SEARCH] Final answer: {final_answer[:150]}...")
            
            return success(final_answer), None
            
        except requests.exceptions.Timeout:
            print("[ICS SEARCH] API timeout")
            return success(
                "I'm sorry, the search is taking longer than expected. Please try again."
            ), None
        except requests.exceptions.ConnectionError:
            print("[ICS SEARCH] Connection error")
            return success(
                "I'm sorry, I'm having trouble connecting to the knowledge base. Please try again."
            ), None
        except Exception as e:
            print(f"[ICS SEARCH] Error: {e}")
            return success(
                "I'm sorry, something went wrong. Please try asking again."
            ), None
    
    async def end_conversation(args: FlowArgs, flow_manager: FlowManager):
        """End the conversation gracefully"""
        print("[END CONVERSATION] User said goodbye")
        return success(
            "Thank you for contacting Indian Consulate Services. Have a great day! Goodbye."
        ), None
    
    # 🚨 OUTBOUND-SPECIFIC: Use system prompt for outbound calls
    # The greeting is delivered via TextFrame directly, not by LLM
    if not outbound_message:
        outbound_message = "your passport application is ready for collection"
    
    # 🎯 CRITICAL FIX: Add MINIMAL context to task_messages
    # Don't include the full message - the user may have interrupted!
    # Just record that bot ATTEMPTED to deliver the greeting
    initial_context = f"I started calling to inform the user about: {outbound_message}"
    
    system_prompt_with_context = OUTBOUND_SYSTEM_PROMPT.format(outbound_message=outbound_message)
    
    return NodeConfig(
        system_prompt=system_prompt_with_context,
        # 🚨 CRITICAL: Minimal task_messages - don't assume user heard everything
        task_messages=[
            {
                "role": "assistant",
                "content": initial_context
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="search_ics",
                handler=tool_search_ics,
                description="Search the Indian Consulate Services knowledge base for information about follow-up questions",
                properties={
                    "query": {
                        "type": "string",
                        "description": "The user's question to search for"
                    }
                },
                required=["query"]
            ),
            FlowsFunctionSchema(
                name="end_conversation",
                handler=end_conversation,
                description="End the conversation when user says goodbye or has no more questions",
                properties={},
                required=[]
            )
        ],
        respond_immediately=False  # 🚨 Greeting already sent via TextFrame, now wait for user
    )
