
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
ICS_SEARCH_API = "https://sacs-backend.vibhohcm.com/api/knowledge/search"
ICS_AGENT_AVAILABLE_API = "https://sacs-backend.vibhohcm.com/api/config/agent-availability"

# HuggingFace configuration
HF_API_URL = "https://router.huggingface.co/v1"
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# 🚨 OUTBOUND-SPECIFIC SYSTEM PROMPT
# Bot speaks FIRST and PROACTIVELY delivers specific information
OUTBOUND_SYSTEM_PROMPT = """
You are a proactive AI assistant for Indian Consulate Services (ICS) making OUTBOUND calls.

🎯 OUTBOUND CALL PURPOSE:
- YOU are calling to PROACTIVELY INFORM the user about specific information
- YOU speak FIRST with greeting, purpose, and the specific message
- YOU deliver the intended message IMMEDIATELY after greeting
- YOU then offer to answer any questions

📢 YOUR OUTBOUND MESSAGE DELIVERY:
When making an outbound call:

1. GREETING + PURPOSE + MESSAGE (all together):
   Example: "Hello! This is Indian Consulate Services calling. 
            I'm calling to inform you that [SPECIFIC MESSAGE/UPDATE/REMINDER].
            Do you have any questions about this?"

2. TYPES OF PROACTIVE MESSAGES:
   - Application status updates: "Your passport application is ready for collection"
   - Appointment reminders: "Your appointment is scheduled for tomorrow at nine A M"
   - Document expiry alerts: "Your passport expires in thirty days"
   - Important announcements: "New visa requirements are now in effect"
   - Service updates: "Our office hours have changed to eight thirty A M"
   - Emergency notifications: "Important security advisory for Indian citizens"

3. MESSAGE STRUCTURE:
   Step 1: Greet → "Hello! This is Indian Consulate Services calling."
   Step 2: State purpose → "I'm calling to inform you that..."
   Step 3: Deliver message → [SPECIFIC INFORMATION]
   Step 4: Offer help → "Do you have any questions?" or "Would you like more details?"

🚨 CRITICAL: YOU HAVE LIMITED KNOWLEDGE 🚨
YOU DO NOT KNOW EVERYTHING - You only know what's in:
1. The outbound message you're delivering (from call_context)
2. Information from search_ics tool (for follow-up questions)

YOUR CAPABILITIES:
- Deliver the specific outbound message
- Use search_ics tool to answer follow-up questions
- Provide clear, helpful responses

CRITICAL RULES:

1. FIRST INTERACTION - DELIVER THE MESSAGE:
   → Start with: "Hello! This is Indian Consulate Services calling. I'm calling to inform you that [MESSAGE]."
   → The MESSAGE comes from flow_manager.state["outbound_message"]
   → Deliver it IMMEDIATELY - don't wait for user to ask
   → After delivering, ask: "Do you have any questions about this?"
   → NO TOOL NEEDED for initial message delivery

2. AFTER MESSAGE DELIVERED:
   → If user says "no questions" or "okay" → Thank them and end call
   → If user asks questions → Call search_ics to answer
   → If user says "goodbye" → Call end_conversation

3. FOR FOLLOW-UP QUESTIONS → Call search_ics:
   → You are FORBIDDEN from answering from your own knowledge
   → You MUST call search_ics for ANY question beyond the initial message
   → Only search_ics tool has accurate answers

4. AFTER search_ics returns:
   → Use ONLY the exact information returned
   → Do NOT add your own context or elaboration
   → Speak conversationally but stick to what was returned
   → CRITICAL: Keep times in word format (eight thirty AM, nine AM, etc.)

5. IF search_ics finds nothing:
   → Say "I don't have information on that specific topic"
   → Offer to transfer to a human agent

6. RESPONSE FORMAT:
   → Plain English only
   → No code, no JSON, no technical terms
   → Natural conversational speech
   → Times in words ALWAYS (eight thirty AM, not 8:30 AM)

🚨 CRITICAL RULE — NEVER ASK FOR PERSONAL INFORMATION:
  You do NOT have access to:
    ❌ Application numbers or reference numbers
    ❌ Personal details (name, date of birth, passport number)
    ❌ Application status for specific individuals
    ❌ Personal documents or records

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

════════════════════════════════════════════════════
REMEMBER: 
- YOU call to deliver a SPECIFIC MESSAGE
- Deliver it IMMEDIATELY after greeting
- Then answer follow-up questions with search_ics
- Times in words ALWAYS
- Natural conversational English only
════════════════════════════════════════════════════
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
    Bot waits for user response after greeting is delivered via TextFrame.
    """

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

        if not user_question or user_question.startswith("ICS INFORMATION:") or user_question.startswith("NO_INFORMATION_FOUND"):
            return "Please ask a specific question so I can search the knowledge base.", None

        print(f"\n[ICS SEARCH] Using user question: {user_question}")

        user_question_stripped = user_question.strip()
        short_valid_words = ['ok', 'no', 'yes', 'hi', 'bye', 'the', 'pcc', 'oci']

        if len(user_question_stripped) < 3 and user_question_stripped.lower() not in short_valid_words:
            return "I'm sorry, I couldn't hear you clearly. Could you please repeat your question?", None

        vowel_count = sum(1 for char in user_question_stripped.lower() if char in 'aeiou')
        total_letters = sum(1 for char in user_question_stripped if char.isalpha())

        if total_letters > 5 and (vowel_count / total_letters) < 0.2:
            return "I'm sorry, I couldn't understand that clearly. Could you please repeat your question?", None

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
            return (
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
                return (
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
                return (
                    "I'm sorry, I don't have information on that specific topic. "
                    "I can assist you with Passport Services, Police Clearance Certificate, "
                    "Overseas Citizen of India, Track Application Status, SOS Emergency Assistance, "
                    "Report Incident, Visa, Miscellaneous services and Contact Information. "
                    "Which service can I help you with today?"
                ), None

            if not answer:
                return "I'm sorry, I couldn't find specific information on that. Would you like to know about other services?", None

            customer_name_val = flow_manager.state.get("customer_name", "")
            reshaped_answer = await reshape_with_huggingface(answer, user_question, customer_name_val)
            final_answer = tts_safe(reshaped_answer)

            print(f"[ICS SEARCH] Final answer: {final_answer[:150]}...")

            return final_answer, None

        except requests.exceptions.Timeout:
            return "I'm sorry, the search is taking longer than expected. Please try again.", None
        except requests.exceptions.ConnectionError:
            return "I'm sorry, I'm having trouble connecting to the knowledge base. Please try again.", None
        except Exception as e:
            print(f"[ICS SEARCH] Error: {e}")
            return "I'm sorry, something went wrong. Please try asking again.", None

    async def end_conversation(args: FlowArgs, flow_manager: FlowManager):
        """End the conversation gracefully"""
        print("[END CONVERSATION] User said goodbye")
        return (
            "Thank you for contacting Indian Consulate Services. Have a great day! Goodbye."
        ), None

    if not outbound_message:
        outbound_message = "your passport application is ready for collection"

    system_prompt_with_context = OUTBOUND_SYSTEM_PROMPT + f"""

🎯 OUTBOUND CALL CONTEXT:
The initial greeting has ALREADY been delivered to the user:
"Hello! This is Indian Consulate Services calling. I'm calling to inform you that {outbound_message}. Do you have any questions about this?"

YOUR JOB NOW:
1. Wait for user's response to the greeting
2. If user says "no", "okay", "that's all" → Thank them and say goodbye
3. If user asks questions → Use search_ics tool to answer
4. If user says "goodbye" → Call end_conversation immediately
5. DO NOT repeat the greeting - it was already delivered
6. DO NOT ask "how can I help you" - you already told them why you called

CRITICAL RULES:
- The outbound message has BEEN delivered already
- Focus on answering follow-up questions about the message
- Use search_ics for ANY service-related questions
- Keep responses focused on Indian Consulate Services ONLY
- DO NOT act like a generic AI assistant

EXAMPLE RESPONSES:
User: "What documents do I need?"
You: [Call search_ics("collection documents")] → Answer with result

User: "Okay, thank you"
You: "Thank you for your time. Have a great day!"

User: "No questions"
You: "Alright, thank you. Have a great day! Goodbye."
"""

    return NodeConfig(
        # ✅ role_message replaces system_prompt — sent as LLM system instruction
        role_message=system_prompt_with_context,
        # ✅ task_messages is required — tells the LLM what to do in this node
        task_messages=[
            {
                "role": "user",
                "content": (
                    f"The outbound message has already been delivered: '{outbound_message}'. "
                    "Wait for the user's response. If they have questions, use search_ics. "
                    "If they say goodbye or have no questions, use end_conversation."
                ),
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
        # ✅ Bot waits for user — greeting already sent via TextFrame
        respond_immediately=False,
    )