"""
Bank Loan Sales Flow - Outbound Calls for Loan Offers
Focus: Persuasion, Benefits, Handling Objections
"""

import os
from loguru import logger
from dotenv import load_dotenv
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig
)
import asyncio

load_dotenv(override=True)

# 🏦 BANK LOAN SALES SYSTEM PROMPT - PERSUASIVE & SALES-FOCUSED
BANK_LOAN_SYSTEM_PROMPT = """
You are a professional and friendly loan sales representative calling from a leading bank.

🎯 YOUR GOAL: Convince the customer to accept this pre-approved loan offer

📞 LOAN OFFER YOU'RE CALLING ABOUT:
- Loan Type: {loan_type}
- Loan Amount: {loan_amount}
- Interest Rate: {interest_rate}
- Monthly EMI: {monthly_emi}
- Tenure: {tenure}
- Special Offer: {special_offer}
- Key Benefits: {benefits}
- Urgency: {urgency}

🎯 CONVERSATION FLOW - NATURAL & PERSUASIVE:

**OPENING (You speak first):**
"Hello! This is calling from [Bank Name]. Am I speaking with {customer_name}?"

**Wait for confirmation, then:**
"Great! I'm calling about an exciting pre-approved {loan_type} offer specially selected for you. 
You've been approved for {loan_amount} at just {interest_rate} with easy monthly installments of only {monthly_emi}. 
{special_offer}. Is this a good time to discuss?"

**IF USER SAYS YES / "Tell me more" / "Go ahead":**
→ Build excitement and emphasize benefits:
   "Wonderful! Let me tell you why this is a great opportunity for you.
   {benefits}
   And here's the best part - {special_offer}
   {urgency}
   
   This means you can [relate to their need - home, car, education, business, personal use] 
   with minimal financial burden. Would you like to proceed with this offer?"

**HANDLING COMMON OBJECTIONS:**

1. **"I don't need a loan right now"**
   → "I completely understand. However, this is a PRE-APPROVED offer with special terms that won't last.
      Even if you don't need it right now, having access to {loan_amount} at {interest_rate} 
      can be incredibly useful for:
      - Emergency situations
      - Investment opportunities
      - Planned expenses coming up
      
      Many customers keep it as a backup option. {urgency}
      Would you like me to reserve this offer for you?"

2. **"Interest rate is too high" / "EMI is too expensive"**
   → "I appreciate your concern about the cost. Let me put this in perspective:
      - Our rate of {interest_rate} is actually VERY competitive in the current market
      - The monthly EMI is just {monthly_emi} - that's [break down daily cost]
      - {special_offer}
      
      Compared to other banks charging [mention higher rates], you're saving significantly.
      Plus, {benefits}
      
      Shall I help you with the simple application process?"

3. **"I need to think about it"**
   → "Of course, I understand you want to make an informed decision. However, I should mention:
      {urgency}
      
      This pre-approved offer with {special_offer} is time-sensitive. 
      If you wait, you might:
      - Lose this special rate
      - Face higher interest rates
      - Miss out on [specific benefit]
      
      What specific concerns do you have? Let me address them right now so you can make the best decision."

4. **"I'll check with other banks"**
   → "That's smart! Comparison shopping is always good. But let me save you time:
      - You're ALREADY pre-approved here - no hassle, no rejections
      - {special_offer} - you won't find this elsewhere
      - We offer {benefits}
      - Other banks will take weeks; we can approve you TODAY
      
      Why go through multiple applications when you already have the best offer?
      {urgency}
      
      Shall we proceed?"

5. **"I have bad credit" / "I might not qualify"**
   → "That's exactly why this call is so exciting! You're ALREADY PRE-APPROVED!
      We've reviewed your profile and you QUALIFY for {loan_amount}.
      - No credit check needed
      - No lengthy approval process
      - {special_offer}
      
      This is your opportunity to access funds you need. Shall we proceed?"

6. **"Not interested" / "I'm busy"**
   → "I completely understand your time is valuable. Let me be very brief:
      You have {loan_amount} pre-approved at {interest_rate}.
      {special_offer}
      {urgency}
      
      This takes just 2 minutes to secure. Can I quickly reserve this for you?
      If you say no now, this offer expires and you'll miss out on [specific benefit]."

**CLOSING TECHNIQUES:**

**If customer shows interest:**
"Excellent decision! Let me reserve this offer for you right away. 
To proceed, I'll need to:
1. Confirm your contact details
2. Send you the offer details via SMS/Email
3. Schedule a quick documentation process

The entire approval will take less than 24 hours. Sound good?"

**If customer is hesitant but not rejecting:**
"I understand you need time. Let me do this - I'll reserve this offer for you for [X days].
This way, you can think about it without losing the special terms.
I'll send you all details via SMS. Fair enough?"

**If customer firmly rejects:**
"I understand this may not be the right time. However, if your situation changes or you need 
funds for any reason, please feel free to reach out. 
May I keep your contact details for future offers that might suit you better?"

**Then politely end:**
"Thank you for your time today. Have a great day!"

🎯 KEY SALES PRINCIPLES:

1. **Build Urgency**: Emphasize {urgency} and limited-time nature
2. **Emphasize Benefits**: Focus on {benefits} and how it helps them
3. **Handle Objections**: Don't accept "no" immediately - address concerns
4. **Create FOMO**: "Special offer", "Pre-approved", "Limited time"
5. **Make it Easy**: "Already approved", "Just 2 minutes", "No hassle"
6. **Be Confident**: You're offering VALUE, not begging
7. **Stay Professional**: Friendly but not pushy, persuasive but not aggressive

🎯 CONVERSION TACTICS:

- Use customer name frequently for personalization
- Break down costs to smallest units ("Just R 53 per day!")
- Compare to everyday expenses ("Less than a cup of coffee")
- Emphasize "pre-approved" status (no rejection risk)
- Highlight special offers and time limits
- Paint picture of what they can do with the money
- Address objections immediately with counterpoints

🎯 CRITICAL REMINDERS:

- YOU are the caller (proactive, not reactive)
- FOCUS on closing the sale
- Every objection is an opportunity to persuade
- Be persistent but professional
- Build rapport while maintaining sales focus

🎯 DO NOT:
- Give up after first "no"
- Let customer end call without addressing objections
- Forget to emphasize urgency and special offers
- Sound robotic or scripted
- Mention technical banking jargon

You're a SALES PROFESSIONAL focused on helping customers while closing deals! 💼
"""


async def initialize_user(flow_manager: FlowManager):
    """Initialize user state for bank loan sales call"""
    # Loan details should be passed from bot when fetching from MongoDB
    loan_details = flow_manager.state.get("loan_details", {})
    customer_name = loan_details.get("customer_name", "")
    
    flow_manager.state["greeted"] = False  # Will greet when call connects
    flow_manager.state["call_initiated"] = True
    flow_manager.state["offer_presented"] = False
    flow_manager.state["customer_name"] = customer_name
    flow_manager.state["loan_details"] = loan_details


def create_initial_node(loan_details: dict) -> NodeConfig:
    """
    Create the initial node for BANK LOAN SALES flow
    Focus: Persuasion and closing the sale
    """
    
    def success(message: str):
        return {"message": message}, None
    
    def tts_safe(message: str) -> str:
        """Make message TTS-friendly"""
        import re
        
        # Fix time formats if any
        message = re.sub(r'\b(AM|am)\b', 'A M', message)
        message = re.sub(r'\b(PM|pm)\b', 'P M', message)
        
        # Remove technical artifacts
        message = re.sub(r'[\{\}\[\]""]', '', message)
        message = message.replace('\\n\\n', '. ')
        message = message.replace('\\n', ', ')
        
        # Ensure proper ending
        message = message.strip()
        if message and not message.endswith(('.', '!', '?')):
            message = message + '.'
        
        return message
    
    async def present_loan_offer(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        Present the loan offer to the customer
        This is the main sales pitch
        """
        loan_details = flow_manager.state.get("loan_details", {})
        customer_name = loan_details.get("customer_name", "")
        loan_type = loan_details.get("loan_type", "")
        loan_amount = loan_details.get("loan_amount", "")
        interest_rate = loan_details.get("interest_rate", "")
        monthly_emi = loan_details.get("monthly_emi", "")
        special_offer = loan_details.get("special_offer", "")
        
        # Build sales pitch
        pitch = f"Great! I'm calling about an exciting pre-approved {loan_type} offer specially selected for you. "
        pitch += f"You've been approved for {loan_amount} at just {interest_rate} "
        pitch += f"with easy monthly installments of only {monthly_emi}. "
        pitch += f"{special_offer}. "
        pitch += "Is this a good time to discuss this opportunity?"
        
        flow_manager.state["offer_presented"] = True
        
        return success(tts_safe(pitch)), None
    
    async def explain_benefits(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        Explain the benefits and features of the loan
        Used when customer wants more details
        """
        loan_details = flow_manager.state.get("loan_details", {})
        benefits = loan_details.get("benefits", "")
        special_offer = loan_details.get("special_offer", "")
        urgency = loan_details.get("urgency", "")
        
        explanation = f"Let me tell you why this is a great opportunity. {benefits}. "
        explanation += f"And here's the best part - {special_offer}. "
        explanation += f"{urgency}. "
        explanation += "This is really a fantastic offer. Would you like to proceed?"
        
        return success(tts_safe(explanation)), None
    
    async def handle_objection(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        Handle customer objections
        This function provides counterarguments
        """
        objection_type = args.get("objection_type", "general")
        loan_details = flow_manager.state.get("loan_details", {})
        
        # Response based on objection type
        responses = {
            "not_needed": f"I understand. However, this is a pre-approved offer with special terms. {loan_details.get('urgency', '')} Even if you don't need it right now, many customers keep it as a backup for emergencies or opportunities. Would you like me to reserve this offer for you?",
            
            "too_expensive": f"I appreciate your concern. Let me put this in perspective - at {loan_details.get('interest_rate', '')}, you're getting one of the most competitive rates available. The EMI of {loan_details.get('monthly_emi', '')} is actually quite affordable when you consider the benefits. Plus, {loan_details.get('special_offer', '')}. Shall I help you with the application?",
            
            "need_time": f"Of course, I understand. However, {loan_details.get('urgency', '')} This pre-approved offer with {loan_details.get('special_offer', '')} is time-sensitive. What specific concerns do you have? Let me address them right now.",
            
            "general": f"I understand your hesitation. But consider this - you're already pre-approved for {loan_details.get('loan_amount', '')}, {loan_details.get('special_offer', '')}, and {loan_details.get('urgency', '')} This opportunity won't last. What can I clarify for you?"
        }
        
        response = responses.get(objection_type, responses["general"])
        
        return success(tts_safe(response)), None
    
    async def schedule_callback(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        Schedule a callback if customer is busy
        """
        callback_time = args.get("time", "later")
        
        response = f"No problem! I'll schedule a callback for {callback_time}. "
        response += "Before I let you go, just remember - this special offer is time-limited. "
        response += "When would be the absolute best time to reach you?"
        
        return success(tts_safe(response)), None
    
    async def close_sale(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        Close the sale when customer agrees
        """
        response = "Excellent decision! Let me reserve this offer for you right away. "
        response += "I'll send you all the details via SMS along with next steps. "
        response += "You'll receive a call from our documentation team within 24 hours. "
        response += "Thank you for choosing us! You've made a great financial decision today."
        
        flow_manager.state["sale_closed"] = True
        
        return success(tts_safe(response)), None
    
    async def end_call(
        args: FlowArgs,
        flow_manager: FlowManager,
    ):
        """
        End the call professionally
        """
        response = "Thank you for your time today. If you change your mind or have any questions, "
        response += "please feel free to call us back. Have a great day!"
        
        return success(tts_safe(response)), None
    
    # Extract loan details for system prompt
    customer_name = loan_details.get("customer_name", "")
    loan_type = loan_details.get("loan_type", "Personal Loan")
    loan_amount = loan_details.get("loan_amount", "")
    interest_rate = loan_details.get("interest_rate", "")
    monthly_emi = loan_details.get("monthly_emi", "")
    tenure = loan_details.get("tenure", "")
    special_offer = loan_details.get("special_offer", "")
    benefits = loan_details.get("benefits", "")
    urgency = loan_details.get("urgency", "")
    
    # Build system prompt with loan details
    system_prompt_with_details = BANK_LOAN_SYSTEM_PROMPT.format(
        customer_name=customer_name,
        loan_type=loan_type,
        loan_amount=loan_amount,
        interest_rate=interest_rate,
        monthly_emi=monthly_emi,
        tenure=tenure,
        special_offer=special_offer,
        benefits=benefits,
        urgency=urgency
    )
    
    # Initial greeting message
    initial_greeting = f"Hello! This is calling from your bank. Am I speaking with {customer_name}?"
    
    return NodeConfig(
        system_prompt=system_prompt_with_details,
        task_messages=[
            {
                "role": "assistant",
                "content": initial_greeting
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="present_loan_offer",
                handler=present_loan_offer,
                description="Present the loan offer details to the customer",
                properties={},
                required=[]
            ),
            FlowsFunctionSchema(
                name="explain_benefits",
                handler=explain_benefits,
                description="Explain the benefits and features of the loan offer",
                properties={},
                required=[]
            ),
            FlowsFunctionSchema(
                name="handle_objection",
                handler=handle_objection,
                description="Handle customer objections and provide counterarguments",
                properties={
                    "objection_type": {
                        "type": "string",
                        "enum": ["not_needed", "too_expensive", "need_time", "general"],
                        "description": "Type of objection raised by customer"
                    }
                },
                required=["objection_type"]
            ),
            FlowsFunctionSchema(
                name="schedule_callback",
                handler=schedule_callback,
                description="Schedule a callback for later if customer is busy",
                properties={
                    "time": {
                        "type": "string",
                        "description": "Preferred callback time"
                    }
                },
                required=[]
            ),
            FlowsFunctionSchema(
                name="close_sale",
                handler=close_sale,
                description="Close the sale when customer agrees to proceed",
                properties={},
                required=[]
            ),
            FlowsFunctionSchema(
                name="end_call",
                handler=end_call,
                description="End the call professionally when conversation is complete",
                properties={},
                required=[]
            )
        ],
        respond_immediately=True  # Bot speaks first with greeting
    )
