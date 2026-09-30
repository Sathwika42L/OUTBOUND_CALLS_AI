"""
Test script to verify system prompt formatting with actual values
"""

from outbound_flow import OUTBOUND_SYSTEM_PROMPT

# Test with sample values
customer_name = "Rajesh Kumar"
outbound_message = "your passport application is ready for collection at our Johannesburg office. You can collect it Monday to Friday between nine A M and three P M. Please bring your original ID and the application receipt."

# Format the system prompt
formatted_prompt = OUTBOUND_SYSTEM_PROMPT.format(
    CustomerName=customer_name,
    outbound_message=outbound_message
)

print("=" * 100)
print("FORMATTED SYSTEM PROMPT:")
print("=" * 100)
print(formatted_prompt)
print("=" * 100)

# Check if placeholders are filled
if "{CustomerName}" in formatted_prompt or "{outbound_message}" in formatted_prompt:
    print("\n❌ ERROR: Placeholders not replaced!")
    print(f"  - CustomerName present: {'{CustomerName}' in formatted_prompt}")
    print(f"  - outbound_message present: {'{outbound_message}' in formatted_prompt}")
else:
    print("\n✅ SUCCESS: All placeholders replaced correctly!")
    print(f"  - Customer Name: {customer_name}")
    print(f"  - Message length: {len(outbound_message)} chars")

# Count how many times actual values appear
print(f"\n📊 Stats:")
print(f"  - '{customer_name}' appears: {formatted_prompt.count(customer_name)} times")
print(f"  - Message text appears: {formatted_prompt.count(outbound_message[:50])} time(s)")
print(f"  - Total length: {len(formatted_prompt)} characters")
