import os
import requests
from dotenv import load_dotenv

load_dotenv()

r = requests.get(
    "https://router.huggingface.co/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
)

print(r.status_code)
print(r.text)
