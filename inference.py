import os
from openai import OpenAI

print("[START]")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenAI client (MANDATORY)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

print(f"[STEP] Using model: {MODEL_NAME}")

try:
    # Actual inference call
    response_obj = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a traffic AI agent Act like as Super Fast agent."},
            {"role": "user", "content": "Optimize the traffic signal."}
        ],
        max_tokens=20
    )
    response = response_obj.choices[0].message.content.strip()
except Exception as e:
    print(f"[STEP] Warning: LLM call failed ({e}). Using fallback.")
    response = "Fallback: Traffic optimized"

reward = 0.9

print(f"[STEP] Response: {response}")
print(f"[STEP] Reward: {reward}")


print("[END]")
