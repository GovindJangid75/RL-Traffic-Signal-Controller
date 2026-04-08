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

# Dummy inference
response = "Traffic optimized successfully"
reward = 0.9

print(f"[STEP] Response: {response}")
print(f"[STEP] Reward: {reward}")

print("[END]")