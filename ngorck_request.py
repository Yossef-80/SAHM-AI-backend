import requests
import json
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Replace this with your actual ngrok public URL
NGROK_URL = "https://eleven-bulgur-curtsy.ngrok-free.dev" 
NGROK_MODEL_ID = "hf.co/jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF:Q4_K_M"

print("Sending request to Ollama API...")

# 1. Prepare the payload (the data we are sending)
payload = {
    "model": NGROK_MODEL_ID,
    "messages": [
        {"role": "user", "content": "What is the name of current egypt presidant"}
    ],
    "stream": False # Set to True if you want the response word-by-word
}

headers = {
    "ngrok-skip-browser-warning": "true",
}

# 2. Send the POST request to the /api/chat endpoint (stripping any trailing slash)
endpoint = f"{NGROK_URL.rstrip('/')}/api/chat"
response = requests.post(
    endpoint, 
    json=payload,
    headers=headers,
    timeout=None,
)

# 3. Parse and view the response
if response.status_code == 200:
    data = response.json()
    print("\nFull JSON Response from API:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)
    print("Model's Message Only:")
    print(data["message"]["content"])
    print("="*50)
else:
    print(f"Error: {response.status_code}")
    print(response.text)