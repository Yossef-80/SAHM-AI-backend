import sys
import requests
import json

# Force UTF-8 on Windows console output
sys.stdout.reconfigure(encoding='utf-8')

base_url = "http://localhost:8000/api/v1"
headers = {
    "X-Workspace-Id": "biz_d335699704e740219870",
    "Content-Type": "application/json"
}

# 1. Test Typo Query: "Tiee Hose"
print("=========================================")
print("1. Testing Typo Query: 'Tiee Hose'")
r1 = requests.post(f"{base_url}/research/run", json={"tool": "analyze_competitor", "query": "Tiee Hose"}, headers=headers)
print("Status:", r1.status_code)
d1 = r1.json()
print("Title:", d1.get("title"))
print("Saved to memory:", d1.get("saved_to_memory"))
print("Verification status:", d1.get("details", {}).get("verification_status"))
print("Summary preview (first 250 chars):\n", d1.get("summary")[:250])
print("\nSources referenced:", len(d1.get("details", {}).get("sources", [])))
for s in d1.get("details", {}).get("sources", []):
    print(" -", s.get("title"), "|", s.get("url"))

# 2. Test Fake Non-Existent Entity: "asdfghjkl9999fake"
print("\n=========================================")
print("2. Testing Fake/Invalid Query: 'asdfghjkl9999fake'")
r2 = requests.post(f"{base_url}/research/run", json={"tool": "analyze_competitor", "query": "asdfghjkl9999fake"}, headers=headers)
print("Status:", r2.status_code)
d2 = r2.json()
print("Title:", d2.get("title"))
print("Saved to memory:", d2.get("saved_to_memory"))
print("Summary:\n", d2.get("summary"))
