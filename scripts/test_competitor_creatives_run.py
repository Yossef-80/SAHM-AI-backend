import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')

base_url = "http://localhost:8000/api/v1"
headers = {
    "X-Workspace-Id": "biz_d335699704e740219870",
    "Content-Type": "application/json"
}

print("Testing POST /research/run for 'analyze_competitor_creatives' (Tie House)...")
res = requests.post(f"{base_url}/research/run", json={"tool": "analyze_competitor_creatives", "query": "Tie House"}, headers=headers)
print("Status:", res.status_code)
data = res.json()
print("Title:", data.get("title"))
print("Saved to memory:", data.get("saved_to_memory"))
print("\nSources referenced:")
for s in data.get("details", {}).get("sources", []):
    print(" -", s.get("title"), "|", s.get("url"))

print("\n--- Summary Report Preview ---\n")
print(data.get("summary")[:1200])
