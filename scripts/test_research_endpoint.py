import asyncio
import httpx
import json

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("1. Testing GET /api/v1/research/items...")
        res = await client.get("http://localhost:8000/api/v1/research/items", headers={"X-Workspace-Id": "biz_d335699704e740219870"})
        print("Status:", res.status_code)
        print("Items before:", len(res.json()) if res.status_code == 200 else res.text)

        print("\n2. Testing POST /api/v1/research/run for 'Tie House'...")
        payload = {
            "tool": "analyze_competitor",
            "query": "Tie House"
        }
        res2 = await client.post("http://localhost:8000/api/v1/research/run", json=payload, headers={"X-Workspace-Id": "biz_d335699704e740219870"})
        print("Status:", res2.status_code)
        if res2.status_code == 200:
            data = res2.json()
            print("Title:", data.get("title"))
            print("Summary preview:\n", data.get("summary")[:400])
            print("Saved to memory:", data.get("saved_to_memory"))
        else:
            print("Error:", res2.text)

        print("\n3. Testing GET /api/v1/research/items after run...")
        res3 = await client.get("http://localhost:8000/api/v1/research/items", headers={"X-Workspace-Id": "biz_d335699704e740219870"})
        print("Items after:", len(res3.json()) if res3.status_code == 200 else res3.text)

asyncio.run(main())
