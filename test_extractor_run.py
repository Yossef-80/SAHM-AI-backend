import asyncio
from app.ai.tools.get_reviews_playwright import extract_reviews_from_url

async def run():
    res = await extract_reviews_from_url(
        "https://www.google.com/maps/place/Qedra+Smart+Village/@30.0830914,30.9012018,43935m/data=!3m1!1e3!4m6!3m5!1s0x14585bbb0fc5c845:0x78a4ee0f662f2f0a!8m2!3d30.0795663!4d31.0136127!16s%2Fg%2F11rzsxdglx?entry=ttu&g_ep=EgoyMDI2MTAwNy4wIKXMDSoASAFQAw%3D%3D",
        max_reviews=5,
        headless=True
    )
    import json
    print(json.dumps(res, ensure_ascii=False, indent=2))

asyncio.run(run())
