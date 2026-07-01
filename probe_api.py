"""
Probe HYROX API using Playwright to capture real network requests.
Runs headlessly, prints the actual XHR calls and response format.
"""
import asyncio
import json
from playwright.async_api import async_playwright

TARGET = (
    "https://results.hyrox.com/season-9/"
    "?pid=start&event=HPRO_LR3MS4JI16AA"
    "&event_main_group=2026+Jakarta"
    "&pidp=ranking_nav&ranking=time_finish_netto"
)

async def main():
    captured = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        async def handle_response(response):
            url = response.url
            if "content=ajax2" in url and "func=getLeaderboard" in url:
                try:
                    body = await response.json()
                    captured.append({"url": url, "body": body})
                    print(f"\n=== Captured: {url[:120]} ===")
                    # Print containers summary
                    for c in body.get("containers", []):
                        rows = c.get("data", {}).get("rows", [])
                        non_empty = [r for r in rows if r]
                        print(f"  Container '{c['title']}': {len(non_empty)}/{len(rows)} rows with data")
                        if non_empty:
                            print(f"  Sample row: {json.dumps(non_empty[0])[:300]}")
                except Exception as e:
                    print(f"Error parsing {url}: {e}")

        page.on("response", handle_response)

        print(f"Navigating to {TARGET}")
        await page.goto(TARGET, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)  # wait for deferred JS

        if not captured:
            print("No getLeaderboard calls captured. Printing ALL ajax2 calls:")
            # Reset and capture any ajax calls
            all_ajax = []
            async def handle_all(response):
                if "ajax2" in response.url:
                    all_ajax.append(response.url)
            page.on("response", handle_all)
            await asyncio.sleep(2)
            for u in all_ajax:
                print(" ", u)

        # Save first captured response for inspection
        if captured:
            with open("api_response_sample.json", "w") as f:
                # Save only first response, truncated
                sample = captured[0]["body"]
                json.dump(sample, f, indent=2)
            print(f"\nSaved sample response to api_response_sample.json")
            print(f"URL: {captured[0]['url']}")

        await browser.close()

asyncio.run(main())
