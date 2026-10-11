"""
Google Maps reviews extractor (Playwright)

Extracts customer reviews and metadata (rating, total reviews count, address)
from any Google Maps place URL using headless Chromium via Playwright.
"""
import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.async_api import Browser, Page, async_playwright

GOOGLE_MAPS_DOMAINS = (
    "google.com",
    "maps.google.com",
    "maps.app.goo.gl",
    "goo.gl",
)
REVIEW_CARD_SELECTOR = "div.jftiEf"
DEBUG_SCREENSHOT = "debug.png"

CONSENT_PATTERN = re.compile(
    r"accept all|reject all|i agree|agree|accept|consent|قبول الكل|رفض الكل|أوافق|موافق|تخصيص الخيارات",
    re.I,
)


def _is_google_maps_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in GOOGLE_MAPS_DOMAINS)


def _clean(value: Any) -> str:
    cleaned = re.sub(r"[\ue000-\uf8ff]", "", str(value or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def _rating_from_label(label: str) -> float | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)", label or "")
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", "."))
        return v if 0 <= v <= 5 else None
    except ValueError:
        return None


def _error(
    url: str,
    code: str,
    message: str,
    started_at: str,
    debug: dict | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "error",
        "source": "google_maps",
        "url": url,
        "place_name": None,
        "overall_rating": None,
        "total_ratings": None,
        "address": None,
        "reviews_fetched": 0,
        "reviews": [],
        "error": {"code": code, "message": message},
        "fetched_at": started_at,
    }
    if debug:
        out["debug"] = debug
    return out


async def _debug_info(page: Page) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        info["final_url"] = page.url
        info["title"] = await page.title()
        await page.screenshot(path=DEBUG_SCREENSHOT, full_page=True)
        info["screenshot"] = DEBUG_SCREENSHOT
        body = (await page.locator("body").inner_text())[:400]
        info["body_preview"] = _clean(body)
    except Exception as e:
        info["debug_error"] = str(e)
    return info


async def _handle_consent(page: Page) -> bool:
    await page.wait_for_timeout(1000)
    for frame in page.frames:
        try:
            btn = frame.get_by_role("button", name=CONSENT_PATTERN).first
            if await btn.count() and await btn.is_visible():
                await btn.click(timeout=3000)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(1000)
                return True
        except Exception:
            continue
    return False


def _normalize_maps_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if "goo.gl" in (parsed.hostname or ""):
        return url

    # Strip volatile experiment parameters (entry, g_ep, etc.) that force Google Maps into limited/lite mode
    query_params = parse_qs(parsed.query)
    new_params: dict[str, list[str]] = {"hl": ["en"]}
    for k in ("authuser",):
        if k in query_params:
            new_params[k] = query_params[k]
    new_query = urlencode(new_params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


async def _dismiss_popups(page: Page):
    try:
        await page.keyboard.press("Escape")
        dismiss_btn = page.locator(
            'button:has-text("Dismiss"), button[aria-label="Dismiss"], button:has-text("Stay signed out")'
        ).first
        if await dismiss_btn.count():
            await dismiss_btn.click(timeout=1000)
    except Exception:
        pass


async def _open_reviews_tab(page: Page) -> bool:
    """Ensures the reviews panel/feed is active and ready."""
    await _dismiss_popups(page)

    # 1. If reviews tab is already active
    if await page.locator(
        'button[role="tab"][aria-label*="Reviews" i][aria-selected="true"]'
    ).count():
        try:
            await page.wait_for_selector(REVIEW_CARD_SELECTOR, state="attached", timeout=10000)
            return True
        except Exception:
            pass

    # 2. Click Reviews tab or More reviews button
    candidate_selectors = [
        'button[role="tab"][aria-label*="Reviews for" i]',
        'button[role="tab"][aria-label*="Reviews" i]',
        'button[role="tab"]:has-text("Reviews")',
        'button:has-text("More reviews")',
        'button[aria-label*="More reviews" i]',
        'div.F7nice button',
        'button[aria-label*="reviews" i]:not([aria-label*="Write"])',
    ]
    combined = ", ".join(candidate_selectors)
    trigger = page.locator(combined).first

    if await trigger.count():
        try:
            await trigger.click(timeout=4000)
            await _dismiss_popups(page)
            await page.wait_for_selector(REVIEW_CARD_SELECTOR, state="attached", timeout=15000)
            return True
        except Exception:
            pass

    # 3. If trigger was not visible yet, wheel down overview panel to reveal sticky tabs
    await page.mouse.move(200, 500)
    await page.mouse.wheel(0, 1500)
    await page.wait_for_timeout(1000)
    await _dismiss_popups(page)

    try:
        await trigger.wait_for(timeout=6000)
        await trigger.click(timeout=3000)
        await _dismiss_popups(page)
        await page.wait_for_selector(REVIEW_CARD_SELECTOR, state="attached", timeout=15000)
        return True
    except Exception:
        pass

    try:
        await page.wait_for_selector(REVIEW_CARD_SELECTOR, state="attached", timeout=5000)
        return True
    except Exception:
        return False


async def _extract_card_data(c) -> dict[str, Any] | None:
    try:
        author = ""
        action_btn = c.locator('button[aria-label*="Actions for " i]').first
        if await action_btn.count():
            lbl = await action_btn.get_attribute("aria-label") or ""
            m = re.search(r"Actions for (.+?)('s review|\.|$)", lbl)
            if m:
                author = m.group(1).strip()
        if not author:
            author_el = c.locator('.d4r55, button.al6KCe').first
            if await author_el.count():
                author = (await author_el.inner_text()).strip()
        if not author:
            author = "Anonymous"

        # Rating
        star_el = c.locator('span[role="img"][aria-label*="star" i]').first
        rating = None
        if await star_el.count():
            lbl = await star_el.get_attribute("aria-label") or ""
            rating = _rating_from_label(lbl)

        # Text
        text_el = c.locator('.wiI7pd').first
        text = (await text_el.inner_text()).strip() if await text_el.count() else ""

        # Relative time
        time_el = c.locator('.rsqaWe').first
        rel_time = (await time_el.inner_text()).strip() if await time_el.count() else None

        return {
            "author": _clean(author),
            "rating": rating,
            "text": _clean(text),
            "relative_time": rel_time,
            "source": "google_maps_playwright",
        }
    except Exception:
        return None


async def _collect_reviews(
    page: Page,
    max_reviews: int,
    max_scrolls: int,
    pause_ms: int,
) -> list[dict[str, Any]]:
    """
    Progressively collects reviews while scrolling. Handles Google Maps feed virtualization
    by recording reviews into memory as they mount in the DOM.
    """
    seen: set[tuple[str, str]] = set()
    reviews: list[dict[str, Any]] = []

    container = page.locator('div.m6QErb.DxyBCb.kA9KIf.dS8AEf, div[role="feed"]').first
    stagnant_count = 0
    last_len = 0

    for _ in range(max_scrolls):
        # 1. Expand "See more" buttons on currently mounted cards
        for btn in await page.locator('button.w8nwRe, button[aria-label="See more"]').all():
            try:
                txt = (await btn.inner_text()).strip().lower()
                if txt in ("more", "see more", "المزيد"):
                    await btn.click(timeout=100)
            except Exception:
                pass

        # 2. Extract from currently mounted cards
        cards = await page.locator(REVIEW_CARD_SELECTOR).all()
        for c in cards:
            data = await _extract_card_data(c)
            if not data:
                continue
            dedup_key = (data["author"], data["text"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            reviews.append(data)
            if len(reviews) >= max_reviews:
                return reviews

        if len(reviews) == last_len:
            stagnant_count += 1
            if stagnant_count >= 5:
                break
        else:
            stagnant_count = 0
        last_len = len(reviews)

        # 3. Scroll container down
        if await container.count():
            try:
                await container.evaluate("el => el.scrollBy(0, 1000)")
            except Exception:
                pass

        # Also scroll last card into view or mouse wheel
        last_card = page.locator(REVIEW_CARD_SELECTOR).last
        if await last_card.count():
            try:
                await last_card.scroll_into_view_if_needed(timeout=1500)
            except Exception:
                pass

        await page.mouse.move(200, 700)
        await page.mouse.wheel(0, 2000)
        await page.wait_for_timeout(pause_ms)
        await _dismiss_popups(page)

    return reviews


async def extract_reviews_from_url(
    url: str,
    max_reviews: int = 50,
    headless: bool = True,
    scroll_pause_ms: int = 1000,
    max_scrolls: int = 35,
) -> dict[str, Any]:
    """
    Extract reviews and place details from a Google Maps URL.

    Args:
        url: Full or shortened Google Maps URL.
        max_reviews: Maximum number of reviews to extract (1 to 500).
        headless: Whether to run Chromium headlessly.
        scroll_pause_ms: Milliseconds to pause between scroll actions.
        max_scrolls: Maximum scroll attempts.

    Returns:
        dict with status, place_name, overall_rating, total_ratings, address, reviews_fetched, reviews.
    """
    started_at = datetime.now(timezone.utc).isoformat()

    if not isinstance(url, str) or not url.strip():
        return _error(url, "INVALID_URL", "A Google Maps URL is required.", started_at)
    url = url.strip()
    if not _is_google_maps_url(url):
        return _error(
            url,
            "UNSUPPORTED_URL",
            "URL must belong to google.com, maps.google.com, or maps.app.goo.gl.",
            started_at,
        )

    max_reviews = max(1, min(int(max_reviews), 500))
    place_name = None
    overall_rating = None
    total_ratings = None
    address = None
    reviews: list[dict[str, Any]] = []
    browser: Browser | None = None

    try:
        target_web_url = _normalize_maps_url(url)

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                ],
            )
            context = await browser.new_context(
                locale="en-US",
                timezone_id="America/Los_Angeles",
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            await page.goto(target_web_url, wait_until="domcontentloaded", timeout=60000)
            await _handle_consent(page)
            await _dismiss_popups(page)

            # Wait for place header
            try:
                await page.wait_for_selector("h1", timeout=25000)
                place_name = _clean(await page.locator("h1").first.inner_text())
            except Exception:
                return _error(
                    url,
                    "PLACE_PAGE_NOT_LOADED",
                    "Place page did not load (consent / captcha / invalid place URL).",
                    started_at,
                    await _debug_info(page),
                )

            # Extract overview metadata
            rating_loc = page.locator(
                'div.F7nice span[aria-hidden="true"], span.ceNzKf[aria-hidden="true"]'
            ).first
            if await rating_loc.count():
                try:
                    overall_rating = float(
                        (await rating_loc.inner_text()).strip().replace(",", ".")
                    )
                except ValueError:
                    pass

            rev_count_loc = page.locator(
                'div.F7nice span[aria-label*="reviews" i], button[aria-label*="reviews" i], button:has-text("reviews")'
            ).first
            if await rev_count_loc.count():
                lbl = (await rev_count_loc.get_attribute("aria-label")) or (
                    await rev_count_loc.inner_text()
                )
                m = re.search(r"(\d[\d\s,.]*)", lbl)
                if m:
                    clean_num = re.sub(r"[^\d]", "", m.group(1))
                    if clean_num:
                        total_ratings = int(clean_num)

            addr_loc = page.locator(
                'button[data-item-id="address"], button[aria-label*="Address:"]'
            ).first
            if await addr_loc.count():
                address = _clean(await addr_loc.inner_text())

            # Switch to reviews view
            reviews_ready = await _open_reviews_tab(page)
            if not reviews_ready:
                err = _error(
                    url,
                    "NO_REVIEWS_FOUND",
                    "Reviews section could not be opened from this place page.",
                    started_at,
                    await _debug_info(page),
                )
                err["place_name"] = place_name
                err["overall_rating"] = overall_rating
                err["total_ratings"] = total_ratings
                err["address"] = address
                return err

            # Collect reviews progressively
            reviews = await _collect_reviews(
                page,
                max_reviews=max_reviews,
                max_scrolls=max_scrolls,
                pause_ms=scroll_pause_ms,
            )

        return {
            "status": "success",
            "source": "google_maps",
            "url": url,
            "place_name": place_name,
            "overall_rating": overall_rating,
            "total_ratings": total_ratings,
            "address": address,
            "reviews_fetched": len(reviews),
            "reviews": reviews,
            "fetched_at": started_at,
        }

    except Exception as e:
        return _error(url, "RUNTIME_ERROR", str(e), started_at)
    finally:
        if browser:
            await browser.close()


DEFAULT_URL = (
    "https://www.google.com/maps/place/Qedra+Smart+Village/@30.0830914,30.9012018,43935m/data=!3m1!1e3!4m6!3m5!1s0x14585bbb0fc5c845:0x78a4ee0f662f2f0a!8m2!3d30.0795663!4d31.0136127!16s%2Fg%2F11rzsxdglx?entry=ttu&g_ep=EgoyMDI2MTAwNy4wIKXMDSoASAFQAw%3D%3D"
)

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    is_colab = any(x.endswith("ipykernel_launcher.py") for x in sys.argv) or any(
        "-f" == x for x in sys.argv
    )

    if len(sys.argv) > 1 and not is_colab:
        parser = argparse.ArgumentParser(description="Extract Google Maps reviews via Playwright.")
        parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Google Maps place URL")
        parser.add_argument("--max", type=int, default=10, help="Maximum reviews to extract")
        parser.add_argument("--headful", action="store_true", help="Run with visible browser UI")
        args = parser.parse_args()
        target_url = args.url
        max_num = args.max
        headless_mode = not args.headful
    else:
        target_url = DEFAULT_URL
        max_num = 10
        headless_mode = True

    result = asyncio.run(
        extract_reviews_from_url(target_url, max_reviews=max_num, headless=headless_mode)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
