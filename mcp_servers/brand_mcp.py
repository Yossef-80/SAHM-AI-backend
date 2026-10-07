"""Brand Research MCP server (one file).

Collect : analyze_website, get_social_profile, get_reviews, get_ads, brand_report
Analyze : engagement_stats, analyze_sentiment, product_insights, full_analysis
Storage : save_data, list_saved, load_saved
Route   : route_file_to_tool

Install:  py -m pip install "mcp<2" requests beautifulsoup4 apify-client
          py -m pip install transformers torch sentencepiece   (only for analyze_sentiment / product_insights / full_analysis)
Env:      APIFY_TOKEN   (needed for social / maps / ads and for the blocked-site fallback)
          BRAND_DATA_DIR (optional, default ./brand_data)
Run:      py brand_mcp.py                       -> MCP server (stdio)
Test:     py brand_mcp.py --test https://ar.shein.com shein Egypt EG      (collect + save)
Analyze:  py brand_mcp.py --analyze "KFC Egypt"                            (engagement + sentiment + products/aspects on saved data)
Products: domain-agnostic — discovers products via web search + frequent terms in reviews/posts (no fixed fashion dictionary)

Big results are saved to disk and only a SHORT summary is returned to the model (saves tokens).

Blocked sites: analyze_website tries, in order
  1. plain request with browser headers
  2. Apify website-content-crawler (real browser + proxies, renders JavaScript)
  3. Jina Reader (free, text/markdown only)
  4. Wayback Machine snapshot
and also extracts social links from the raw HTML/JSON, not only from <a> tags.
These only work for PUBLIC pages. Respect each site's terms of service.
"""
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(os.getenv("BRAND_DATA_DIR", "brand_data"))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en;q=0.8",
}
# verify names in the Apify Store if a run fails
ACTORS = {
    "facebook_page": "apify/facebook-pages-scraper",
    "facebook_posts": "apify/facebook-posts-scraper",
    "instagram": "apify/instagram-profile-scraper",
    "tiktok": "clockworks/tiktok-scraper",
    "maps": "compass/crawler-google-places",
    "ads": "apify/facebook-ads-scraper",
    "web": "apify/website-content-crawler",
}
SOC = {"instagram.com": "instagram", "facebook.com": "facebook", "fb.com": "facebook", "x.com": "x",
       "twitter.com": "x", "tiktok.com": "tiktok", "youtube.com": "youtube", "linkedin.com": "linkedin",
       "t.me": "telegram", "wa.me": "whatsapp", "snapchat.com": "snapchat", "pinterest.com": "pinterest"}
SOC_RE = re.compile(r"https?:(?:\\?/){2}(?:www\.)?(instagram\.com|facebook\.com|fb\.com|tiktok\.com|youtube\.com|"
                    r"twitter\.com|x\.com|linkedin\.com|t\.me|wa\.me|snapchat\.com|pinterest\.com)"
                    r"((?:\\?/)[A-Za-z0-9_.@%\-/]+)")
BAD_SOC = {"p", "reel", "reels", "explore", "stories", "sharer", "share", "intent", "tr", "plugins", "dialog", "login", "watch"}
TECH = {"facebook_pixel": "fbevents.js", "google_analytics": "gtag(", "google_tag_manager": "googletagmanager.com",
        "tiktok_pixel": "analytics.tiktok.com", "snap_pixel": "sc-static.net/scevent", "shopify": "cdn.shopify.com",
        "wordpress": "wp-content", "woocommerce": "woocommerce"}
BLOCK_HINTS = ("just a moment", "access denied", "captcha", "cf-chl", "attention required", "enable javascript",
               "are you a robot", "request blocked", "perimeterx", "px-captcha")


# ---------------------------------------------------------------- helpers
def _slug(s: str) -> str:
    return re.sub(r"\W+", "_", s.strip(), flags=re.UNICODE).strip("_").lower() or "brand"


def _brand_dir(brand: str) -> Path:
    d = DATA_DIR / _slug(brand)
    (d / "images").mkdir(parents=True, exist_ok=True)
    return d


def _run_actor(actor: str, run_input: dict, limit: int = 100) -> list:
    token = os.getenv("APIFY_TOKEN")
    if not token:
        return []
    from apify_client import ApifyClient
    c = ApifyClient(token)
    try:
        run = c.actor(actor).call(run_input=run_input)
        ds = run["defaultDatasetId"] if isinstance(run, dict) else getattr(run, "default_dataset_id", None)
        return list(c.dataset(ds).iterate_items(limit=limit))
    except Exception as e:
        print(f"[apify] {actor}: {str(e)[:150]}", file=sys.stderr)
        return []


def _uname(url: str) -> str:
    return [p for p in urlparse(url).path.split("/") if p][-1].lstrip("@")


def _pick(d: dict, *keys):
    return next((d[k] for k in keys if d.get(k) not in (None, "")), None)


def _looks_blocked(status: int, html: str) -> bool:
    low = html[:6000].lower()
    return status in (401, 403, 406, 429, 503) or len(html) < 1500 or any(h in low for h in BLOCK_HINTS)


# ---------------------------------------------------------------- fetching with fallbacks
def fetch_page(url: str) -> dict:
    """Return {'html', 'final_url', 'method', 'tried', 'note'} using progressively stronger methods."""
    tried = []

    # 1) direct
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        tried.append(f"direct:{r.status_code}")
        if not _looks_blocked(r.status_code, r.text):
            return {"html": r.text, "final_url": r.url, "method": "direct", "tried": tried}
    except Exception as e:
        tried.append(f"direct:error:{type(e).__name__}")

    # 2) real browser through Apify (JS rendering + proxies)
    if os.getenv("APIFY_TOKEN"):
        items = _run_actor(ACTORS["web"], {"startUrls": [{"url": url}], "maxCrawlPages": 1, "crawlerType": "playwright:firefox",
                                           "saveHtml": True, "proxyConfiguration": {"useApifyProxy": True}}, 1)
        html = (items[0].get("html") or "") if items else ""
        tried.append(f"apify_browser:{'ok' if html else 'empty'}")
        if html and not _looks_blocked(200, html):
            return {"html": html, "final_url": url, "method": "apify_browser", "tried": tried}

    # 3) Jina Reader (markdown text, free)
    try:
        r = requests.get("https://r.jina.ai/" + url, headers={"Accept": "text/plain"}, timeout=40)
        tried.append(f"jina:{r.status_code}")
        if r.status_code == 200 and len(r.text) > 300:
            return {"html": r.text, "final_url": url, "method": "jina_reader", "tried": tried,
                    "note": "text only (no images/colors/fonts)"}
    except Exception as e:
        tried.append(f"jina:error:{type(e).__name__}")

    # 4) Wayback snapshot
    try:
        a = requests.get("https://archive.org/wayback/available", params={"url": url}, timeout=20).json()
        snap = (a.get("archived_snapshots") or {}).get("closest", {}).get("url")
        if snap:
            r = requests.get(snap, headers=HEADERS, timeout=30)
            tried.append(f"wayback:{r.status_code}")
            if r.status_code == 200:
                return {"html": r.text, "final_url": snap, "method": "wayback", "tried": tried,
                        "note": "archived copy, may be old"}
    except Exception as e:
        tried.append(f"wayback:error:{type(e).__name__}")

    return {"html": "", "final_url": url, "method": "failed", "tried": tried,
            "note": "all methods blocked. Paste the social URLs manually or add APIFY_TOKEN."}


def _social_links(html: str, base: str) -> dict:
    found: dict = {}

    def add(u: str):
        u = u.replace("\\/", "/")
        p = urlparse(u)
        host = (p.hostname or "").removeprefix("www.")
        first = [x for x in p.path.split("/") if x][:1]
        if not first or first[0].lower() in BAD_SOC:
            return
        for d, n in SOC.items():
            if host == d or host.endswith("." + d):
                found.setdefault(n, [])
                if u.rstrip("/") not in found[n]:
                    found[n].append(u.rstrip("/"))

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        add(urljoin(base, a["href"]))
    for m in SOC_RE.finditer(html):                      # social links hidden in scripts / JSON / markdown
        add("https://" + m.group(1) + m.group(2).replace("\\/", "/"))
    return found


def _download_image(url: str, dest: Path, prefix: str = "") -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        if len(r.content) > 8_000_000:
            return None
        ext = Path(urlparse(url).path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".avif"}:
            ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(
                r.headers.get("content-type", "").split(";")[0], ".jpg")
        p = dest / "images" / f"{prefix}{hashlib.md5(url.encode()).hexdigest()[:10]}{ext}"
        p.write_bytes(r.content)
        return str(p.relative_to(dest))
    except Exception:
        return None


# ---------------------------------------------------------------- core logic (plain functions)
def _analyze_website(url: str) -> dict:
    url = url if url.startswith("http") else "https://" + url
    page = fetch_page(url)
    html, base = page["html"], page["final_url"]
    out = {"url": base, "domain": urlparse(url).hostname, "fetch_method": page["method"], "fetch_attempts": page["tried"],
           "note": page.get("note")}
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser")
    meta = lambda **a: (lambda t: t.get("content") if t else None)(soup.find("meta", attrs=a))
    og = {m["property"]: m.get("content", "") for m in soup.find_all("meta", property=True) if m["property"].startswith("og:")}

    imgs, seen = [], set()

    def add(u, alt="", logo=False):
        if not u or u.startswith("data:"):
            return
        f = urljoin(base, u)
        if f not in seen:
            seen.add(f)
            imgs.append({"url": f, "alt": alt, "logo": logo})
    add(og.get("og:image"))
    for im in soup.find_all("img"):
        src = im.get("data-src") or im.get("data-lazy-src") or im.get("src")
        ss = im.get("srcset") or im.get("data-srcset")
        if ss:
            src = ss.split(",")[-1].strip().split()[0]
        parents = " ".join(" ".join(p.get("class", [])) for p in im.parents if getattr(p, "get", None))
        hint = " ".join([im.get("src", ""), im.get("alt", ""), *im.get("class", []), parents]).lower()
        add(src, im.get("alt", ""), "logo" in hint)
    for m in re.findall(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", html):       # markdown images (Jina fallback)
        add(m)

    ld = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            ld.append(json.loads(s.string or ""))
        except Exception:
            pass
    css = " ".join(s.get_text() for s in soup.find_all("style")) + " ".join(e["style"] for e in soup.find_all(style=True))
    colors = [c for c, _ in Counter(h.lower() for h in re.findall(r"#(?:[0-9a-fA-F]{6})\b", css)).most_common(8)]
    fonts = set()
    for l in soup.find_all("link", href=True):
        if "fonts.googleapis.com" in l["href"]:
            fonts.update(f.replace("+", " ") for f in re.findall(r"family=([^:&]+)", l["href"]))

    social = _social_links(html, base)
    low = html.lower()
    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()
    text = soup.get_text(" ", strip=True)
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text()) > 60]
    out.update({
        "title": soup.title.get_text(strip=True) if soup.title else None,
        "description": meta(name="description") or og.get("og:description"),
        "keywords": meta(name="keywords"),
        "language": soup.html.get("lang") if soup.html else None,
        "open_graph": og, "structured_data": ld[:5],
        "h1": [h.get_text(" ", strip=True) for h in soup.find_all("h1")][:5],
        "h2": [h.get_text(" ", strip=True) for h in soup.find_all("h2")][:15],
        "intro_text": " ".join(paras[:5])[:1500] or text[:800],
        "emails": sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+[A-Za-z]", text)))[:10],
        "phones": sorted(set(re.findall(r"\+?\d[\d\s-]{8,14}\d", text)))[:10],
        "social": social, "images": imgs, "fonts": sorted(fonts), "colors": colors,
        "ad_pixels_and_tech": [k for k, s in TECH.items() if s in low],
    })
    return out


def _norm_post(platform: str, it: dict) -> dict:
    img = _pick(it, "displayUrl", "imageUrl", "thumbnailUrl", "full_picture", "media_url")
    if not img and isinstance(it.get("videoMeta"), dict):
        img = it["videoMeta"].get("coverUrl")
    if not img and isinstance(it.get("media"), list) and it["media"] and isinstance(it["media"][0], dict):
        img = _pick(it["media"][0], "thumbnail", "photo_image", "url")
    return {"platform": platform, "id": str(_pick(it, "id", "postId", "shortCode") or ""),
            "url": _pick(it, "url", "postUrl", "webVideoUrl", "permalink", "topLevelUrl"),
            "caption": _pick(it, "caption", "text", "message", "postText") or "",
            "timestamp": str(_pick(it, "timestamp", "time", "createTimeISO", "created_time") or ""),
            "likes": _pick(it, "likesCount", "likes", "diggCount", "like_count") or 0,
            "comments": _pick(it, "commentsCount", "comments", "commentCount", "comments_count") or 0,
            "shares": _pick(it, "sharesCount", "shares", "shareCount") or 0, "image_url": img}


def _engagement(posts: list, followers=None) -> dict:
    if not posts:
        return {"posts": 0, "verdict": "no posts"}
    inter = [(p["likes"] or 0) + (p["comments"] or 0) + (p["shares"] or 0) for p in posts]
    avg = statistics.mean(inter)
    rate = round(avg / followers * 100, 3) if followers else None
    v = "unknown followers" if rate is None else "strong" if rate >= 3 else "healthy" if rate >= 1 else "low" if rate >= .3 else "very low"
    top = sorted(zip(inter, [p["url"] for p in posts]), key=lambda x: -x[0])[:3]
    return {"posts": len(posts), "avg_interactions": round(avg, 1), "engagement_rate_pct": rate, "verdict": v,
            "top_posts": [{"interactions": i, "url": u} for i, u in top]}


def _social(platform: str, url: str, limit: int = 20) -> dict:
    platform = platform.lower()
    if platform == "instagram":
        items = _run_actor(ACTORS["instagram"], {"usernames": [_uname(url)]}, 5)
        prof = items[0] if items else {}
        posts = [_norm_post("instagram", p) for p in (prof.get("latestPosts") or [])[:limit]]
        info = {k: prof.get(k) for k in ("username", "fullName", "followersCount", "followsCount", "postsCount", "biography", "verified")}
        fol = info.get("followersCount")
    elif platform == "facebook":
        pg = _run_actor(ACTORS["facebook_page"], {"startUrls": [{"url": url}]}, 3)
        info = {k: (pg[0].get(k) if pg else None) for k in ("title", "likes", "followers", "categories", "rating", "ratingCount", "address", "phone", "website")}
        posts = [_norm_post("facebook", p) for p in _run_actor(ACTORS["facebook_posts"], {"startUrls": [{"url": url}], "resultsLimit": limit}, limit)]
        fol = info.get("followers") or info.get("likes")
    elif platform == "tiktok":
        raw = _run_actor(ACTORS["tiktok"], {"profiles": [_uname(url)], "resultsPerPage": limit}, limit)
        a = (raw[0].get("authorMeta") or {}) if raw else {}
        info = {k: a.get(k) for k in ("name", "nickName", "fans", "following", "heart", "video", "signature", "verified")}
        posts = [_norm_post("tiktok", p) for p in raw]
        fol = info.get("fans")
    else:
        return {"error": f"unsupported platform '{platform}' (facebook | instagram | tiktok)"}
    return {"platform": platform, "url": url, "profile": info, "engagement": _engagement(posts, fol), "posts": posts}


def _reviews(brand: str, location: str = "Egypt", max_reviews: int = 50) -> list:
    items = _run_actor(ACTORS["maps"], {"searchStringsArray": [f"{brand} {location}"], "maxCrawledPlacesPerSearch": 5,
                                        "maxReviews": max_reviews, "language": "ar"}, 10)
    return [{"place": p.get("title"), "rating": p.get("totalScore"), "reviews_count": p.get("reviewsCount"),
             "address": p.get("address"), "phone": p.get("phone"), "website": p.get("website"),
             "category": p.get("categoryName"), "maps_url": p.get("url"),
             "reviews": [{"text": r.get("text"), "stars": r.get("stars"), "date": r.get("publishedAtDate"), "likes": r.get("likesCount")}
                         for r in (p.get("reviews") or [])]} for p in items]


def _ads(brand: str, country_code: str = "EG", limit: int = 30) -> list:
    u = ("https://www.facebook.com/ads/library/?active_status=active&ad_type=all"
         f"&country={country_code}&q={quote(brand)}&search_type=keyword_unordered")
    return _run_actor(ACTORS["ads"], {"startUrls": [{"url": u}], "resultsLimit": limit}, limit)


def _write_csv(path: Path, rows: list):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows)


def _save_report(brand: str, report: dict) -> dict:
    """Write report.json, posts.csv, reviews.csv, images_dataset.jsonl and update the global index."""
    d = _brand_dir(brand)
    posts = report.get("posts", [])
    site = report.get("website", {})
    for im in sorted(site.get("images", []), key=lambda i: not i.get("logo"))[:report.get("max_images", 30)]:
        im["local_file"] = _download_image(im["url"], d, "site_")
    for p in posts:
        p["local_image"] = _download_image(p["image_url"], d, p["platform"] + "_") if p.get("image_url") else None
    (d / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(d / "posts.csv", posts)
    _write_csv(d / "reviews.csv", [dict(place=p["place"], **r) for p in report.get("google_maps", []) for r in p["reviews"]])
    _write_csv(d / "ads.csv", report.get("ads", []))
    n_img = 0
    with open(d / "images_dataset.jsonl", "w", encoding="utf-8") as f:
        for src, items in (("website", site.get("images", [])), ("social", posts)):
            for it in items:
                lf = it.get("local_file") or it.get("local_image")
                if not lf:
                    continue
                path = d / lf
                f.write(json.dumps({"brand": report["brand"], "source": src, "platform": it.get("platform"),
                                    "post_url": it.get("url"), "caption": it.get("caption") or it.get("alt"),
                                    "image_url": it.get("image_url") or it.get("url"),
                                    "mime_type": mimetypes.guess_type(path.name)[0] or "image/jpeg",
                                    "image_base64": base64.b64encode(path.read_bytes()).decode()}, ensure_ascii=False) + "\n")
                n_img += 1
    # index: one row per brand
    idx = DATA_DIR / "brands_index.jsonl"
    rows = [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()] if idx.exists() else []
    rows = [r for r in rows if r["brand"] != report["brand"]] + [{
        "brand": report["brand"], "folder": str(d), "website": site.get("url"), "posts": len(posts),
        "places": len(report.get("google_maps", [])), "ads": len(report.get("ads", [])), "images": n_img}]
    idx.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return {"folder": str(d), "files": sorted(p.name for p in d.iterdir()), "images_saved": n_img}


def _summary(report: dict, saved: dict) -> dict:
    site = report.get("website", {})
    rated = [(p["place"], p["rating"], p["reviews_count"]) for p in report.get("google_maps", []) if p.get("rating")]
    return {
        "brand": report["brand"], "saved": saved,
        "website": {"title": site.get("title"), "description": site.get("description"), "fetch_method": site.get("fetch_method"),
                    "social": site.get("social"), "colors": site.get("colors"), "fonts": site.get("fonts"),
                    "ad_pixels_and_tech": site.get("ad_pixels_and_tech"), "emails": site.get("emails"), "image_count": len(site.get("images", []))},
        "social": {k: {"profile": v["profile"], "engagement": v["engagement"]} for k, v in report.get("social", {}).items()},
        "google_maps": rated[:5], "ads_count": len(report.get("ads", [])), "posts_count": len(report.get("posts", [])),
    }


# ---------------------------------------------------------------- analysis: engagement, sentiment, products/aspects
# These tools work on data already saved by brand_report (brand_data/<brand>/report.json).
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")  # multilingual, handles Arabic + English
_LABELS = {"label_0": "negative", "label_1": "neutral", "label_2": "positive", "neg": "negative", "neu": "neutral", "pos": "positive"}
_SCORE = {"positive": 1, "neutral": 0, "negative": -1}
_clf = None

# ---------------------------------------------------------------- product / aspect discovery (domain-agnostic)
# Aspects stay keyword-based (service, price, quality… apply to almost any brand).
# Products are NO longer a fixed fashion list: they are discovered per brand via
#   1) web search (DuckDuckGo) for "<brand> products / menu / best sellers"
#   2) frequent meaningful terms extracted from reviews + post captions
# then matched back into the texts for sentiment aggregation.

ASPECTS = {
    "quality":           r"quality|material|fabric|stitch|taste|flavor|fresh|جودة|جوده|خامة|خامه|قماش|خياطة|طعم|مذاق|فريش|طازة",
    "price":             r"price|expensive|cheap|cost|worth|سعر|اسعار|أسعار|غالي|غاليه|رخيص|ارخص|بكام|قيمة",
    "size":              r"\bsize|\bfit\b|tight|loose|small|large|portion|مقاس|مقاسات|ضيق|واسع|صغير|كبير|كمية|حجم",
    "delivery/shipping": r"deliver|shipping|\bship|late|delay|arriv|توصيل|شحن|شحنة|تأخير|تاخير|اتأخر|وصل",
    "customer service":  r"service|staff|manager|rude|employee|cashier|support|friendly|helpful|خدمة العملاء|موظف|معاملة|تعامل|مدير|كاشير",
    "returns/refund":    r"return|refund|exchange|استرجاع|ارجاع|إرجاع|استبدال|ريفند",
    "availability":      r"out of stock|in stock|not available|مش متوفر|غير متوفر|نفذ|خلص",
    "store/branch":      r"store|branch|crowd|clean|queue|\bline\b|فرع|زحمة|ازدحام|طابور|نظافة",
    "design/style":      r"style|design|trend|colou?r|موديل|ستايل|تصميم|لون|الوان",
    "packaging":         r"packag|box|bag|wrap|تغليف|علبة|كيس|باكت",
    "hygiene/cleanliness": r"clean|dirty|hygiene|sanitar|نظافة|وسخ|قذر|صحي",
}

# Generic stopwords (EN + AR) — used when mining product terms from text
STOPWORDS = set("""
the and for was were this that with have has had not you your but are they them their from out all one she her his
there when just back what about would could been into than then also very will can its it's i'm i've don't didn't
very more most some any each other such only same so than too very just over after also before being
اللي في من على مع ده دي دا هو هي ان انا مش لا ما كان كانت الى إلى عن حتى اذا إذا او أو لم لن قد كل تم يكون
يكونوا كانوا يكون هناك هنا هناك جدا جداً يعني يعني كده كدا عشان علشان اللي ال الي الى
order food place location time times people really really good bad nice great love hate like
طلب اكل مكان فرع وقت ناس حلو وحش كويس سيء احب اكره زي مثل
""".split())

# Extra noise words that often look like "products" but aren't
PRODUCT_NOISE = set("""
review reviews star stars rating ratings google maps facebook instagram tiktok app website phone number
branch branches store stores restaurant restaurants location locations customer customers service staff
today yesterday tomorrow morning evening night day week month year
اليوم امبارح بكرة صباح مساء ليل نهار اسبوع شهر سنة تقييم نجوم ريفيو
""".split())


def _web_search_products(brand: str, max_terms: int = 25) -> list:
    """Discover likely product names for any brand via DuckDuckGo HTML (no API key).
    Queries a few variants and extracts capitalized / repeated noun-like phrases.
    Returns a list of lowercase product candidate strings.
    """
    queries = [
        f"{brand} products",
        f"{brand} menu",
        f"{brand} best sellers",
        f"{brand} أشهر المنتجات",
        f"{brand} منتجات",
    ]
    blob = []
    for q in queries:
        try:
            r = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": q},
                headers={**HEADERS, "Accept-Language": "en,ar;q=0.8"},
                timeout=20,
            )
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for sel in (".result__snippet", ".result__a", "a.result__a", ".snippet"):
                for el in soup.select(sel):
                    blob.append(el.get_text(" ", strip=True))
            # also plain text dump as fallback
            blob.append(soup.get_text(" ", strip=True)[:4000])
        except Exception as e:
            print(f"[web_search] {q}: {type(e).__name__}", file=sys.stderr)

    text = " ".join(blob)
    # candidate tokens: latin words 3+ chars OR arabic words 3+ chars
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'\-]{2,}|[\u0600-\u06FF]{3,}", text)
    # also multi-word product-ish phrases (Title Case sequences)
    phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)

    counts: Counter = Counter()
    brand_parts = {p.lower() for p in re.findall(r"[A-Za-z\u0600-\u06FF]{2,}", brand)}
    for t in tokens + [p for ph in phrases for p in ph.split()]:
        w = t.lower().strip("-'&")
        if len(w) < 3 or w in STOPWORDS or w in PRODUCT_NOISE or w in brand_parts:
            continue
        if w.isdigit():
            continue
        counts[w] += 1

    # keep terms that appeared at least twice across search results
    ranked = [w for w, n in counts.most_common(80) if n >= 2][:max_terms]
    return ranked


def _mine_products_from_texts(texts: list, max_terms: int = 30) -> list:
    """Extract frequent content words from reviews/captions as product candidates (domain-agnostic)."""
    counts: Counter = Counter()
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9&'\-]{2,}|[\u0600-\u06FF]{3,}", (t or "").lower()):
            w = w.strip("-'&")
            if len(w) < 3 or w in STOPWORDS or w in PRODUCT_NOISE or w.isdigit():
                continue
            counts[w] += 1
    # need at least 2 mentions in the corpus
    return [w for w, n in counts.most_common(60) if n >= 2][:max_terms]


def _build_product_patterns(brand: str, texts: list) -> dict:
    """Merge web-search products + mined terms into a {name: regex} dict for _tag()."""
    web = _web_search_products(brand)
    mined = _mine_products_from_texts(texts)
    # prefer terms that show up in BOTH sources, then web, then mined
    web_set, mined_set = set(web), set(mined)
    ordered = []
    for w in web:
        if w in mined_set and w not in ordered:
            ordered.append(w)
    for w in web + mined:
        if w not in ordered:
            ordered.append(w)
    patterns = {}
    for name in ordered[:40]:
        # word-boundary-ish match; escape regex specials
        esc = re.escape(name)
        patterns[name] = rf"(?i)(?<!\w){esc}(?!\w)"
    return patterns


def _analysis_dir(brand: str) -> Path:
    d = _brand_dir(brand) / "analysis"
    d.mkdir(exist_ok=True)
    return d


def _load_report(brand: str) -> dict:
    f = _brand_dir(brand) / "report.json"
    if not f.exists():
        raise FileNotFoundError(f"nothing saved for '{brand}'. Run brand_report first.")
    return json.loads(f.read_text(encoding="utf-8"))


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _place_key(brand: str, place_filter: str) -> str:
    """'auto' keeps only places whose name contains the first word of the brand (drops unrelated Maps results)."""
    if place_filter == "auto":
        return brand.split()[0].lower() if brand.split() else ""
    return (place_filter or "").lower()


def _review_rows(report: dict, brand: str, place_filter: str = "auto") -> list:
    rows = [{"place": p.get("place"), "text": r.get("text") or "", "stars": r.get("stars"), "date": r.get("date"), "likes": r.get("likes")}
            for p in report.get("google_maps", []) for r in p.get("reviews", [])]
    key = _place_key(brand, place_filter)
    kept = [r for r in rows if key and key in str(r["place"]).lower()]
    return kept or rows


def _engagement_stats(brand: str, place_filter: str = "auto") -> dict:
    rep = _load_report(brand)
    reviews = _review_rows(rep, brand, place_filter)
    stars = [s for s in (_num(r["stars"]) for r in reviews) if s is not None]
    key = _place_key(brand, place_filter)
    places = [{"place": p.get("place"), "rating": p.get("rating"), "reviews_count": p.get("reviews_count"), "address": p.get("address")}
              for p in rep.get("google_maps", [])]
    places = [p for p in places if key and key in str(p["place"]).lower()] or places

    social = {}
    for p in rep.get("posts", []):
        s = social.setdefault(p["platform"], {"posts": 0, "likes": 0, "comments": 0, "shares": 0})
        s["posts"] += 1
        for k in ("likes", "comments", "shares"):
            s[k] += int(_num(p.get(k)) or 0)
    for pf, s in social.items():
        s["avg_interactions_per_post"] = round((s["likes"] + s["comments"] + s["shares"]) / s["posts"], 1)
        info = rep.get("social", {}).get(pf, {})
        prof = info.get("profile", {})
        s["followers"] = prof.get("followersCount") or prof.get("followers") or prof.get("fans") or prof.get("likes")
        s["engagement_rate_pct"] = info.get("engagement", {}).get("engagement_rate_pct")

    total = lambda p: sum(int(_num(p.get(k)) or 0) for k in ("likes", "comments", "shares"))
    top = sorted(rep.get("posts", []), key=total, reverse=True)[:5]
    out = {
        "brand": brand, "reviews_collected": len(reviews),
        "reviews_with_text": sum(1 for r in reviews if len(_clean_text(r["text"])) >= 2),
        "avg_stars": round(statistics.mean(stars), 2) if stars else None,
        "stars_distribution": {k: sum(1 for s in stars if int(s) == k) for k in range(1, 6)},
        "maps_places": places, "social": social,
        "top_posts": [{"platform": p["platform"], "url": p.get("url"), "caption": (p.get("caption") or "")[:100],
                       "likes": p.get("likes"), "comments": p.get("comments"), "shares": p.get("shares")} for p in top],
    }
    (_analysis_dir(brand) / "engagement.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def _clean_text(t) -> str:
    t = str(t or "")
    t = re.sub(r"https?://\S+|@\w+", " ", t)
    t = re.sub(r"[ً-ْـ]", "", t)     # Arabic diacritics and tatweel
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)             # collapse letter repeats ("soooo" -> "soo")
    return re.sub(r"\s+", " ", t).strip()


def _predict(texts: list, batch: int = 32) -> list:
    """Run the Hugging Face model. Returns [(label, confidence), ...]. Imports are lazy so the other tools need no ML libraries."""
    global _clf
    if _clf is None:
        from transformers import pipeline
        import torch
        _clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, truncation=True,
                        max_length=256, device=0 if torch.cuda.is_available() else -1)
    out = _clf(list(texts), batch_size=batch)
    return [(_LABELS.get(o["label"].lower(), o["label"].lower()), float(o["score"])) for o in out]


def _stars_label(s):
    s = _num(s)
    return None if s is None else "positive" if s >= 4 else "neutral" if s == 3 else "negative"


def _score_rows(rows: list, text_key: str, stars_key: str = "") -> list:
    """Add sentiment to each row. Rows without text fall back to their star rating."""
    for r in rows:
        r["clean_text"] = _clean_text(r.get(text_key))
    idx = [i for i, r in enumerate(rows) if len(r["clean_text"]) >= 2]
    preds = _predict([rows[i]["clean_text"] for i in idx]) if idx else []
    for i, (label, conf) in zip(idx, preds):
        rows[i].update(sentiment=label, confidence=round(conf, 3), sentiment_source="text")
    for r in rows:
        if "sentiment" not in r:
            label = _stars_label(r.get(stars_key)) if stars_key else None
            r.update(sentiment=label, confidence=None, sentiment_source="stars" if label else None)
        r["score"] = _SCORE.get(r["sentiment"])
    return rows


def _pct(rows: list) -> dict:
    labels = [r["sentiment"] for r in rows if r.get("sentiment")]
    return {k: round(labels.count(k) / len(labels) * 100, 1) for k in ("positive", "neutral", "negative")} if labels else {}


def _analyze_sentiment(brand: str, place_filter: str = "auto") -> dict:
    rep = _load_report(brand)
    reviews = _score_rows(_review_rows(rep, brand, place_filter), "text", "stars")
    posts = _score_rows([dict(p) for p in rep.get("posts", [])], "caption")
    d = _analysis_dir(brand)
    cols = ("sentiment", "confidence", "sentiment_source", "score")
    _write_csv(d / "reviews_sentiment.csv", [{k: r.get(k) for k in ("place", "text", "stars", "date", "likes", *cols)} for r in reviews])
    _write_csv(d / "posts_sentiment.csv", [{k: r.get(k) for k in ("platform", "url", "caption", "likes", "comments", "shares", *cols)} for r in posts])
    both = [r for r in reviews if r["sentiment_source"] == "text" and _stars_label(r.get("stars"))]
    agree = round(sum(r["sentiment"] == _stars_label(r["stars"]) for r in both) / len(both) * 100, 1) if both else None
    return {"model": SENTIMENT_MODEL, "reviews": {"n": len(reviews), "from_text": sum(r["sentiment_source"] == "text" for r in reviews),
                                                  "from_stars_only": sum(r["sentiment_source"] == "stars" for r in reviews), "sentiment_pct": _pct(reviews)},
            "posts": {"n": len(posts), "sentiment_pct": _pct(posts)}, "text_vs_stars_agreement_pct": agree,
            "saved_to": str(d)}


def _read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _tag(text: str, patterns: dict) -> list:
    low = text.lower()
    return [k for k, p in patterns.items() if re.search(p, low)]


def _aggregate(items: list, key: str) -> list:
    groups: dict = {}
    for it in items:
        for t in it[key]:
            groups.setdefault(t, []).append(it)
    rows = []
    for name, g in groups.items():
        stars = [x["stars"] for x in g if x["stars"] is not None]
        rows.append({"name": name, "mentions": len(g), "avg_score": round(statistics.mean(x["score"] for x in g), 2),
                     "avg_stars": round(statistics.mean(stars), 2) if stars else None,
                     "positive_pct": round(sum(x["sentiment"] == "positive" for x in g) / len(g) * 100, 1),
                     "negative_pct": round(sum(x["sentiment"] == "negative" for x in g) / len(g) * 100, 1)})
    return sorted(rows, key=lambda r: -r["avg_score"])


def _product_insights(brand: str, place_filter: str = "auto") -> dict:
    """Domain-agnostic product + aspect insights.
    Products are discovered via web search + frequent terms in the saved reviews/posts
    (no fixed fashion dictionary). Aspects stay on a general keyword list.
    """
    d = _analysis_dir(brand)
    if not (d / "reviews_sentiment.csv").exists() and not (d / "posts_sentiment.csv").exists():
        _analyze_sentiment(brand, place_filter)

    review_rows = _read_csv(d / "reviews_sentiment.csv")
    post_rows = _read_csv(d / "posts_sentiment.csv")

    # collect raw texts first so we can mine product candidates
    all_texts = []
    for r in review_rows:
        all_texts.append(_clean_text(r.get("text")))
    for r in post_rows:
        all_texts.append(_clean_text(r.get("caption")))

    product_patterns = _build_product_patterns(brand, all_texts)
    # persist discovered dictionary for transparency
    (d / "discovered_products.json").write_text(
        json.dumps({"brand": brand, "products": list(product_patterns.keys())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    items = []
    for source, rows, col in (("google_reviews", review_rows, "text"),
                              ("posts", post_rows, "caption")):
        for r in rows:
            txt, label = _clean_text(r.get(col)), r.get("sentiment")
            if len(txt) > 2 and label in _SCORE:
                items.append({
                    "source": r.get("platform") or source,
                    "text": txt,
                    "sentiment": label,
                    "score": _SCORE[label],
                    "stars": _num(r.get("stars")),
                    "products": _tag(txt, product_patterns),
                    "aspects": _tag(txt, ASPECTS),
                })

    products, aspects = _aggregate(items, "products"), _aggregate(items, "aspects")
    _write_csv(d / "products_summary.csv", products)
    _write_csv(d / "aspects_summary.csv", aspects)

    neg = [i for i in items if i["sentiment"] == "negative"]
    complaints = Counter(a for i in neg for a in i["aspects"])
    praises = Counter(a for i in items if i["sentiment"] == "positive" for a in i["aspects"])
    examples = {a: [i["text"][:180] for i in neg if a in i["aspects"]][:3] for a, _ in complaints.most_common(4)}
    words = Counter(w for i in neg for w in re.findall(r"[\w؀-ۿ]{3,}", i["text"].lower()) if w not in STOPWORDS)
    solid = lambda rows: [r for r in rows if r["mentions"] >= 2]
    bp, ba = solid(products), solid(aspects)
    out = {
        "texts_analysed": len(items),
        "with_product_mention": sum(bool(i["products"]) for i in items),
        "with_aspect_mention": sum(bool(i["aspects"]) for i in items),
        "discovered_products": list(product_patterns.keys()),
        "happiest_product": bp[0]["name"] if bp else None,
        "unhappiest_product": bp[-1]["name"] if len(bp) > 1 else None,
        "best_aspect": ba[0]["name"] if ba else None,
        "worst_aspect": ba[-1]["name"] if len(ba) > 1 else None,
        "products": products,
        "aspects": aspects,
        "top_complaints": dict(complaints.most_common(5)),
        "top_praises": dict(praises.most_common(5)),
        "complaint_examples": examples,
        "frequent_negative_words": words.most_common(15),
        "note": (
            "Products are discovered automatically (web search + terms in reviews/posts), "
            "not from a fixed category dictionary. Aspects use a general keyword list."
        ),
    }
    (d / "insights.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def _full_analysis(brand: str, place_filter: str = "auto") -> dict:
    eng = _engagement_stats(brand, place_filter)
    sent = _analyze_sentiment(brand, place_filter)
    ins = _product_insights(brand, place_filter)
    summary = {"brand": brand, "avg_stars": eng["avg_stars"], "stars_distribution": eng["stars_distribution"], "social": eng["social"],
               "sentiment": {"reviews": sent["reviews"]["sentiment_pct"], "posts": sent["posts"]["sentiment_pct"],
                             "text_vs_stars_agreement_pct": sent["text_vs_stars_agreement_pct"]},
               "happiest_product": ins["happiest_product"], "unhappiest_product": ins["unhappiest_product"],
               "best_aspect": ins["best_aspect"], "worst_aspect": ins["worst_aspect"], "top_complaints": ins["top_complaints"],
               "top_praises": ins["top_praises"], "saved_to": sent["saved_to"]}
    (_analysis_dir(brand) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- MCP tools
def analyze_website(url: str, save_as: str = "") -> dict:
    """Analyze a brand website (metadata, headings, contacts, social links, images, colors, fonts, ad pixels).
    Works on blocked sites through automatic fallbacks. Pass save_as=<brand> to also save the full result to disk."""
    res = _analyze_website(url)
    if save_as:
        d = _brand_dir(save_as)
        (d / "website.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        res["saved_to"] = str(d / "website.json")
    res["images"] = [i["url"] for i in res.get("images", [])[:10]] + (["..."] if len(res.get("images", [])) > 10 else [])
    return res


def get_social_profile(platform: str, url: str, posts_limit: int = 20) -> dict:
    """Social profile + recent posts + engagement for facebook | instagram | tiktok (needs APIFY_TOKEN)."""
    r = _social(platform, url, posts_limit)
    r["posts"] = r.get("posts", [])[:5]          # short preview only
    return r


def get_reviews(brand: str, location: str = "Egypt", max_reviews: int = 50) -> list:
    """Google Maps rating, review count, address and latest reviews for the brand's places (needs APIFY_TOKEN)."""
    out = _reviews(brand, location, max_reviews)
    return [{**p, "reviews": p["reviews"][:5]} for p in out]


def get_ads(brand: str, country_code: str = "EG", limit: int = 30) -> list:
    """Active ads of a brand or competitor from the Meta Ad Library (needs APIFY_TOKEN)."""
    return _ads(brand, country_code, limit)[:10]


def brand_report(brand: str, website: str = "", country: str = "Egypt", country_code: str = "EG",
                 posts_limit: int = 20, reviews_max: int = 50, social_urls_json: str = "", analyze: bool = False) -> dict:
    """FULL pipeline: website + images + social posts + Google Maps ratings + ads, saved to brand_data/<brand>/.
    social_urls_json: optional JSON like {"instagram": "https://instagram.com/x", "facebook": "...", "tiktok": "..."}
    to use when the site hides its social links. analyze=True also runs full_analysis. Returns a short summary; full data is on disk."""
    report = {"brand": brand, "country": country, "website_url": website, "max_images": 30}
    report["website"] = _analyze_website(website) if website else {}
    manual = json.loads(social_urls_json) if social_urls_json else {}
    report["social"], posts = {}, []
    for pf in ("instagram", "facebook", "tiktok"):
        link = manual.get(pf) or (report["website"].get("social", {}).get(pf) or [None])[0]
        if link:
            r = _social(pf, link, posts_limit)
            posts += r.pop("posts", [])
            report["social"][pf] = r
    report["posts"] = posts
    report["google_maps"] = _reviews(brand, country, reviews_max)
    report["ads"] = _ads(brand, country_code)
    saved = _save_report(brand, report)
    out = _summary(report, saved)
    if analyze:
        out["analysis"] = full_analysis(brand)
    return out


def engagement_stats(brand: str, place_filter: str = "auto") -> dict:
    """Counts for a saved brand: star distribution, Google Maps ratings, and likes / comments / shares per social platform.
    place_filter: 'auto' keeps Maps places whose name contains the brand's first word, '' disables filtering."""
    return _engagement_stats(brand, place_filter)


def analyze_sentiment(brand: str, place_filter: str = "auto") -> dict:
    """Sentiment analysis (positive / neutral / negative) of saved reviews and post captions with a Hugging Face model.
    Reviews without text use their star rating. Needs: pip install transformers torch sentencepiece."""
    try:
        return _analyze_sentiment(brand, place_filter)
    except ImportError:
        return {"error": "install the ML libraries first: pip install transformers torch sentencepiece"}


def product_insights(brand: str, place_filter: str = "auto") -> dict:
    """Which products and which aspects (quality, price, size, delivery, service, returns...) people are happy or unhappy about, and why.
    Uses saved sentiment results (runs analyze_sentiment first if needed)."""
    try:
        r = _product_insights(brand, place_filter)
    except ImportError:
        return {"error": "install the ML libraries first: pip install transformers torch sentencepiece"}
    return {**r, "products": r["products"][:10], "aspects": r["aspects"][:10]}


def full_analysis(brand: str, place_filter: str = "auto") -> dict:
    """Run engagement_stats + analyze_sentiment + product_insights on saved data and return a short summary (full files in brand_data/<brand>/analysis/)."""
    try:
        return _full_analysis(brand, place_filter)
    except ImportError:
        return {"error": "install the ML libraries first: pip install transformers torch sentencepiece"}


def save_data(brand: str, data_json: str, filename: str = "notes.json") -> dict:
    """Save any JSON you collected (e.g. from other tools) under brand_data/<brand>/<filename>."""
    d = _brand_dir(brand)
    p = d / re.sub(r"[^\w.\-]", "_", filename)
    p.write_text(json.dumps(json.loads(data_json), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved_to": str(p)}


def list_saved() -> list:
    """List the brands already saved on disk."""
    idx = DATA_DIR / "brands_index.jsonl"
    return [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()] if idx.exists() else []


def load_saved(brand: str, part: str = "summary") -> dict | list:
    """Load saved data for a brand. part: summary | website | social | google_maps | ads | posts."""
    f = _brand_dir(brand) / "report.json"
    if not f.exists():
        return {"error": f"nothing saved for '{brand}'. Run brand_report first."}
    r = json.loads(f.read_text(encoding="utf-8"))
    if part == "summary":
        return _summary(r, {"folder": str(f.parent)})
    v = r.get(part)
    return v[:20] if isinstance(v, list) else v

def route_file_to_tool(file_path: str, brand: str = "") -> dict:
    """
    Read a local file precisely, analyze its structure and content, 
    and match it with the most suitable MCP tool to process it.
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File '{file_path}' does not exist."}
        
    ext = path.suffix.lower()
    
    # 1. Handle JSON files (e.g., report.json)
    if ext == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                keys = data.keys()
                # Matches a full data collection report
                if "google_maps" in keys and "social" in keys:
                    return {
                        "file": file_path,
                        "detected_type": "Full Brand Report",
                        "suitable_tool": "full_analysis",
                        "reason": "The JSON contains comprehensive brand data (maps, social, website).",
                        "action": f"Call full_analysis('{brand or data.get('brand', 'YOUR_BRAND')}')"
                    }
                # Matches a previously analyzed summary
                elif "avg_stars" in keys and "sentiment" in keys:
                    return {
                        "file": file_path,
                        "detected_type": "Analysis Summary",
                        "suitable_tool": "load_saved",
                        "reason": "This looks like an already processed analysis summary.",
                        "action": f"Call load_saved('{brand or data.get('brand', 'YOUR_BRAND')}', part='summary')"
                    }
            return {
                "file": file_path,
                "detected_type": "Generic JSON",
                "suitable_tool": "save_data",
                "reason": "Generic JSON data can be stored using the save_data tool.",
            }
        except Exception as e:
            return {"error": f"Failed to parse JSON: {e}"}

    # 2. Handle CSV files (e.g., reviews.csv, posts.csv)
    elif ext == ".csv":
        rows = _read_csv(path)
        if not rows:
            return {"error": "CSV file is empty."}
            
        columns = list(rows[0].keys())
        # Matches raw Maps reviews
        if "stars" in columns and "text" in columns:
            return {
                "file": file_path,
                "detected_type": "Reviews Dataset",
                "suitable_tool": "analyze_sentiment",
                "reason": "Contains text and star ratings, perfect for sentiment analysis.",
                "action": f"Call analyze_sentiment('{brand}')" if brand else "Provide brand name to run analyze_sentiment"
            }
        # Matches raw Social media posts
        elif "likes" in columns and "comments" in columns:
            return {
                "file": file_path,
                "detected_type": "Social Posts Dataset",
                "suitable_tool": "engagement_stats",
                "reason": "Contains social engagement metrics (likes, comments, shares).",
                "action": f"Call engagement_stats('{brand}')" if brand else "Provide brand name to run engagement_stats"
            }
        # Matches already-scored sentiment data
        elif "sentiment" in columns and "score" in columns:
            return {
                "file": file_path,
                "detected_type": "Scored Sentiments",
                "suitable_tool": "product_insights",
                "reason": "Contains sentiment scores. Ready for product and aspect extraction.",
                "action": f"Call product_insights('{brand}')" if brand else "Provide brand name to run product_insights"
            }
        else:
            return {
                "file": file_path,
                "detected_type": "Generic CSV",
                "suitable_tool": "Unknown",
                "reason": f"Columns {columns} do not strongly match a specific analysis tool."
            }
            
    # 3. Handle Images
    elif ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        return {
            "file": file_path,
            "detected_type": "Image",
            "suitable_tool": "None",
            "reason": "This is an image file. It can be used as a reference in reports, but no specific analysis tool applies."
        }
        
    # 4. Fallback for other formats
    else:
        try:
            content = path.read_text(encoding="utf-8")[:300]
            return {
                "file": file_path,
                "detected_type": "Text/Unknown",
                "suitable_tool": "None",
                "reason": "Unsupported file extension.",
                "preview": content + "..."
            }
        except Exception:
            return {"error": "Binary or unreadable file format."}


# ---------------------------------------------------------------- entry
TOOLS = (analyze_website, get_social_profile, get_reviews, get_ads, brand_report,
         engagement_stats, analyze_sentiment, product_insights, full_analysis, save_data, list_saved, load_saved, route_file_to_tool)


def run_server():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("brand-research")
    for fn in TOOLS:
        mcp.tool()(fn)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        print(json.dumps(full_analysis(sys.argv[sys.argv.index("--analyze") + 1]), ensure_ascii=False, indent=2, default=str))
    elif "--test" in sys.argv:
        _, _, site, brand, *rest = sys.argv
        print(json.dumps(brand_report(brand, site, rest[0] if rest else "Egypt", rest[1] if len(rest) > 1 else "EG"),
                         ensure_ascii=False, indent=2, default=str)[:5000])
    else:
        run_server()
