import json
import time
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.errors import ProviderError
from app.ai.loggers.tools.get_reviews import logger
from app.ai.providers.reviews import get_apify, get_google_places
from app.ai.settings import get_ai_settings

MIN_MATCH = 0.50    # below this we return NO reviews (avoid another business's feedback)
CONFIDENT = 0.75
MAX_OUTPUT_CHARS = 15_000  # Law 7: bounded output


class GetReviewsInput(BaseModel):
    business_name: str = Field(min_length=1, description="Business or brand name, e.g. 'McDonald's'.")
    location: Optional[str] = Field(None, description="City/country to pick the right branch, e.g. 'Cairo, Egypt'.")
    provider: Literal["auto", "apify"] = Field(
        "auto", description="Use 'apify' to search and fetch reviews through Apify only."
    )
    max_reviews: int = Field(5, ge=1, le=200, description="Up to 5 = fast (Google). More = slow and paid (Apify).")
    reviews_sort: Optional[Literal["newest", "mostRelevant", "highestRanking", "lowestRanking"]] = Field(
        None, description="Order of reviews. Setting it uses Apify. 'lowestRanking' surfaces complaints.")
    language: str = Field("en", description="Language code for place names, e.g. 'en' or 'ar'.")


def _result(status: str, **fields) -> str:
    """JSON for the LLM; trims reviews from the end until it fits the size cap."""
    data = {"status": status, **{k: v for k, v in fields.items() if v not in (None, [], "")}}
    reviews = data.get("reviews", [])
    while reviews and len(json.dumps(data, ensure_ascii=False)) > MAX_OUTPUT_CHARS:
        reviews.pop()
        data["warnings"] = data.get("warnings", []) + ["Output trimmed to protect the context window."]
    return json.dumps(data, ensure_ascii=False)


def run_get_reviews(p: GetReviewsInput, *, google=None, apify=None) -> str:
    """The whole logic. Never raises. ``google`` / ``apify`` are injectable for tests."""
    warnings: list[str] = []
    # Both Google Places and Apify may incur usage charges. Tests may inject
    # deterministic fakes; real provider calls require a separate explicit opt-in.
    if google is None and apify is None and not get_ai_settings().allow_paid_review_calls:
        return _result(
            "unavailable",
            unavailable_reason=(
                "Live review providers are metered and disabled until an administrator enables "
                "AI_ALLOW_PAID_REVIEW_CALLS. No provider was contacted."
            ),
        )

    try:
        if p.provider == "apify":
            places = (apify or get_apify()).search(
                p.business_name,
                p.location,
                max_reviews=p.max_reviews,
                sort=p.reviews_sort,
                language=p.language,
            )
        else:
            places = (google or get_google_places()).search(p.business_name, p.location, p.language)
    except ProviderError as exc:
        return _result("unavailable", unavailable_reason=str(exc))
    if not places or places[0].match_score < MIN_MATCH:
        closest = f" Closest result: '{places[0].name}'." if places else ""
        return _result(
            "partial",
            unavailable_reason=f"No confident match for '{p.business_name}'.{closest} "
                               "Reviews were NOT returned to avoid mixing up businesses.",
            other_candidates=[{"name": x.name, "address": x.address} for x in places[:3]])
    best = places[0]
    if best.match_score < CONFIDENT:
        warnings.append(f"Low-confidence match: asked '{p.business_name}', got '{best.name}'. Verify.")
    if any(x.match_score >= CONFIDENT for x in places[1:]):
        warnings.append(f"Other branches share this name. Reviews are from one place only: {best.address}.")

    reviews = best.reviews[:p.max_reviews if p.provider == "apify" else 5]
    degraded = False
    if p.provider == "auto" and (p.max_reviews > 5 or p.reviews_sort):
        try:
            fetched = (apify or get_apify()).fetch(
                best.place_id, max_reviews=p.max_reviews, sort=p.reviews_sort, language=p.language)
            if fetched:
                reviews = fetched
            else:
                warnings.append("Apify returned no reviews; showing Google's 5 instead.")
                degraded = True
        except ProviderError as exc:
            warnings.append(f"Apify unavailable ({exc}); showing Google's 5 instead.")
            degraded = True
    reviews = reviews[:p.max_reviews]

    skewed = p.reviews_sort in ("lowestRanking", "highestRanking")
    return _result(
        "ok" if reviews and not degraded and best.match_score >= CONFIDENT else "partial",
        place_name=best.name, address=best.address, maps_uri=best.maps_uri,
        average_rating=best.rating, total_ratings=best.total_ratings,
        reviews=[r.model_dump(exclude_none=True) for r in reviews],
        warnings=warnings,
        unavailable_reason=None if reviews else "Place found but no reviews were returned.",
        note=("average_rating/total_ratings cover ALL reviews; the reviews below are only a sample"
              + (" deliberately skewed by the sort order: do not generalise." if skewed else ".")))


@tool(args_schema=GetReviewsInput)
def get_reviews_tool(
    business_name: str,
    location: Optional[str] = None,
    provider: str = "auto",
    max_reviews: int = 5,
    reviews_sort: Optional[str] = None,
    language: str = "en",
) -> str:
    """
    Fetch REAL Google Maps customer reviews and the overall rating of a business.

    Guidelines:
    1. Always pass `location` (city/country) so the right branch is matched.
     2. Set `provider="apify"` to search and fetch reviews through Apify only.
     3. Default max_reviews=5 is fast and cheap. Use more (up to 200) only when you need depth:
       it is slow (1-2 min) and paid.
     4. reviews_sort="lowestRanking" shows complaints, "newest" shows recent experience. These
       are skewed samples: never present them as the overall opinion.
     5. status "partial": read `warnings` / `unavailable_reason` first. status "unavailable":
       say there is no data. NEVER invent reviews.
    This tool does not analyse sentiment; do that on the returned text.
    """
    started = time.monotonic()
    out = run_get_reviews(GetReviewsInput(
        business_name=business_name, location=location, provider=provider, max_reviews=max_reviews,
        reviews_sort=reviews_sort, language=language))
    logger.info("reviews_done", business=business_name, requested=max_reviews,
                status=json.loads(out)["status"], duration_ms=int((time.monotonic() - started) * 1000))
    return out
