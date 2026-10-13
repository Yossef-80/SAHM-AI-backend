"""Google Places and Apify providers for customer reviews."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field
from unidecode import unidecode

from app.ai.errors import ProviderNotConfiguredError, ProviderRequestError
from app.ai.loggers.providers.reviews import logger
from app.ai.settings import AISettings, get_ai_settings

_PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
_PLACES_FIELDS = (
    "places.id,places.displayName,places.formattedAddress,places.googleMapsUri,"
    "places.rating,places.userRatingCount,places.reviews"
)
class PlaceReview(BaseModel):
    rating: float | None = None
    text: str
    relative_time: str | None = None


class PlaceResult(BaseModel):
    place_id: str
    name: str
    address: str = ""
    maps_uri: str | None = None
    rating: float | None = None
    total_ratings: int | None = None
    match_score: float
    reviews: list[PlaceReview] = Field(default_factory=list)


def _review_from_google(raw: dict[str, Any]) -> PlaceReview:
    text = raw.get("text") or raw.get("originalText") or {}
    if isinstance(text, dict):
        text = text.get("text", "")
    relative_time = raw.get("relativePublishTimeDescription")
    if not relative_time and raw.get("publishTime"):
        relative_time = raw["publishTime"]
    return PlaceReview(rating=raw.get("rating"), text=str(text), relative_time=relative_time)


def _name_score(requested_name: str, found_name: str) -> float:
    normalize = lambda value: " ".join(re.findall(r"[\w]+", value.casefold()))
    requested, found = normalize(requested_name), normalize(found_name)
    if not requested or not found:
        return 0.0
    requested_tokens, found_tokens = set(requested.split()), set(found.split())
    token_score = len(requested_tokens & found_tokens) / len(requested_tokens)
    transliterate = lambda value: "".join(
        character for character in unidecode(value).casefold() if character.isalnum()
    )
    transliterated_score = SequenceMatcher(
        None, transliterate(requested_name), transliterate(found_name)
    ).ratio()
    return max(SequenceMatcher(None, requested, found).ratio(), token_score, transliterated_score)


class GooglePlacesProvider:
    def __init__(self, settings: AISettings) -> None:
        if not settings.google_places_api_key:
            raise ProviderNotConfiguredError("GOOGLE_PLACES_API_KEY is not configured.")
        self._key = settings.google_places_api_key
        self._timeout = settings.reviews_timeout_seconds

    def search(self, business_name: str, location: str | None, language: str) -> list[PlaceResult]:
        query = ", ".join(part for part in (business_name, location) if part)
        try:
            response = httpx.post(
                _PLACES_URL,
                headers={
                    "X-Goog-Api-Key": self._key,
                    "X-Goog-FieldMask": _PLACES_FIELDS,
                    "Content-Type": "application/json",
                },
                json={"textQuery": query, "languageCode": language, "maxResultCount": 5},
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw_places = response.json().get("places", [])
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.error("google_places_request_failed", query=query, error=type(exc).__name__)
            raise ProviderRequestError(f"Google Places request failed: {type(exc).__name__}") from exc

        results = []
        for raw in raw_places:
            display_name = raw.get("displayName") or {}
            name = display_name.get("text", "") if isinstance(display_name, dict) else str(display_name)
            results.append(
                PlaceResult(
                    place_id=raw.get("id", ""),
                    name=name,
                    address=raw.get("formattedAddress", ""),
                    maps_uri=raw.get("googleMapsUri"),
                    rating=raw.get("rating"),
                    total_ratings=raw.get("userRatingCount"),
                    match_score=_name_score(business_name, name),
                    reviews=[_review_from_google(review) for review in raw.get("reviews", [])],
                )
            )
        return sorted(results, key=lambda place: place.match_score, reverse=True)


class ApifyReviewsProvider:
    def __init__(self, settings: AISettings) -> None:
        if not settings.apify_api_token:
            raise ProviderNotConfiguredError("APIFY_API_TOKEN is not configured.")
        self._token = settings.apify_api_token
        self._actor_id = settings.google_maps_reviews_actor_id
        self._timeout = settings.reviews_timeout_seconds

    def _run(self, actor_input: dict[str, Any]) -> list[dict[str, Any]]:
        actor = quote(self._actor_id.replace("/", "~"), safe="~")
        url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
        try:
            response = httpx.post(
                url,
                params={"format": "json"},
                headers={"Authorization": f"Bearer {self._token}"},
                json=actor_input,
                timeout=self._timeout,
            )
            response.raise_for_status()
            rows = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json()
                error = payload.get("error") or {}
                detail = error.get("message") or error.get("type") or "request rejected"
            except (ValueError, TypeError):
                detail = "request rejected"
            logger.error(
                "apify_reviews_request_failed",
                status=exc.response.status_code,
                detail=detail,
            )
            raise ProviderRequestError(
                f"Apify returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.error("apify_reviews_request_failed", error=type(exc).__name__)
            raise ProviderRequestError(f"Apify reviews request failed: {type(exc).__name__}") from exc
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def search(
        self,
        business_name: str,
        location: str | None,
        *,
        max_reviews: int,
        sort: str | None,
        language: str,
    ) -> list[PlaceResult]:
        actor_input: dict[str, Any] = {
            "searchStringsArray": [business_name],
            "maxCrawledPlacesPerSearch": 1,
            "scrapePlaceDetailPage": True,
            "maxReviews": max_reviews,
            "reviewsSort": sort or "mostRelevant",
            "language": language,
            "scrapeReviewsPersonalData": False,
        }
        if location:
            actor_input["locationQuery"] = location
        rows = self._run(actor_input)
        results = []
        for row in rows:
            name = row.get("title") or row.get("name") or row.get("placeName") or ""
            raw_reviews = row.get("reviewsData") or row.get("reviews") or []
            if isinstance(raw_reviews, dict):
                raw_reviews = [raw_reviews]
            results.append(
                PlaceResult(
                    place_id=str(row.get("placeId") or row.get("place_id") or ""),
                    name=str(name),
                    address=str(row.get("address") or ""),
                    maps_uri=row.get("url") or row.get("googleMapsUri"),
                    rating=row.get("totalScore") or row.get("rating"),
                    total_ratings=row.get("reviewsCount") or row.get("userRatingCount"),
                    match_score=_name_score(business_name, str(name)),
                    reviews=[_review_from_apify(review) for review in raw_reviews if isinstance(review, dict)],
                )
            )
        return sorted(results, key=lambda place: place.match_score, reverse=True)

    def fetch(
        self, place_id: str, *, max_reviews: int, sort: str | None, language: str
    ) -> list[PlaceReview]:
        actor_input = {
            "startUrls": [{"url": f"https://www.google.com/maps/search/?api=1&query_place_id={quote(place_id)}"}],
            "maxReviews": max_reviews,
            "reviewsSort": sort or "mostRelevant",
            "language": language,
        }
        rows = self._run(actor_input)
        return [_review_from_apify(row) for row in rows[:max_reviews]]


def _review_from_apify(row: dict[str, Any]) -> PlaceReview:
    text = row.get("text") or row.get("reviewText") or row.get("textTranslated") or ""
    rating = row.get("stars") or row.get("rating")
    try:
        rating = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    return PlaceReview(
        rating=rating,
        text=str(text),
        relative_time=row.get("publishedAtDate") or row.get("publishedAt") or row.get("timeAgo"),
    )


def get_google_places() -> GooglePlacesProvider:
    return GooglePlacesProvider(get_ai_settings())


def get_apify() -> ApifyReviewsProvider:
    return ApifyReviewsProvider(get_ai_settings())


__all__ = ["ApifyReviewsProvider", "GooglePlacesProvider", "PlaceResult", "PlaceReview", "get_apify", "get_google_places"]
