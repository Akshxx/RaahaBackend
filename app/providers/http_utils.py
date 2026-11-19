"""Shared HTTP helpers with timeout, retries and backoff."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.core.settings import get_settings

logger = logging.getLogger("raaha.providers.http")


def request_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    timeout = timeout if timeout is not None else settings.provider_timeout_seconds
    max_retries = (
        max_retries if max_retries is not None else settings.provider_max_retries
    )
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                )
            latency = (time.perf_counter() - started) * 1000.0
            if response.status_code in {401, 403}:
                raise PermissionError(
                    f"Auth failed for {url}: HTTP {response.status_code}"
                )
            if response.status_code == 429:
                raise RuntimeError(f"Rate limited by {url}")
            if response.status_code >= 500:
                raise RuntimeError(f"Server error {response.status_code} from {url}")
            response.raise_for_status()
            payload = response.json() if response.content else {}
            if isinstance(payload, dict):
                payload["_http_latency_ms"] = latency
            return payload if isinstance(payload, dict) else {"data": payload}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "HTTP %s %s attempt=%s failed: %s", method, url, attempt + 1, exc
            )
            if attempt >= max_retries:
                break
            time.sleep(0.4 * (2**attempt))
    assert last_error is not None
    raise last_error
