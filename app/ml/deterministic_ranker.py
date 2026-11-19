"""Deterministic three-preference ranking over a shared candidate pool.

Primary objectives (hard):
- cheapest  → lowest comparable fare
- fastest   → lowest door-to-door duration
- comfortable → fewest transfers, then walking, then waiting

ML scores may only break near-ties; they never override the primary objective.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


PREFERENCES = ("cheapest", "comfortable", "fastest")


def _duration(c: dict) -> float:
    return float(c.get("estimated_duration_minutes") or c.get("duration_minutes") or 1e9)


def _cost(c: dict) -> float:
    if c.get("fare_unavailable") and c.get("estimated_cost_inr") in (None, 0, 0.0):
        return 1e12  # unusable for cheapest primary
    if c.get("estimated_cost_inr") is not None:
        return float(c["estimated_cost_inr"])
    return float(c.get("cost_inr") or 1e12)


def _transfers(c: dict) -> float:
    return float(c.get("transfers") or 0)


def _walking(c: dict) -> float:
    return float(c.get("walking_meters") or 0)


def _waiting(c: dict) -> float:
    return float(c.get("waiting_minutes") or 0)


def _ml(c: dict, preference: str) -> float:
    scores = c.get("preference_scores") or {}
    return float(scores.get(preference, 0.0))


def _comparable_fare_candidates(candidates: List[dict]) -> List[dict]:
    usable = [
        c
        for c in candidates
        if not (
            c.get("fare_unavailable")
            and float(c.get("estimated_cost_inr") or 0) <= 0
        )
    ]
    return usable or candidates


def rank_cheapest(candidates: List[dict]) -> Tuple[dict, float, str]:
    pool = _comparable_fare_candidates(candidates)
    ordered = sorted(
        pool,
        key=lambda c: (_cost(c), _duration(c), _transfers(c), _walking(c), -_ml(c, "cheapest")),
    )
    winner = ordered[0]
    reason = (
        f"Selected because it has the lowest comparable estimated total fare "
        f"(₹{_cost(winner):.0f}), then lower duration/transfers."
    )
    if winner.get("fare_partial") or any(
        (leg.get("fare_status") == "estimated") for leg in (winner.get("legs") or [])
    ):
        reason += " Some fare components are estimated, not live ticket quotes."
    score = 1.0 / (1.0 + _cost(winner) / 100.0)
    return winner, score, reason


def rank_fastest(candidates: List[dict]) -> Tuple[dict, float, str]:
    ordered = sorted(
        candidates,
        key=lambda c: (
            _duration(c),
            _waiting(c),
            _transfers(c),
            _cost(c),
            -_ml(c, "fastest"),
        ),
    )
    winner = ordered[0]
    reason = (
        f"Selected because it has the lowest door-to-door duration "
        f"({_duration(winner):.0f} min), including walking/waiting where provided."
    )
    score = 1.0 / (1.0 + _duration(winner) / 60.0)
    return winner, score, reason


def _comfort_mode_penalty(c: dict) -> float:
    modes = {str(m).lower() for m in (c.get("modes") or [])}
    # Soft preference: avoid making two-wheeler the default "comfortable" pick
    # when a direct car/bus/train alternative has equal transfers.
    if "rapido" in modes or "bike" in modes:
        return 1.0
    return 0.0


def rank_comfortable(candidates: List[dict]) -> Tuple[dict, float, str]:
    ordered = sorted(
        candidates,
        key=lambda c: (
            _transfers(c),
            _comfort_mode_penalty(c),
            _walking(c),
            _waiting(c),
            _duration(c),
            _cost(c),
            -_ml(c, "comfortable"),
        ),
    )
    winner = ordered[0]
    reason = (
        f"Selected because it minimises transfers ({int(_transfers(winner))}), "
        f"then walking ({int(_walking(winner))} m) and waiting."
    )
    score = 1.0 / (1.0 + _transfers(winner) + _walking(winner) / 1000.0)
    return winner, score, reason


def route_signature(candidate: dict) -> Tuple:
    modes = tuple(str(m).lower() for m in (candidate.get("modes") or []))
    stops = tuple(
        (
            str(leg.get("from_location", "")),
            str(leg.get("to_location", "")),
            str(leg.get("mode", "")),
        )
        for leg in (candidate.get("legs") or [])
    )
    return modes, stops


def routes_similar(a: dict, b: dict) -> bool:
    if a.get("name") == b.get("name"):
        return True
    return route_signature(a) == route_signature(b)


def rank_all(
    candidates: List[dict],
    preferences: Optional[List[str]] = None,
    prefer_distinct: bool = True,
) -> Dict[str, dict]:
    preferences = [p for p in (preferences or list(PREFERENCES)) if p in PREFERENCES]
    if not preferences:
        preferences = list(PREFERENCES)
    if not candidates:
        return {}

    rankers = {
        "cheapest": rank_cheapest,
        "fastest": rank_fastest,
        "comfortable": rank_comfortable,
    }
    # Independent full orderings for diversity.
    orderings = {
        pref: _full_order(candidates, pref) for pref in preferences if pref in rankers
    }

    results: Dict[str, dict] = {}
    chosen: List[dict] = []
    # Choose cheapest/fastest first so comfort can diversify if near-identical.
    select_order = [p for p in ("cheapest", "fastest", "comfortable") if p in orderings]

    for preference in select_order:
        ordered = orderings[preference]
        best = ordered[0]
        selected = best
        reason_suffix = ""
        if prefer_distinct and chosen:
            for alt in ordered[1:]:
                if any(routes_similar(alt, prev) for prev in chosen):
                    continue
                if _near_tie(preference, best, alt):
                    selected = alt
                    reason_suffix = (
                        " Chosen as a near-tied distinct alternative for this preference."
                    )
                    break
            if any(routes_similar(selected, prev) for prev in chosen) and selected is best:
                reason_suffix = (
                    " Same route also ranks best for another preference because "
                    "no near-tied distinct alternative exists."
                )

        winner, score, reason = rankers[preference]([selected, best] if selected is not best else [selected])
        # Recompute reason against selected using primary metrics.
        if preference == "cheapest":
            _, score, reason = rank_cheapest([selected])
        elif preference == "fastest":
            _, score, reason = rank_fastest([selected])
        else:
            _, score, reason = rank_comfortable([selected])

        payload = dict(selected)
        payload["suitability_score"] = round(float(score), 4)
        payload["preference"] = preference
        payload["recommendation_reason"] = reason + reason_suffix
        payload["ranking_note"] = payload["recommendation_reason"]
        payload["ranking_method"] = "deterministic_primary"
        results[preference] = payload
        chosen.append(selected)

    return results


def _full_order(candidates: List[dict], preference: str) -> List[dict]:
    if preference == "cheapest":
        pool = _comparable_fare_candidates(candidates)
        return sorted(
            pool,
            key=lambda c: (_cost(c), _duration(c), _transfers(c), _walking(c)),
        )
    if preference == "fastest":
        return sorted(
            candidates,
            key=lambda c: (_duration(c), _waiting(c), _transfers(c), _cost(c)),
        )
    return sorted(
        candidates,
        key=lambda c: (
            _transfers(c),
            _comfort_mode_penalty(c),
            _walking(c),
            _waiting(c),
            _duration(c),
            _cost(c),
        ),
    )


def _near_tie(preference: str, best: dict, alt: dict) -> bool:
    """Allow diversity only when the primary objective stays essentially equal."""
    if preference == "cheapest":
        # Never pick a materially more expensive route just for diversity.
        return _cost(alt) <= _cost(best) + 5.0
    if preference == "fastest":
        # Never pick a slower route as "fastest" for diversity.
        return _duration(alt) <= _duration(best) + 2.0
    # comfortable: same transfer count and walking within 300 m
    return _transfers(best) == _transfers(alt) and abs(_walking(best) - _walking(alt)) <= 300
