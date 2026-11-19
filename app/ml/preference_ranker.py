"""Preference-specific ranking: ML scores + transparent optimization guardrails."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from app.ml.optimization_config import DEFAULT_OPTIMIZATION_CONFIG, OptimizationConfig


def _modes(candidate: dict) -> List[str]:
    modes = candidate.get("modes") or []
    if modes:
        return [str(m).lower() for m in modes]
    legs = candidate.get("legs") or []
    return [str(leg.get("mode", "")).lower() for leg in legs if leg.get("mode")]


def _duration(c: dict) -> float:
    return float(c.get("estimated_duration_minutes") or c.get("duration_minutes") or 999)


def _cost(c: dict) -> float:
    # Prefer explicit total; otherwise sum leg fares.
    if c.get("estimated_cost_inr") is not None:
        return float(c["estimated_cost_inr"])
    legs = c.get("legs") or []
    if legs:
        return float(sum(float(leg.get("estimated_cost_inr") or 0) for leg in legs))
    return float(c.get("cost_inr") or 0)


def _walking(c: dict) -> float:
    return float(c.get("walking_meters") or 0)


def _transfers(c: dict) -> float:
    return float(c.get("transfers") or 0)


def _waiting(c: dict) -> float:
    return float(c.get("waiting_minutes") or 0)


def _crowding(c: dict) -> float:
    return float(c.get("crowding_estimate") or 0.5)


def _comfort(c: dict) -> float:
    return float(c.get("comfort_index") or 0.5)


def _ml_score(c: dict, preference: str) -> float:
    scores = c.get("preference_scores") or {}
    return float(scores.get(preference, 0.5))


def mode_hierarchy_score(candidate: dict, cfg: OptimizationConfig) -> float:
    """Journey safety from configured mode hierarchy (min / average blend)."""
    table = cfg.safest.mode_safety_scores
    modes = _modes(candidate) or ["walk"]
    # Prefer the public-transport trunk when present. Last-mile auto/walk is
    # scored as exposure separately so it cannot sink metro/train below bus.
    trunk = [mode for mode in modes if mode in {"metro", "train", "bus"}]
    if trunk:
        transit_modes = trunk
    else:
        transit_modes = [mode for mode in modes if mode != "walk"] or modes
    values = [table.get(mode, 0.45) for mode in transit_modes]
    return 0.65 * min(values) + 0.35 * (sum(values) / len(values))


def route_signature(candidate: dict) -> Tuple:
    modes = tuple(_modes(candidate))
    stops = tuple(
        (str(leg.get("from_location", "")), str(leg.get("to_location", "")), str(leg.get("mode", "")))
        for leg in (candidate.get("legs") or [])
    )
    return modes, stops


def routes_similar(a: dict, b: dict) -> bool:
    if a.get("name") == b.get("name"):
        return True
    return route_signature(a) == route_signature(b)


class PreferenceRanker:
    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        self.config = config or DEFAULT_OPTIMIZATION_CONFIG

    def _ranked_list(
        self,
        preference: str,
        candidates: List[dict],
    ) -> List[Tuple[float, dict, str]]:
        method = {
            "fastest": self.rank_fastest,
            "cheapest": self.rank_cheapest,
            "comfortable": self.rank_comfortable,
            "safest": self.rank_safest,
        }[preference]
        # Score every candidate with the preference objective by reusing pairwise logic
        # via a full re-score helper when available.
        scorer = {
            "fastest": self._score_fastest_all,
            "cheapest": self._score_cheapest_all,
            "comfortable": self._score_comfortable_all,
            "safest": self._score_safest_all,
        }[preference]
        return scorer(candidates)

    def rank_all(
        self,
        candidates: List[dict],
        preferences: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        preferences = preferences or ["fastest", "cheapest", "comfortable", "safest"]
        methods: Dict[str, Callable[[List[dict]], Tuple[dict, float, str]]] = {
            "fastest": self.rank_fastest,
            "cheapest": self.rank_cheapest,
            "comfortable": self.rank_comfortable,
            "safest": self.rank_safest,
        }
        # Independent ranking first.
        scored_lists = {
            preference: self._ranked_list(preference, candidates)
            for preference in preferences
            if preference in methods and candidates
        }
        results: Dict[str, dict] = {}
        chosen: List[dict] = []

        order = [p for p in self.config.diversity.prefer_distinct_order if p in scored_lists]
        for preference in order:
            ranked = scored_lists[preference]
            if not ranked:
                continue
            best_score, best, reason = ranked[0]
            selected_score, selected, selected_reason = best_score, best, reason

            if self.config.diversity.enabled and chosen:
                for score, candidate, cand_reason in ranked[1:]:
                    if any(routes_similar(candidate, prev) for prev in chosen):
                        continue
                    if best_score <= 0:
                        break
                    if not self._near_tie_for_preference(
                        preference, best, best_score, candidate, score
                    ):
                        continue
                    selected_score, selected, selected_reason = score, candidate, (
                        cand_reason
                        + " Chosen as the best distinct alternative for this preference "
                        "among near-tied feasible options."
                    )
                    break
                # If top is duplicate and no near-tie alternate, keep top and explain.
                if any(routes_similar(selected, prev) for prev in chosen) and selected is best:
                    selected_reason = (
                        reason
                        + " Same route also ranks best for another preference because "
                        "no near-tied distinct alternative exists in the feasible pool."
                    )

            payload = dict(selected)
            payload["suitability_score"] = round(float(selected_score), 4)
            payload["preference"] = preference
            payload["recommendation_reason"] = selected_reason
            payload["ranking_note"] = selected_reason
            if preference == "safest":
                payload["safety_disclaimer"] = self.config.safest.disclaimer
            results[preference] = payload
            chosen.append(selected)

        return results

    def _near_tie_for_preference(
        self,
        preference: str,
        best: dict,
        best_score: float,
        candidate: dict,
        score: float,
    ) -> bool:
        """Relative score near-tie plus preference-specific absolute bands."""
        gap = (best_score - score) / max(best_score, 1e-6)
        if gap > self.config.diversity.near_tie_ratio:
            return False
        if preference == "cheapest":
            return abs(_cost(candidate) - _cost(best)) <= self.config.diversity.cheapest_near_tie_inr
        if preference == "fastest":
            return abs(_duration(candidate) - _duration(best)) <= (
                self.config.diversity.fastest_near_tie_minutes
            )
        return True

    def _score_fastest_all(self, candidates: List[dict]) -> List[Tuple[float, dict, str]]:
        cfg = self.config.fastest
        ranked_by_time = sorted(candidates, key=lambda c: (_duration(c), _cost(c)))
        fastest = ranked_by_time[0]
        fastest_eta = _duration(fastest)
        near = [c for c in candidates if _duration(c) <= fastest_eta + cfg.near_fastest_minutes] or [fastest]
        reference = min(near, key=_cost)
        base_cost = _cost(reference)
        base_eta = _duration(reference)

        def fails_cost_guardrail(candidate: dict) -> bool:
            premium = max(0.0, _cost(candidate) - base_cost)
            saved = max(0.0, base_eta - _duration(candidate))
            if premium <= cfg.max_cost_premium_without_strong_gain:
                return False
            return saved < cfg.substantial_time_advantage_minutes

        viable = [c for c in near if not fails_cost_guardrail(c)] or [reference]
        scored: List[Tuple[float, dict, str]] = []
        for candidate in viable:
            eta = _duration(candidate)
            cost = _cost(candidate)
            time_saved = max(0.0, base_eta - eta)
            cost_premium = max(0.0, cost - base_cost)
            if cost_premium > cfg.max_cost_premium_without_strong_gain:
                efficiency = 0.78 + min(0.18, time_saved / 50.0)
                note = (
                    f"Selected because it offers a substantial time advantage "
                    f"({time_saved:.0f} min saved) despite higher cost (₹{cost:.0f})."
                )
            elif time_saved <= 0.5 and candidate is reference:
                efficiency = 0.92
                note = (
                    f"Selected because it is only {eta - fastest_eta:.0f} min slower than "
                    f"the absolute fastest option but costs ₹{cost:.0f} instead of "
                    f"₹{_cost(fastest):.0f}."
                )
            elif time_saved <= 0.5:
                efficiency = 0.88
                note = (
                    f"Among options within {cfg.near_fastest_minutes:.0f} min of the fastest ETA, "
                    f"favours practical cost (₹{cost:.0f})."
                )
            else:
                rupees_per_min = cost_premium / max(time_saved, 0.5)
                if rupees_per_min > cfg.max_rupees_per_minute_saved:
                    efficiency = 0.40
                    note = (
                        f"Time gain of {time_saved:.0f} min costs ₹{rupees_per_min:.0f}/min saved, "
                        f"above the ₹{cfg.max_rupees_per_minute_saved:.0f}/min guardrail."
                    )
                else:
                    efficiency = 0.86 + min(0.12, time_saved / 40.0)
                    note = (
                        f"Selected for a meaningful time advantage ({int(eta)} min) without an "
                        f"excessive cost increase (₹{cost:.0f})."
                    )
            time_component = 1.0 - min(1.0, (eta - fastest_eta) / max(cfg.near_fastest_minutes, 1.0))
            ml = _ml_score(candidate, "fastest")
            final = (
                (1.0 - cfg.ml_blend_weight) * (0.62 * efficiency + 0.38 * time_component)
                + cfg.ml_blend_weight * ml
            )
            final -= 0.03 * _transfers(candidate) + 0.01 * (_waiting(candidate) / 10.0)
            scored.append((final, candidate, note))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            best_score, best, reason = scored[0]
            if best is reference and best is not fastest:
                reason = (
                    f"Selected because it is only {_duration(best) - fastest_eta:.0f} min slower than "
                    f"the absolute fastest option but costs ₹{_cost(best):.0f} instead of "
                    f"₹{_cost(fastest):.0f}."
                )
                scored[0] = (best_score, best, reason)
            elif best is fastest and _cost(best) - base_cost > cfg.max_cost_premium_without_strong_gain:
                reason = (
                    f"Selected because it offers a substantial time advantage "
                    f"({base_eta - _duration(best):.0f} min saved) despite higher cost "
                    f"(₹{_cost(best):.0f} vs ₹{base_cost:.0f})."
                )
                scored[0] = (best_score, best, reason)
        return scored

    def rank_fastest(self, candidates: List[dict]) -> Tuple[dict, float, str]:
        scored = self._score_fastest_all(candidates)
        return scored[0][1], scored[0][0], scored[0][2]

    def _score_cheapest_all(self, candidates: List[dict]) -> List[Tuple[float, dict, str]]:
        cfg = self.config.cheapest
        durations = [_duration(c) for c in candidates]
        median_duration = sorted(durations)[len(durations) // 2]
        max_cost = max(_cost(c) for c in candidates) or 1.0
        min_cost = min(_cost(c) for c in candidates)
        scored: List[Tuple[float, dict, str]] = []
        for candidate in candidates:
            cost = _cost(candidate)
            duration = _duration(candidate)
            walking = _walking(candidate)
            cost_score = 1.0 if max_cost == min_cost else 1.0 - (cost - min_cost) / (max_cost - min_cost)
            penalty = 0.0
            reasons = [f"lowest practical total fare ₹{cost:.0f}"]
            if duration > median_duration * cfg.max_practical_duration_multiplier:
                penalty += cfg.extreme_duration_penalty
                reasons.append(f"duration {duration:.0f} min is long vs typical options")
            if walking > cfg.max_practical_walking_meters:
                penalty += cfg.extreme_walking_penalty
                reasons.append(f"walking {walking:.0f} m is high")
            # Mild bonus for suburban-train trunks (low fare + structured corridor).
            modes = _modes(candidate)
            if "train" in modes:
                cost_score = min(1.0, cost_score + 0.04)
            ml = _ml_score(candidate, "cheapest")
            final = (1.0 - cfg.ml_blend_weight) * max(0.0, cost_score - penalty) + cfg.ml_blend_weight * ml
            final -= 0.0005 * duration
            if penalty > 0:
                reason = (
                    f"Cost-first pick at ₹{cost:.0f}, with practicality checks applied "
                    f"({'; '.join(reasons[1:])})."
                )
            elif abs(cost - min_cost) < 0.5:
                reason = (
                    f"Selected because it has the lowest estimated total journey cost (₹{cost:.0f})."
                )
            else:
                reason = f"Selected as the best cost-first option at ₹{cost:.0f}."
            scored.append((final, candidate, reason))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def rank_cheapest(self, candidates: List[dict]) -> Tuple[dict, float, str]:
        scored = self._score_cheapest_all(candidates)
        return scored[0][1], scored[0][0], scored[0][2]

    def _score_comfortable_all(self, candidates: List[dict]) -> List[Tuple[float, dict, str]]:
        cfg = self.config.comfortable
        scored: List[Tuple[float, dict, str]] = []
        for candidate in candidates:
            transfers = _transfers(candidate)
            walking_km = _walking(candidate) / 1000.0
            crowding = _crowding(candidate)
            waiting = _waiting(candidate)
            comfort = _comfort(candidate)
            duration = _duration(candidate)
            modes = _modes(candidate)
            non_walk = [m for m in modes if m != "walk"]

            mode_bonus = 0.0
            # Door-to-door private modes help, but must not dominate a simple metro/train.
            if len(non_walk) <= 1 and transfers == 0:
                mode_bonus += 0.08
            if set(non_walk) <= {"car", "cab"}:
                mode_bonus += 0.08
            # Structured transit with light first/last-mile access is highly comfortable.
            has_rail = any(m in ("metro", "train") for m in modes)
            if has_rail and transfers <= 1 and walking_km <= 0.6:
                mode_bonus += 0.22
            elif has_rail and transfers == 2 and walking_km <= 0.35 and set(non_walk) <= {
                "metro",
                "train",
                "auto",
            }:
                # Auto↔rail↔auto is structured access, not a chaotic multi-hop.
                mode_bonus += 0.16
            if "rapido" in modes or "bike" in modes:
                mode_bonus -= 0.14
            if "auto" in non_walk and len(non_walk) == 1:
                mode_bonus -= 0.04  # direct auto is convenient but not premium comfort

            transfer_penalty = cfg.transfer_penalty * transfers
            # A single paid transit trunk with walk access is not a "hard" transfer chain.
            if transfers == 1 and len(non_walk) == 1:
                transfer_penalty *= 0.35
            if transfers == 2 and has_rail and walking_km <= 0.35:
                transfer_penalty *= 0.55
            if transfers >= 2:
                transfer_penalty += cfg.two_transfer_extra_penalty * (
                    0.45 if has_rail and walking_km <= 0.35 else 1.0
                )
            if transfers >= 3:
                transfer_penalty += cfg.three_plus_transfer_extra_penalty

            base = (
                0.28 * comfort
                + 0.14 * (1.0 - crowding)
                + mode_bonus
                - transfer_penalty
                - cfg.walking_penalty_per_km * walking_km
                - cfg.wait_penalty_per_10_min * (waiting / 10.0)
                - cfg.duration_soft_penalty * min(1.0, duration / 150.0)
            )
            # Soft cost pressure — comfort ≠ most expensive by default.
            base -= min(0.22, max(0.0, (_cost(candidate) - 120.0) / 900.0))
            ml = _ml_score(candidate, "comfortable")
            final = (1.0 - cfg.ml_blend_weight) * max(0.0, min(1.0, base)) + cfg.ml_blend_weight * ml

            reason = (
                f"Selected because it minimises travel effort "
                f"({int(transfers)} transfers, {int(_walking(candidate))} m walking) "
                f"using modes: {', '.join(modes) or 'n/a'}."
            )
            scored.append((final, candidate, reason))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def rank_comfortable(self, candidates: List[dict]) -> Tuple[dict, float, str]:
        scored = self._score_comfortable_all(candidates)
        return scored[0][1], scored[0][0], scored[0][2]

    def _score_safest_all(self, candidates: List[dict]) -> List[Tuple[float, dict, str]]:
        cfg = self.config.safest
        scored: List[Tuple[float, dict, str]] = []
        for candidate in candidates:
            # Feasibility must already have passed; still hard-reject flagged invalids.
            if candidate.get("feasible") is False:
                continue
            hierarchy = mode_hierarchy_score(candidate, self.config)
            walking_km = _walking(candidate) / 1000.0
            modes = _modes(candidate)
            trunk = [m for m in modes if m in {"metro", "train", "bus"}]
            # Last-mile auto next to a rail trunk is lighter road exposure.
            if trunk and any(m in {"metro", "train"} for m in trunk):
                road_modes = [m for m in modes if m in cfg.road_exposure_modes and m not in {"auto"}]
                road_share = len(road_modes) / max(len(modes), 1)
            else:
                road_share = sum(1 for m in modes if m in cfg.road_exposure_modes) / max(
                    len(modes), 1
                )
            verified = float(candidate.get("verified_provider_ratio") or 0.5)
            base = (
                0.82 * hierarchy
                + 0.10 * verified
                - cfg.walking_exposure_penalty_per_km * walking_km
                - cfg.road_exposure_penalty * road_share
            )
            ml = _ml_score(candidate, "safest")
            blend = min(cfg.ml_blend_weight, 0.15)
            final = (1.0 - blend) * max(0.0, min(1.0, base)) + blend * ml
            ranked_modes = trunk or [m for m in modes if m != "walk"] or modes or ["walk"]
            top_mode = max(ranked_modes, key=lambda m: cfg.mode_safety_scores.get(m, 0.4))
            reason = (
                f"Selected because it prioritises higher-ranked transport modes "
                f"(led by {top_mode}) and reduces exposure to less preferred modes "
                f"({', '.join(modes)})."
            )
            scored.append((final, candidate, reason))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def rank_safest(self, candidates: List[dict]) -> Tuple[dict, float, str]:
        scored = self._score_safest_all(candidates)
        return scored[0][1], scored[0][0], scored[0][2]
