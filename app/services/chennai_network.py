"""Simulated Chennai multimodal hub graph for prototype candidate generation.

This is NOT a live GTFS feed. Coordinates are approximate public locations used
to build geographically plausible transfer chains (e.g. Potheri ↔ Velachery
suburban corridor). Every candidate built from this graph must be labelled
simulated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Approximate hub coordinates (lat, lon).
HUBS: Dict[str, Dict[str, Any]] = {
    "srm_kattankulathur": {
        "name": "SRM Institute of Science and Technology, Kattankulathur",
        "lat": 12.8231,
        "lon": 80.0442,
    },
    "potheri_railway": {
        "name": "Potheri Railway Station",
        "lat": 12.8250,
        "lon": 80.0455,
    },
    "guindy_railway": {
        "name": "Guindy Railway Station",
        "lat": 13.0085,
        "lon": 80.2130,
    },
    "velachery_railway": {
        "name": "Velachery Railway Station",
        "lat": 12.9758,
        "lon": 80.2209,
    },
    "tirusulam_metro": {
        "name": "Tirusulam Metro Station",
        "lat": 12.9732,
        "lon": 80.1642,
    },
    "guindy_metro": {
        "name": "Guindy Metro Station",
        "lat": 13.0069,
        "lon": 80.2133,
    },
    "phoenix_marketcity": {
        "name": "Phoenix Marketcity, Velachery",
        "lat": 12.9916,
        "lon": 80.2170,
    },
    "marina_beach": {
        "name": "Marina Beach",
        "lat": 13.0500,
        "lon": 80.2824,
    },
    "chennai_central": {
        "name": "Chennai Central",
        "lat": 13.0827,
        "lon": 80.2707,
    },
}

# Directed edges: (from_hub, to_hub, mode, duration_min, fare_inr, distance_km)
# Suburban / metro edges are simulated network connectivity for the prototype.
EDGES: List[Tuple[str, str, str, int, float, float]] = [
    # First/last mile around SRM
    ("srm_kattankulathur", "potheri_railway", "walk", 8, 0.0, 0.45),
    ("potheri_railway", "srm_kattankulathur", "walk", 8, 0.0, 0.45),
    ("srm_kattankulathur", "potheri_railway", "auto", 6, 40.0, 0.6),
    # Suburban corridor (simulated)
    ("potheri_railway", "guindy_railway", "train", 38, 15.0, 28.0),
    ("potheri_railway", "velachery_railway", "train", 45, 15.0, 32.0),
    ("guindy_railway", "velachery_railway", "train", 12, 10.0, 6.0),
    ("velachery_railway", "guindy_railway", "train", 12, 10.0, 6.0),
    ("guindy_railway", "potheri_railway", "train", 38, 15.0, 28.0),
    ("velachery_railway", "potheri_railway", "train", 45, 15.0, 32.0),
    # Metro (simulated)
    ("tirusulam_metro", "guindy_metro", "metro", 18, 40.0, 12.0),
    ("guindy_metro", "tirusulam_metro", "metro", 18, 40.0, 12.0),
    # Station adjacency walks / autos (feasible short links only)
    ("guindy_railway", "guindy_metro", "walk", 6, 0.0, 0.35),
    ("guindy_metro", "guindy_railway", "walk", 6, 0.0, 0.35),
    ("velachery_railway", "phoenix_marketcity", "auto", 12, 80.0, 2.2),
    ("phoenix_marketcity", "velachery_railway", "auto", 12, 80.0, 2.2),
    # NOTE: Velachery ↔ Phoenix walk is intentionally NOT an edge (~2km+ street walk).
    ("guindy_metro", "phoenix_marketcity", "auto", 18, 110.0, 4.5),
    ("phoenix_marketcity", "guindy_metro", "auto", 18, 110.0, 4.5),
    ("tirusulam_metro", "srm_kattankulathur", "auto", 35, 220.0, 18.0),
    ("srm_kattankulathur", "tirusulam_metro", "auto", 35, 220.0, 18.0),
    # Bus-style simulated hops
    ("potheri_railway", "velachery_railway", "bus", 70, 35.0, 34.0),
    ("velachery_railway", "phoenix_marketcity", "bus", 15, 10.0, 2.5),
    ("srm_kattankulathur", "phoenix_marketcity", "bus", 75, 45.0, 36.0),
]


def nearest_hub(lat: float, lon: float, max_km: float = 8.0) -> Optional[str]:
    from app.services.route_feasibility import haversine_meters

    best = None
    best_m = max_km * 1000.0
    for hub_id, hub in HUBS.items():
        dist = haversine_meters(lat, lon, hub["lat"], hub["lon"])
        if dist <= best_m:
            best_m = dist
            best = hub_id
    return best


def edge_lookup(src: str, dst: str, mode: Optional[str] = None) -> Optional[Tuple]:
    for edge in EDGES:
        if edge[0] == src and edge[1] == dst and (mode is None or edge[2] == mode):
            return edge
    return None


def build_leg_from_edge(edge: Tuple, waiting_minutes: int = 0) -> Dict[str, Any]:
    src, dst, mode, duration, fare, distance_km = edge
    origin = HUBS[src]
    dest = HUBS[dst]
    return {
        "mode": mode,
        "from_location": origin["name"],
        "to_location": dest["name"],
        "from_coordinates": (origin["lat"], origin["lon"]),
        "to_coordinates": (dest["lat"], dest["lon"]),
        "estimated_duration_minutes": int(duration + waiting_minutes),
        "estimated_cost_inr": float(fare),
        "distance_km": float(distance_km),
        "distance_meters": float(distance_km * 1000.0),
        "data_source": "simulated_chennai_network",
        "waiting_minutes": waiting_minutes,
    }


def path_to_candidate(
    name: str,
    hub_path: List[str],
    mode_sequence: List[str],
    wait_per_transfer: int = 6,
    comfort: float = 0.55,
    crowding: float = 0.5,
    safety: float = 0.7,
    verified: float = 0.85,
    notes: str = "",
) -> Optional[Dict[str, Any]]:
    """Assemble a candidate from hub hops; returns None if any edge missing."""
    if len(hub_path) != len(mode_sequence) + 1:
        return None
    legs: List[Dict[str, Any]] = []
    for i, mode in enumerate(mode_sequence):
        edge = edge_lookup(hub_path[i], hub_path[i + 1], mode)
        if edge is None:
            return None
        wait = wait_per_transfer if i > 0 and mode != "walk" else 0
        legs.append(build_leg_from_edge(edge, waiting_minutes=wait))

    duration = sum(int(leg["estimated_duration_minutes"]) for leg in legs)
    cost = round(sum(float(leg["estimated_cost_inr"]) for leg in legs), 2)
    walking = sum(
        float(leg["distance_meters"])
        for leg in legs
        if leg["mode"] == "walk"
    )
    waiting = sum(int(leg.get("waiting_minutes") or 0) for leg in legs)
    modes = [leg["mode"] for leg in legs]
    transfers = max(0, len([m for m in modes if m != "walk"]) - 1)
    # Count mode changes more accurately
    non_walk = [m for m in modes if m != "walk"]
    transfers = max(0, len(non_walk) - 1)

    return {
        "name": name,
        "modes": modes,
        "legs": legs,
        "estimated_duration_minutes": duration,
        "estimated_cost_inr": cost,
        "transfers": transfers,
        "description": notes
        or "Simulated Chennai network candidate. Not a live timetable.",
        "duration_minutes": float(duration),
        "cost_inr": cost,
        "distance_km": round(sum(float(leg["distance_km"]) for leg in legs), 2),
        "walking_meters": walking,
        "waiting_minutes": float(waiting),
        "mode_count": float(len(set(modes))),
        "comfort_index": comfort,
        "crowding_estimate": crowding,
        "route_complexity": min(0.95, 0.25 + 0.15 * transfers + walking / 5000.0),
        "safety_proxy": safety,
        "verified_provider_ratio": verified,
        "paid_changes": float(len([m for m in non_walk if m not in ("walk",)])),
        "contains_simulated_data": True,
        "data_sources": ["simulated_chennai_network"],
        "safety_disclaimer": (
            "Safety attributes use a provisional mode policy and simulated proxies."
        ),
    }


def corridor_candidates_for_endpoints(
    origin: Dict[str, Any],
    destination: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate diverse simulated corridor paths when hubs match known places."""
    o_hub = nearest_hub(origin["latitude"], origin["longitude"], max_km=6.0)
    d_hub = nearest_hub(destination["latitude"], destination["longitude"], max_km=6.0)
    if not o_hub or not d_hub:
        return []

    # Normalize SRM / Phoenix aliases to canonical hubs for path templates.
    start = "srm_kattankulathur" if o_hub in {"srm_kattankulathur", "potheri_railway"} else o_hub
    end = "phoenix_marketcity" if d_hub in {"phoenix_marketcity", "velachery_railway"} else d_hub

    templates: List[Dict[str, Any]] = []

    if start == "srm_kattankulathur" and end == "phoenix_marketcity":
        templates = [
            {
                "name": "Suburban train + auto (Potheri→Velachery)",
                "hubs": ["srm_kattankulathur", "potheri_railway", "velachery_railway", "phoenix_marketcity"],
                "modes": ["walk", "train", "auto"],
                "comfort": 0.58,
                "crowding": 0.70,
                "safety": 0.84,
                "notes": "Simulated: walk to Potheri, suburban train to Velachery, auto last-mile to Phoenix. Not a live schedule.",
            },
            {
                "name": "Suburban train via Guindy + auto",
                "hubs": ["srm_kattankulathur", "potheri_railway", "guindy_railway", "velachery_railway", "phoenix_marketcity"],
                "modes": ["walk", "train", "train", "auto"],
                "comfort": 0.50,
                "crowding": 0.75,
                "safety": 0.82,
                "notes": "Simulated multi-stop suburban path with auto last-mile.",
            },
            {
                "name": "Auto to Tirusulam Metro + Metro + Auto",
                "hubs": ["srm_kattankulathur", "tirusulam_metro", "guindy_metro", "phoenix_marketcity"],
                "modes": ["auto", "metro", "auto"],
                "comfort": 0.72,
                "crowding": 0.55,
                "safety": 0.88,
                "notes": "Simulated metro-assisted journey. Crowding estimate is simulated.",
            },
            {
                "name": "Direct simulated bus (SRM→Phoenix)",
                "hubs": ["srm_kattankulathur", "phoenix_marketcity"],
                "modes": ["bus"],
                "comfort": 0.46,
                "crowding": 0.68,
                "safety": 0.78,
                "notes": "Simulated direct bus corridor estimate.",
            },
            {
                "name": "Bus to Velachery + Auto",
                "hubs": ["srm_kattankulathur", "potheri_railway", "velachery_railway", "phoenix_marketcity"],
                "modes": ["auto", "bus", "auto"],
                "comfort": 0.48,
                "crowding": 0.66,
                "safety": 0.70,
                "notes": "Simulated bus trunk with auto access legs.",
            },
        ]
    elif start == "phoenix_marketcity" and end == "srm_kattankulathur":
        templates = [
            {
                "name": "Auto + suburban train (Velachery→Potheri)",
                "hubs": ["phoenix_marketcity", "velachery_railway", "potheri_railway", "srm_kattankulathur"],
                "modes": ["auto", "train", "walk"],
                "comfort": 0.58,
                "crowding": 0.70,
                "safety": 0.84,
                "notes": "Simulated reverse suburban corridor.",
            },
            {
                "name": "Auto + Metro + Auto (reverse)",
                "hubs": ["phoenix_marketcity", "guindy_metro", "tirusulam_metro", "srm_kattankulathur"],
                "modes": ["auto", "metro", "auto"],
                "comfort": 0.72,
                "crowding": 0.55,
                "safety": 0.88,
                "notes": "Simulated reverse metro-assisted journey.",
            },
        ]

    # Intentionally impossible template used only in tests via helper, not here.
    out: List[Dict[str, Any]] = []
    for template in templates:
        candidate = path_to_candidate(
            name=template["name"],
            hub_path=template["hubs"],
            mode_sequence=template["modes"],
            comfort=template["comfort"],
            crowding=template["crowding"],
            safety=template["safety"],
            notes=template["notes"],
        )
        if candidate:
            out.append(candidate)
    return out


def impossible_potheri_thiruvallur_walk_candidate() -> Dict[str, Any]:
    """Test fixture: long walk that must fail feasibility."""
    # Thiruvallur approx
    thiruvallur = {"name": "Thiruvallur", "lat": 13.1439, "lon": 79.9080}
    potheri = HUBS["potheri_railway"]
    return {
        "name": "Impossible walk Potheri→Thiruvallur",
        "modes": ["walk"],
        "legs": [
            {
                "mode": "walk",
                "from_location": potheri["name"],
                "to_location": thiruvallur["name"],
                "from_coordinates": (potheri["lat"], potheri["lon"]),
                "to_coordinates": (thiruvallur["lat"], thiruvallur["lon"]),
                "estimated_duration_minutes": 480,
                "estimated_cost_inr": 0.0,
                "distance_km": 40.0,
                "distance_meters": 40000.0,
                "data_source": "test_fixture",
            }
        ],
        "estimated_duration_minutes": 480,
        "estimated_cost_inr": 0.0,
        "transfers": 0,
        "description": "Invalid long walk fixture",
        "walking_meters": 40000.0,
        "waiting_minutes": 0.0,
        "contains_simulated_data": True,
        "data_sources": ["test_fixture"],
    }
