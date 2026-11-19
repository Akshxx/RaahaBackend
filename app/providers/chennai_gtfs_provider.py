"""Chennai GTFS loader and transit candidate builder (MTC / CMRL / suburban)."""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.settings import get_settings
from app.providers.base_provider import (
    BaseProvider,
    DataStatus,
    NormalizedLeg,
    NormalizedRoute,
    ProviderStatus,
)

logger = logging.getLogger("raaha.providers.gtfs")

FEED_MODE = {
    "mtc-bus": "bus",
    "metro-cmrl": "metro",
    "suburban-rail-mrts": "train",
}

# Suburban rail still has no published fare file. Metro uses fare_rules.txt.
# MTC buses use data/fares/mtc_fares_official.json.
FARE_ESTIMATE_INR = {
    "train": lambda km: max(5.0, min(40.0, 5.0 + km * 0.8)),
}

# MTC's own note: one fare stage is about 2 km. The feed has no per-stop stage numbers.
MTC_STAGE_METERS = 2000.0
MTC_FARE_PATH = Path(__file__).resolve().parents[2] / "data" / "fares" / "mtc_fares_official.json"


@dataclass
class GtfsStop:
    stop_id: str
    name: str
    lat: float
    lon: float
    feed: str
    mode: str
    zone: str = ""


@dataclass
class GtfsFeedIndex:
    feed: str
    mode: str
    source_url: str
    downloaded_at: str
    stops: Dict[str, GtfsStop] = field(default_factory=dict)
    # stop_id -> list of (trip_id, stop_sequence, arrival_sec, departure_sec)
    stop_trips: Dict[str, List[Tuple[str, int, int, int]]] = field(default_factory=dict)
    # trip_id -> ordered list of (stop_id, seq, arr, dep)
    trip_stops: Dict[str, List[Tuple[str, int, int, int]]] = field(default_factory=dict)
    route_names: Dict[str, str] = field(default_factory=dict)
    trip_route: Dict[str, str] = field(default_factory=dict)
    # fare_id -> price in INR, from fare_attributes.txt
    fare_prices: Dict[str, float] = field(default_factory=dict)
    # (origin zone, destination zone) -> price in INR, from fare_rules.txt
    fare_pairs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


class ChennaiGTFSProvider(BaseProvider):
    name = "chennai_gtfs"

    def __init__(self) -> None:
        self._feeds: Dict[str, GtfsFeedIndex] = {}
        self._loaded = False
        self._load_error: Optional[str] = None

    def status(self) -> ProviderStatus:
        self.ensure_loaded()
        return ProviderStatus(
            name=self.name,
            available=bool(self._feeds),
            configured=bool(self._feeds),
            message=self._load_error or f"{len(self._feeds)} GTFS feeds loaded",
            details={
                feed: {
                    "stops": len(index.stops),
                    "trips": len(index.trip_stops),
                    "mode": index.mode,
                    "source_url": index.source_url,
                    "downloaded_at": index.downloaded_at,
                    "validation": index.validation,
                }
                for feed, index in self._feeds.items()
            },
        )

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        settings = get_settings()
        cache_dir = settings.gtfs_data_dir / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "feed_meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                meta = {}

        for feed_name, mode in FEED_MODE.items():
            feed_dir = settings.gtfs_data_dir / feed_name
            if not (feed_dir / "stops.txt").exists():
                continue
            cache_file = cache_dir / f"{feed_name}.pkl"
            try:
                index: Optional[GtfsFeedIndex] = None
                if cache_file.exists():
                    try:
                        with cache_file.open("rb") as fh:
                            index = pickle.load(fh)
                    except Exception:  # noqa: BLE001
                        index = None
                if index is None or not _fare_cache_current(index, feed_dir):
                    index = self._parse_feed(feed_name, mode, feed_dir)
                    with cache_file.open("wb") as fh:
                        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
                self._feeds[feed_name] = index
                meta[feed_name] = {
                    "source_url": index.source_url,
                    "downloaded_at": index.downloaded_at,
                    "stops": len(index.stops),
                    "trips": len(index.trip_stops),
                    "validation": index.validation,
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed loading GTFS feed %s", feed_name)
                self._load_error = f"{feed_name}: {exc}"
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
        except OSError as exc:
            logger.warning("Could not write GTFS meta cache: %s", exc)
        self._loaded = True
        logger.info("GTFS feeds ready: %s", list(self._feeds))

    def nearby_stops(
        self,
        lat: float,
        lon: float,
        radius_m: float = 1200.0,
        limit: int = 8,
        modes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        self.ensure_loaded()
        found: List[Dict[str, Any]] = []
        for index in self._feeds.values():
            if modes and index.mode not in modes:
                continue
            for stop in index.stops.values():
                dist = haversine_m(lat, lon, stop.lat, stop.lon)
                if dist <= radius_m:
                    found.append(
                        {
                            "stop_id": stop.stop_id,
                            "name": stop.name,
                            "lat": stop.lat,
                            "lon": stop.lon,
                            "feed": stop.feed,
                            "mode": stop.mode,
                            "distance_m": dist,
                        }
                    )
        found.sort(key=lambda s: s["distance_m"])
        return found[:limit]

    def routes(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        **kwargs: Any,
    ) -> List[NormalizedRoute]:
        self.ensure_loaded()
        settings = get_settings()
        # Walk-radius stops for local bus; larger radius for rail/metro hubs (auto access).
        origin_stops = self.nearby_stops(
            origin["latitude"],
            origin["longitude"],
            radius_m=settings.max_last_mile_walk_meters,
            limit=10,
        )
        dest_stops = self.nearby_stops(
            destination["latitude"],
            destination["longitude"],
            radius_m=settings.max_last_mile_walk_meters,
            limit=10,
        )
        origin_hubs = self.nearby_stops(
            origin["latitude"],
            origin["longitude"],
            radius_m=8000,
            limit=20,
            modes=["train", "metro"],
        )
        dest_hubs = self.nearby_stops(
            destination["latitude"],
            destination["longitude"],
            radius_m=8000,
            limit=25,
            modes=["train", "metro"],
        )
        # Merge unique stops (hubs may be farther than walk preference).
        by_key = {}
        for stop in origin_stops + origin_hubs:
            by_key[(stop["feed"], stop["stop_id"])] = stop
        origin_stops = list(by_key.values())
        by_key = {}
        for stop in dest_stops + dest_hubs:
            by_key[(stop["feed"], stop["stop_id"])] = stop
        dest_stops = list(by_key.values())
        if not origin_stops or not dest_stops:
            return []

        out: List[NormalizedRoute] = []
        # Direct same-feed / same-trip connections.
        for o_stop in origin_stops:
            for d_stop in dest_stops:
                if o_stop["feed"] != d_stop["feed"]:
                    continue
                direct = self._direct_trip_candidate(
                    origin, destination, o_stop, d_stop
                )
                if direct:
                    out.append(direct)
                if len(out) >= 8:
                    return _dedupe_routes(out)

        # Conservative one-transfer: only short walk between different-feed nearby stops
        # that each already have a direct ride from origin / to destination.
        out.extend(
            self._simple_interfeed_transfers(origin, destination, origin_stops, dest_stops)
        )
        return _dedupe_routes(out)[:12]

    def _direct_trip_candidate(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        o_stop: Dict[str, Any],
        d_stop: Dict[str, Any],
    ) -> Optional[NormalizedRoute]:
        index = self._feeds.get(o_stop["feed"])
        if not index:
            return None
        o_trips = {
            trip_id: (seq, arr, dep)
            for trip_id, seq, arr, dep in index.stop_trips.get(o_stop["stop_id"], [])
        }
        best: Optional[Tuple[str, int, int, int, int]] = None
        for trip_id, seq, arr, dep in index.stop_trips.get(d_stop["stop_id"], []):
            if trip_id not in o_trips:
                continue
            o_seq, o_arr, o_dep = o_trips[trip_id]
            if o_seq >= seq:
                continue
            duration_sec = max(60, arr - o_dep)
            if best is None or duration_sec < best[1]:
                best = (trip_id, duration_sec, o_dep, arr, o_seq)
        if not best:
            return None

        trip_id, duration_sec, o_dep, d_arr, _ = best
        mode = index.mode
        dist_m = haversine_m(o_stop["lat"], o_stop["lon"], d_stop["lat"], d_stop["lon"])
        # Reject absurdly short/long declared trip durations for geometry.
        duration_min = duration_sec / 60.0
        if duration_min > get_settings().max_route_duration_minutes:
            return None
        fare, fare_status = _leg_fare(index, o_stop["stop_id"], d_stop["stop_id"], dist_m)
        legs: List[NormalizedLeg] = []
        access_to = _access_legs(
            origin,
            o_stop,
            direction="to_stop",
            max_walk=get_settings().max_last_mile_walk_meters,
        )
        access_from = _access_legs(
            destination,
            d_stop,
            direction="from_stop",
            max_walk=get_settings().max_last_mile_walk_meters,
        )
        # If stop is far and access builder refused (>12km), reject.
        o_dist = haversine_m(origin["latitude"], origin["longitude"], o_stop["lat"], o_stop["lon"])
        d_dist = haversine_m(
            destination["latitude"], destination["longitude"], d_stop["lat"], d_stop["lon"]
        )
        if (o_dist >= 60 and not access_to) or (d_dist >= 60 and not access_from):
            return None
        legs.extend(access_to)
        route_id = index.trip_route.get(trip_id, "")
        route_name = index.route_names.get(route_id, route_id or trip_id)
        legs.append(
            NormalizedLeg(
                mode=mode,
                from_location=o_stop["name"],
                to_location=d_stop["name"],
                from_coordinates=(o_stop["lat"], o_stop["lon"]),
                to_coordinates=(d_stop["lat"], d_stop["lon"]),
                duration_minutes=duration_min,
                distance_meters=dist_m * 1.15,
                fare_inr=fare,
                waiting_minutes=6.0,
                provider=self.name,
                data_source=_fare_source(index, fare_status),
                duration_status=DataStatus.SCHEDULED,
                fare_status=fare_status,
                confidence=0.75,
                departure_time=_sec_to_hhmm(o_dep),
                arrival_time=_sec_to_hhmm(d_arr),
                metadata={"trip_id": trip_id, "route": route_name},
            )
        )
        legs.extend(access_from)
        if not _legs_feasible(legs):
            return None
        return NormalizedRoute(
            name=f"GTFS {mode}: {route_name} ({o_stop['name']}→{d_stop['name']})",
            legs=legs,
            provider=self.name,
            contains_simulated_data=False,
            data_sources=[f"gtfs:{index.feed}"],
            metadata={
                "description": _fare_description(index, fare_status),
                "comfort_index": {"metro": 0.78, "train": 0.6, "bus": 0.48}.get(mode, 0.5),
                "crowding_estimate": 0.65,
                "confidence": 0.75,
                "data_freshness": index.downloaded_at,
                "feed_validation": index.validation,
            },
        )

    def _simple_interfeed_transfers(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        origin_stops: List[Dict[str, Any]],
        dest_stops: List[Dict[str, Any]],
        limit: int = 3,
    ) -> List[NormalizedRoute]:
        """Origin feed ride → short walk → destination feed ride."""
        settings = get_settings()
        results: List[NormalizedRoute] = []
        mid_hubs = self.nearby_stops(
            (origin["latitude"] + destination["latitude"]) / 2.0,
            (origin["longitude"] + destination["longitude"]) / 2.0,
            radius_m=12000,
            limit=25,
        )
        for o_stop in origin_stops[:5]:
            for hub_a in mid_hubs:
                if hub_a["feed"] != o_stop["feed"]:
                    continue
                first = self._direct_trip_candidate(
                    origin,
                    {"name": hub_a["name"], "latitude": hub_a["lat"], "longitude": hub_a["lon"]},
                    o_stop,
                    hub_a,
                )
                if not first:
                    continue
                for hub_b in mid_hubs:
                    if hub_b["feed"] == hub_a["feed"]:
                        continue
                    transfer_m = haversine_m(
                        hub_a["lat"], hub_a["lon"], hub_b["lat"], hub_b["lon"]
                    )
                    if transfer_m > settings.max_transfer_walk_meters:
                        continue
                    for d_stop in dest_stops[:5]:
                        if d_stop["feed"] != hub_b["feed"]:
                            continue
                        second = self._direct_trip_candidate(
                            {
                                "name": hub_b["name"],
                                "latitude": hub_b["lat"],
                                "longitude": hub_b["lon"],
                            },
                            destination,
                            hub_b,
                            d_stop,
                        )
                        if not second:
                            continue
                        merged = _merge_transfer_routes(
                            first, second, hub_a, hub_b, transfer_m
                        )
                        if merged:
                            results.append(merged)
                        if len(results) >= limit:
                            return results
        return results

    def _parse_feed(self, feed_name: str, mode: str, feed_dir: Path) -> GtfsFeedIndex:
        started = time.perf_counter()
        source_url = (
            "https://github.com/reclaimchennai/India_GTFS/tree/main/tamil-nadu/chennai"
        )
        downloaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(feed_dir.stat().st_mtime))
        index = GtfsFeedIndex(
            feed=feed_name,
            mode=mode,
            source_url=source_url,
            downloaded_at=downloaded_at,
        )

        with (feed_dir / "stops.txt").open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    lat = float(row.get("stop_lat") or 0)
                    lon = float(row.get("stop_lon") or 0)
                except ValueError:
                    continue
                if not lat and not lon:
                    continue
                # Keep boarding stops and stations (0/1). Skip entrances/generic nodes.
                loc_type = str(row.get("location_type") or "0")
                if loc_type not in {"0", "1", ""}:
                    continue
                stop_id = row["stop_id"]
                index.stops[stop_id] = GtfsStop(
                    stop_id=stop_id,
                    name=row.get("stop_name") or stop_id,
                    lat=lat,
                    lon=lon,
                    feed=feed_name,
                    mode=mode,
                    zone=(row.get("zone_id") or row.get("stop_code") or "").strip(),
                )

        routes_file = feed_dir / "routes.txt"
        if routes_file.exists():
            with routes_file.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    index.route_names[row["route_id"]] = (
                        row.get("route_short_name")
                        or row.get("route_long_name")
                        or row["route_id"]
                    )

        trips_file = feed_dir / "trips.txt"
        if trips_file.exists():
            with trips_file.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    index.trip_route[row["trip_id"]] = row.get("route_id") or ""

        # stop_times — can be large; stream once.
        with (feed_dir / "stop_times.txt").open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                trip_id = row["trip_id"]
                stop_id = row["stop_id"]
                if stop_id not in index.stops:
                    continue
                try:
                    seq = int(float(row.get("stop_sequence") or 0))
                except ValueError:
                    continue
                arr = _parse_gtfs_time(row.get("arrival_time") or row.get("departure_time"))
                dep = _parse_gtfs_time(row.get("departure_time") or row.get("arrival_time"))
                index.trip_stops.setdefault(trip_id, []).append((stop_id, seq, arr, dep))
                index.stop_trips.setdefault(stop_id, []).append((trip_id, seq, arr, dep))

        for trip_id, stops in index.trip_stops.items():
            stops.sort(key=lambda item: item[1])

        _load_fare_table(index, feed_dir)

        elapsed = time.perf_counter() - started
        fare_note = (
            f" Metro token fares loaded from fare_rules.txt ({len(index.fare_pairs)} pairs)."
            if index.fare_pairs
            else ""
        )
        index.validation = {
            "stops": len(index.stops),
            "trips_indexed": len(index.trip_stops),
            "fare_pairs": len(index.fare_pairs),
            "parse_seconds": round(elapsed, 2),
            "status": "ok" if index.trip_stops else "empty_trips",
            "notes": (
                "Static GTFS — schedules may be stale. "
                "Not a live vehicle position feed."
                + fare_note
            ),
        }
        logger.info(
            "Parsed GTFS %s stops=%s trips=%s in %.1fs",
            feed_name,
            len(index.stops),
            len(index.trip_stops),
            elapsed,
        )
        return index


def _fare_cache_current(index: GtfsFeedIndex, feed_dir: Path) -> bool:
    """Cached indexes built before fare files were read must be rebuilt."""
    if not (feed_dir / "fare_rules.txt").exists():
        return True
    return bool(getattr(index, "fare_pairs", None))


def _load_fare_table(index: GtfsFeedIndex, feed_dir: Path) -> None:
    attributes = feed_dir / "fare_attributes.txt"
    rules = feed_dir / "fare_rules.txt"
    if not attributes.exists() or not rules.exists():
        return
    with attributes.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fare_id = (row.get("fare_id") or "").strip()
            if not fare_id:
                continue
            try:
                price = float(row.get("price") or "")
            except ValueError:
                continue
            index.fare_prices[fare_id] = price
    with rules.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            origin = (row.get("origin_id") or "").strip()
            destination = (row.get("destination_id") or "").strip()
            fare_id = (row.get("fare_id") or "").strip()
            if not origin or not destination or fare_id not in index.fare_prices:
                continue
            index.fare_pairs[(origin, destination)] = index.fare_prices[fare_id]


def _load_mtc_stage_fares() -> Dict[str, List[int]]:
    if not MTC_FARE_PATH.exists():
        return {}
    payload = json.loads(MTC_FARE_PATH.read_text(encoding="utf-8"))
    fares = payload.get("fares") or {}
    loaded: Dict[str, List[int]] = {}
    for key, values in fares.items():
        if isinstance(values, list) and values:
            loaded[str(key)] = [int(value) for value in values]
    return loaded


MTC_STAGE_FARES = _load_mtc_stage_fares()


def _mtc_stage_fare(dist_m: float) -> Tuple[Optional[float], DataStatus]:
    """Ordinary adult fare from MTC's published stage table.

    The GTFS feed does not label a route as express, deluxe, or AC, and it
    does not give each stop's official stage. One stage is counted per 2 km,
    which is MTC's published stage length, then looked up in the ordinary column.
    """
    table = MTC_STAGE_FARES.get("ordinary") or []
    if not table:
        return None, DataStatus.UNAVAILABLE
    stages = 1 if dist_m <= 0 else math.ceil(dist_m / MTC_STAGE_METERS)
    stages = max(1, min(stages, len(table)))
    return float(table[stages - 1]), DataStatus.ESTIMATED


def _leg_fare(
    index: GtfsFeedIndex,
    origin_stop_id: str,
    destination_stop_id: str,
    dist_m: float,
) -> Tuple[Optional[float], DataStatus]:
    if index.fare_pairs:
        origin_zone = index.stops[origin_stop_id].zone if origin_stop_id in index.stops else ""
        dest_zone = (
            index.stops[destination_stop_id].zone
            if destination_stop_id in index.stops
            else ""
        )
        fare = index.fare_pairs.get((origin_zone, dest_zone))
        if fare is None:
            return None, DataStatus.UNAVAILABLE
        return float(fare), DataStatus.SCHEDULED
    if index.mode == "bus":
        return _mtc_stage_fare(dist_m)
    fare_fn = FARE_ESTIMATE_INR.get(index.mode)
    if not fare_fn:
        return None, DataStatus.UNAVAILABLE
    return round(fare_fn(dist_m / 1000.0), 2), DataStatus.ESTIMATED


def _fare_source(index: GtfsFeedIndex, fare_status: DataStatus) -> str:
    if index.mode == "bus" and MTC_STAGE_FARES and fare_status == DataStatus.ESTIMATED:
        return "mtc_fares_official"
    if fare_status == DataStatus.SCHEDULED:
        return f"gtfs:{index.feed}/fare_rules"
    return f"gtfs:{index.feed}"


def _fare_description(index: GtfsFeedIndex, fare_status: DataStatus) -> str:
    if index.mode == "bus" and MTC_STAGE_FARES and fare_status == DataStatus.ESTIMATED:
        return (
            f"Scheduled GTFS {index.feed} connection. "
            "Fare is MTC's official ordinary stage fare from G.O. (Ms.) No. 48 "
            "(mtc_fares_official.json). One stage is counted per 2 km."
        )
    if index.fare_pairs and fare_status == DataStatus.SCHEDULED:
        return (
            f"Scheduled GTFS {index.feed} connection. "
            "Fare is the CMRL station-pair token fare from fare_rules.txt."
        )
    if index.fare_pairs:
        return (
            f"Scheduled GTFS {index.feed} connection. "
            "No fare_rules entry for this station pair."
        )
    return (
        f"Scheduled GTFS {index.feed} connection. "
        "Fare is an estimated band, not a live ticket quote."
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_gtfs_time(value: Optional[str]) -> int:
    if not value:
        return 0
    parts = value.strip().split(":")
    if len(parts) < 2:
        return 0
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(float(parts[2])) if len(parts) > 2 else 0
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return 0


def _sec_to_hhmm(sec: int) -> str:
    sec = int(sec) % (24 * 3600)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}"


def _access_legs(
    place: Dict[str, Any],
    stop: Dict[str, Any],
    direction: str,
    max_walk: float,
) -> List[NormalizedLeg]:
    dist = haversine_m(place["latitude"], place["longitude"], stop["lat"], stop["lon"])
    if dist < 60:
        return []
    # Hard reject absurd last-mile (no silent mega-walks).
    if dist > 12000:
        return []
    if dist > max_walk:
        # Feasible auto last-mile instead of impossible walk.
        mode = "auto"
        fare = round(30 + dist / 1000 * 14, 2)
        duration = max(5.0, dist / 250.0)
        fare_status = DataStatus.ESTIMATED
        distance_meters = dist
    else:
        mode = "walk"
        fare = 0.0
        duration = max(2.0, dist / 80.0)
        fare_status = DataStatus.ESTIMATED
        distance_meters = dist * 1.25  # approximate pedestrian factor
    if direction == "to_stop":
        frm, to = place["name"], stop["name"]
        frm_c = (place["latitude"], place["longitude"])
        to_c = (stop["lat"], stop["lon"])
    else:
        frm, to = stop["name"], place["name"]
        frm_c = (stop["lat"], stop["lon"])
        to_c = (place["latitude"], place["longitude"])
    return [
        NormalizedLeg(
            mode=mode,
            from_location=frm,
            to_location=to,
            from_coordinates=frm_c,
            to_coordinates=to_c,
            duration_minutes=duration,
            distance_meters=distance_meters,
            fare_inr=fare,
            provider="chennai_gtfs",
            data_source="gtfs_access_estimate",
            duration_status=DataStatus.ESTIMATED,
            fare_status=fare_status,
            confidence=0.55,
        )
    ]


def _legs_feasible(legs: List[NormalizedLeg]) -> bool:
    settings = get_settings()
    for leg in legs:
        if leg.mode == "walk" and (leg.distance_meters or 0) > settings.max_walk_only_meters:
            return False
    return True


def _dedupe_routes(routes: List[NormalizedRoute]) -> List[NormalizedRoute]:
    seen = set()
    out = []
    for route in routes:
        key = (
            tuple(leg.mode for leg in route.legs),
            tuple((leg.from_location, leg.to_location) for leg in route.legs),
            int(sum(leg.duration_minutes or 0 for leg in route.legs)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(route)
    return out


def _merge_transfer_routes(
    first: NormalizedRoute,
    second: NormalizedRoute,
    hub_a: Dict[str, Any],
    hub_b: Dict[str, Any],
    transfer_m: float,
) -> Optional[NormalizedRoute]:
    if transfer_m > get_settings().max_transfer_walk_meters:
        return None

    def strip_outer_access(legs: List[NormalizedLeg], keep_end_name: str, from_start: bool) -> List[NormalizedLeg]:
        """Keep transit core; drop OD access legs that bookend a segment."""
        if not legs:
            return []
        core = list(legs)
        if from_start and core and core[0].mode in {"walk", "auto"}:
            core = core[1:]
        if core and core[-1].mode in {"walk", "auto"} and core[-1].to_location != keep_end_name:
            core = core[:-1]
        return core

    first_core = strip_outer_access(first.legs, hub_a["name"], from_start=True)
    second_core = strip_outer_access(second.legs, hub_b["name"], from_start=True)
    legs: List[NormalizedLeg] = []
    # Origin access from first.
    if first.legs and first.legs[0].mode in {"walk", "auto"}:
        legs.append(first.legs[0])
    legs.extend(first_core)
    if transfer_m > 80:
        legs.append(
            NormalizedLeg(
                mode="walk",
                from_location=hub_a["name"],
                to_location=hub_b["name"],
                from_coordinates=(hub_a["lat"], hub_a["lon"]),
                to_coordinates=(hub_b["lat"], hub_b["lon"]),
                duration_minutes=max(2.0, transfer_m / 80.0),
                distance_meters=transfer_m * 1.25,
                fare_inr=0.0,
                provider="chennai_gtfs",
                data_source="gtfs_transfer_walk",
                duration_status=DataStatus.ESTIMATED,
                fare_status=DataStatus.ESTIMATED,
                confidence=0.6,
            )
        )
    legs.extend(second_core)
    if second.legs and second.legs[-1].mode in {"walk", "auto"}:
        legs.append(second.legs[-1])

    paid = [lg for lg in legs if lg.mode not in {"walk"}]
    if len(paid) - 1 > get_settings().max_transfers:
        return None
    return NormalizedRoute(
        name=f"GTFS transfer {hub_a['name']} → {hub_b['name']}",
        legs=legs,
        provider="chennai_gtfs",
        contains_simulated_data=False,
        data_sources=list(set(first.data_sources + second.data_sources)),
        metadata={
            "description": "One-transfer GTFS multimodal candidate with validated short walk.",
            "comfort_index": 0.5,
            "crowding_estimate": 0.7,
            "confidence": 0.65,
        },
    )
