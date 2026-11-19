"""Metro legs use the CMRL station-pair fare files, not the km formula."""

from pathlib import Path

from app.providers.base_provider import DataStatus
from app.providers.chennai_gtfs_provider import (
    ChennaiGTFSProvider,
    GtfsFeedIndex,
    GtfsStop,
    _leg_fare,
    haversine_m,
)

ROOT = Path(__file__).resolve().parents[1]
METRO_FEED = ROOT / "data" / "gtfs" / "metro-cmrl"


def _stop(stop_id: str, zone: str, mode: str = "metro") -> GtfsStop:
    return GtfsStop(
        stop_id=stop_id,
        name=stop_id,
        lat=0.0,
        lon=0.0,
        feed="metro-cmrl",
        mode=mode,
        zone=zone,
    )


def test_missing_metro_pair_is_not_filled_by_distance_formula():
    index = GtfsFeedIndex(
        feed="metro-cmrl",
        mode="metro",
        source_url="",
        downloaded_at="",
    )
    index.stops["a"] = _stop("a", "SAP")
    index.stops["b"] = _stop("b", "NOPE")
    index.fare_pairs[("SAP", "SGU")] = 30.0
    fare, status = _leg_fare(index, "a", "b", 12_000)
    assert fare is None
    assert status == DataStatus.UNAVAILABLE


def test_bus_fare_uses_official_mtc_stage_table():
    index = GtfsFeedIndex(
        feed="mtc-bus",
        mode="bus",
        source_url="",
        downloaded_at="",
    )
    index.stops["a"] = _stop("a", "", mode="bus")
    index.stops["b"] = _stop("b", "", mode="bus")
    # 10 km is 5 stages. Ordinary stage 5 in G.O. 48 is ₹9, not the old ₹23 formula.
    fare, status = _leg_fare(index, "a", "b", 10_000)
    assert fare == 9.0
    assert status == DataStatus.ESTIMATED
    # 2 km is exactly one stage.
    fare, status = _leg_fare(index, "a", "b", 2_000)
    assert fare == 5.0
    # The ordinary column stops at stage 30 (₹24).
    fare, status = _leg_fare(index, "a", "b", 80_000)
    assert fare == 24.0


def test_cmrl_feed_fare_replaces_km_formula():
    index = ChennaiGTFSProvider()._parse_feed("metro-cmrl", "metro", METRO_FEED)
    assert index.fare_pairs[("SAP", "SME")] == 10.0
    assert index.fare_pairs[("SAP", "SGU")] == 30.0
    assert index.fare_pairs[("SAP", "SWN")] == 50.0

    sap = next(
        stop
        for stop in index.stops.values()
        if stop.zone == "SAP" and stop.stop_id in index.stop_trips
    )
    sme = next(
        stop
        for stop in index.stops.values()
        if stop.zone == "SME" and stop.stop_id in index.stop_trips
    )
    distance_m = haversine_m(sap.lat, sap.lon, sme.lat, sme.lon)
    formula = round(max(10.0, min(50.0, 10.0 + (distance_m / 1000.0) * 2.0)), 2)
    fare, status = _leg_fare(index, sap.stop_id, sme.stop_id, distance_m)
    assert status == DataStatus.SCHEDULED
    assert fare == 10.0
    assert fare != formula

    provider = ChennaiGTFSProvider()
    provider._feeds["metro-cmrl"] = index
    route = provider._direct_trip_candidate(
        {"name": "Airport", "latitude": sap.lat, "longitude": sap.lon},
        {"name": "Meenambakkam", "latitude": sme.lat, "longitude": sme.lon},
        {
            "stop_id": sap.stop_id,
            "name": sap.name,
            "lat": sap.lat,
            "lon": sap.lon,
            "feed": "metro-cmrl",
        },
        {
            "stop_id": sme.stop_id,
            "name": sme.name,
            "lat": sme.lat,
            "lon": sme.lon,
            "feed": "metro-cmrl",
        },
    )
    assert route is not None
    metro_legs = [leg for leg in route.legs if leg.mode == "metro"]
    assert len(metro_legs) == 1
    assert metro_legs[0].fare_inr == 10.0
    assert metro_legs[0].fare_status == DataStatus.SCHEDULED
    assert metro_legs[0].data_source == "gtfs:metro-cmrl/fare_rules"
