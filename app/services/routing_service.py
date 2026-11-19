
import json
import subprocess

# FOSSGIS routing servers
ROUTING_URLS = {
    "car": (
        "https://routing.openstreetmap.de/"
        "routed-car/route/v1/driving"
    ),
    "bike": (
        "https://routing.openstreetmap.de/"
        "routed-bike/route/v1/driving"
    ),
    "walk": (
        "https://routing.openstreetmap.de/"
        "routed-foot/route/v1/driving"
    ),
}


def get_routes(
    origin: dict,
    destination: dict,
    mode: str = "car",
) -> list[dict]:

    if mode not in ROUTING_URLS:
        raise ValueError(f"Unsupported routing mode: {mode}")

    start = (
        f'{origin["longitude"]},'
        f'{origin["latitude"]}'
    )

    end = (
        f'{destination["longitude"]},'
        f'{destination["latitude"]}'
    )

    url = f"{ROUTING_URLS[mode]}/{start};{end}"

    try:
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time", "20",
                "--get",
                url,
                "--data-urlencode", "overview=full",
                "--data-urlencode", "geometries=geojson",
                "--data-urlencode", "alternatives=true",
                "--data-urlencode", "steps=true",
            ],
            capture_output=True,
            text=True,
            timeout=25,
            check=True,
        )

        data = json.loads(result.stdout)

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Routing request timed out."
        ) from exc

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Routing provider request failed."
        ) from exc

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid response from routing provider."
        ) from exc

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"No {mode} route found.")

    routes = []

    for index, route in enumerate(data["routes"]):
        distance_meters = route["distance"]
        duration_seconds = route["duration"]

        coordinates = route.get(
            "geometry", {}
        ).get("coordinates", [])

        instructions = []

        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                maneuver = step.get("maneuver", {})

                instructions.append({
                    "instruction": maneuver.get("type", ""),
                    "modifier": maneuver.get("modifier"),
                    "road_name": step.get("name", ""),
                    "distance_meters": round(
                        step.get("distance", 0)
                    ),
                    "duration_seconds": round(
                        step.get("duration", 0)
                    ),
                    "location": maneuver.get("location"),
                })

        routes.append({
            "route_id": f"osrm-{mode}-{index + 1}",
            "mode": mode,
            "distance_km": round(
                distance_meters / 1000, 2
            ),
            "distance_meters": round(distance_meters),
            "estimated_duration_minutes": round(
                duration_seconds / 60
            ),
            "estimated_cost_inr": None,
            "cost_status": "unavailable",
            "data_source": f"FOSSGIS OSRM ({mode})",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "instructions": instructions,
        })

    return routes


def get_driving_routes(
    origin: dict,
    destination: dict,
) -> list[dict]:
    """Backward-compatible car routing."""
    return get_routes(origin, destination, "car")


def get_bike_routes(
    origin: dict,
    destination: dict,
) -> list[dict]:
    return get_routes(origin, destination, "bike")


def get_walking_routes(
    origin: dict,
    destination: dict,
) -> list[dict]:
    return get_routes(origin, destination, "walk")

