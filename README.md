# RAaHA Backend

Multimodal journey planning API for the RAaHA iOS app.

**Ranking categories:** Cheapest · Comfortable · Fastest  
**SAFEST has been removed.**

## Architecture

1. Geocode origin/destination (Nominatim + demo aliases)
2. Collect candidates from the provider registry:
   - **OSRM** — driving / cycling / walking geometry
   - **Google Routes API** — traffic-aware drive, two-wheeler, transit (requires key)
   - **Chennai GTFS** — MTC bus, CMRL metro, suburban/MRTS (local static feeds)
   - **RailRadar** — trains between stations (requires key)
3. Validate geographical feasibility (walk limits, continuity)
4. Rank the **same candidate pool** with deterministic objectives:
   - Cheapest → lowest comparable fare
   - Fastest → lowest door-to-door duration
   - Comfortable → fewest transfers, then walking/waiting
5. Optional experimental ML assist may break near-ties only; it never overrides the primary objective

## Setup

```bash
cd ~/Desktop/RAAHA/raaha-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with keys you obtain manually
./start_server.sh
```

Open Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Credentials checklist (manual)

| Variable | Where to get it | Required? |
|----------|-----------------|-----------|
| `GOOGLE_MAPS_API_KEY` | Google Cloud Console → enable **Routes API** | Optional (enables Google transit/traffic) |
| `RAILRADAR_API_KEY` | [railradar.in](https://railradar.in/docs) (`rg_…`) | Optional (live/scheduled train search) |

Without paid keys the API still works using **OSRM + Chennai GTFS**.

Never put API keys in the iOS app. Keep them in backend `.env` only (gitignored).

## GTFS data

Feeds live under `data/gtfs/`:

- `mtc-bus/` — Chennai MTC (from reclaimchennai/India_GTFS)
- `metro-cmrl/` — Chennai Metro
- `suburban-rail-mrts/` — Suburban / MRTS

Refresh:

```bash
./scripts/refresh_gtfs.sh
```

First startup parses stop_times into `data/gtfs/_cache/*.pkl` (MTC can take several minutes once).

Treat GTFS as **scheduled/static**, not live vehicle positions. CMRL calendars/fares may be stale — structural connectivity is still useful.

## Primary endpoint

`POST /api/journey/plan`

```json
{
  "origin": "SRM Institute of Science and Technology, Kattankulathur",
  "destination": "Marina Beach, Chennai",
  "preferences": ["cheapest", "comfortable", "fastest"]
}
```

Provider status: `GET /api/providers/status`

## ML component

- **Layer A (always on):** deterministic ranker (`app/ml/deterministic_ranker.py`)
- **Layer B (optional):** Random Forest artifact trained historically on **synthetic** data. It is labelled experimental and is only a soft near-tie assist when `ENABLE_ML_ASSIST=true` and the model file exists.

Do not treat the synthetic RF as real-world Chennai behaviour.

## Tests

```bash
pytest -q
```

## Known limitations

- Google / RailRadar features stay disabled until you add keys and verify authenticated calls.
- GTFS fares are estimated bands unless a verified fare file is applied.
- Google Transit does not return every multimodal combination; RAaHA builds additional combinations from GTFS + road providers.
- Nominatim/OSRM public endpoints have rate limits — prefer self-hosted OSRM for production.
