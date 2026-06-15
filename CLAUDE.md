# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Databricks Apps & Agents for Good Hackathon 2026 — Track 2: Medical Desert Planner**

Two datasets. One question: *"Where are people actually suffering because care is missing or untrustworthy?"*

The app combines facility coverage (what exists and how credible it is) with real health outcomes (what's actually happening to people) to identify true crisis zones — not just data gaps.

## Datasets

### 1. Facilities
`databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities`
- 10,000 Indian healthcare facility records, 51 columns
- Key fields: `name`, `address_stateOrRegion`, `address_city`, `address_zipOrPostcode`, `latitude`, `longitude`, `specialties` (JSON array), `description`, `capability`, `procedure`, `equipment`, `capacity`, `numberDoctors`, `yearEstablished`, `facilityTypeId`, `operatorTypeId`, `affiliationTypeIds`
- Free-text fields (`description`, `capability`, `procedure`, `equipment`) are noisy and unevenly populated — treat as claims to verify, not ground truth
- `specialties` JSON array is the most reliable structured capability signal

### 2. NFHS-5 District Health Indicators
`databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.nfhs_5_district_health_indicators`
- 706 districts × 109 columns — actual health outcomes from India's largest household survey
- Key fields for Medical Desert scoring:
  - `institutional_birth_rate` — proxy for maternity facility access
  - `anc_4_visits_pct` — proxy for prenatal care access
  - `vaccination_pct` — child healthcare access
  - `stunting_pct` — long-term nutrition/care failure
  - `health_insurance_coverage` — financial access
  - `skilled_birth_attendance` — emergency obstetric access
- **Data quality issue**: 45% of columns stored as STRING, not DOUBLE. 99.7% have trailing whitespace. Parentheses values e.g. `(12.3)` = unreliable estimate (small sample) — flag as `low_confidence`, do not drop.

**Cleaning pattern for NFHS-5:**
```python
from pyspark.sql import functions as F

df_clean = df.select([
    F.try_cast(F.trim(F.col(c)), "double").alias(c)
    if c not in ["district_name", "state_ut"]
    else F.trim(F.col(c)).alias(c)
    for c in df.columns
])
```

## Architecture

```
Facilities (Delta)          NFHS-5 (Delta)
      ↓                           ↓
  Evidence Scoring          Outcome Scoring
  per facility-capability   per district
  (field coverage → 0–1)    (6 key indicators → 0–1 risk score)
      ↓                           ↓
  Spatial Join: facility lat/lon → district
  (India district shapefile from GADM)
      ↓
  GraphFrames — vertex/edge tables persisted to Delta:
    Vertices: facilities, districts, states
    Edges:    located_in (facility→district, w/ distance)
              claims_capability (facility→capability, w/ evidence_score)
              part_of (district→state)
              health_risk (district→indicator, w/ risk_score)
      ↓
  AgentBricks — answers planner queries by traversing the graph
      ↓
  Lakebase — persists planner scenarios, notes, region flags, overrides
      ↓
  Databricks Apps — choropleth map + drill-down UI
```

## The Core 2×2

Every district is classified by crossing two signals:

|  | Strong facility evidence | Weak / no facility evidence |
|---|---|---|
| **Poor health outcomes** | Investigate — claims don't match reality | **True crisis zone** |
| **Good health outcomes** | Served | Data gap — lower urgency |

## Evidence Scoring (Facilities)

Per facility-capability pair, score 0.0–1.0:

| Signal | Weight |
|---|---|
| `specialties` JSON contains capability | +0.35 |
| `capability` field mentions it | +0.30 |
| `description` mentions it | +0.20 |
| `procedure` supports it | +0.10 |
| `equipment` supports it | +0.05 |
| Cross-field contradiction | −0.20 |
| `affiliationTypeIds` includes government/academic | +0.10 bonus |

Thresholds: `≥0.7` strong · `0.4–0.69` partial · `<0.4` weak/suspicious

## Health Risk Scoring (NFHS-5)

Per district, composite risk score 0.0–1.0 (higher = more at-risk):
- Invert positive indicators (high institutional birth rate = low risk)
- Normalize each indicator 0–1 across all 706 districts
- Weighted average (weights TBD once data is explored)
- Flag districts where NFHS-5 estimates are unreliable (parentheses values)

## Spatial Join Strategy

Facilities have `latitude`/`longitude`. NFHS-5 has `district_name` + `state_ut`. Link them via:
- **India district shapefile** from GADM (public, free): `gadm41_IND_2.shp` = district level
- Spatial join: point-in-polygon using `sedona` (Apache Sedona, Spark-native) or `geopandas` + broadcast
- Result: each facility gets a `district_id` matching NFHS-5 districts
- Fallback: fuzzy match `address_city` → `district_name` for facilities missing lat/lon

## GraphFrames Schema

**Vertices** (`graph_vertices` Delta table)
| column | values |
|---|---|
| id | `fac_{unique_id}`, `dist_{district_name}_{state}`, `state_{state_ut}` |
| type | `facility` · `district` · `state` |
| name | human-readable name |
| lat / lon | facilities only |
| health_risk_score | districts only (0–1) |
| nfhs_confidence | districts only: `high` · `low` (parentheses flag) |

**Edges** (`graph_edges` Delta table)
| column | values |
|---|---|
| src / dst | vertex ids |
| relationship | `located_in` · `claims_capability` · `part_of` · `health_risk` |
| capability | ICU, maternity, dialysis… (claims edges only) |
| evidence_score | 0–1 (claims edges only) |
| evidence_fields | which fields contributed (claims edges only) |
| distance_km | facility→district centroid (located_in edges) |

Vertex/edge tables are **rebuilt on ingest and frozen**. No mutations during a user session.

## Lakebase Schema

```sql
CREATE TABLE scenarios (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    capability  TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE scenario_regions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id     UUID REFERENCES scenarios(id),
    region_id       TEXT NOT NULL,          -- matches vertex id
    classification  TEXT,                   -- 'crisis_zone' | 'data_gap' | 'served' | 'investigate'
    note            TEXT,
    override_score  FLOAT,
    reviewed_at     TIMESTAMPTZ DEFAULT now()
);
```

## Development Setup

```bash
pip install -r requirements.txt

export LAKEBASE_CONNECTION_STRING="..."
export ANTHROPIC_API_KEY="..."
export DATABRICKS_HOST="..."
export DATABRICKS_TOKEN="..."
```

## Databricks Agent Skills (run once per machine)

```bash
databricks aitools install --agents claude-code --skills databricks-core,databricks-lakebase,databricks-apps,databricks-dabs
databricks aitools update
databricks aitools list
```

## Commands

```bash
pytest
ruff check . && ruff format --check .
mypy .
```

## Key Conventions

- Evidence scores live on graph **edges** (facility→capability), not on facility vertices.
- NFHS-5 health risk scores live on **district vertices**.
- All planner actions (notes, flags, overrides, scenarios) go to Lakebase only — never mutate Delta or graph tables.
- Agent system prompts use prompt caching (`cache_control: {"type": "ephemeral"}`).
- All Lakebase schema changes go through migration files.
- Every UI score shown to the planner must display the evidence fields that produced it.
- Parentheses values in NFHS-5 = unreliable estimate — show `low confidence` label, never silently drop.
