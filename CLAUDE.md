# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Databricks Apps & Agents for Good Hackathon 2026 — Track 2: Medical Desert Planner**

Given 10,000 messy Indian healthcare facility records (51 columns, uneven free-text fields), build a Databricks App that helps a non-technical planner identify where real gaps in care exist vs. where we simply lack reliable data.

The app answers: *"Is this region underserved, or just under-documented?"*

## Stack

- **Lakebase** — managed Postgres (Free Edition). Stores planner-created scenarios, notes, overrides, and flagged regions. Connect via psycopg2 / SQLAlchemy / asyncpg.
- **AgentBricks** — Databricks agent orchestration. Handles capability extraction from free-text fields and evidence scoring.
- **Databricks Apps** — hosted UI, no separate frontend server.
- **GraphFrames** — Spark-native graph library. Represents facilities, districts, and states as nodes; relationships (located_in, claims_capability, part_of) as edges. Runs on Delta Tables directly — no external graph database.
- **Delta Lake** — stores raw facility data and derived GraphFrame vertex/edge tables.
- **Anthropic SDK** — Claude for extracting structured capability claims from noisy free-text. Use prompt caching on system prompts.

## Architecture

```
Raw CSV (10k facility records)
        ↓
  Delta Lake — raw + cleaned facility table
        ↓
  Evidence Scoring — per facility per capability, based on field coverage
  (description, capability, procedure, equipment fields)
        ↓
  GraphFrames — two Delta tables:
    vertices: facilities, districts, states
    edges:    located_in, claims_capability (with evidence_score), part_of
        ↓
  AgentBricks — answers planner queries by traversing the graph
  e.g. "which districts have no verified ICU within them?"
        ↓
  Lakebase — persists scenarios, planner notes, region overrides
        ↓
  Databricks Apps — choropleth map + drill-down UI
```

## GraphFrames Data Model

**Vertices table** (`graph_vertices`)
| column | example |
|---|---|
| id | `facility_001`, `district_patna`, `state_bihar` |
| type | `facility`, `district`, `state` |
| name | "AIIMS Patna", "Patna", "Bihar" |
| lat / lon | for facilities |
| population | for districts (from Census of India) |

**Edges table** (`graph_edges`)
| column | example |
|---|---|
| src | `facility_001` |
| dst | `district_patna` |
| relationship | `located_in`, `claims_capability`, `part_of` |
| capability | `ICU`, `maternity`, `dialysis` (for claims edges) |
| evidence_score | 0.0–1.0 (for claims edges) |
| evidence_fields | which fields supported the score |

Both tables persist to Delta so the graph loads instantly in the app without recomputing.

## Evidence Scoring Logic

Each facility-capability pair gets a score 0.0–1.0 built from field coverage:

- `description` mentions capability → +0.3
- `capability` field confirms it → +0.3
- `procedure` field supports it → +0.2
- `equipment` field supports it → +0.2
- Contradiction between fields → penalty

Score thresholds: `≥0.7` strong, `0.4–0.69` partial, `<0.4` weak/suspicious.

## Lakebase Schema (planned)

```sql
-- Planner-saved scenarios
CREATE TABLE scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    capability TEXT NOT NULL,
    geography_level TEXT,   -- 'state' | 'district'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Regions flagged within a scenario
CREATE TABLE scenario_regions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID REFERENCES scenarios(id),
    region_id TEXT NOT NULL,   -- matches vertex id in GraphFrames
    flag TEXT,                 -- 'true_desert' | 'data_gap' | 'needs_survey'
    note TEXT,
    override_score FLOAT,
    reviewed_at TIMESTAMPTZ DEFAULT now()
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
databricks aitools update   # keep current
databricks aitools list     # verify
```

## Commands

```bash
pytest
ruff check . && ruff format --check .
mypy .
```

## Key Conventions

- Evidence scores live on graph **edges**, not vertices — a facility node has no score; a facility→capability edge does.
- Vertex/edge tables are rebuilt from Delta on ingest, then frozen. No live mutations to the graph during a user session.
- All planner actions (notes, flags, overrides) go to Lakebase only — never mutate the graph or Delta source tables.
- Agent system prompts use prompt caching (`cache_control: {"type": "ephemeral"}`).
- All Lakebase schema changes go through migration files.
- Every UI element that shows a score must also show the evidence fields that produced it.
