-- Medical Desert Planner — Lakebase schema
-- Run against your Lakebase instance via psql or any Postgres client

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS scenarios (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT        NOT NULL,
    capability    TEXT        NOT NULL,
    description   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scenario_regions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id     UUID        NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    district_id     TEXT        NOT NULL,
    district_name   TEXT        NOT NULL,
    state_ut        TEXT        NOT NULL,
    classification  TEXT        NOT NULL CHECK (classification IN (
                        'crisis_zone', 'investigate', 'data_gap', 'served'
                    )),
    health_risk_score   FLOAT,
    coverage_score      FLOAT,
    planner_flag    TEXT        CHECK (planner_flag IN (
                        'confirmed', 'needs_survey', 'disputed', 'resolved', NULL
                    )),
    note            TEXT,
    override_score  FLOAT       CHECK (override_score BETWEEN 0 AND 1),
    reviewed_by     TEXT,
    reviewed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scenario_regions_scenario
    ON scenario_regions (scenario_id);

CREATE INDEX IF NOT EXISTS idx_scenario_regions_district
    ON scenario_regions (district_id);

CREATE TABLE IF NOT EXISTS facility_overrides (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id     UUID        NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    facility_id     TEXT        NOT NULL,
    capability      TEXT        NOT NULL,
    planner_score   FLOAT       CHECK (planner_score BETWEEN 0 AND 1),
    note            TEXT,
    reviewed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Trigger to keep updated_at current on scenarios
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS scenarios_updated_at ON scenarios;
CREATE TRIGGER scenarios_updated_at
    BEFORE UPDATE ON scenarios
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
