"""
AgentBricks-style agent using Anthropic tool use.
Answers natural-language planner queries by running SQL against Delta tables.
"""
import json
import os
import anthropic
from databricks import sql as dbsql

CATALOG   = os.getenv("CATALOG",   "main")
SCHEMA    = os.getenv("SCHEMA",    "medical_desert")
DB_HOST   = os.getenv("DATABRICKS_HOST")
DB_TOKEN  = os.getenv("DATABRICKS_TOKEN")
DB_WH_ID  = os.getenv("DATABRICKS_WAREHOUSE_ID")

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _sql(query: str) -> list[dict]:
    with dbsql.connect(
        server_hostname=DB_HOST,
        http_path=f"/sql/1.0/warehouses/{DB_WH_ID}",
        access_token=DB_TOKEN,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "list_crisis_zones",
        "description": (
            "List districts classified as crisis zones or investigate for a given capability. "
            "Returns district name, state, health risk score, coverage score, and facility count. "
            "Use when the planner asks where the worst gaps are."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "enum": ["icu", "maternity", "emergency", "dialysis", "nicu", "oncology"],
                    "description": "The type of medical capability to check.",
                },
                "state": {
                    "type": "string",
                    "description": "Optional — filter to a specific Indian state.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 20).",
                    "default": 20,
                },
            },
            "required": ["capability"],
        },
    },
    {
        "name": "get_district_facilities",
        "description": (
            "Get facilities in a specific district for a given capability, "
            "ranked by evidence score. Includes evidence quotes for citation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "district_name": {"type": "string"},
                "state":         {"type": "string"},
                "capability":    {
                    "type": "string",
                    "enum": ["icu", "maternity", "emergency", "dialysis", "nicu", "oncology"],
                },
            },
            "required": ["district_name", "state", "capability"],
        },
    },
    {
        "name": "compare_states",
        "description": (
            "Compare states by average health risk score and facility coverage "
            "for a given capability. Useful for prioritisation across regions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "enum": ["icu", "maternity", "emergency", "dialysis", "nicu", "oncology"],
                },
            },
            "required": ["capability"],
        },
    },
    {
        "name": "get_district_detail",
        "description": (
            "Get full detail for one district and capability: classification, "
            "NFHS indicator values, and top-ranked facilities with evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "district_name": {"type": "string"},
                "state":         {"type": "string"},
                "capability":    {
                    "type": "string",
                    "enum": ["icu", "maternity", "emergency", "dialysis", "nicu", "oncology"],
                },
            },
            "required": ["district_name", "state", "capability"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

def list_crisis_zones(capability: str, state: str = None, limit: int = 20) -> list[dict]:
    state_filter = f"AND LOWER(state_ut) = LOWER('{state}')" if state else ""
    return _sql(f"""
        SELECT
            district_name, state_ut,
            classification,
            ROUND(health_risk_score, 3) AS health_risk,
            ROUND(coverage_score, 3)    AS coverage,
            facility_count,
            strong_count,
            nfhs_low_confidence
        FROM {CATALOG}.{SCHEMA}.district_capability_classification
        WHERE capability = '{capability}'
          AND classification IN ('crisis_zone', 'investigate')
          {state_filter}
        ORDER BY health_risk_score DESC, coverage_score ASC
        LIMIT {limit}
    """)


def get_district_facilities(district_name: str, state: str, capability: str) -> list[dict]:
    return _sql(f"""
        SELECT
            f.name,
            f.facilityTypeId    AS facility_type,
            f.operatorTypeId    AS operator,
            ROUND(s.evidence_score, 3) AS evidence_score,
            s.strength,
            s.evidence_fields,
            s.evidence_quotes
        FROM {CATALOG}.{SCHEMA}.facility_capability_scores s
        JOIN {CATALOG}.{SCHEMA}.facilities_located f
          ON s.facility_id = f.unique_id
        WHERE s.capability = '{capability}'
          AND LOWER(f.gadm_district) = LOWER('{district_name}')
          AND LOWER(f.gadm_state)    = LOWER('{state}')
        ORDER BY s.evidence_score DESC
        LIMIT 20
    """)


def compare_states(capability: str) -> list[dict]:
    return _sql(f"""
        SELECT
            state_ut,
            COUNT(*)                              AS total_districts,
            SUM(CASE WHEN classification = 'crisis_zone'  THEN 1 ELSE 0 END) AS crisis_zones,
            SUM(CASE WHEN classification = 'investigate'  THEN 1 ELSE 0 END) AS investigate,
            ROUND(AVG(health_risk_score), 3)      AS avg_health_risk,
            ROUND(AVG(coverage_score), 3)         AS avg_coverage,
            SUM(facility_count)                   AS total_facilities
        FROM {CATALOG}.{SCHEMA}.district_capability_classification
        WHERE capability = '{capability}'
        GROUP BY state_ut
        ORDER BY crisis_zones DESC, avg_health_risk DESC
    """)


def get_district_detail(district_name: str, state: str, capability: str) -> dict:
    classification = _sql(f"""
        SELECT *
        FROM {CATALOG}.{SCHEMA}.district_capability_classification
        WHERE capability = '{capability}'
          AND LOWER(district_name) = LOWER('{district_name}')
          AND LOWER(state_ut)      = LOWER('{state}')
        LIMIT 1
    """)
    facilities = get_district_facilities(district_name, state, capability)
    return {"classification": classification, "facilities": facilities}


def dispatch_tool(name: str, inputs: dict):
    if name == "list_crisis_zones":
        return list_crisis_zones(**inputs)
    if name == "get_district_facilities":
        return get_district_facilities(**inputs)
    if name == "compare_states":
        return compare_states(**inputs)
    if name == "get_district_detail":
        return get_district_detail(**inputs)
    raise ValueError(f"Unknown tool: {name}")


# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM = """You are a medical desert planning assistant for India. You help non-technical
healthcare planners, NGO coordinators, and analysts understand where care gaps are real vs.
where data is simply missing.

You have access to 10,000 healthcare facility records and district-level NFHS-5 health
outcome data. Always cite the evidence behind your answers. When data confidence is low,
say so explicitly. Never present weak evidence as fact."""


def run(user_message: str, history: list[dict] = None) -> str:
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    while True:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            return next(
                (b.text for b in resp.content if hasattr(b, "text")), ""
            )

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                result = dispatch_tool(block.name, block.input)
            except Exception as e:
                result = {"error": str(e)}
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})
