"""
Medical Desert Planner — Databricks App
Run with: gradio app.py
"""
import json
import os
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import gradio as gr
from databricks import sql as dbsql
from agent import run as agent_run

CATALOG   = os.getenv("CATALOG",   "main")
SCHEMA    = os.getenv("SCHEMA",    "medical_desert")
DB_HOST   = os.getenv("DATABRICKS_HOST")
DB_TOKEN  = os.getenv("DATABRICKS_TOKEN")
DB_WH_ID  = os.getenv("DATABRICKS_WAREHOUSE_ID")
SQLITE_PATH = os.getenv("SQLITE_PATH", "/tmp/medical_desert.db")

CAPABILITIES = ["icu", "maternity", "emergency", "dialysis", "nicu", "oncology"]

CLASS_COLORS = {
    "crisis_zone": "#d7191c",
    "investigate":  "#fdae61",
    "data_gap":     "#abd9e9",
    "served":       "#1a9641",
}

CLASS_LABELS = {
    "crisis_zone": "Crisis Zone — high need, weak coverage",
    "investigate":  "Investigate — high need, unverified claims",
    "data_gap":     "Data Gap — unknown need or coverage",
    "served":       "Served — covered with evidence",
}


# ── DB helpers ────────────────────────────────────────────────────────────────

def _sql(query: str) -> pd.DataFrame:
    with dbsql.connect(
        server_hostname=DB_HOST,
        http_path=f"/sql/1.0/warehouses/{DB_WH_ID}",
        access_token=DB_TOKEN,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)


def _sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        capability TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS scenario_regions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scenario_id INTEGER REFERENCES scenarios(id),
        district_id TEXT NOT NULL,
        district_name TEXT NOT NULL,
        state_ut TEXT NOT NULL,
        classification TEXT NOT NULL,
        health_risk_score REAL,
        coverage_score REAL,
        planner_flag TEXT,
        note TEXT,
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return conn


# ── Map tab ───────────────────────────────────────────────────────────────────

def load_state_map(capability: str) -> go.Figure:
    df = _sql(f"""
        SELECT
            state_ut,
            COUNT(*) AS total,
            SUM(CASE WHEN classification='crisis_zone' THEN 1 ELSE 0 END) AS crisis,
            SUM(CASE WHEN classification='investigate'  THEN 1 ELSE 0 END) AS investigate,
            ROUND(AVG(health_risk_score), 3) AS avg_risk,
            ROUND(AVG(coverage_score),    3) AS avg_coverage
        FROM {CATALOG}.{SCHEMA}.district_capability_classification
        WHERE capability = '{capability}'
        GROUP BY state_ut
    """)

    df["crisis_pct"] = (df["crisis"] / df["total"] * 100).round(1)
    df["label"] = df.apply(
        lambda r: (
            f"{r.state_ut}<br>"
            f"Crisis zones: {r.crisis}/{r.total} districts ({r.crisis_pct}%)<br>"
            f"Avg health risk: {r.avg_risk}<br>"
            f"Avg coverage: {r.avg_coverage}"
        ), axis=1
    )

    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
        locations="state_ut",
        featureidkey="properties.NAME_1",
        color="crisis_pct",
        color_continuous_scale=["#1a9641", "#fdae61", "#d7191c"],
        range_color=[0, 100],
        labels={"crisis_pct": "% Crisis Districts"},
        hover_name="state_ut",
        hover_data={"crisis": True, "total": True, "avg_risk": True, "crisis_pct": True},
        title=f"Medical Desert Map — {capability.upper()} coverage",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=500)
    return fig


def load_district_table(capability: str, state: str) -> pd.DataFrame:
    if not state:
        return pd.DataFrame()
    df = _sql(f"""
        SELECT
            district_name       AS District,
            classification      AS Classification,
            ROUND(health_risk_score, 3) AS `Health Risk`,
            ROUND(coverage_score, 3)    AS `Coverage Score`,
            facility_count              AS `Facilities`,
            strong_count                AS `Verified`,
            CASE WHEN nfhs_low_confidence THEN '⚠ Low confidence' ELSE '' END AS `NFHS Note`
        FROM {CATALOG}.{SCHEMA}.district_capability_classification
        WHERE capability = '{capability}'
          AND LOWER(state_ut) = LOWER('{state}')
        ORDER BY health_risk_score DESC
    """)
    return df


def load_facility_detail(capability: str, district: str, state: str) -> pd.DataFrame:
    if not district:
        return pd.DataFrame()
    df = _sql(f"""
        SELECT
            f.name                          AS Facility,
            f.facilityTypeId                AS Type,
            f.operatorTypeId                AS Operator,
            ROUND(s.evidence_score, 3)      AS `Evidence Score`,
            s.strength                      AS Strength,
            s.evidence_fields               AS `Evidence Fields`,
            s.evidence_quotes               AS `Cited Evidence`
        FROM {CATALOG}.{SCHEMA}.facility_capability_scores s
        JOIN {CATALOG}.{SCHEMA}.facilities_located f ON s.facility_id = f.unique_id
        WHERE s.capability = '{capability}'
          AND LOWER(f.gadm_district) = LOWER('{district}')
          AND LOWER(f.gadm_state)    = LOWER('{state}')
        ORDER BY s.evidence_score DESC
        LIMIT 20
    """)
    return df


# ── Lakebase: save scenario ───────────────────────────────────────────────────

def save_scenario(name: str, capability: str, description: str) -> str:
    if not name.strip():
        return "Scenario name is required."
    try:
        with _sqlite() as conn:
            cur = conn.execute(
                "INSERT INTO scenarios (name, capability, description) VALUES (?,?,?)",
                (name.strip(), capability, description.strip()),
            )
            sid = cur.lastrowid
        return f"Saved scenario '{name}' (ID: {sid})"
    except Exception as e:
        return f"Error: {e}"


def flag_region(
    scenario_id: str, district_name: str, state: str,
    classification: str, health_risk: float, coverage: float,
    flag: str, note: str
) -> str:
    if not scenario_id or not district_name:
        return "Provide a scenario ID and district."
    try:
        with _sqlite() as conn:
            conn.execute("""
                INSERT INTO scenario_regions
                    (scenario_id, district_id, district_name, state_ut,
                     classification, health_risk_score, coverage_score,
                     planner_flag, note)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                int(scenario_id),
                district_name.lower().replace(" ", "_") + "__" + state.lower().replace(" ", "_"),
                district_name, state,
                classification, health_risk, coverage, flag, note,
            ))
        return f"Flagged {district_name} as '{flag}'"
    except Exception as e:
        return f"Error: {e}"


def list_scenarios() -> pd.DataFrame:
    try:
        with _sqlite() as conn:
            rows = conn.execute(
                "SELECT id, name, capability, created_at FROM scenarios ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            return pd.DataFrame([dict(r) for r in rows])
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Medical Desert Planner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Medical Desert Planner\nIdentify where care gaps are real vs. where data is simply missing.")

    with gr.Tabs():

        # ── Tab 1: Map ──────────────────────────────────────────────────────
        with gr.Tab("Map"):
            with gr.Row():
                cap_map  = gr.Dropdown(CAPABILITIES, value="icu", label="Capability")
                btn_map  = gr.Button("Load Map", variant="primary")

            map_plot = gr.Plot(label="India — crisis zone % per state")

            gr.Markdown("### Select a state to drill into districts")
            with gr.Row():
                state_input = gr.Textbox(label="State name (type exactly)", placeholder="Bihar")
                btn_dist    = gr.Button("Load Districts")

            district_table = gr.DataFrame(label="Districts", interactive=False)

            gr.Markdown("### Select a district to see its facilities")
            with gr.Row():
                district_input = gr.Textbox(label="District name", placeholder="Patna")
                btn_fac        = gr.Button("Load Facilities")

            facility_table = gr.DataFrame(label="Facilities with evidence", interactive=False)

            btn_map.click(load_state_map,   inputs=[cap_map],  outputs=[map_plot])
            btn_dist.click(load_district_table, inputs=[cap_map, state_input], outputs=[district_table])
            btn_fac.click(load_facility_detail, inputs=[cap_map, district_input, state_input], outputs=[facility_table])

        # ── Tab 2: Agent ────────────────────────────────────────────────────
        with gr.Tab("Ask the Agent"):
            gr.Markdown(
                "Ask plain-English questions. The agent cites evidence and "
                "flags uncertainty.\n\n"
                "**Examples:**\n"
                "- *Which districts in Bihar have no verified ICU?*\n"
                "- *Compare states by dialysis coverage*\n"
                "- *What's the situation for maternity care in Arunachal Pradesh?*"
            )
            chatbot  = gr.Chatbot(height=400)
            msg_box  = gr.Textbox(placeholder="Ask a question...", label="")
            chat_hist = gr.State([])

            def respond(message, history, chat_history):
                reply = agent_run(message, chat_history)
                history.append((message, reply))
                chat_history.append({"role": "user",      "content": message})
                chat_history.append({"role": "assistant", "content": reply})
                return history, chat_history, ""

            msg_box.submit(respond, [msg_box, chatbot, chat_hist], [chatbot, chat_hist, msg_box])

        # ── Tab 3: Scenarios ────────────────────────────────────────────────
        with gr.Tab("Save Scenario"):
            gr.Markdown("Save your planning work to Lakebase so you can return to it.")

            with gr.Row():
                sc_name = gr.Textbox(label="Scenario name", placeholder="Bihar ICU audit June 2026")
                sc_cap  = gr.Dropdown(CAPABILITIES, label="Capability")
            sc_desc = gr.Textbox(label="Notes", lines=3)
            btn_save = gr.Button("Save Scenario", variant="primary")
            save_status = gr.Textbox(label="", interactive=False)

            gr.Markdown("### Flag a Region")
            with gr.Row():
                fl_sid    = gr.Textbox(label="Scenario ID")
                fl_dist   = gr.Textbox(label="District name")
                fl_state  = gr.Textbox(label="State")
            with gr.Row():
                fl_class  = gr.Dropdown(["crisis_zone","investigate","data_gap","served"], label="Classification")
                fl_risk   = gr.Number(label="Health risk score", precision=3)
                fl_cov    = gr.Number(label="Coverage score",    precision=3)
            with gr.Row():
                fl_flag   = gr.Dropdown(["confirmed","needs_survey","disputed","resolved"], label="Planner flag")
                fl_note   = gr.Textbox(label="Note")
            btn_flag = gr.Button("Flag Region")
            flag_status = gr.Textbox(label="", interactive=False)

            gr.Markdown("### My Scenarios")
            btn_list = gr.Button("Refresh")
            sc_list  = gr.DataFrame(interactive=False)

            btn_save.click(save_scenario, [sc_name, sc_cap, sc_desc], save_status)
            btn_flag.click(flag_region,
                [fl_sid, fl_dist, fl_state, fl_class, fl_risk, fl_cov, fl_flag, fl_note],
                flag_status)
            btn_list.click(list_scenarios, outputs=sc_list)


if __name__ == "__main__":
    demo.launch()
