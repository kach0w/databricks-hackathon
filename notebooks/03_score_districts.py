# Databricks notebook source
# MAGIC %md # 03 — Score Districts
# MAGIC Computes a composite health risk score (0–1) per district from NFHS-5 indicators.
# MAGIC Higher score = higher risk = more urgent need.

# COMMAND ----------
from pyspark.sql import functions as F, Window

CATALOG = "main"
SCHEMA  = "medical_desert"

# COMMAND ----------
nfhs = spark.read.table(f"{CATALOG}.{SCHEMA}.nfhs_clean")

# ── Indicator config ──────────────────────────────────────────────────────────
# direction: "invert" means high value = good (low risk), so we flip it
# weight: contribution to composite risk score (must sum to 1.0)

INDICATORS = [
    {"col": "institutional_birth_5y_pct",                                       "direction": "invert", "weight": 0.20},
    {"col": "births_attended_by_skilled_hp_5y_10_pct",                          "direction": "invert", "weight": 0.15},
    {"col": "mothers_who_had_at_least_4_anc_visits_lb5y_pct",                   "direction": "invert", "weight": 0.15},
    {"col": "child_12_23m_fully_vaccinated_based_on_information_from_eit_pct",   "direction": "invert", "weight": 0.15},
    {"col": "hh_member_covered_health_insurance_pct",                            "direction": "invert", "weight": 0.10},
    {"col": "child_u5_who_are_stunted_height_for_age_18_pct",                   "direction": "keep",   "weight": 0.15},
    {"col": "non_pregnant_w15_49_who_are_anaemic_lt_12_0_g_dl_22_pct",         "direction": "keep",   "weight": 0.10},
]

assert abs(sum(i["weight"] for i in INDICATORS) - 1.0) < 0.001

# COMMAND ----------
# ── Normalize each indicator 0–1 across all 706 districts ─────────────────────
# Use min-max normalization; nulls get the worst-case value for their direction

df = nfhs

for ind in INDICATORS:
    col = ind["col"]
    direction = ind["direction"]

    col_min = df.agg(F.min(col)).collect()[0][0]
    col_max = df.agg(F.max(col)).collect()[0][0]

    if col_min is None or col_max is None or col_min == col_max:
        df = df.withColumn(f"{col}_norm", F.lit(0.5))
        continue

    # Normalize to 0-1 where 1 always means "worse outcome" (higher risk)
    if direction == "invert":
        # high value = good = low risk → flip so 1 = worst
        df = df.withColumn(
            f"{col}_norm",
            F.when(F.col(col).isNull(), F.lit(1.0))  # missing = assume worst
             .otherwise(F.lit(1.0) - (F.col(col) - col_min) / (col_max - col_min))
        )
    else:
        # high value = bad = high risk → keep direction
        df = df.withColumn(
            f"{col}_norm",
            F.when(F.col(col).isNull(), F.lit(1.0))
             .otherwise((F.col(col) - col_min) / (col_max - col_min))
        )

# COMMAND ----------
# ── Composite risk score ──────────────────────────────────────────────────────
risk_expr = sum(
    F.col(f"{ind['col']}_norm") * ind["weight"]
    for ind in INDICATORS
)

df = df.withColumn("health_risk_score", F.round(risk_expr, 4))

# Flag districts where ANY key indicator has a low-confidence estimate
low_conf_cols = [f"{ind['col']}_low_conf" for ind in INDICATORS
                 if f"{ind['col']}_low_conf" in df.columns]

if low_conf_cols:
    df = df.withColumn(
        "nfhs_low_confidence",
        F.greatest(*[F.col(c).cast("int") for c in low_conf_cols]).cast("boolean")
    )
else:
    df = df.withColumn("nfhs_low_confidence", F.lit(False))

# ── Risk band ─────────────────────────────────────────────────────────────────
df = df.withColumn(
    "risk_band",
    F.when(F.col("health_risk_score") >= 0.70, "high")
     .when(F.col("health_risk_score") >= 0.45, "medium")
     .otherwise("low")
)

out_cols = [
    "district_id", "district_name", "state_ut",
    "health_risk_score", "risk_band", "nfhs_low_confidence",
] + [ind["col"] for ind in INDICATORS]

df.select(out_cols).write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.district_scores")

high  = df.filter("risk_band = 'high'").count()
med   = df.filter("risk_band = 'medium'").count()
low   = df.filter("risk_band = 'low'").count()
print(f"district_scores: {high} high-risk, {med} medium, {low} low")

# Top 10 highest-risk districts
df.select("district_name", "state_ut", "health_risk_score", "risk_band") \
  .orderBy(F.col("health_risk_score").desc()).show(10, truncate=False)
