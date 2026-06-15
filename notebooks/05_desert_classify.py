# Databricks notebook source
# MAGIC %md # 05 — Classify Medical Deserts
# MAGIC Crosses facility coverage strength per district with NFHS-5 health risk score
# MAGIC to assign each district × capability to one of four buckets.
# MAGIC
# MAGIC Buckets:
# MAGIC   crisis_zone  — poor outcomes + weak/no facility coverage  ← most urgent
# MAGIC   investigate  — poor outcomes + facilities exist but claims are weak
# MAGIC   data_gap     — good outcomes but no/weak data (send a surveyor)
# MAGIC   served       — good outcomes + strong facility coverage

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "main"
SCHEMA  = "medical_desert"
CAPABILITIES = ["icu", "maternity", "emergency", "dialysis", "nicu", "oncology"]

# COMMAND ----------
fac_loc = spark.read.table(f"{CATALOG}.{SCHEMA}.facilities_located")
scores  = spark.read.table(f"{CATALOG}.{SCHEMA}.facility_capability_scores")
dist    = spark.read.table(f"{CATALOG}.{SCHEMA}.district_scores")

# Join facilities → scores → districts
fac_scores = (
    fac_loc
    .filter("district_id IS NOT NULL AND district_id != ''")
    .join(scores, fac_loc.unique_id == scores.facility_id, how="inner")
    .select(
        F.col("district_id"),
        F.col("capability"),
        F.col("evidence_score"),
        F.col("strength"),
    )
)

# Per district × capability: aggregate coverage signal
coverage = (
    fac_scores
    .groupBy("district_id", "capability")
    .agg(
        F.count("*").alias("facility_count"),
        F.sum(F.when(F.col("strength") == "strong",  1).otherwise(0)).alias("strong_count"),
        F.sum(F.when(F.col("strength") == "partial", 1).otherwise(0)).alias("partial_count"),
        F.max("evidence_score").alias("best_evidence_score"),
        F.avg("evidence_score").alias("avg_evidence_score"),
    )
)

# Coverage score: weighted combination favouring strong evidence
coverage = coverage.withColumn(
    "coverage_score",
    F.round(
        (F.col("strong_count")  * 1.0 +
         F.col("partial_count") * 0.5) /
        F.greatest(F.col("facility_count"), F.lit(1)),
        4
    )
)

# COMMAND ----------
# ── Cross product: all districts × all capabilities ───────────────────────────
# Ensures every district appears for every capability (even with zero facilities)

all_districts = dist.select("district_id", "district_name", "state_ut",
                             "health_risk_score", "risk_band", "nfhs_low_confidence")
all_caps = spark.createDataFrame(
    [(c,) for c in CAPABILITIES], ["capability"]
)

full_grid = all_districts.crossJoin(all_caps)

classified = (
    full_grid
    .join(coverage, on=["district_id", "capability"], how="left")
    .fillna({
        "facility_count":    0,
        "strong_count":      0,
        "partial_count":     0,
        "best_evidence_score": 0.0,
        "avg_evidence_score":  0.0,
        "coverage_score":      0.0,
    })
)

# ── 2×2 classification ────────────────────────────────────────────────────────
# Axis 1 — health need:     health_risk_score ≥ 0.50 → high need
# Axis 2 — facility cover:  coverage_score   ≥ 0.40 → covered
#                           strong_count ≥ 1        → at least one strong claim

classified = classified.withColumn(
    "classification",
    F.when(
        (F.col("health_risk_score") >= 0.50) & (F.col("coverage_score") < 0.40),
        F.when(F.col("facility_count") == 0, F.lit("crisis_zone"))
         .otherwise(F.lit("investigate"))   # claims exist but too weak
    ).when(
        (F.col("health_risk_score") < 0.50) & (F.col("coverage_score") < 0.40),
        F.lit("data_gap")
    ).otherwise(
        F.lit("served")
    )
)

classified.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.district_capability_classification")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = (
    classified
    .groupBy("capability", "classification")
    .agg(F.count("*").alias("district_count"))
    .orderBy("capability", "classification")
)
summary.show(50, truncate=False)

crisis_total = classified.filter("classification = 'crisis_zone'").count()
print(f"\nTotal crisis zone district-capability pairs: {crisis_total}")
