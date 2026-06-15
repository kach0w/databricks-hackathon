# Databricks notebook source
# MAGIC %md # 04 — Build Graph Tables
# MAGIC Matches facilities to NFHS districts via state name + city→district fuzzy text matching.
# MAGIC No shapefile needed — runs fully on serverless.

# COMMAND ----------
import pandas as pd
from difflib import get_close_matches
from pyspark.sql import functions as F

CATALOG = "main"
SCHEMA  = "medical_desert"

# COMMAND ----------
# ── Build district lookup from NFHS ──────────────────────────────────────────
nfhs        = spark.read.table(f"{CATALOG}.{SCHEMA}.nfhs_clean")
dist_scores = spark.read.table(f"{CATALOG}.{SCHEMA}.district_scores")
fac         = spark.read.table(f"{CATALOG}.{SCHEMA}.facilities_clean")

nfhs_pd = nfhs.select("district_id", "district_name", "state_ut").toPandas()
nfhs_pd["state_norm"]    = nfhs_pd["state_ut"].str.lower().str.strip()
nfhs_pd["district_norm"] = nfhs_pd["district_name"].str.lower().str.strip()

# state → list of (district_norm, district_id)
state_districts = (
    nfhs_pd.groupby("state_norm")
    .apply(lambda g: list(zip(g["district_norm"], g["district_id"])))
    .to_dict()
)

# Common alternate state name spellings in facility data
STATE_ALIASES = {
    "uttarakhand": "uttarakhand", "uttaranchal": "uttarakhand",
    "jammu & kashmir": "jammu and kashmir", "j&k": "jammu and kashmir",
    "andaman and nicobar": "andaman & nicobar islands",
    "andaman & nicobar": "andaman & nicobar islands",
    "dadra and nagar haveli": "dadra & nagar haveli and daman & diu",
    "daman and diu": "dadra & nagar haveli and daman & diu",
    "delhi": "delhi", "new delhi": "delhi",
    "pondicherry": "puducherry",
}

def norm_state(s):
    if not s:
        return ""
    s = s.lower().strip()
    return STATE_ALIASES.get(s, s)

def match_district(state_raw, city_raw):
    """Return district_id or empty string."""
    state = norm_state(state_raw)
    if state not in state_districts:
        # try partial match on state name
        candidates = [k for k in state_districts if state in k or k in state]
        if not candidates:
            return ""
        state = candidates[0]

    districts = state_districts[state]   # [(district_norm, district_id), ...]
    dist_names = [d[0] for d in districts]
    dist_map   = {d[0]: d[1] for d in districts}

    if not city_raw:
        return ""

    city = city_raw.lower().strip()

    # 1. Exact match
    if city in dist_map:
        return dist_map[city]

    # 2. City is contained in a district name or vice versa
    for dname, did in districts:
        if city in dname or dname in city:
            return did

    # 3. Fuzzy match (cutoff=0.7 is conservative enough to avoid bad matches)
    close = get_close_matches(city, dist_names, n=1, cutoff=0.7)
    if close:
        return dist_map[close[0]]

    return ""

# COMMAND ----------
# ── Match all facilities ──────────────────────────────────────────────────────
fac_pd = fac.select(
    "unique_id", "name", "address_stateOrRegion", "address_city",
    "latitude", "longitude", "facilityTypeId", "operatorTypeId", "affiliationTypeIds"
).toPandas()

fac_pd["district_id"] = fac_pd.apply(
    lambda r: match_district(r["address_stateOrRegion"], r["address_city"]), axis=1
)

matched = (fac_pd["district_id"] != "").sum()
print(f"Matched {matched}/{len(fac_pd)} facilities to an NFHS district")

# Sample matches to verify
print(fac_pd[fac_pd["district_id"] != ""][
    ["name", "address_stateOrRegion", "address_city", "district_id"]
].head(10).to_string())

# COMMAND ----------
fac_located_spark = spark.createDataFrame(
    fac_pd[["unique_id", "district_id"]]
)
fac_with_district = fac.join(fac_located_spark, on="unique_id", how="left")
fac_with_district.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.facilities_located")

# COMMAND ----------
# ── Vertices ──────────────────────────────────────────────────────────────────
fac_loc = spark.read.table(f"{CATALOG}.{SCHEMA}.facilities_located")

v_fac = fac_loc.select(
    F.concat(F.lit("fac__"), F.col("unique_id")).alias("id"),
    F.lit("facility").alias("type"),
    F.col("name"),
    F.col("latitude").alias("lat"),
    F.col("longitude").alias("lon"),
    F.col("facilityTypeId"),
    F.col("operatorTypeId"),
    F.col("district_id"),
    F.lit(None).cast("double").alias("health_risk_score"),
    F.lit(None).cast("string").alias("risk_band"),
    F.lit(None).cast("boolean").alias("nfhs_low_confidence"),
)

v_dist = dist_scores.select(
    F.concat(F.lit("dist__"), F.col("district_id")).alias("id"),
    F.lit("district").alias("type"),
    F.col("district_name").alias("name"),
    F.lit(None).cast("double").alias("lat"),
    F.lit(None).cast("double").alias("lon"),
    F.lit(None).cast("string").alias("facilityTypeId"),
    F.lit(None).cast("string").alias("operatorTypeId"),
    F.col("district_id"),
    F.col("health_risk_score"),
    F.col("risk_band"),
    F.col("nfhs_low_confidence"),
)

v_state = nfhs.select(
    F.concat(F.lit("state__"),
        F.lower(F.regexp_replace(F.trim(F.col("state_ut")), r"\s+", "_"))
    ).alias("id"),
    F.lit("state").alias("type"),
    F.col("state_ut").alias("name"),
    F.lit(None).cast("double").alias("lat"),
    F.lit(None).cast("double").alias("lon"),
    F.lit(None).cast("string").alias("facilityTypeId"),
    F.lit(None).cast("string").alias("operatorTypeId"),
    F.lit(None).cast("string").alias("district_id"),
    F.lit(None).cast("double").alias("health_risk_score"),
    F.lit(None).cast("string").alias("risk_band"),
    F.lit(None).cast("boolean").alias("nfhs_low_confidence"),
).distinct()

vertices = v_fac.unionByName(v_dist).unionByName(v_state)
vertices.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.graph_vertices")
print(f"Vertices: {vertices.count()}")

# COMMAND ----------
# ── Edges ─────────────────────────────────────────────────────────────────────
scores = spark.read.table(f"{CATALOG}.{SCHEMA}.facility_capability_scores")

e_located = fac_loc.filter("district_id IS NOT NULL AND district_id != ''").select(
    F.concat(F.lit("fac__"), F.col("unique_id")).alias("src"),
    F.concat(F.lit("dist__"), F.col("district_id")).alias("dst"),
    F.lit("located_in").alias("relationship"),
    F.lit(None).cast("string").alias("capability"),
    F.lit(None).cast("double").alias("evidence_score"),
    F.lit(None).cast("string").alias("strength"),
    F.array().cast("array<string>").alias("evidence_fields"),
    F.array().cast("array<string>").alias("evidence_quotes"),
)

e_claims = scores.select(
    F.concat(F.lit("fac__"), F.col("facility_id")).alias("src"),
    F.concat(F.lit("cap__"), F.col("capability")).alias("dst"),
    F.lit("claims_capability").alias("relationship"),
    F.col("capability"),
    F.col("evidence_score"),
    F.col("strength"),
    F.col("evidence_fields"),
    F.col("evidence_quotes"),
)

e_part_of = dist_scores.select(
    F.concat(F.lit("dist__"), F.col("district_id")).alias("src"),
    F.concat(F.lit("state__"),
        F.lower(F.regexp_replace(F.trim(F.col("state_ut")), r"\s+", "_"))
    ).alias("dst"),
    F.lit("part_of").alias("relationship"),
    F.lit(None).cast("string").alias("capability"),
    F.lit(None).cast("double").alias("evidence_score"),
    F.lit(None).cast("string").alias("strength"),
    F.array().cast("array<string>").alias("evidence_fields"),
    F.array().cast("array<string>").alias("evidence_quotes"),
)

edges = e_located.unionByName(e_claims).unionByName(e_part_of)
edges.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.graph_edges")
print(f"Edges: {edges.count()}")

# COMMAND ----------
# ── Validate ──────────────────────────────────────────────────────────────────
spark.sql(f"""
    SELECT e.dst AS district_id, COUNT(*) AS facility_count
    FROM {CATALOG}.{SCHEMA}.graph_edges e
    WHERE e.relationship = 'located_in'
    GROUP BY e.dst
    ORDER BY facility_count DESC
    LIMIT 10
""").show(truncate=False)
