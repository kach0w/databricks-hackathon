# Databricks notebook source
# MAGIC %md # 04 — Build Graph Tables
# MAGIC Spatial join maps facility lat/lon → India district using GADM shapefile.
# MAGIC Builds vertex and edge Delta tables traversed via SQL joins in notebook 05 and the app.

# COMMAND ----------
%pip install geopandas shapely requests

# COMMAND ----------
import os
import io
import zipfile
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pyspark.sql import functions as F

CATALOG = "main"
SCHEMA  = "medical_desert"

# COMMAND ----------
# ── Download GADM India district shapefile ────────────────────────────────────
GADM_URL  = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_IND_shp.zip"
GADM_DIR  = "/tmp/gadm_india"
GADM_FILE = f"{GADM_DIR}/gadm41_IND_2.shp"

if not os.path.exists(GADM_FILE):
    os.makedirs(GADM_DIR, exist_ok=True)
    print("Downloading GADM India shapefile (~30MB)...")
    r = requests.get(GADM_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(GADM_DIR)
    print("Done.")

districts_gdf = gpd.read_file(GADM_FILE)[["NAME_1", "NAME_2", "geometry"]]
districts_gdf.columns = ["gadm_state", "gadm_district", "geometry"]
districts_gdf = districts_gdf.set_crs("EPSG:4326")
print(f"Loaded {len(districts_gdf)} district polygons")

# COMMAND ----------
# ── Spatial join: facilities → districts ──────────────────────────────────────
fac = spark.read.table(f"{CATALOG}.{SCHEMA}.facilities_clean")

fac_pd = fac.select(
    "unique_id", "name", "address_stateOrRegion", "address_city",
    "latitude", "longitude", "facilityTypeId", "operatorTypeId", "affiliationTypeIds"
).toPandas()

# Only spatially join rows with valid coords; rest get empty district
fac_valid = fac_pd[fac_pd["latitude"].notna() & fac_pd["longitude"].notna()].copy()
fac_invalid = fac_pd[~fac_pd["unique_id"].isin(fac_valid["unique_id"])].copy()

fac_gdf = gpd.GeoDataFrame(
    fac_valid,
    geometry=[Point(lon, lat) for lat, lon in zip(fac_valid.longitude, fac_valid.latitude)],
    crs="EPSG:4326"
)

joined = gpd.sjoin(fac_gdf, districts_gdf, how="left", predicate="within")
joined["gadm_district"] = joined["gadm_district"].fillna("")
joined["gadm_state"]    = joined["gadm_state"].fillna("")
joined["district_id"] = (
    joined["gadm_district"].str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
    + "__"
    + joined["gadm_state"].str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
)

fac_invalid["gadm_district"] = ""
fac_invalid["gadm_state"]    = ""
fac_invalid["district_id"]   = ""

fac_located_pd = pd.concat(
    [joined[["unique_id", "district_id", "gadm_district", "gadm_state"]],
     fac_invalid[["unique_id", "district_id", "gadm_district", "gadm_state"]]],
    ignore_index=True
)

fac_located_spark = spark.createDataFrame(fac_located_pd)
fac_with_district = fac.join(fac_located_spark, on="unique_id", how="left")
fac_with_district.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.facilities_located")

matched = fac_located_pd[fac_located_pd.district_id != ""].shape[0]
print(f"Spatial join: {matched}/{len(fac_pd)} facilities matched to a district")

# COMMAND ----------
# ── Vertices ──────────────────────────────────────────────────────────────────
nfhs        = spark.read.table(f"{CATALOG}.{SCHEMA}.nfhs_clean")
fac_loc     = spark.read.table(f"{CATALOG}.{SCHEMA}.facilities_located")
dist_scores = spark.read.table(f"{CATALOG}.{SCHEMA}.district_scores")

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
# ── Validate with plain SQL ───────────────────────────────────────────────────
strong_icu = spark.sql(f"""
    SELECT DISTINCT v_dist.name AS district
    FROM {CATALOG}.{SCHEMA}.graph_edges e_claims
    JOIN {CATALOG}.{SCHEMA}.graph_edges e_located
      ON REPLACE(e_claims.src, 'fac__', '') = REPLACE(e_located.src, 'fac__', '')
    JOIN {CATALOG}.{SCHEMA}.graph_vertices v_dist
      ON e_located.dst = v_dist.id
    WHERE e_claims.relationship = 'claims_capability'
      AND e_claims.capability   = 'icu'
      AND e_claims.strength     = 'strong'
      AND e_located.relationship = 'located_in'
      AND v_dist.type = 'district'
""")
print(f"Districts with strong ICU evidence: {strong_icu.count()}")
