# Databricks notebook source
# MAGIC %md # 01 — Ingest & Clean
# MAGIC Reads raw Delta Sharing tables, validates coordinates, parses JSON array columns,
# MAGIC and writes clean tables to the writable catalog.

# COMMAND ----------
import json
import re
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, DoubleType

# ── Config ────────────────────────────────────────────────────────────────────
SRC_CATALOG = "databricks_virtue_foundation_dataset_dais_2026"
SRC_SCHEMA  = "virtue_foundation_dataset"
CATALOG     = "main"          # change to your writable catalog
SCHEMA      = "medical_desert"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ── India state bounding boxes for coordinate validation ──────────────────────
# (min_lat, max_lat, min_lon, max_lon)
STATE_BOUNDS = {
    "andhra pradesh":    (12.6, 19.9, 76.8, 84.8),
    "arunachal pradesh": (26.6, 29.5, 91.5, 97.4),
    "assam":             (24.1, 27.9, 89.7, 96.0),
    "bihar":             (24.3, 27.5, 83.3, 88.3),
    "chhattisgarh":      (17.8, 24.1, 80.2, 84.4),
    "goa":               (14.9, 15.8, 73.9, 74.4),
    "gujarat":           (20.1, 24.7, 68.2, 74.5),
    "haryana":           (27.7, 30.9, 74.5, 77.6),
    "himachal pradesh":  (30.4, 33.2, 75.6, 79.0),
    "jharkhand":         (21.9, 25.3, 83.3, 87.9),
    "karnataka":         (11.6, 18.4, 74.1, 78.6),
    "kerala":            (8.3,  12.8, 74.9, 77.4),
    "madhya pradesh":    (21.1, 26.9, 74.0, 82.8),
    "maharashtra":       (15.6, 22.0, 72.7, 80.9),
    "manipur":           (23.8, 25.7, 93.0, 94.8),
    "meghalaya":         (25.0, 26.1, 89.8, 92.8),
    "mizoram":           (21.9, 24.5, 92.2, 93.5),
    "nagaland":          (25.2, 27.0, 93.3, 95.3),
    "odisha":            (17.8, 22.6, 81.4, 87.5),
    "punjab":            (29.5, 32.5, 73.9, 76.9),
    "rajasthan":         (23.0, 30.2, 69.5, 78.3),
    "sikkim":            (27.1, 28.1, 88.0, 88.9),
    "tamil nadu":        (8.1,  13.6, 76.2, 80.3),
    "telangana":         (15.8, 19.9, 77.3, 81.3),
    "tripura":           (22.9, 24.5, 91.2, 92.3),
    "uttar pradesh":     (23.9, 30.4, 77.1, 84.6),
    "uttarakhand":       (28.7, 31.5, 77.6, 81.0),
    "west bengal":       (21.5, 27.2, 85.8, 89.9),
    "delhi":             (28.4, 28.9, 76.8, 77.4),
    "jammu and kashmir": (32.3, 36.6, 73.7, 80.3),
    "ladakh":            (32.0, 36.0, 75.0, 80.5),
}

def coord_valid(state_raw, lat, lon):
    if lat is None or lon is None:
        return False
    state = (state_raw or "").lower().strip()
    bounds = STATE_BOUNDS.get(state)
    if bounds is None:
        return (6.0 <= lat <= 37.0) and (68.0 <= lon <= 97.5)
    min_lat, max_lat, min_lon, max_lon = bounds
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

coord_valid_udf = F.udf(coord_valid, "boolean")

def parse_json_array(s):
    if s is None:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x) for x in val if x is not None]
        return [str(val)]
    except Exception:
        return [s] if s.strip() else []

parse_json_udf = F.udf(parse_json_array, ArrayType(StringType()))

def dedup_list(lst):
    if lst is None:
        return []
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

dedup_udf = F.udf(dedup_list, ArrayType(StringType()))

# COMMAND ----------
# ── Facilities ────────────────────────────────────────────────────────────────
raw_fac = spark.read.table(f"{SRC_CATALOG}.{SRC_SCHEMA}.facilities")

JSON_ARRAY_COLS = ["specialties", "affiliationTypeIds", "capability", "procedure", "equipment"]

fac = raw_fac
for col in JSON_ARRAY_COLS:
    if col in raw_fac.columns:
        fac = fac.withColumn(col, dedup_udf(parse_json_udf(F.col(col).cast("string"))))

fac = (
    fac
    .withColumn("lat_valid", coord_valid_udf(
        F.col("address_stateOrRegion"),
        F.col("latitude").cast(DoubleType()),
        F.col("longitude").cast(DoubleType()),
    ))
    .withColumn("latitude",  F.col("latitude").cast(DoubleType()))
    .withColumn("longitude", F.col("longitude").cast(DoubleType()))
    .withColumn("capacity_num",
        F.regexp_extract(F.col("capacity").cast("string"), r"(\d+)", 1).cast("int"))
    .withColumn("doctors_num",
        F.regexp_extract(F.col("numberDoctors").cast("string"), r"(\d+)", 1).cast("int"))
    .withColumn("year_num",
        F.regexp_extract(F.col("yearEstablished").cast("string"), r"(\d{4})", 1).cast("int"))
)

fac.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.facilities_clean")
print(f"facilities_clean: {fac.count()} rows, {fac.filter('lat_valid').count()} with valid coords")

# COMMAND ----------
# ── NFHS-5 ───────────────────────────────────────────────────────────────────
raw_nfhs = spark.read.table(f"{SRC_CATALOG}.{SRC_SCHEMA}.nfhs_5_district_health_indicators")

NFHS_KEY_COLS = [
    "district_name", "state_ut",
    "institutional_birth_5y_pct",
    "births_attended_by_skilled_hp_5y_10_pct",
    "mothers_who_had_at_least_4_anc_visits_lb5y_pct",
    "child_12_23m_fully_vaccinated_based_on_information_from_eit_pct",
    "hh_member_covered_health_insurance_pct",
    "child_u5_who_are_stunted_height_for_age_18_pct",
    "non_pregnant_w15_49_who_are_anaemic_lt_12_0_g_dl_22_pct",
]

nfhs = raw_nfhs.select(NFHS_KEY_COLS)

# Flag parentheses (unreliable estimates) before casting
METRIC_COLS = [c for c in NFHS_KEY_COLS if c not in ("district_name", "state_ut")]
for col in METRIC_COLS:
    nfhs = nfhs.withColumn(
        f"{col}_low_conf",
        F.col(col).cast("string").rlike(r"^\s*\(|\*")
    )

# Clean: strip whitespace, remove parens/asterisks, cast to double
def clean_numeric(s):
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"[()* ]", "", s)
    try:
        return float(s)
    except Exception:
        return None

clean_num_udf = F.udf(clean_numeric, DoubleType())

for col in METRIC_COLS:
    nfhs = nfhs.withColumn(col, clean_num_udf(F.col(col).cast("string")))

nfhs = (
    nfhs
    .withColumn("district_name", F.trim(F.col("district_name")))
    .withColumn("state_ut",      F.trim(F.col("state_ut")))
    .withColumn("district_id",
        F.concat_ws("__",
            F.lower(F.regexp_replace(F.trim(F.col("district_name")), r"\s+", "_")),
            F.lower(F.regexp_replace(F.trim(F.col("state_ut")),      r"\s+", "_")),
        )
    )
)

nfhs.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.nfhs_clean")
print(f"nfhs_clean: {nfhs.count()} rows")
