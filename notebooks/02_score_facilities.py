# Databricks notebook source
# MAGIC %md # 02 — Score Facilities
# MAGIC For each facility × capability, compute a 0–1 evidence score and capture
# MAGIC the exact strings that produced it (for UI citation).

# COMMAND ----------
import json
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import pandas as pd

CATALOG = "main"
SCHEMA  = "medical_desert"

# COMMAND ----------
# ── Capability definitions ────────────────────────────────────────────────────
# specialties: camelCase values in the specialties JSON array (high reliability)
# keywords: substrings to match in capability/procedure/equipment/description text

CAPABILITIES = {
    "icu": {
        "specialties": ["criticalCareMedicine", "intensiveCare", "anesthesiology", "anesthesia"],
        "keywords": [
            "intensive care unit", "icu", "critical care", "ventilator bed",
            "micu", "sicu", "iccu", "picu", "intensive care",
        ],
    },
    "maternity": {
        "specialties": [
            "gynecologyAndObstetrics", "maternalFetalMedicineOrPerinatology",
            "neonatologyPerinatalMedicine", "obstetricsAndGynaecology",
            "foetalMedicine", "familyPlanningAndComplexContraception",
        ],
        "keywords": [
            "maternity", "labour room", "labor room", "delivery ward",
            "antenatal", "postnatal", "obstetric", "birthing", "labour ward",
            "normal delivery", "caesarean", "gynaecolog",
        ],
    },
    "emergency": {
        "specialties": [
            "emergencyMedicine", "emergencyPreparednessAndDisasterResponse",
            "pediatricEmergencyMedicine",
        ],
        "keywords": [
            "emergency", "casualty", "trauma centre", "trauma center",
            "accident and emergency", "24x7 emergency", "24/7 emergency",
            "emergency department", "emergency ward", "emergency services",
        ],
    },
    "dialysis": {
        "specialties": ["nephrology"],
        "keywords": [
            "dialysis", "hemodialysis", "haemodialysis", "peritoneal dialysis",
            "dialysis unit", "dialysis machine", "dialysis centre", "dialysis center",
            "dialysis bed",
        ],
    },
    "nicu": {
        "specialties": ["neonatologyPerinatalMedicine"],
        "keywords": [
            "nicu", "neonatal intensive care", "neonatal icu",
            "newborn intensive care", "special newborn care", "sncu",
            "neonatal intensive care unit",
        ],
    },
    "oncology": {
        "specialties": [
            "medicalOncology", "surgicalOncology", "radiationOncology",
            "gynecologicalOncology", "gynecologicOncology", "pediatricOncology",
            "paediatricOncology", "paediatricHematologyOncology", "orthopaedicOncology",
        ],
        "keywords": [
            "oncology", "cancer", "chemotherapy", "radiotherapy",
            "radiation therapy", "tumour", "tumor", "oncologist", "cancer centre",
        ],
    },
}

TRUSTED_AFFILIATIONS = {"government", "academic"}

# COMMAND ----------
def score_facility(
    specialties, affiliations, capability_arr, procedure_arr,
    equipment_arr, description, cap_name
):
    """
    Returns (score: float, matched_fields: list[str], evidence_quotes: list[str])
    """
    cfg = CAPABILITIES[cap_name]
    spec_set = set(specialties or [])
    aff_set  = set(affiliations or [])
    kws      = cfg["keywords"]

    score = 0.0
    fields = []
    quotes = []

    def text_match(text, label):
        if not text:
            return False
        tl = text.lower()
        for kw in kws:
            if kw in tl:
                return True, text[:200]
        return False, None

    # specialties (highest weight — structured, reliable)
    matched_specs = spec_set & set(cfg["specialties"])
    if matched_specs:
        score += 0.35
        fields.append("specialties")
        quotes.extend(list(matched_specs)[:2])

    # capability array
    cap_hits = []
    for item in (capability_arr or []):
        hit, q = text_match(item, "capability")
        if hit and q:
            cap_hits.append(q)
    if cap_hits:
        score += 0.30
        fields.append("capability")
        quotes.extend(cap_hits[:2])

    # description
    hit, q = text_match(description, "description")
    if hit and q:
        score += 0.20
        fields.append("description")
        quotes.append(q)

    # procedure array
    proc_hits = []
    for item in (procedure_arr or []):
        hit, q = text_match(item, "procedure")
        if hit and q:
            proc_hits.append(q)
    if proc_hits:
        score += 0.10
        fields.append("procedure")
        quotes.extend(proc_hits[:1])

    # equipment array
    equip_hits = []
    for item in (equipment_arr or []):
        hit, q = text_match(item, "equipment")
        if hit and q:
            equip_hits.append(q)
    if equip_hits:
        score += 0.05
        fields.append("equipment")
        quotes.extend(equip_hits[:1])

    # trusted affiliation bonus
    if aff_set & TRUSTED_AFFILIATIONS:
        score = min(1.0, score + 0.10)
        fields.append("affiliation_bonus")

    # contradiction penalty: specialties say yes but description says "listed under"
    # (common data-aggregation artefact we saw in sample data)
    if description and "listed" in description.lower() and score > 0:
        score = max(0.0, score - 0.10)

    score = round(min(1.0, score), 4)

    if score >= 0.70:
        strength = "strong"
    elif score >= 0.40:
        strength = "partial"
    elif score > 0.0:
        strength = "weak"
    else:
        strength = "none"

    return score, fields, quotes[:5], strength

SCHEMA_OUT = StructType([
    StructField("facility_id",     StringType()),
    StructField("capability",      StringType()),
    StructField("evidence_score",  DoubleType()),
    StructField("evidence_fields", ArrayType(StringType())),
    StructField("evidence_quotes", ArrayType(StringType())),
    StructField("strength",        StringType()),
])

def score_partition(pdf_iter):
    import pandas as pd
    for pdf in pdf_iter:
        rows = []
        for _, row in pdf.iterrows():
            for cap_name in CAPABILITIES:
                score, fields, quotes, strength = score_facility(
                    row.get("specialties") or [],
                    row.get("affiliationTypeIds") or [],
                    row.get("capability") or [],
                    row.get("procedure") or [],
                    row.get("equipment") or [],
                    row.get("description"),
                    cap_name,
                )
                rows.append({
                    "facility_id":     row["unique_id"],
                    "capability":      cap_name,
                    "evidence_score":  float(score),
                    "evidence_fields": fields,
                    "evidence_quotes": quotes,
                    "strength":        strength,
                })
        yield pd.DataFrame(rows)

# COMMAND ----------
fac = spark.read.table(f"{CATALOG}.{SCHEMA}.facilities_clean")

df_scores = fac.mapInPandas(score_partition, schema=SCHEMA_OUT)

# Only keep rows where there's at least a weak signal
df_scores = df_scores.filter(F.col("evidence_score") > 0)

df_scores.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.facility_capability_scores")

total = df_scores.count()
strong = df_scores.filter("strength = 'strong'").count()
partial = df_scores.filter("strength = 'partial'").count()
print(f"Scored {total} facility-capability pairs: {strong} strong, {partial} partial")
