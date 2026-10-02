import os
import sys
import pickle
import pandas as pd
import numpy as np
import shap
from flask import Flask, render_template, request, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.biomass_utilization import calculate_utilization
from src.industry_matching import match_industries_for_biomass, calculate_company_deal, FACTORY_DATABASE, PRODUCT_SPECS
from src.database import get_database_statistics, query_table, save_deal, get_db_connection


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "groundnut_biomass_dataset_2000_2017.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_biomass_model.pkl")
COMPARISON_CSV = os.path.join(PROJECT_ROOT, "outputs", "model_comparison_results.csv")
SHAP_CSV = os.path.join(PROJECT_ROOT, "outputs", "shap_detailed_analysis.csv")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

FEATURES = [
    "mean_temperature_C",
    "mean_relative_humidity_pct",
    "mean_root_zone_soil_wetness",
    "annual_precipitation_mm",
    "mean_ndvi",
    "mean_evi",
    "ndvi_std",
    "evi_std",
    "valid_observations"
]

FEATURE_NAMES = {
    "mean_temperature_C": "Mean Temperature (°C)",
    "mean_relative_humidity_pct": "Relative Humidity (%)",
    "mean_root_zone_soil_wetness": "Root Zone Soil Moisture",
    "annual_precipitation_mm": "Annual Precipitation (mm)",
    "mean_ndvi": "Mean NDVI (Vegetation Greenness)",
    "mean_evi": "Mean EVI (Enhanced Canopy)",
    "ndvi_std": "NDVI Seasonal Variability",
    "evi_std": "EVI Seasonal Variability",
    "valid_observations": "Satellite Clean Observations"
}

TARGET = "estimated_shell_biomass_kg_ha"

df = pd.read_csv(DATA_PATH)
DISTRICTS = sorted(df["Dist Name"].unique())
YEARS = sorted(df["year"].unique(), reverse=True)

# Baseline climatology
DISTRICT_MEANS = df.groupby("Dist Name")[FEATURES].mean().round(3).to_dict(orient="index")
STATE_MEANS = df[FEATURES].mean().round(3).to_dict()

# 1. Models initialization
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor

rf_model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1)
rf_model.fit(df[FEATURES], df[TARGET])

xgb_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)
xgb_model.fit(df[FEATURES], df[TARGET])

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        hybrid_model = pickle.load(f)
else:
    et_model = ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
    hybrid_model = VotingRegressor(
        estimators=[('rf', rf_model), ('xgb', xgb_model), ('et', et_model)],
        weights=[0.25, 0.15, 0.60]
    )
    hybrid_model.fit(df[FEATURES], df[TARGET])

AVAILABLE_MODELS = {
    "hybrid": ("Improved Hybrid Model (Ensemble)", hybrid_model),
    "rf": ("Random Forest Regressor (Baseline)", rf_model),
    "xgb": ("XGBoost Regressor (Baseline)", xgb_model)
}

# 2. Fast TreeSHAP Explainer on RF
shap_explainer = shap.TreeExplainer(rf_model)
SHAP_BASE_VAL = float(shap_explainer.expected_value[0]) if isinstance(shap_explainer.expected_value, (np.ndarray, list)) else float(shap_explainer.expected_value)

# 3. Model comparison records
comparison_records = []
if os.path.exists(COMPARISON_CSV):
    comp_df = pd.read_csv(COMPARISON_CSV)
    comparison_records = comp_df.to_dict(orient="records")

# 4. Global SHAP records
global_shap_records = []
if os.path.exists(SHAP_CSV):
    shap_df = pd.read_csv(SHAP_CSV)
    global_shap_records = shap_df.to_dict(orient="records")


def generate_shap_narrative(pred, base_val, shap_records, place_name, context_label):
    """
    Translates raw SHAP numerical outputs into a detailed, plain-English
    explanation answering: 'Why is the prediction like that?'
    """
    diff = pred - base_val
    pct_shift = (diff / base_val) * 100.0

    # Verdict
    if diff >= 50:
        verdict = f"🌟 High Biomass Season ({pct_shift:+.1f}% Above State Average)"
        verdict_color = "#1b5e20"
        sentiment = "exceptionally favorable"
    elif diff >= -30:
        verdict = f"⚖️ Near-Average Biomass Season ({pct_shift:+.1f}% of Baseline)"
        verdict_color = "#0277bd"
        sentiment = "moderate and balanced"
    else:
        verdict = f"⚠️ Below-Average / Constrained Biomass Season ({pct_shift:.1f}% Below State Average)"
        verdict_color = "#c62828"
        sentiment = "environmentally constrained"

    # Separate positive and negative forces
    pos_forces = [r for r in shap_records if r["impact"] > 0]
    neg_forces = [r for r in shap_records if r["impact"] < 0]

    max_impact = max([abs(r["impact"]) for r in shap_records]) if shap_records else 1.0

    # Human-readable narratives for top positive drivers
    pos_explanations = []
    for r in pos_forces[:3]:
        feat = r["key"]
        val = r["val"]
        imp = r["impact"]
        if feat == "mean_temperature_C":
            pos_explanations.append(f"🌡️ Moderate Thermal Conditions (Temp = {val}°C): Canopy temperature remained within the optimal vegetative window, adding +{imp:.1f} kg/ha.")
        elif feat == "mean_relative_humidity_pct":
            pos_explanations.append(f"💧 Adequate Atmospheric Humidity (RH = {val}%): Favorable vapor pressure prevented excessive plant transpiration, contributing +{imp:.1f} kg/ha.")
        elif feat == "mean_ndvi":
            pos_explanations.append(f"🌿 Vigorous Satellite Greenness (NDVI = {val}): Strong optical chlorophyll signal indicated dense vegetative growth, lifting yield by +{imp:.1f} kg/ha.")
        elif feat == "mean_root_zone_soil_wetness":
            pos_explanations.append(f"🌱 Root Zone Moisture (Wetness = {val}): Adequate subterranean water aided geocarpic peg penetration and pod filling (+{imp:.1f} kg/ha).")
        elif feat == "annual_precipitation_mm":
            pos_explanations.append(f"🌧️ Favorable Rainfall (Precip = {val:.0f} mm): Plentiful precipitation supported continuous crop development (+{imp:.1f} kg/ha).")
        else:
            pos_explanations.append(f"✅ {r['feature']} ({val}): Provided a positive vegetative boost of +{imp:.1f} kg/ha.")

    # Human-readable narratives for top negative drivers
    neg_explanations = []
    for r in neg_forces[:3]:
        feat = r["key"]
        val = r["val"]
        imp = abs(r["impact"])
        if feat == "mean_temperature_C":
            neg_explanations.append(f"🔥 Thermal Stress Penalty (Temp = {val}°C): High heat during flowering and pegging suppressed pod formation, pulling down biomass by -{imp:.1f} kg/ha.")
        elif feat == "mean_relative_humidity_pct":
            neg_explanations.append(f"💨 Dry Atmospheric Air (RH = {val}%): Subdued relative humidity accelerated moisture loss and crop stress, subtracting -{imp:.1f} kg/ha.")
        elif feat == "mean_ndvi":
            neg_explanations.append(f"🍂 Sparse Canopy Greenness (NDVI = {val}): Below-average vegetative cover restricted photosynthesis, lowering yield by -{imp:.1f} kg/ha.")
        elif feat == "annual_precipitation_mm":
            neg_explanations.append(f"☀️ Rainfall Deficit (Precip = {val:.0f} mm): Deficient seasonal rainfall created drought stress during pod enlargement (-{imp:.1f} kg/ha).")
        elif feat == "mean_root_zone_soil_wetness":
            neg_explanations.append(f"🏜️ Subsoil Moisture Stress (Wetness = {val}): Inadequate moisture in the root zone hardened soil and hampered pod development (-{imp:.1f} kg/ha).")
        else:
            neg_explanations.append(f"⚠️ {r['feature']} ({val}): Imposed an environmental penalty of -{imp:.1f} kg/ha.")

    # Executive Summary Paragraph
    summary_p = (
        f"The AI model estimated {pred:.1f} kg/ha of shell biomass for {place_name} ({context_label}), "
        f"which represents a net shift of {diff:+.1f} kg/ha compared to the state historical baseline of {base_val:.1f} kg/ha. "
        f"The prediction is {sentiment}. "
    )
    if neg_forces and (not pos_forces or abs(neg_forces[0]["impact"]) > pos_forces[0]["impact"]):
        summary_p += f"The primary limiting factor was {neg_forces[0]['feature']}, which dragged the yield down by {neg_forces[0]['impact']:.1f} kg/ha."
    elif pos_forces:
        summary_p += f"The strongest positive catalyst was {pos_forces[0]['feature']}, which lifted the yield by +{pos_forces[0]['impact']:.1f} kg/ha."

    # Practical agronomic takeaway
    if any(r["key"] == "mean_root_zone_soil_wetness" and r["impact"] < 0 for r in neg_forces):
        recommendation = "💡 Recommendation: Ground moisture was a key bottleneck. Applying supplemental irrigation or organic mulching can preserve root-zone moisture and recover up to 15-25 kg/ha of lost shell biomass."
    elif any(r["key"] == "mean_temperature_C" and r["impact"] < 0 for r in neg_forces):
        recommendation = "💡 Recommendation: Thermal stress reduced pod formation. Shifting sowing dates by 10-14 days to escape peak mid-season temperatures can alleviate heat penalties."
    elif diff > 30:
        recommendation = "💡 Opportunity: Favorable agro-climatic conditions maximized biomass yield. This harvest offers prime potential for commercial briquette aggregation and biochar pyrolysis."
    else:
        recommendation = "💡 Agronomic Note: Conditions aligned closely with historical norms. Standard bioenergy feedstock planning is well supported."

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "summary": summary_p,
        "pos_explanations": pos_explanations,
        "neg_explanations": neg_explanations,
        "max_impact": max_impact,
        "recommendation": recommendation,
        "diff": diff,
        "pct_shift": pct_shift
    }


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


# Page 1: Home & Overview
@app.route("/")
def home():
    return render_template(
        "index.html",
        active_page="home",
        districts=DISTRICTS,
        years=YEARS
    )


# Page 2: Future Land & Biomass Predictor
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    util_data = None
    row_features = {}
    shap_attribution = []
    narrative = None
    
    industry_matches = None
    
    custom_place = "Ananthapur"
    land_area_val = 50.0
    land_area_unit = "acres"
    soil_type = "normal"
    weather_type = "normal"
    selected_model_key = "hybrid"
    effective_area_ha = 50.0 * 0.40468564

    if request.method == "POST":
        custom_place = request.form.get("place", "Ananthapur")
        try:
            land_area_val = float(request.form.get("area", 50.0))
        except ValueError:
            land_area_val = 50.0

        land_area_unit = request.form.get("unit", "acres")
        soil_type = request.form.get("soil_condition", "normal")
        weather_type = request.form.get("weather_condition", "normal")
        selected_model_key = request.form.get("model", "hybrid")
    else:
        custom_place = request.args.get("place", "Ananthapur")
        try:
            land_area_val = float(request.args.get("area", 50.0))
        except (ValueError, TypeError):
            land_area_val = 50.0

        land_area_unit = request.args.get("unit", "acres")
        soil_type = request.args.get("soil_condition", "normal")
        weather_type = request.args.get("weather_condition", "normal")
        selected_model_key = request.args.get("model", "hybrid")

    if land_area_unit == "acres":
        effective_area_ha = land_area_val * 0.40468564
    else:
        effective_area_ha = land_area_val

    base_feats = DISTRICT_MEANS.get(custom_place, STATE_MEANS).copy()

    if soil_type == "dry":
        base_feats["mean_root_zone_soil_wetness"] = max(0.2, base_feats["mean_root_zone_soil_wetness"] * 0.75)
        base_feats["mean_ndvi"] = max(0.1, base_feats["mean_ndvi"] * 0.85)
    elif soil_type == "irrigated":
        base_feats["mean_root_zone_soil_wetness"] = min(0.9, base_feats["mean_root_zone_soil_wetness"] * 1.25)
        base_feats["mean_ndvi"] = min(0.65, base_feats["mean_ndvi"] * 1.20)

    if weather_type == "drought":
        base_feats["annual_precipitation_mm"] = base_feats["annual_precipitation_mm"] * 0.65
        base_feats["mean_temperature_C"] = base_feats["mean_temperature_C"] + 1.2
        base_feats["mean_relative_humidity_pct"] = base_feats["mean_relative_humidity_pct"] * 0.85
    elif weather_type == "abundant":
        base_feats["annual_precipitation_mm"] = base_feats["annual_precipitation_mm"] * 1.35
        base_feats["mean_relative_humidity_pct"] = min(90.0, base_feats["mean_relative_humidity_pct"] * 1.15)

    input_df = pd.DataFrame([[base_feats[f] for f in FEATURES]], columns=FEATURES)

    model_tuple = AVAILABLE_MODELS.get(selected_model_key, AVAILABLE_MODELS["hybrid"])
    active_model = model_tuple[1]
    prediction = float(active_model.predict(input_df)[0])

    area_in_thousand_ha = effective_area_ha / 1000.0
    util_data = calculate_utilization(prediction, area_in_thousand_ha)

    # Match Nearby Industries & Calculate Profits
    industry_matches = match_industries_for_biomass(
        origin_district=custom_place,
        raw_wastage_tonnes=util_data["total_biomass_tonnes"],
        max_radius_km=300.0
    )

    # Real-time SHAP explanation
    s_vals = shap_explainer.shap_values(input_df)[0]
    max_imp = max([abs(v) for v in s_vals]) if len(s_vals) > 0 else 1.0

    for f, impact in zip(FEATURES, s_vals):
        shap_attribution.append({
            "feature": FEATURE_NAMES[f],
            "key": f,
            "val": round(float(base_feats[f]), 2),
            "impact": float(impact),
            "rel_pct": round(abs(float(impact)) / max_imp * 100.0, 1)
        })
    shap_attribution.sort(key=lambda x: abs(x["impact"]), reverse=True)

    for f in FEATURES:
        row_features[FEATURE_NAMES[f]] = round(float(base_feats[f]), 2)

    narrative = generate_shap_narrative(prediction, SHAP_BASE_VAL, shap_attribution, custom_place, "Custom Land Forecast")

    return render_template(
        "predict.html",
        active_page="predict",
        districts=DISTRICTS,
        models=[(k, v[0]) for k, v in AVAILABLE_MODELS.items()],
        selected_model=selected_model_key,
        custom_place=custom_place,
        land_area_val=land_area_val,
        land_area_unit=land_area_unit,
        soil_type=soil_type,
        weather_type=weather_type,
        effective_area_ha=effective_area_ha,
        prediction=prediction,
        util_data=util_data,
        row_features=row_features,
        shap_attribution=shap_attribution,
        shap_base_val=SHAP_BASE_VAL,
        narrative=narrative,
        industry_matches=industry_matches
    )


# Page: Circular Economy & Factory Marketplace
@app.route("/marketplace", methods=["GET", "POST"])
def marketplace():
    selected_district = request.values.get("district", "Ananthapur")
    try:
        biomass_tonnes = float(request.values.get("biomass_tonnes", 100.0))
    except ValueError:
        biomass_tonnes = 100.0
    
    selected_product = request.values.get("product", "all")
    try:
        max_radius = float(request.values.get("radius", 250.0))
    except ValueError:
        max_radius = 250.0

    selected_factory_id = request.values.get("factory_id", "")

    match_results = match_industries_for_biomass(
        origin_district=selected_district,
        raw_wastage_tonnes=biomass_tonnes,
        max_radius_km=max_radius,
        product_preference=selected_product
    )

    # Select specific factory to calculate exact transport & profit
    selected_factory = None
    if selected_factory_id:
        selected_factory = next((f for f in match_results["matched_factories"] if f["factory_id"] == selected_factory_id), None)
    if not selected_factory and match_results["matched_factories"]:
        selected_factory = match_results["matched_factories"][0]

    return render_template(
        "marketplace.html",
        active_page="marketplace",
        districts=DISTRICTS,
        selected_district=selected_district,
        biomass_tonnes=biomass_tonnes,
        selected_product=selected_product,
        max_radius=max_radius,
        match_results=match_results,
        selected_factory=selected_factory,
        product_specs=PRODUCT_SPECS
    )


@app.route("/api/match_industries")
def api_match_industries():
    district = request.args.get("district", "Ananthapur")
    try:
        biomass = float(request.args.get("biomass", 50.0))
    except ValueError:
        biomass = 50.0
    try:
        radius = float(request.args.get("radius", 250.0))
    except ValueError:
        radius = 250.0
    product = request.args.get("product", "all")
    res = match_industries_for_biomass(district, biomass, radius, product)
    return res


@app.route("/api/calculate_deal")
def api_calculate_deal():
    district = request.args.get("district", "Ananthapur")
    factory_id = request.args.get("factory_id", "")
    try:
        biomass = float(request.args.get("biomass", 100.0))
    except ValueError:
        biomass = 100.0
    product = request.args.get("product", "all")
    deal = calculate_company_deal(district, factory_id, biomass, product)
    if not deal:
        return {"error": "Factory not found"}, 404
    return deal




@app.route("/database", methods=["GET", "POST"])
def database_explorer():
    selected_table = request.values.get("table", "biomass_observations")
    try:
        limit = int(request.values.get("limit", 50))
    except ValueError:
        limit = 50
    search = request.values.get("search", "").strip()

    db_stats = get_database_statistics()
    table_names = [t["table_name"] for t in db_stats["table_stats"]]
    if selected_table not in table_names:
        selected_table = "biomass_observations"

    rows = query_table(selected_table, limit=limit, search=search if search else None)
    current_table_info = next((t for t in db_stats["table_stats"] if t["table_name"] == selected_table), None)

    return render_template(
        "database.html",
        active_page="database",
        db_stats=db_stats,
        selected_table=selected_table,
        current_table_info=current_table_info,
        rows=rows,
        limit=limit,
        search=search
    )


@app.route("/api/confirm_deal", methods=["POST"])
def api_confirm_deal():
    data = request.get_json() or request.form
    deal_ref = save_deal(dict(data))
    return {"status": "success", "deal_reference": deal_ref, "message": f"Deal confirmed and stored in SQLite database under reference {deal_ref}."}


# Page 3: Model Performance Comparison

@app.route("/comparison")
def comparison():
    return render_template(
        "comparison.html",
        active_page="comparison",
        comparison_records=comparison_records
    )


# Page 4: Explainable AI & SHAP
@app.route("/xai", methods=["GET", "POST"])
def xai():
    selected_district = "Ananthapur"
    selected_year = 2017
    local_shap_records = []
    base_val = SHAP_BASE_VAL
    predicted_val = 0.0
    narrative = None

    if request.method == "POST":
        selected_district = request.form.get("district", "Ananthapur")
        selected_year = int(request.form.get("year", 2017))

    row = df[(df["Dist Name"] == selected_district) & (df["year"] == selected_year)]
    if not row.empty:
        x_row = row[FEATURES]
        feature_vals = {f: float(row[f].iloc[0]) for f in FEATURES}
    else:
        base_feats = DISTRICT_MEANS.get(selected_district, STATE_MEANS).copy()
        year_drift = (selected_year - 2010) * 0.02
        base_feats["mean_temperature_C"] = round(max(18.0, min(38.0, base_feats["mean_temperature_C"] + year_drift)), 2)
        x_row = pd.DataFrame([[base_feats[f] for f in FEATURES]], columns=FEATURES)
        feature_vals = base_feats

    predicted_val = float(hybrid_model.predict(x_row)[0])
    s_vals = shap_explainer.shap_values(x_row)[0]
    max_imp = max([abs(v) for v in s_vals]) if len(s_vals) > 0 else 1.0

    for f, impact in zip(FEATURES, s_vals):
        local_shap_records.append({
            "feature": FEATURE_NAMES[f],
            "key": f,
            "val": round(float(feature_vals[f]), 2),
            "impact": float(impact),
            "rel_pct": round(abs(float(impact)) / max_imp * 100.0, 1)
        })
    local_shap_records.sort(key=lambda x: abs(x["impact"]), reverse=True)

    narrative = generate_shap_narrative(predicted_val, base_val, local_shap_records, selected_district, str(selected_year))

    return render_template(
        "xai.html",
        active_page="xai",
        districts=DISTRICTS,
        years=YEARS,
        selected_district=selected_district,
        selected_year=selected_year,
        base_val=base_val,
        predicted_val=predicted_val,
        local_shap_records=local_shap_records,
        global_shap_records=global_shap_records,
        narrative=narrative
    )


# Page 5: Historical District Records
@app.route("/historical", methods=["GET", "POST"])
def historical():
    prediction = None
    actual_shell_biomass = None
    util_data = None
    row_features = {}
    
    selected_district = request.form.get("district") or request.args.get("district") or (DISTRICTS[0] if DISTRICTS else "Ananthapur")
    year_input = request.form.get("year") or request.args.get("year")
    try:
        selected_year = int(year_input) if year_input else (YEARS[0] if YEARS else 2017)
    except (ValueError, TypeError):
        selected_year = YEARS[0] if YEARS else 2017
        
    selected_model_key = request.form.get("model") or request.args.get("model") or "hybrid"

    row = df[(df["Dist Name"] == selected_district) & (df["year"] == selected_year)]
    is_archived_record = not row.empty

    model_tuple = AVAILABLE_MODELS.get(selected_model_key, AVAILABLE_MODELS["hybrid"])
    active_model = model_tuple[1]

    if is_archived_record:
        # Exact verified historical archive row
        x_vals = row[FEATURES]
        prediction = float(active_model.predict(x_vals)[0])
        actual_shell_biomass = float(row[TARGET].iloc[0])

        area_k_ha = float(row["Groundnut Area"].iloc[0])
        util_data = calculate_utilization(prediction, area_k_ha)

        for f in FEATURES:
            row_features[FEATURE_NAMES[f]] = round(float(row[f].iloc[0]), 2)
    else:
        # Dynamic unarchived / custom year prediction:
        # Synthesize baseline agro-meteorological climatology for this district with secular trend
        base_feats = DISTRICT_MEANS.get(selected_district, STATE_MEANS).copy()
        
        # Apply climatic drift (+0.02°C thermal trend per year relative to 2010 baseline)
        year_drift = (selected_year - 2010) * 0.02
        base_feats["mean_temperature_C"] = round(max(18.0, min(38.0, base_feats["mean_temperature_C"] + year_drift)), 2)
        
        input_df = pd.DataFrame([[base_feats[f] for f in FEATURES]], columns=FEATURES)
        prediction = float(active_model.predict(input_df)[0])
        actual_shell_biomass = None  # No ground truth survey recorded for this custom year

        dist_df = df[df["Dist Name"] == selected_district]
        area_k_ha = float(dist_df["Groundnut Area"].mean()) if not dist_df.empty else 100.0
        util_data = calculate_utilization(prediction, area_k_ha)

        for f in FEATURES:
            row_features[FEATURE_NAMES[f]] = round(float(base_feats[f]), 2)

    # 198 complete historical observations for manual search & live filtering
    all_historical_records = []
    for _, r in df.sort_values(by=["Dist Name", "year"], ascending=[True, False]).iterrows():
        all_historical_records.append({
            "district": str(r["Dist Name"]),
            "year": int(r["year"]),
            "area": round(float(r["Groundnut Area"]), 1),
            "production": round(float(r["Groundnut Production"]), 1),
            "pod_yield": round(float(r["Groundnut Yield"]), 1),
            "shell_biomass": round(float(r[TARGET]), 1),
            "temp": round(float(r["mean_temperature_C"]), 1),
            "humidity": round(float(r["mean_relative_humidity_pct"]), 1),
            "rainfall": round(float(r["annual_precipitation_mm"]), 1),
            "ndvi": round(float(r["mean_ndvi"]), 3),
            "soil_wetness": round(float(r["mean_root_zone_soil_wetness"]), 3)
        })

    return render_template(
        "historical.html",
        active_page="historical",
        districts=DISTRICTS,
        years=YEARS,
        models=[(k, v[0]) for k, v in AVAILABLE_MODELS.items()],
        selected_model=selected_model_key,
        selected_district=selected_district,
        selected_year=selected_year,
        prediction=prediction,
        actual_shell_biomass=actual_shell_biomass,
        is_archived_record=is_archived_record,
        util_data=util_data,
        row_features=row_features,
        historical_records=all_historical_records
    )


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html", active_page="404"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html", active_page="500"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f"  Starting Flask Web Application")
    print(f"  > Localhost (this device):   http://127.0.0.1:{port}")
    print(f"  > Network (any device on Wi-Fi): http://192.168.1.5:{port}")
    print(f"=======================================================\n")
    app.run(debug=False, host=host, port=port)


