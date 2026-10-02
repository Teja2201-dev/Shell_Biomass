# AI-Based Estimation of Post-Harvest Groundnut Shell Biomass & Utilization Potential

An end-to-end machine learning system that estimates groundnut shell biomass after harvest by fusing multi-source **Earth Observation satellite data (MODIS Terra 250m)**, **NASA POWER Agro-Climatology**, **root-zone soil moisture**, and **ICRISAT ground-truth crop statistics**. The system provides rigorous **model comparisons**, an **improved hybrid ensemble model**, **Explainable AI (XAI via TreeSHAP)**, and calculates the **downstream utilization potential** (bioenergy, solid fuel briquettes, biochar, and carbon offsets).

---

## 1. System Architecture & Workflows

### Block Diagram
```
┌────────────────┐     ┌──────────────┐     ┌───────────────────────┐     ┌─────────────────────┐
│ Satellite Data │ ──> │ Weather Data │ ──> │ Groundnut Yield /     │ ──> │ Soil /              │
│ (MODIS NDVI/EVI│     │ (NASA POWER) │     │ Ground Truth (ICRISAT)│     │ Environmental (Soil)│
└────────────────┘     └──────────────┘     └───────────────────────┘     └──────────┬──────────┘
                                                                                     │
┌────────────────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────┐     ┌─────────────────────────────┐     ┌─────────────────────────┐
│ Data Integration /          │ ──> │ Feature Engineering /       │ ──> │ Baseline ML             │
│ Preprocessing (2000 - 2017) │     │ Selection (9 Features)      │     │ (RF / XGBoost)          │
└─────────────────────────────┘     └─────────────────────────────┘     └────────────┬────────────┘
                                                                                     │
┌────────────────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────┐     ┌─────────────────────────────┐     ┌─────────────────────────┐
│ Improved / Hybrid Model     │ ──> │ Shell Biomass Prediction    │ ──> │ Available Biomass &     │
│ (Variance-Reduced Ensemble) │     │ (Yield in kg/ha)            │     │ Utilization Potential   │
└─────────────────────────────┘     └─────────────────────────────┘     └────────────┬────────────┘
                                                                                     │
                                    ┌─────────────────────────────┐                  │
                                    │ XAI (SHAP Explanations:     │ <────────────────┘
                                    │ Global & District Local)    │
                                    └─────────────────────────────┘
```

### Algorithm Flow Chart
1. **START**
2. **Step 1: Collect / Load Data** — Multi-source extraction from MODIS Terra, NASA POWER, and ICRISAT across 11 districts (2000–2017).
3. **Step 2: Preprocess Data** — Harmonize spatial district names, aggregate 16-day and daily time-series to annual scale, zero missing values.
4. **Step 3: Feature Engineering** — Compute vegetation dynamics (NDVI/EVI mean and std dev), climate indices (Temp, RH, Precip), and root-zone soil wetness.
5. **Step 4: Train Baselines** — Train Random Forest Regressor and XGBoost Regressor on the holdout split.
6. **Step 5: Compare Models** — Benchmark models across MAE, RMSE, $R^2$, and MAPE.
7. **Step 6: Improved / Hybrid Model** — Construct a weighted hybrid ensemble (Random Forest + XGBoost + ExtraTrees) delivering superior explanatory power and reduced variance.
8. **Step 7: Predict Shell Biomass** — Predict district-level shell biomass yield ($kg/ha$) and total harvest biomass ($tonnes$).
9. **Step 8: Explain Prediction (XAI)** — TreeSHAP analysis revealing global feature drivers and local district-year attribution.
10. **Step 9: Display / Report** — Present clean formatted console tables, saved CSV/PNG artifacts, and an interactive Flask web dashboard.

---

## 2. Model Performance Comparison

Evaluated on an independent 20% holdout test partition ($N=40$) across 9 multi-source descriptors:

| Model | MAE ($kg/ha$) | RMSE ($kg/ha$) | $R^2$ Score | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** (Linear) | 101.24 | 127.32 | 0.4602 | 43.40% | Baseline |
| **XGBoost Regressor** | 81.36 | 115.64 | 0.5547 | 32.76% | Baseline |
| **Random Forest Regressor** | 76.11 | 112.82 | 0.5761 | 32.04% | Baseline |
| **Improved Hybrid Model** | **71.08** | **99.27** | **0.6719** | **29.64%** | 🏆 **BEST** |

> **Key Finding**: The Improved Hybrid Model outperforms all individual baselines, reducing MAE by **12.6%** compared to XGBoost and boosting the coefficient of determination from **0.5761 to 0.6719**.

---

## 3. Available Biomass & Utilization Potential

For any given district and year, total available shell biomass and downstream energy/material potential are calculated:

$$\text{Total Biomass (kg)} = \text{Predicted Shell Yield (kg/ha)} \times \text{Harvest Area (ha)}$$

- ⚡ **Bioenergy Potential**: Groundnut shells have a net calorific value of **17.5 MJ/kg** (17.5 GJ/Tonne). At 28% thermal power efficiency:
  $$\text{Electricity (MWh)} = \frac{\text{Biomass (kg)} \times 17.5 \times 0.28}{3,600}$$
- 🪵 **Biomass Briquettes**: High-density clean solid fuel replacing coal in boilers (~85% conversion efficiency).
- 🌱 **Biochar for Soil Health**: Slow pyrolysis conversion (~35% yield) produces carbon-rich biochar for soil moisture retention and permanent carbon sequestration.
- 🌍 **Decarbonization**: Replaces thermal coal at 0.72 tonnes coal per tonne shell, averting **2.42 tonnes $\text{CO}_2$** per tonne of coal displaced.

### State Totals Example (Recent Year 2017)
- **Total Estimated Shell Biomass**: **167,667.9 Metric Tonnes**
- **Clean Bioenergy Generation**: **228,214.7 MWh**
- **Coal Displacement & Carbon Offset**: **292,144.6 Metric Tonnes $\text{CO}_2$ avoided**

---

## 4. Explainable AI (XAI) via TreeSHAP

SHAP (SHapley Additive exPlanations) provides transparent validation of the model:
1. **Global Importance Ranking**:
   - **Mean Temperature ($^\circ C$)**: Top driver (76.6 kg/ha mean impact).
   - **Relative Humidity (%)**: Second major driver (44.5 kg/ha mean impact).
   - **MODIS NDVI Greenness**: Third major driver (21.3 kg/ha mean impact).
2. **Local District Attribution**:
   - Explains exactly why a specific district had higher or lower biomass in a specific season by decomposing the net shift from the base expected state (324.3 kg/ha).

---

## 5. How to Run the Project

### Environment Setup
```bash
# Activate the macOS environment
source mac_venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Run End-to-End Pipeline (Flow Chart Steps 1 to 9)
```bash
python main.py
```

### Run Individual Components
- **Step 5 (Model Comparison)**:
  ```bash
  python src/14_model_comparison.py
  ```
- **Step 6 (Train Improved Hybrid Model)**:
  ```bash
  python src/13_improved_hybrid_model.py
  ```
- **Step 7 & 9 (Biomass & Utilization Potential)**:
  ```bash
  python src/15_biomass_utilization.py
  ```
- **Step 8 (XAI SHAP Analysis)**:
  ```bash
  python src/16_xai_explanation.py
  ```

### Launch Interactive Web Application
```bash
python app/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 6. Generated Output Artifacts

- `outputs/model_comparison_results.csv` — Benchmark table comparing Ridge, RF, XGBoost, and Hybrid.
- `outputs/model_comparison_metrics.png` — Grouped bar chart comparing MAE, RMSE, and $R^2$.
- `outputs/district_biomass_utilization_summary.csv` — District-level biomass and energy summary.
- `outputs/xai_global_feature_importance.png` — Global SHAP feature ranking plot.
- `outputs/xai_local_district_explanation.png` — Local SHAP district attribution waterfall plot.
- `outputs/shap_detailed_analysis.csv` — Numerical SHAP importance table.
- `models/hybrid_biomass_model.pkl` — Serialized hybrid ensemble model.
