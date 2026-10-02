"""
Industry & Factory Matching Engine for Groundnut Shell Biomass Circular Economy.
Matches district-level biomass residue/wastage to nearby industrial buyers,
calculating logistics, product conversion, producer net profit, and buyer fuel savings.
"""

import math
from typing import Dict, List, Any, Optional

# District Geographic Coordinates (Andhra Pradesh Groundnut Corridor)
DISTRICT_COORDINATES = {
    "Ananthapur": {"lat": 14.6819, "lon": 77.6006, "region": "Rayalaseema"},
    "Kadapa YSR": {"lat": 14.4673, "lon": 78.8242, "region": "Rayalaseema"},
    "Kurnool": {"lat": 15.8281, "lon": 78.0373, "region": "Rayalaseema"},
    "Chittoor": {"lat": 13.2172, "lon": 79.1003, "region": "Rayalaseema"},
    "Guntur": {"lat": 16.3067, "lon": 80.4365, "region": "Coastal Andhra"},
    "Krishna": {"lat": 16.5062, "lon": 80.6480, "region": "Coastal Andhra"},
    "S.P.S. Nellore": {"lat": 14.4426, "lon": 79.9865, "region": "Coastal Andhra"},
    "West Godavari": {"lat": 16.7107, "lon": 81.0952, "region": "Coastal Andhra"},
    "East Godavari": {"lat": 16.9891, "lon": 82.2475, "region": "Coastal Andhra"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "region": "North Coastal"},
    "Srikakulam": {"lat": 18.2949, "lon": 83.8938, "region": "North Coastal"}
}

# Major Industries, Factories, and Power Stations in Andhra Pradesh
# that can utilize groundnut shell biomass products (Briquettes, Pellets, Raw Co-Firing, Biochar, Steam)
FACTORY_DATABASE = [
    # Cement Plants (Massive thermal energy demand for rotary kilns & precalciners)
    {
        "id": "ind-cem-01",
        "name": "UltraTech Cement Limited (Tadipatri Works)",
        "category": "Cement Manufacturing",
        "sub_category": "Rotary Kiln & Captive Thermal Plant",
        "location": "Tadipatri, Ananthapur District",
        "district": "Ananthapur",
        "lat": 14.9125,
        "lon": 78.0125,
        "primary_product_needed": "Biomass Briquettes / Pellets",
        "accepted_products": ["briquettes", "pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 85000,
        "current_baseline_fuel": "Thermal / Imported Steam Coal",
        "buying_price_per_tonne_inr": 6850,
        "coal_replacement_ratio": 0.88,
        "baseline_coal_cost_per_tonne_inr": 8400,
        "carbon_credit_value_per_tonne_inr": 1100,
        "description": "One of India's largest integrated cement plants. Seeks high-calorific groundnut shell briquettes to replace imported Indonesian steam coal in kiln pre-calciners."
    },
    {
        "id": "ind-cem-02",
        "name": "Bharathi Cement Corp (Vicat Group)",
        "category": "Cement Manufacturing",
        "sub_category": "Alternative Fuel & Raw Material (AFR) Unit",
        "location": "Yerraguntla, Kadapa YSR District",
        "district": "Kadapa YSR",
        "lat": 14.6335,
        "lon": 78.5385,
        "primary_product_needed": "Biomass Briquettes / Pellets",
        "accepted_products": ["briquettes", "pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 65000,
        "current_baseline_fuel": "Petcoke & Grade-9 Domestic Coal",
        "buying_price_per_tonne_inr": 6700,
        "coal_replacement_ratio": 0.85,
        "baseline_coal_cost_per_tonne_inr": 8200,
        "carbon_credit_value_per_tonne_inr": 1050,
        "description": "Equipped with automated AFR feeding system. Continuously sources agricultural residue briquettes for cement clinker manufacturing."
    },
    {
        "id": "ind-cem-03",
        "name": "Dalmia Bharat Cement Limited",
        "category": "Cement Manufacturing",
        "sub_category": "Clinker Calciner & Cogeneration",
        "location": "Chinnakomerla, Kadapa YSR District",
        "district": "Kadapa YSR",
        "lat": 14.5420,
        "lon": 78.3610,
        "primary_product_needed": "Biomass Briquettes",
        "accepted_products": ["briquettes", "pellets"],
        "annual_biomass_capacity_tonnes": 50000,
        "current_baseline_fuel": "Grade B Thermal Coal",
        "buying_price_per_tonne_inr": 6900,
        "coal_replacement_ratio": 0.90,
        "baseline_coal_cost_per_tonne_inr": 8600,
        "carbon_credit_value_per_tonne_inr": 1150,
        "description": "Industry leader in low-carbon cement production with an aggressive net-zero roadmap; pays premium rates for high-calorific groundnut briquettes."
    },
    {
        "id": "ind-cem-04",
        "name": "Penna Cement Industries Limited",
        "category": "Cement Manufacturing",
        "sub_category": "Kiln Combustion & Boiler Unit",
        "location": "Boyareddypalli, Ananthapur District",
        "district": "Ananthapur",
        "lat": 15.0210,
        "lon": 77.8920,
        "primary_product_needed": "Biomass Briquettes / Pellets",
        "accepted_products": ["briquettes", "pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 45000,
        "current_baseline_fuel": "Lignite & Coal Blends",
        "buying_price_per_tonne_inr": 6650,
        "coal_replacement_ratio": 0.84,
        "baseline_coal_cost_per_tonne_inr": 8100,
        "carbon_credit_value_per_tonne_inr": 1000,
        "description": "Heavy industrial consumer in the Rayalaseema cluster utilizing agro-briquettes to reduce furnace carbon intensity."
    },
    {
        "id": "ind-cem-05",
        "name": "Rain Cements Limited",
        "category": "Cement Manufacturing",
        "sub_category": "Waste Heat Recovery & Kiln Feed",
        "location": "Ramapuram, Kurnool District",
        "district": "Kurnool",
        "lat": 15.4120,
        "lon": 78.1180,
        "primary_product_needed": "Biomass Briquettes",
        "accepted_products": ["briquettes", "pellets"],
        "annual_biomass_capacity_tonnes": 40000,
        "current_baseline_fuel": "Grade C Coal & Coke Breeze",
        "buying_price_per_tonne_inr": 6750,
        "coal_replacement_ratio": 0.86,
        "baseline_coal_cost_per_tonne_inr": 8300,
        "carbon_credit_value_per_tonne_inr": 1050,
        "description": "Utilizes agro-waste fuels to substitute fossil fuels in high-temperature precalciner burners."
    },
    {
        "id": "ind-cem-06",
        "name": "Andhra Cements Limited (Jaypee Group)",
        "category": "Cement Manufacturing",
        "sub_category": "Limestone Processing & Kiln Unit",
        "location": "Dachepalli, Guntur District",
        "district": "Guntur",
        "lat": 16.5980,
        "lon": 79.7340,
        "primary_product_needed": "Biomass Briquettes / Pellets",
        "accepted_products": ["briquettes", "pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 38000,
        "current_baseline_fuel": "SCCL Coal",
        "buying_price_per_tonne_inr": 6600,
        "coal_replacement_ratio": 0.85,
        "baseline_coal_cost_per_tonne_inr": 8000,
        "carbon_credit_value_per_tonne_inr": 980,
        "description": "Major Guntur cement manufacturing facility actively shifting towards biomass fuel substitution."
    },

    # Thermal Power Stations (Biomass Co-firing mandated by Govt of India / Ministry of Power)
    {
        "id": "ind-pwr-01",
        "name": "Rayalaseema Thermal Power Project (APGENCO RTPP)",
        "category": "Thermal Power Generation",
        "sub_category": "Supercritical & Subcritical Boilers (1650 MW)",
        "location": "Muddanur, Kadapa YSR District",
        "district": "Kadapa YSR",
        "lat": 14.6850,
        "lon": 78.3320,
        "primary_product_needed": "Raw Biomass Pellets & Pulverized Shells",
        "accepted_products": ["pellets", "raw_biomass", "briquettes"],
        "annual_biomass_capacity_tonnes": 120000,
        "current_baseline_fuel": "Singareni Thermal Coal",
        "buying_price_per_tonne_inr": 5400,
        "coal_replacement_ratio": 0.82,
        "baseline_coal_cost_per_tonne_inr": 6800,
        "carbon_credit_value_per_tonne_inr": 850,
        "description": "State-owned 1,650 MW power station with a statutory 5-7% biomass co-firing mandate from the Ministry of Power."
    },
    {
        "id": "ind-pwr-02",
        "name": "Dr. Narla Tata Rao Thermal Power Station (APGENCO VTPS)",
        "category": "Thermal Power Generation",
        "sub_category": "Pulverized Coal Boilers (1760 MW)",
        "location": "Ibrahimpatnam, Krishna District",
        "district": "Krishna",
        "lat": 16.5920,
        "lon": 80.5280,
        "primary_product_needed": "Biomass Pellets / Shredded Shells",
        "accepted_products": ["pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 110000,
        "current_baseline_fuel": "Mahanadi & Singareni Coal",
        "buying_price_per_tonne_inr": 5350,
        "coal_replacement_ratio": 0.80,
        "baseline_coal_cost_per_tonne_inr": 6700,
        "carbon_credit_value_per_tonne_inr": 850,
        "description": "Massive coastal power generation station operating continuous biomass co-firing to reduce sulfur and particulate emissions."
    },
    {
        "id": "ind-pwr-03",
        "name": "Simhadri Super Thermal Power Plant (NTPC)",
        "category": "Thermal Power Generation",
        "sub_category": "2000 MW Central Generating Station",
        "location": "Parawada, Visakhapatnam District",
        "district": "Visakhapatnam",
        "lat": 17.6180,
        "lon": 83.0820,
        "primary_product_needed": "Torrefied / Non-Torrefied Biomass Pellets",
        "accepted_products": ["pellets", "briquettes"],
        "annual_biomass_capacity_tonnes": 95000,
        "current_baseline_fuel": "Domestic High-Ash Coal",
        "buying_price_per_tonne_inr": 5800,
        "coal_replacement_ratio": 0.85,
        "baseline_coal_cost_per_tonne_inr": 7200,
        "carbon_credit_value_per_tonne_inr": 920,
        "description": "NTPC flagship coastal power plant running National Biomass Mission procurement tenders for agro-waste pellets."
    },
    {
        "id": "ind-pwr-04",
        "name": "Sri Damodaram Sanjeevaiah TPS (APPDCL Krishnapatnam)",
        "category": "Thermal Power Generation",
        "sub_category": "Supercritical Coastal Power Plant (1600 MW)",
        "location": "Nelaturu / Krishnapatnam, S.P.S. Nellore District",
        "district": "S.P.S. Nellore",
        "lat": 14.3310,
        "lon": 80.1250,
        "primary_product_needed": "Biomass Pellets / Briquettes",
        "accepted_products": ["pellets", "briquettes", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 80000,
        "current_baseline_fuel": "Imported Steam Coal",
        "buying_price_per_tonne_inr": 5700,
        "coal_replacement_ratio": 0.84,
        "baseline_coal_cost_per_tonne_inr": 7500,
        "carbon_credit_value_per_tonne_inr": 950,
        "description": "Supercritical coastal station blending agro-biomass with coal to suppress ash volumes and meet green energy targets."
    },

    # Textile Mills, Spinning Mills & Dyeing Units (Industrial Steam Boilers)
    {
        "id": "ind-tex-01",
        "name": "Suryalakshmi Cotton Mills Limited",
        "category": "Textiles & Spinning",
        "sub_category": "Yarn Processing Steam Boiler",
        "location": "Guntakal, Ananthapur District",
        "district": "Ananthapur",
        "lat": 15.1740,
        "lon": 77.3750,
        "primary_product_needed": "Biomass Briquettes (90mm)",
        "accepted_products": ["briquettes"],
        "annual_biomass_capacity_tonnes": 18000,
        "current_baseline_fuel": "High-Cost Furnace Oil / Light Diesel",
        "buying_price_per_tonne_inr": 7200,
        "coal_replacement_ratio": 0.92,
        "baseline_coal_cost_per_tonne_inr": 9200,
        "carbon_credit_value_per_tonne_inr": 1250,
        "description": "High-capacity textile yarn spinning mill. Uses biomass briquettes for boiler steam to slash process heat costs vs furnace oil."
    },
    {
        "id": "ind-tex-02",
        "name": "Kallam Textiles Limited",
        "category": "Textiles & Spinning",
        "sub_category": "Dyeing & Weaving Steam Generation",
        "location": "Perecherla, Guntur District",
        "district": "Guntur",
        "lat": 16.3240,
        "lon": 80.3420,
        "primary_product_needed": "Biomass Briquettes",
        "accepted_products": ["briquettes", "pellets"],
        "annual_biomass_capacity_tonnes": 14000,
        "current_baseline_fuel": "Boiler Coal & Firewood",
        "buying_price_per_tonne_inr": 7100,
        "coal_replacement_ratio": 0.90,
        "baseline_coal_cost_per_tonne_inr": 8900,
        "carbon_credit_value_per_tonne_inr": 1200,
        "description": "Integrated ginning, spinning, and fabric mill running fluidized bed biomass boilers for clean steam generation."
    },
    {
        "id": "ind-tex-03",
        "name": "Precot Meridian Limited",
        "category": "Textiles & Spinning",
        "sub_category": "Spinning & Mercerizing Boilers",
        "location": "Hindupur, Ananthapur District",
        "district": "Ananthapur",
        "lat": 13.8290,
        "lon": 77.4920,
        "primary_product_needed": "Biomass Briquettes",
        "accepted_products": ["briquettes"],
        "annual_biomass_capacity_tonnes": 12500,
        "current_baseline_fuel": "Commercial Wood / Coal",
        "buying_price_per_tonne_inr": 7150,
        "coal_replacement_ratio": 0.89,
        "baseline_coal_cost_per_tonne_inr": 9000,
        "carbon_credit_value_per_tonne_inr": 1180,
        "description": "Major textile manufacturing export unit located on the Karnataka-AP corridor seeking steady supply of dry groundnut briquettes."
    },
    {
        "id": "ind-tex-04",
        "name": "Brandix India Apparel City (BIAC)",
        "category": "Textiles & Apparel",
        "sub_category": "Common Steam Boiler & Garment Washing",
        "location": "Atchutapuram SEZ, Visakhapatnam District",
        "district": "Visakhapatnam",
        "lat": 17.5250,
        "lon": 82.9810,
        "primary_product_needed": "Biomass Briquettes / Pellets",
        "accepted_products": ["briquettes", "pellets"],
        "annual_biomass_capacity_tonnes": 22000,
        "current_baseline_fuel": "Low Sulfur Heavy Stock (LSHS)",
        "buying_price_per_tonne_inr": 7400,
        "coal_replacement_ratio": 0.94,
        "baseline_coal_cost_per_tonne_inr": 9600,
        "carbon_credit_value_per_tonne_inr": 1350,
        "description": "Global green apparel manufacturing zone. Requires certified eco-friendly biomass briquettes for compliance with European brand supply chains."
    },

    # Food Processing, Sugar Mills, Distilleries & Agro-Solvent Units
    {
        "id": "ind-foo-01",
        "name": "The Andhra Sugars Limited (Chemical & Sugar Complex)",
        "category": "Agro-Processing & Chemical",
        "sub_category": "Industrial Steam & High-Pressure Boilers",
        "location": "Tanuku, West Godavari District",
        "district": "West Godavari",
        "lat": 16.7580,
        "lon": 81.6840,
        "primary_product_needed": "Biomass Briquettes / Bagasse-Shell Mix",
        "accepted_products": ["briquettes", "pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 28000,
        "current_baseline_fuel": "Sub-Bituminous Coal",
        "buying_price_per_tonne_inr": 6950,
        "coal_replacement_ratio": 0.88,
        "baseline_coal_cost_per_tonne_inr": 8700,
        "carbon_credit_value_per_tonne_inr": 1150,
        "description": "Diversified industrial complex producing industrial chemicals, caustic soda, and sugar; burns biomass fuels in off-season boilers."
    },
    {
        "id": "ind-foo-02",
        "name": "K.C.P. Sugar and Industries Corporation Ltd",
        "category": "Agro-Processing & Sugar",
        "sub_category": "Cogeneration Plant & Distillery Boilers",
        "location": "Vuyyuru, Krishna District",
        "district": "Krishna",
        "lat": 16.3680,
        "lon": 80.8460,
        "primary_product_needed": "Biomass Briquettes / Raw Shells",
        "accepted_products": ["briquettes", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 24000,
        "current_baseline_fuel": "Bagasse & Coal Mix",
        "buying_price_per_tonne_inr": 6800,
        "coal_replacement_ratio": 0.87,
        "baseline_coal_cost_per_tonne_inr": 8500,
        "carbon_credit_value_per_tonne_inr": 1100,
        "description": "Operates high-efficiency cogeneration power plants and distillery evaporators requiring supplemental high-calorific biomass."
    },
    {
        "id": "ind-foo-03",
        "name": "Chittoor Agro & Fruit Processors Association",
        "category": "Food & Beverage Processing",
        "sub_category": "Mango Pulp & Aseptic Packaging Steam Boilers",
        "location": "Chittoor Agro-Industrial Park",
        "district": "Chittoor",
        "lat": 13.2450,
        "lon": 79.0850,
        "primary_product_needed": "Biomass Briquettes (Clean Burning)",
        "accepted_products": ["briquettes"],
        "annual_biomass_capacity_tonnes": 16000,
        "current_baseline_fuel": "Diesel & Wood Logs",
        "buying_price_per_tonne_inr": 7300,
        "coal_replacement_ratio": 0.91,
        "baseline_coal_cost_per_tonne_inr": 9400,
        "carbon_credit_value_per_tonne_inr": 1280,
        "description": "Asia's largest mango pulp manufacturing cluster. Operates clean automated steam boilers for fruit pasteurization and canning."
    },
    {
        "id": "ind-foo-04",
        "name": "Mondelez India Foods (Sri City Mega Plant)",
        "category": "Food & FMCG",
        "sub_category": "Zero-Carbon Sustainability Boilers",
        "location": "Sri City SEZ, Tirupati / Chittoor Border",
        "district": "Chittoor",
        "lat": 13.5510,
        "lon": 80.0240,
        "primary_product_needed": "High-Density Biomass Pellets",
        "accepted_products": ["pellets", "briquettes"],
        "annual_biomass_capacity_tonnes": 15000,
        "current_baseline_fuel": "LPG & High-Cost Grid Electricity",
        "buying_price_per_tonne_inr": 7600,
        "coal_replacement_ratio": 0.95,
        "baseline_coal_cost_per_tonne_inr": 9900,
        "carbon_credit_value_per_tonne_inr": 1400,
        "description": "State-of-the-art confectionery manufacturing facility operating 100% sustainable biomass steam boilers."
    },
    {
        "id": "ind-foo-05",
        "name": "Kurnool Solvent Extraction & Agro-Feeds",
        "category": "Agro-Processing & Edible Oil",
        "sub_category": "Steam Desolventizing Boilers",
        "location": "Kurnool Industrial Estate",
        "district": "Kurnool",
        "lat": 15.8450,
        "lon": 78.0120,
        "primary_product_needed": "Raw Groundnut Shells / Briquettes",
        "accepted_products": ["raw_biomass", "briquettes"],
        "annual_biomass_capacity_tonnes": 20000,
        "current_baseline_fuel": "Lignite & Coal",
        "buying_price_per_tonne_inr": 6700,
        "coal_replacement_ratio": 0.86,
        "baseline_coal_cost_per_tonne_inr": 8300,
        "carbon_credit_value_per_tonne_inr": 1050,
        "description": "Oilseed crushing and solvent extraction plant requiring high volumes of low-cost steam for hexane recovery."
    },

    # Paper & Pulp Mills (Heavy Continuous Steam & Power Generation)
    {
        "id": "ind-pap-01",
        "name": "Andhra Paper Limited (Rajahmundry Mill)",
        "category": "Paper & Pulp Manufacturing",
        "sub_category": "Recovery Boiler & Multi-Fuel Cogeneration",
        "location": "Rajahmundry, East Godavari District",
        "district": "East Godavari",
        "lat": 17.0080,
        "lon": 81.7780,
        "primary_product_needed": "Biomass Briquettes & Chips",
        "accepted_products": ["briquettes", "pellets", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 35000,
        "current_baseline_fuel": "Coal & Wood Waste",
        "buying_price_per_tonne_inr": 6900,
        "coal_replacement_ratio": 0.88,
        "baseline_coal_cost_per_tonne_inr": 8600,
        "carbon_credit_value_per_tonne_inr": 1150,
        "description": "Pulp and paper manufacturing facility running high-pressure captive power boilers to drive drying rolls."
    },
    {
        "id": "ind-pap-02",
        "name": "Sree Rayalaseema Paper Mills Ltd",
        "category": "Paper & Pulp Manufacturing",
        "sub_category": "Fluidized Bed Biomass Boiler",
        "location": "Gondiparla, Kurnool District",
        "district": "Kurnool",
        "lat": 15.8620,
        "lon": 78.0650,
        "primary_product_needed": "Biomass Briquettes",
        "accepted_products": ["briquettes", "raw_biomass"],
        "annual_biomass_capacity_tonnes": 21000,
        "current_baseline_fuel": "Industrial Coal",
        "buying_price_per_tonne_inr": 6800,
        "coal_replacement_ratio": 0.87,
        "baseline_coal_cost_per_tonne_inr": 8450,
        "carbon_credit_value_per_tonne_inr": 1100,
        "description": "Kraft and packaging paper manufacturer burning agro-briquettes for economical steam generation."
    },

    # Biochar & Soil Amendment Off-Takers (Agriculture, Horticulture & Carbon Removal)
    {
        "id": "ind-agr-01",
        "name": "Rayalaseema Horticulture Soil Improvement Collective",
        "category": "Agriculture & Biochar Off-Taker",
        "sub_category": "Orchard Soil Moisture & Fertility Enrichment",
        "location": "Kadiri / Ananthapur Horticultural Zone",
        "district": "Ananthapur",
        "lat": 14.1120,
        "lon": 78.1610,
        "primary_product_needed": "Biochar / Pyrolyzed Soil Conditioner",
        "accepted_products": ["biochar"],
        "annual_biomass_capacity_tonnes": 15000,
        "current_baseline_fuel": "Commercial Organic Manure & Synthetic Gypsum",
        "buying_price_per_tonne_inr": 21500,
        "coal_replacement_ratio": 1.0,
        "baseline_coal_cost_per_tonne_inr": 28000,
        "carbon_credit_value_per_tonne_inr": 3500,
        "description": "Consortium of sweet orange, pomegranate, and banana orchard growers purchasing groundnut shell biochar to boost water retention by 35% in arid red soils."
    },
    {
        "id": "ind-agr-02",
        "name": "Coromandel International & Agro-Carbon Hub",
        "category": "Agriculture & Biochar Off-Taker",
        "sub_category": "Biochar-Enhanced Organic Fertilizer Granulation",
        "location": "Visakhapatnam Fertilizer Complex",
        "district": "Visakhapatnam",
        "lat": 17.6980,
        "lon": 83.2450,
        "primary_product_needed": "Milled Biochar (Grade-1)",
        "accepted_products": ["biochar"],
        "annual_biomass_capacity_tonnes": 25000,
        "current_baseline_fuel": "Chemical Soil Additives",
        "buying_price_per_tonne_inr": 23000,
        "coal_replacement_ratio": 1.0,
        "baseline_coal_cost_per_tonne_inr": 30000,
        "carbon_credit_value_per_tonne_inr": 4000,
        "description": "Manufactures carbon-negative biochar-coated fertilizers for export and soil rejuvenation."
    }
]

# Economic Parameters
FREIGHT_COST_PER_TONNE_KM = 4.50  # INR per Tonne-Km (Standard medium tipper truck logistics)
BASE_LOADING_UNLOADING_COST = 250.0  # INR per Tonne

# Processing & Conversion Parameters (from Raw Groundnut Shell Wastage)
PRODUCT_SPECS = {
    "briquettes": {
        "name": "Biomass Briquettes (90mm)",
        "yield_factor": 0.85,  # 1 Tonne raw shell -> 0.85 Tonne briquettes
        "processing_cost_per_tonne": 1850.0,  # INR to press & extrude
        "calorific_value_kcal_kg": 4100,
        "market_price_inr_tonne": 6800.0,
        "icon": "🪵",
        "badge": "Solid Industrial Fuel"
    },
    "pellets": {
        "name": "High-Density Biomass Pellets (8mm)",
        "yield_factor": 0.80,  # 1 Tonne raw shell -> 0.80 Tonne pellets
        "processing_cost_per_tonne": 2200.0,  # Pelletizer & binder cost
        "calorific_value_kcal_kg": 4250,
        "market_price_inr_tonne": 7400.0,
        "icon": "⚡",
        "badge": "Power Plant Co-Firing"
    },
    "raw_biomass": {
        "name": "Raw Groundnut Shell Shreds",
        "yield_factor": 0.95,  # 1 Tonne raw shell -> 0.95 Tonne cleaned shreds
        "processing_cost_per_tonne": 450.0,  # Screen & shred
        "calorific_value_kcal_kg": 3850,
        "market_price_inr_tonne": 4200.0,
        "icon": "🌾",
        "badge": "Bulk Agricultural Residue"
    },
    "biochar": {
        "name": "High-Fixed-Carbon Biochar",
        "yield_factor": 0.35,  # 1 Tonne raw shell -> 0.35 Tonne biochar (Slow pyrolysis)
        "processing_cost_per_tonne": 4800.0,  # Retort kiln & condensation
        "calorific_value_kcal_kg": 5200,
        "market_price_inr_tonne": 22000.0,
        "icon": "🌱",
        "badge": "Soil Health & Carbon Removal"
    }
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in kilometers,
    with an empirical 1.25x road detour coefficient for real highway routing.
    """
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    aerial_km = R * c
    # Apply Indian highway network tortuosity factor (~1.25)
    return round(aerial_km * 1.25, 1)


def match_industries_for_biomass(
    origin_district: str,
    raw_wastage_tonnes: float,
    max_radius_km: float = 250.0,
    product_preference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Given a district location and total raw groundnut shell wastage weight:
    1. Computes product quantities created from the wastage (Briquettes, Pellets, Biochar, Raw).
    2. Identifies all nearby factories/industries matching the products.
    3. Calculates distance, logistics freight cost, Producer Net Profit, and Factory Fuel Savings / Profit.
    4. Ranks industries by economic viability.
    """
    origin_geo = DISTRICT_COORDINATES.get(
        origin_district,
        {"lat": 14.6819, "lon": 77.6006, "region": "Rayalaseema"}
    )
    origin_lat = origin_geo["lat"]
    origin_lon = origin_geo["lon"]

    # Calculate quantities and production economics for each product line
    products_generated = {}
    for p_key, p_spec in PRODUCT_SPECS.items():
        qty_tonnes = raw_wastage_tonnes * p_spec["yield_factor"]
        mfg_cost = qty_tonnes * p_spec["processing_cost_per_tonne"]
        base_gross_val = qty_tonnes * p_spec["market_price_inr_tonne"]
        products_generated[p_key] = {
            "name": p_spec["name"],
            "icon": p_spec["icon"],
            "badge": p_spec["badge"],
            "yield_factor": p_spec["yield_factor"],
            "quantity_tonnes": round(qty_tonnes, 2),
            "mfg_cost_inr": round(mfg_cost, 2),
            "base_gross_value_inr": round(base_gross_val, 2),
            "calorific_value_kcal_kg": p_spec["calorific_value_kcal_kg"]
        }

    matched_factories = []

    for f in FACTORY_DATABASE:
        dist_km = haversine_distance(origin_lat, origin_lon, f["lat"], f["lon"])

        if dist_km > max_radius_km:
            continue

        # Check product compatibility
        compatible_products = [p for p in f["accepted_products"] if p in PRODUCT_SPECS]
        if product_preference and product_preference != "all":
            if product_preference not in compatible_products:
                continue
            active_product = product_preference
        else:
            # Pick best match product for this factory
            active_product = compatible_products[0] if compatible_products else "briquettes"

        spec = PRODUCT_SPECS[active_product]
        product_qty_tonnes = raw_wastage_tonnes * spec["yield_factor"]

        # Logistics calculation
        transport_rate_per_tonne = BASE_LOADING_UNLOADING_COST + (dist_km * FREIGHT_COST_PER_TONNE_KM)
        total_transport_cost_inr = product_qty_tonnes * transport_rate_per_tonne
        total_processing_cost_inr = product_qty_tonnes * spec["processing_cost_per_tonne"]

        # Factory purchase price
        buying_price_per_tonne = f["buying_price_per_tonne_inr"]
        gross_sales_revenue_inr = product_qty_tonnes * buying_price_per_tonne

        # 1. PRODUCER / FARMER / AGGREGATOR PROFIT (INR)
        producer_net_profit_inr = gross_sales_revenue_inr - total_processing_cost_inr - total_transport_cost_inr
        producer_profit_per_tonne_waste = producer_net_profit_inr / max(raw_wastage_tonnes, 0.001)

        # 2. BUYING INDUSTRY / FACTORY PROFIT & SAVINGS (INR)
        # Baseline fuel cost if buying factory bought equivalent coal/petcoke instead
        equivalent_coal_replaced_tonnes = product_qty_tonnes * f["coal_replacement_ratio"]
        baseline_fuel_expense_inr = equivalent_coal_replaced_tonnes * f["baseline_coal_cost_per_tonne_inr"]
        biomass_procurement_expense_inr = gross_sales_revenue_inr

        # Direct fuel cost savings
        direct_fuel_savings_inr = baseline_fuel_expense_inr - biomass_procurement_expense_inr

        # Environmental Carbon Benefit / ESG incentive (CO2 avoided = ~2.42 tonnes CO2 per tonne coal displaced)
        co2_avoided_tonnes = equivalent_coal_replaced_tonnes * 2.42
        carbon_credit_benefit_inr = product_qty_tonnes * f.get("carbon_credit_value_per_tonne_inr", 950)

        # Total Factory Net Economic Benefit (Savings + Carbon/ESG value)
        factory_total_benefit_inr = direct_fuel_savings_inr + carbon_credit_benefit_inr
        factory_savings_per_tonne = factory_total_benefit_inr / max(product_qty_tonnes, 0.001)

        matched_factories.append({
            "factory_id": f["id"],
            "name": f["name"],
            "category": f["category"],
            "sub_category": f["sub_category"],
            "location": f["location"],
            "factory_district": f["district"],
            "distance_km": dist_km,
            "product_offered": spec["name"],
            "product_key": active_product,
            "product_icon": spec["icon"],
            "product_quantity_tonnes": round(product_qty_tonnes, 2),
            "buying_price_per_tonne_inr": buying_price_per_tonne,
            "transport_cost_inr": round(total_transport_cost_inr, 2),
            "processing_cost_inr": round(total_processing_cost_inr, 2),
            "gross_revenue_inr": round(gross_sales_revenue_inr, 2),
            # Key Profit Metrics for Both Sides:
            "producer_net_profit_inr": round(producer_net_profit_inr, 2),
            "producer_profit_per_tonne_waste": round(producer_profit_per_tonne_waste, 2),
            "factory_fuel_savings_inr": round(direct_fuel_savings_inr, 2),
            "carbon_credit_benefit_inr": round(carbon_credit_benefit_inr, 2),
            "factory_total_benefit_inr": round(factory_total_benefit_inr, 2),
            "factory_savings_per_tonne": round(factory_savings_per_tonne, 2),
            "co2_avoided_tonnes": round(co2_avoided_tonnes, 2),
            "description": f["description"],
            "capacity_tonnes_year": f["annual_biomass_capacity_tonnes"],
            "baseline_fuel": f["current_baseline_fuel"]
        })

    # Sort matches by combination of proximity and producer profit
    matched_factories.sort(key=lambda x: (x["distance_km"], -x["producer_net_profit_inr"]))

    # Summary Statistics
    total_potential_producer_profit = max(0, sum(m["producer_net_profit_inr"] for m in matched_factories[:3]))
    total_factory_savings = max(0, sum(m["factory_total_benefit_inr"] for m in matched_factories[:3]))
    total_co2_offset = sum(m["co2_avoided_tonnes"] for m in matched_factories[:3])

    return {
        "origin_district": origin_district,
        "raw_wastage_tonnes": round(raw_wastage_tonnes, 2),
        "products_generated": products_generated,
        "matched_factories": matched_factories,
        "total_matched": len(matched_factories),
        "top_3_producer_profit_inr": round(total_potential_producer_profit, 2),
        "top_3_factory_savings_inr": round(total_factory_savings, 2),
        "top_3_co2_offset_tonnes": round(total_co2_offset, 2),
        "closest_factory": matched_factories[0] if matched_factories else None
    }


def calculate_company_deal(
    origin_district: str,
    factory_id: str,
    raw_wastage_tonnes: float,
    product_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculates the exact transport cost, logistics breakdown, producer net profit,
    and buying company fuel savings specifically after a company has been selected.
    """
    f = next((item for item in FACTORY_DATABASE if item["id"] == factory_id), None)
    if not f:
        return None

    origin_geo = DISTRICT_COORDINATES.get(
        origin_district,
        {"lat": 14.6819, "lon": 77.6006, "region": "Rayalaseema"}
    )
    dist_km = haversine_distance(origin_geo["lat"], origin_geo["lon"], f["lat"], f["lon"])

    # Determine product type
    if not product_type or product_type == "all" or product_type not in PRODUCT_SPECS:
        product_type = f["accepted_products"][0] if f["accepted_products"] else "briquettes"

    spec = PRODUCT_SPECS[product_type]
    product_qty_tonnes = raw_wastage_tonnes * spec["yield_factor"]

    # Distance-specific freight calculation
    freight_rate_per_tonne = BASE_LOADING_UNLOADING_COST + (dist_km * FREIGHT_COST_PER_TONNE_KM)
    total_transport_cost_inr = product_qty_tonnes * freight_rate_per_tonne
    total_processing_cost_inr = product_qty_tonnes * spec["processing_cost_per_tonne"]

    buying_price_per_tonne = f["buying_price_per_tonne_inr"]
    gross_sales_revenue_inr = product_qty_tonnes * buying_price_per_tonne

    # Producer Net Profit
    producer_net_profit_inr = gross_sales_revenue_inr - total_processing_cost_inr - total_transport_cost_inr
    producer_profit_per_tonne_waste = producer_net_profit_inr / max(raw_wastage_tonnes, 0.001)

    # Factory Fuel Savings
    equivalent_coal_replaced_tonnes = product_qty_tonnes * f["coal_replacement_ratio"]
    baseline_fuel_expense_inr = equivalent_coal_replaced_tonnes * f["baseline_coal_cost_per_tonne_inr"]
    direct_fuel_savings_inr = baseline_fuel_expense_inr - gross_sales_revenue_inr
    co2_avoided_tonnes = equivalent_coal_replaced_tonnes * 2.42
    carbon_credit_benefit_inr = product_qty_tonnes * f.get("carbon_credit_value_per_tonne_inr", 950)
    factory_total_benefit_inr = direct_fuel_savings_inr + carbon_credit_benefit_inr
    factory_savings_per_tonne = factory_total_benefit_inr / max(product_qty_tonnes, 0.001)

    return {
        "factory_id": f["id"],
        "name": f["name"],
        "category": f["category"],
        "sub_category": f["sub_category"],
        "location": f["location"],
        "factory_district": f["district"],
        "origin_district": origin_district,
        "distance_km": dist_km,
        "product_offered": spec["name"],
        "product_key": product_type,
        "product_icon": spec["icon"],
        "product_quantity_tonnes": round(product_qty_tonnes, 2),
        "raw_wastage_tonnes": round(raw_wastage_tonnes, 2),
        "freight_rate_per_tonne": round(freight_rate_per_tonne, 2),
        "transport_cost_inr": round(total_transport_cost_inr, 2),
        "processing_cost_inr": round(total_processing_cost_inr, 2),
        "buying_price_per_tonne_inr": buying_price_per_tonne,
        "gross_revenue_inr": round(gross_sales_revenue_inr, 2),
        "producer_net_profit_inr": round(producer_net_profit_inr, 2),
        "producer_profit_per_tonne_waste": round(producer_profit_per_tonne_waste, 2),
        "factory_fuel_savings_inr": round(direct_fuel_savings_inr, 2),
        "carbon_credit_benefit_inr": round(carbon_credit_benefit_inr, 2),
        "factory_total_benefit_inr": round(factory_total_benefit_inr, 2),
        "factory_savings_per_tonne": round(factory_savings_per_tonne, 2),
        "co2_avoided_tonnes": round(co2_avoided_tonnes, 2),
        "description": f["description"],
        "capacity_tonnes_year": f["annual_biomass_capacity_tonnes"],
        "baseline_fuel": f["current_baseline_fuel"]
    }

