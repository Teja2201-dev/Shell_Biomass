# Planned Data Sources

This document records data sources used or planned for the project. Items marked **To be verified** require confirmation before use.

## 1. Groundnut agricultural data

- **Source:** ICRISAT District Level Database (DLD)
- **Dataset:** Crops → Area production yield → Apportioned
- **State:** Andhra Pradesh
- **District:** Kadapa YSR
- **Crop:** Groundnut
- **Variable currently downloaded:** Groundnut Yield (kg per ha)
- **Years:** 2000–2017
- **Local file:** `data/raw/ICRISAT-District Level Data.csv`
- **Official URL:** https://data.icrisat.org/dld/dashboard/dld/

## 2. Weather data

- **Source:** NASA POWER
- **Dataset:** NASA POWER Point Daily
- **Location:** Latitude 14.47, Longitude 78.82
- **Period:** 2000-01-01 to 2017-12-31
- **Frequency:** Daily
- **Variables:** T2M, RH2M, GWETROOT, PRECTOTCORR
- **Local file:** `data/raw/POWER_Point_Daily_20000101_20171231_014d47N_078d82E_LST.csv`
- **Official URL:** https://power.larc.nasa.gov/

## 3. Satellite data

- **Source name:** MODIS
- **Variables we plan to use:** NDVI
- **Time period needed:** Historical period matching the agricultural data and crop seasons; exact period is **To be verified**
- **Spatial level/resolution:** MODIS pixel or district-level aggregated data; exact product and spatial resolution are **To be verified**
- **Official source URL:** https://modis.gsfc.nasa.gov/

## 4. Soil and environmental data

- **Source name:** Reliable official or public soil/environment dataset
- **Variables we plan to use:** Relevant soil or environmental variables, such as soil properties or moisture, only if reliable data is available; the final variables are **To be verified**
- **Time period needed:** Period appropriate to the selected soil/environment variables; **To be verified**
- **Spatial level/resolution:** District level or a suitable gridded resolution matched to the study area; exact resolution is **To be verified**
- **Official source URL:** **To be verified**
