import os
from dotenv import load_dotenv, find_dotenv

# Only load .env file if it exists (for local development)
# Railway and other cloud platforms provide environment variables directly
try:
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
except Exception:
    # If dotenv loading fails, continue with system environment variables
    pass

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

_origins = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()] or ["*"]

# NASA API Configuration (use environment variables or fall back to DEMO_KEY)
NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# NASA API Base URLs
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_MODIS_BASE_URL = "https://modis.gsfc.nasa.gov/data/"
NASA_EARTHDATA_BASE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Weather Underground API Configuration
WEATHER_UNDERGROUND_API_KEY = os.getenv("WEATHER_UNDERGROUND_API_KEY", "")
WEATHER_UNDERGROUND_BASE_URL = "https://api.weather.com/v2"

# FAO (Food and Agriculture Organization) API
FAO_API_BASE_URL = "http://www.fao.org/faostat/api/v1"
FAO_DATAMART_URL = "https://datalab.review.fao.org/datalab/api"

# Bangladesh Agricultural Data Sources
BARC_API_URL = "http://www.barc.gov.bd"  # Bangladesh Agricultural Research Council
DAE_API_URL = "http://www.dae.gov.bd"    # Department of Agricultural Extension
BRRI_API_URL = "http://www.brri.gov.bd"  # Bangladesh Rice Research Institute
BARI_API_URL = "http://www.bari.gov.bd"  # Bangladesh Agricultural Research Institute
