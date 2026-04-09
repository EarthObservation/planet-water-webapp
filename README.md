# Planet Water Web App

A Streamlit application for interactive water monitoring over user-defined areas of interest (AOI).

Users can:

- define an AOI by drawing on a map or uploading GeoJSON
- choose a date range and time-series interval
- run NDWI-based water detection against a Planet BYOC collection through Sentinel Hub
- inspect a web map with the AOI, true-color composite, water mask, and water-frequency overlay
- review time-series charts for mean NDWI, water area, and valid-pixel coverage

## Workflow summary

The app runs this end-to-end workflow:

1. AOI selection:
   draw a polygon/rectangle on the map or upload GeoJSON.
2. Time setup:
   select start and end dates plus interval cadence (weekly, monthly, quarterly).
3. Data access:
   request Planet BYOC imagery through the Sentinel Hub SDK.
4. Water analysis:
   compute NDWI from green/nir bands, apply clear/data masks, and classify water by threshold.
5. Time-series metrics:
   calculate interval water area, mean NDWI, and valid-pixel coverage.
6. Visual outputs:
   linked timeline points, interactive map layers, and charts on the same page.
7. Exports:
   save time-series CSV, water frequency as GeoTIFF, and water-area masks as PNG.

## Quick start

1. Create and activate a virtual environment (venv):

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

If you get an execution-policy error in PowerShell, run:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Or use Command Prompt (CMD):

```bash
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

1. Install the project:

```bash
pip install -e .
```

1. Start the app:

```bash
streamlit run app.py
```

1. In the sidebar, provide:

- your Sentinel Hub / Planet OAuth client ID
- your Sentinel Hub / Planet OAuth client secret
- the Planet BYOC collection ID

You can also rely on the standard Sentinel Hub configuration file or environment variables if they are already present on your machine.

## Notes

- The default collection ID targets Cerkniško jezero.
- Monthly analysis is the safest default for larger date ranges.
- Lower resolutions and smaller AOIs will return results faster.
