from __future__ import annotations

import io
import json
from datetime import date
from typing import Any

from branca.element import MacroElement, Template
import folium
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from planet_water_app.analysis import (
    AnalysisResult,
    analyze_time_series,
    build_config,
    build_web_map,
    fetch_false_color_composite,
    fetch_true_color_composite,
    geometry_from_geojson,
)

DEFAULT_COLLECTION_ID = "8523c8f2-e983-4a9c-8e69-88365d946cbf"
DEFAULT_CENTER = (46.1199, 14.8153)
# Rectangle similar to the AOI shown in the user's screenshot.
DEFAULT_AOI_BOUNDS = (14.36048, 45.72536, 14.39034, 45.74489)


def default_aoi_geojson() -> dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = DEFAULT_AOI_BOUNDS
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Default AOI (Cerknisko jezero)"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat],
                        ]
                    ],
                },
            }
        ],
    }


def ensure_state() -> None:
    st.session_state.setdefault("aoi_geojson", default_aoi_geojson())
    st.session_state.setdefault("aoi_source", "Default AOI (Cerknisko jezero)")
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_settings", None)
    st.session_state.setdefault("analysis_error", None)
    st.session_state.setdefault("selected_interval_label", None)
    st.session_state.setdefault("composite_cache", {})
    st.session_state.setdefault("show_oauth_values", False)
    st.session_state.setdefault("map_layer_settings", None)


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4fbff 0%, #ecf7fb 40%, #f7fcff 100%);
        }
        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b607d 0%, #0e7a9f 60%, #118ab2 100%);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6 {
            color: #f3fbff !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] input::placeholder {
            color: #64748b !important;
            opacity: 1;
        }
        .pw-card {
            border: 1px solid #cfe3ef;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.85);
            padding: 0.5rem 0.75rem;
            box-shadow: 0 4px 12px rgba(13, 74, 102, 0.08);
        }
        .pw-timeline {
            border: 1px solid #bfd8e7;
            border-radius: 12px;
            background: linear-gradient(180deg, #ffffff 0%, #f1f8fc 100%);
            padding: 0.75rem;
            margin-bottom: 0.9rem;
        }
        .pw-timeline-title {
            font-weight: 600;
            color: #0b4f6c;
            margin-bottom: 0.3rem;
        }
        .pw-legend {
            border: 1px solid #cfe3ef;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.92);
            padding: 0.8rem 0.9rem;
            margin-top: 0.75rem;
        }
        .pw-legend-title {
            font-weight: 600;
            color: #0b4f6c;
            margin-bottom: 0.45rem;
        }
        .pw-legend-ramp {
            height: 14px;
            border-radius: 999px;
            margin: 0.25rem 0 0.35rem 0;
            border: 1px solid #cbd5e1;
        }
        .pw-legend-scale {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_collection_from_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": geometry}],
    }


def parse_uploaded_geojson(uploaded_file: Any) -> dict[str, Any]:
    return json.loads(uploaded_file.getvalue().decode("utf-8"))


def add_aoi_map_legend(base_map: folium.Map) -> None:
    legend_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    bottom: 28px;
    left: 28px;
    z-index: 9999;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
    line-height: 1.35;
    color: #111827;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
">
  <div style="font-weight: 600; margin-bottom: 6px;">Legend</div>
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="width:16px; height:12px; background:#60a5fa; border:1px solid #1d4ed8; display:inline-block;"></span>
    <span>Selected AOI</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="width:16px; height:12px; background:#f3f4f6; border:1px dashed #6b7280; display:inline-block;"></span>
    <span>Draw tools (polygon/rectangle)</span>
  </div>
</div>
{% endmacro %}
"""
    legend = MacroElement()
    legend._template = Template(legend_html)
    base_map.add_child(legend)


def aoi_input_map(geojson_payload: dict[str, Any] | None) -> folium.Map:
    base_map = folium.Map(location=DEFAULT_CENTER, zoom_start=8, tiles="CartoDB positron")

    draw_plugin = Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    )
    draw_plugin.add_to(base_map)

    if geojson_payload:
        try:
            geometry = geometry_from_geojson(geojson_payload)
            bounds = geometry.bounds
            folium.GeoJson(
                geojson_payload,
                name="Selected AOI",
                style_function=lambda _: {
                    "color": "#1d4ed8",
                    "weight": 2,
                    "fillColor": "#60a5fa",
                    "fillOpacity": 0.12,
                },
            ).add_to(base_map)
            base_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        except Exception:
            pass

    add_aoi_map_legend(base_map)
    folium.LayerControl().add_to(base_map)
    return base_map


def render_header() -> None:
    st.set_page_config(page_title="Planet Water Monitor", layout="wide")
    apply_custom_theme()
    st.title("Planet Water Monitor")
    st.caption("Interactive water monitoring application.")


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("Inputs")
        show_oauth_values = st.checkbox(
            "Show OAuth values",
            value=st.session_state["show_oauth_values"],
            help="Reveal or hide the OAuth client ID and secret fields.",
        )
        st.session_state["show_oauth_values"] = show_oauth_values
        oauth_type = "default" if show_oauth_values else "password"

        client_id = st.text_input(
            "OAuth client ID",
            value="",
            type=oauth_type,
            help="Optional if already configured locally.",
        )
        client_secret = st.text_input(
            "OAuth client secret",
            value="",
            type=oauth_type,
            help="Optional if already configured locally.",
        )
        collection_id = st.text_input("Planet BYOC collection ID", value=DEFAULT_COLLECTION_ID)

        date_cols = st.columns(2)
        start_date = date_cols[0].date_input("Start date", value=date(2025, 1, 1), format="YYYY-MM-DD")
        end_date = date_cols[1].date_input("End date", value=date(2025, 12, 31), format="YYYY-MM-DD")
        interval = st.selectbox("Time-series interval", options=["Weekly", "Monthly", "Quarterly"], index=1)

        resolution = st.slider("Resolution (m)", min_value=3, max_value=12, value=3)
        water_threshold = st.slider("Water threshold (NDWI)", min_value=-1.0, max_value=1.0, value=-0.4, step=0.05)
        minimum_valid_fraction = st.slider(
            "Minimum valid-pixel fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Intervals below this coverage are filtered out from charts and maps.",
        )
        analyze = st.button("Run analysis", type="primary", use_container_width=True)

    return {
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
        "collection_id": collection_id.strip(),
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval.lower(),
        "resolution": resolution,
        "water_threshold": water_threshold,
        "minimum_valid_fraction": minimum_valid_fraction,
        "analyze": analyze,
    }


def render_aoi_picker() -> None:
    st.subheader("Area of Interest")
    help_col, button_col = st.columns([5, 1])
    help_col.write("Draw a polygon or rectangle on the map, or upload a GeoJSON file.")

    if button_col.button("Clear AOI", use_container_width=True):
        st.session_state["aoi_geojson"] = default_aoi_geojson()
        st.session_state["aoi_source"] = "Default AOI (Cerknisko jezero)"
        st.session_state["analysis_result"] = None
        st.session_state["analysis_settings"] = None
        st.session_state["selected_interval_label"] = None
        st.session_state["composite_cache"] = {}

    uploaded = st.file_uploader("Upload GeoJSON AOI", type=["geojson", "json"])
    if uploaded is not None:
        st.session_state["aoi_geojson"] = parse_uploaded_geojson(uploaded)
        st.session_state["aoi_source"] = f"Uploaded file: {uploaded.name}"

    map_state = st_folium(
        aoi_input_map(st.session_state["aoi_geojson"]),
        width=None,
        height=460,
        returned_objects=["all_drawings"],
        use_container_width=True,
    )

    drawings = map_state.get("all_drawings") if map_state else None
    if drawings:
        geometry = drawings[-1]["geometry"]
        st.session_state["aoi_geojson"] = feature_collection_from_geometry(geometry)
        st.session_state["aoi_source"] = "Drawn on map"

    if st.session_state["aoi_geojson"]:
        geometry = geometry_from_geojson(st.session_state["aoi_geojson"])
        bounds = geometry.bounds
        st.success(
            f"AOI ready from {st.session_state['aoi_source']}. "
            f"Bounds: {bounds[0]:.5f}, {bounds[1]:.5f}, {bounds[2]:.5f}, {bounds[3]:.5f}"
        )
    else:
        st.info("No AOI selected yet.")


def render_metrics(result: AnalysisResult) -> None:
    latest = result.series.iloc[-1]
    peak = result.series.loc[result.series["water_area_ha"].idxmax()]
    cols = st.columns(4)
    cols[0].metric("Intervals kept", f"{len(result.series)}")
    cols[1].metric("Latest water area", f"{latest['water_area_ha']:.2f} ha")
    cols[2].metric("Peak water area", f"{peak['water_area_ha']:.2f} ha")
    cols[3].metric("Latest mean NDWI", f"{latest['mean_ndwi']:.3f}")


def render_charts(result: AnalysisResult, selected_start: date | None = None) -> None:
    chart_df = result.series.copy()
    chart_df["date"] = chart_df["start"]

    area_chart = px.line(
        chart_df,
        x="date",
        y="water_area_ha",
        markers=True,
        title="Water area through time",
        labels={"date": "Interval start", "water_area_ha": "Water area (ha)"},
        color_discrete_sequence=["#0ea5b7"],
    )
    ndwi_chart = px.line(
        chart_df,
        x="date",
        y="mean_ndwi",
        markers=True,
        title="Mean NDWI through time",
        labels={"date": "Interval start", "mean_ndwi": "Mean NDWI"},
        color_discrete_sequence=["#0369a1"],
    )

    if selected_start is not None:
        selected_iso = selected_start.isoformat()
        area_chart.add_vline(x=selected_iso, line_dash="dash", line_color="#ef4444")
        ndwi_chart.add_vline(x=selected_iso, line_dash="dash", line_color="#ef4444")

    area_chart.update_layout(template="plotly_white", margin={"l": 15, "r": 15, "t": 45, "b": 15})
    ndwi_chart.update_layout(template="plotly_white", margin={"l": 15, "r": 15, "t": 45, "b": 15})

    st.plotly_chart(area_chart, use_container_width=True)
    st.plotly_chart(ndwi_chart, use_container_width=True)

    with st.expander("Valid-pixel coverage", expanded=False):
        coverage_chart = px.bar(
            chart_df,
            x="date",
            y="valid_fraction",
            title="Valid-pixel coverage",
            labels={"date": "Interval start", "valid_fraction": "Coverage"},
            color_discrete_sequence=["#38bdf8"],
        )
        if selected_start is not None:
            coverage_chart.add_vline(x=selected_iso, line_dash="dash", line_color="#ef4444")
        coverage_chart.update_layout(template="plotly_white", margin={"l": 15, "r": 15, "t": 45, "b": 15})
        st.plotly_chart(coverage_chart, use_container_width=True)


def render_frequency_preview(result: AnalysisResult) -> None:
    frequency = np.nan_to_num(result.water_frequency, nan=0.0)
    preview = np.zeros((*frequency.shape, 4), dtype=np.uint8)
    preview[..., 2] = (frequency * 255).astype(np.uint8)
    preview[..., 1] = (frequency * 170).astype(np.uint8)
    preview[..., 3] = np.where(np.isnan(result.water_frequency), 0, 210).astype(np.uint8)
    st.image(preview, caption="Water frequency inside the AOI", use_container_width=True)


def _interval_options(result: AnalysisResult) -> dict[str, Any]:
    return {interval.label: interval for interval in result.interval_rasters}


def _selected_interval(result: AnalysisResult) -> Any:
    option_map = _interval_options(result)
    options = list(option_map)
    preferred = st.session_state.get("selected_interval_label")
    default_value = preferred if preferred in option_map else options[-1]

    st.markdown('<div class="pw-timeline">', unsafe_allow_html=True)
    st.markdown('<div class="pw-timeline-title">Interval timeline</div>', unsafe_allow_html=True)
    selected_label = st.select_slider(
        "Date points",
        options=options,
        value=default_value,
        help="Select a date point to update both the map and charts.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["selected_interval_label"] = selected_label
    selected_interval = option_map[selected_label]
    st.caption(f"Selected interval: {selected_interval.start.isoformat()} to {selected_interval.end.isoformat()}")
    return selected_interval


def _cached_true_color(result: AnalysisResult, settings: dict[str, Any], interval: Any) -> np.ndarray:
    return _cached_composite(result, settings, interval, composite_kind="true")


def _cached_false_color(result: AnalysisResult, settings: dict[str, Any], interval: Any) -> np.ndarray:
    return _cached_composite(result, settings, interval, composite_kind="false")


def _cached_composite(
    result: AnalysisResult,
    settings: dict[str, Any],
    interval: Any,
    *,
    composite_kind: str,
) -> np.ndarray:
    cache_key = (
        composite_kind,
        settings["collection_id"],
        interval.start.isoformat(),
        interval.end.isoformat(),
        tuple(round(v, 8) for v in result.bounds),
        tuple(result.size),
    )
    cache: dict[Any, np.ndarray] = st.session_state["composite_cache"]
    if cache_key in cache:
        return cache[cache_key]

    label = "true-color" if composite_kind == "true" else "false-color"
    with st.spinner(f"Fetching {label} composite for the selected interval..."):
        config = build_config(settings["client_id"], settings["client_secret"])
        if composite_kind == "true":
            cache[cache_key] = fetch_true_color_composite(
                collection_id=settings["collection_id"],
                config=config,
                bounds=result.bounds,
                size=result.size,
                start_date=interval.start,
                end_date=interval.end,
            )
        else:
            cache[cache_key] = fetch_false_color_composite(
                collection_id=settings["collection_id"],
                config=config,
                bounds=result.bounds,
                size=result.size,
                start_date=interval.start,
                end_date=interval.end,
            )
    st.session_state["composite_cache"] = cache
    return cache[cache_key]


def render_layer_controls() -> dict[str, bool]:
    settings = st.session_state.get("map_layer_settings") or {
        "show_true_color": False,
        "show_false_color": False,
        "show_water_mask": True,
        "show_water_frequency": False,
        "show_aoi": False,
    }
    st.markdown("#### Map layers")
    cols = st.columns(5)
    settings["show_true_color"] = cols[0].checkbox("True color", value=settings["show_true_color"])
    settings["show_false_color"] = cols[1].checkbox("False color", value=settings["show_false_color"])
    settings["show_water_mask"] = cols[2].checkbox("Water mask", value=settings["show_water_mask"])
    settings["show_water_frequency"] = cols[3].checkbox("Water frequency", value=settings["show_water_frequency"])
    settings["show_aoi"] = cols[4].checkbox("AOI", value=settings["show_aoi"])
    st.session_state["map_layer_settings"] = settings
    return settings


def render_map_legends(settings: dict[str, Any], selected_interval: Any) -> None:
    legend_cols = st.columns(3)
    legend_cols[0].markdown(
        f"""
        <div class="pw-legend">
          <div class="pw-legend-title">Detected water mask</div>
          <div>Rule: NDWI &gt; {settings['water_threshold']:.2f}</div>
          <div style="margin-top:0.35rem; color:#0f172a;">Blue overlay marks pixels classified as water in the selected interval.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    legend_cols[1].markdown(
        """
        <div class="pw-legend">
          <div class="pw-legend-title">Water frequency</div>
          <div class="pw-legend-ramp" style="background:linear-gradient(90deg,#e0f2fe 0%,#7dd3fc 35%,#0ea5e9 70%,#0369a1 100%);"></div>
          <div class="pw-legend-scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
          <div style="margin-top:0.35rem; color:#0f172a;">Computed across the full selected date range from all processed intervals.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    legend_cols[2].markdown(
        f"""
        <div class="pw-legend">
          <div class="pw-legend-title">Composites</div>
          <div>True color: Red / Green / Blue</div>
          <div>False color: NIR / Red / Green</div>
          <div style="margin-top:0.35rem; color:#0f172a;">Selected interval: {selected_interval.start.isoformat()} to {selected_interval.end.isoformat()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_linked_map_and_charts(result: AnalysisResult, settings: dict[str, Any]) -> None:
    st.markdown('<div class="pw-card">', unsafe_allow_html=True)
    selected_interval = _selected_interval(result)
    layer_settings = render_layer_controls()

    true_color = None
    if layer_settings["show_true_color"]:
        true_color = _cached_true_color(result, settings, selected_interval)

    false_color = None
    if layer_settings["show_false_color"]:
        false_color = _cached_false_color(result, settings, selected_interval)

    overlay_map = build_web_map(
        geometry=result.geometry,
        bounds=result.bounds,
        ndwi_raster=selected_interval.raster,
        water_frequency=result.water_frequency,
        true_color=true_color,
        false_color=false_color,
        water_threshold=settings["water_threshold"],
        show_water_mask=layer_settings["show_water_mask"],
        show_water_frequency=layer_settings["show_water_frequency"],
        show_aoi=layer_settings["show_aoi"],
    )
    st_folium(overlay_map, width=None, height=580, use_container_width=True)
    render_map_legends(settings, selected_interval)
    render_charts(result, selected_start=selected_interval.start)
    st.markdown("</div>", unsafe_allow_html=True)


def _series_for_display(result: AnalysisResult) -> pd.DataFrame:
    table = result.series.copy()
    table["start"] = table["start"].apply(lambda d: d.isoformat())
    table["end"] = table["end"].apply(lambda d: d.isoformat())
    return table


def _water_mask_rgba(ndwi_raster: np.ndarray, water_threshold: float) -> np.ndarray:
    water_mask = np.isfinite(ndwi_raster) & (ndwi_raster > water_threshold)
    rgba = np.zeros(ndwi_raster.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = np.where(water_mask, 16, 0)
    rgba[..., 1] = np.where(water_mask, 96, 0)
    rgba[..., 2] = np.where(water_mask, 224, 0)
    rgba[..., 3] = np.where(water_mask, 200, 0)
    return rgba


def _png_bytes(image_rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _water_frequency_geotiff_bytes(result: AnalysisResult) -> bytes:
    try:
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError("GeoTIFF export requires rasterio. Install with: pip install rasterio") from exc

    min_x, min_y, max_x, max_y = result.bounds
    width, height = result.size
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    nodata = -9999.0
    data = np.nan_to_num(result.water_frequency, nan=nodata).astype(np.float32)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
            compress="deflate",
        ) as dataset:
            dataset.write(data, 1)
        return memfile.read()


def render_frequency_exports(result: AnalysisResult, selected_interval: Any, settings: dict[str, Any]) -> None:
    geotiff_error: str | None = None
    geotiff_data: bytes | None = None
    try:
        geotiff_data = _water_frequency_geotiff_bytes(result)
    except Exception as exc:
        geotiff_error = str(exc)

    selected_png = _png_bytes(_water_mask_rgba(selected_interval.raster, settings["water_threshold"]))

    cols = st.columns(2)
    cols[0].download_button(
        "Export selected water area PNG",
        data=selected_png,
        file_name=f"water_area_{selected_interval.start.isoformat()}_{selected_interval.end.isoformat()}.png",
        mime="image/png",
        use_container_width=True,
    )
    if geotiff_data is not None:
        cols[1].download_button(
            "Export water frequency GeoTIFF",
            data=geotiff_data,
            file_name="water_frequency_inside_aoi.tif",
            mime="image/tiff",
            use_container_width=True,
        )
    elif geotiff_error:
        cols[1].warning(geotiff_error)


def run_analysis(settings: dict[str, Any]) -> None:
    if not st.session_state["aoi_geojson"]:
        st.session_state["aoi_geojson"] = default_aoi_geojson()
        st.session_state["aoi_source"] = "Default AOI (Cerknisko jezero)"

    if settings["start_date"] > settings["end_date"]:
        st.session_state["analysis_error"] = "The start date must be before or equal to the end date."
        st.session_state["analysis_result"] = None
        return

    try:
        geometry = geometry_from_geojson(st.session_state["aoi_geojson"])
        config = build_config(settings["client_id"], settings["client_secret"])
        progress_text = st.empty()
        progress_bar = st.progress(0.0, text="Preparing interval processing...")

        def on_progress(done: int, total: int, label: str) -> None:
            if total <= 0:
                progress_bar.progress(0.0, text="Preparing interval processing...")
                return
            fraction = min(max(done / total, 0.0), 1.0)
            progress_bar.progress(fraction, text=f"Processed {done}/{total} intervals ({label})")
            progress_text.caption(f"Current interval: {label}")

        result = analyze_time_series(
            geometry=geometry,
            collection_id=settings["collection_id"],
            config=config,
            start_date=settings["start_date"],
            end_date=settings["end_date"],
            interval=settings["interval"],
            resolution=settings["resolution"],
            water_threshold=settings["water_threshold"],
            minimum_valid_fraction=settings["minimum_valid_fraction"],
            progress_callback=on_progress,
        )

        progress_bar.empty()
        progress_text.empty()
        st.session_state["analysis_result"] = result
        st.session_state["analysis_settings"] = settings.copy()
        st.session_state["analysis_error"] = None
        st.session_state["selected_interval_label"] = result.interval_rasters[-1].label
        st.session_state["composite_cache"] = {}
    except Exception as exc:
        st.session_state["analysis_result"] = None
        st.session_state["analysis_settings"] = None
        st.session_state["analysis_error"] = str(exc)


def render_results(result: AnalysisResult, settings: dict[str, Any]) -> None:
    st.subheader("Results")
    render_metrics(result)

    render_linked_map_and_charts(result, settings)

    selected_label = st.session_state.get("selected_interval_label")
    option_map = _interval_options(result)
    if selected_label not in option_map:
        selected_label = result.interval_rasters[-1].label
        st.session_state["selected_interval_label"] = selected_label
    selected_interval = option_map[selected_label]

    with st.expander("Frequency preview and export", expanded=False):
        render_frequency_preview(result)
        render_frequency_exports(result, selected_interval, settings)

    with st.expander("Data table and time-series CSV", expanded=False):
        table = _series_for_display(result)
        st.dataframe(table, use_container_width=True)
        csv_data = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download time-series CSV",
            data=csv_data,
            file_name="planet_water_time_series.csv",
            mime="text/csv",
        )


def main() -> None:
    ensure_state()
    render_header()
    settings = render_sidebar()
    render_aoi_picker()

    if settings["analyze"]:
        run_analysis(settings)

    if st.session_state["analysis_error"]:
        st.error(st.session_state["analysis_error"])

    result = st.session_state["analysis_result"]
    if result is not None:
        active_settings = st.session_state["analysis_settings"] or settings
        render_results(result, active_settings)


if __name__ == "__main__":
    main()
