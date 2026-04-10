from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Literal

from branca.element import MacroElement, Template
import folium
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from shapely import contains_xy
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from planet_water_app.evalscripts import FALSE_COLOR_EVALSCRIPT, MEAN_NDWI_EVALSCRIPT, TRUE_COLOR_EVALSCRIPT

IntervalName = Literal["weekly", "monthly", "quarterly"]


@dataclass(frozen=True)
class IntervalRaster:
    label: str
    start: date
    end: date
    raster: np.ndarray
    mean_ndwi: float
    water_area_ha: float
    valid_fraction: float


@dataclass(frozen=True)
class AnalysisResult:
    geometry: BaseGeometry
    bounds: tuple[float, float, float, float]
    size: tuple[int, int]
    interval_rasters: list[IntervalRaster]
    series: pd.DataFrame
    water_frequency: np.ndarray


def geometry_from_geojson(payload: dict[str, Any]) -> BaseGeometry:
    payload_type = payload.get("type")

    if payload_type == "FeatureCollection":
        features = payload.get("features", [])
        geometries = [shape(feature["geometry"]) for feature in features if feature.get("geometry")]
    elif payload_type == "Feature":
        geometries = [shape(payload["geometry"])]
    else:
        geometries = [shape(payload)]

    if not geometries:
        raise ValueError("The provided AOI does not contain any geometry.")

    geometry = unary_union(geometries)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError("The provided AOI is empty after validation.")
    return geometry


def build_config(client_id: str | None = None, client_secret: str | None = None) -> SHConfig:
    config = SHConfig()
    if client_id:
        config.sh_client_id = client_id
    if client_secret:
        config.sh_client_secret = client_secret

    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Missing Sentinel Hub / Planet credentials. Provide them in the sidebar or in your local Sentinel Hub config."
        )
    return config


def _date_to_time_range(start_date: date, end_date: date) -> tuple[str, str]:
    start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
    return (
        start_dt.isoformat().replace("+00:00", "Z"),
        end_dt.isoformat().replace("+00:00", "Z"),
    )


def _iter_intervals(start_date: date, end_date: date, interval: IntervalName) -> list[tuple[date, date, str]]:
    current = start_date
    intervals: list[tuple[date, date, str]] = []

    while current <= end_date:
        if interval == "weekly":
            next_start = current + timedelta(days=7)
        elif interval == "monthly":
            next_start = current + relativedelta(months=1)
        else:
            next_start = current + relativedelta(months=3)

        current_end = min(next_start - timedelta(days=1), end_date)
        label = f"{current:%Y-%m-%d} to {current_end:%Y-%m-%d}"
        intervals.append((current, current_end, label))
        current = current_end + timedelta(days=1)

    return intervals


def _request_raster(
    *,
    collection_id: str,
    config: SHConfig,
    evalscript: str,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
    start_date: date,
    end_date: date,
) -> np.ndarray:
    start_iso, end_iso = _date_to_time_range(start_date, end_date)
    bbox = BBox(bbox=bounds, crs=CRS.WGS84)
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.define_byoc(collection_id),
                time_interval=(start_iso, end_iso),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    raster = request.get_data()[0]
    if raster.ndim == 3 and raster.shape[-1] == 1:
        raster = raster[..., 0]
    return np.asarray(raster)


def _build_aoi_mask(
    geometry: BaseGeometry,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    min_x, min_y, max_x, max_y = bounds
    x_step = (max_x - min_x) / width
    y_step = (max_y - min_y) / height
    xs = np.linspace(min_x + (x_step / 2), max_x - (x_step / 2), width)
    ys = np.linspace(max_y - (y_step / 2), min_y + (y_step / 2), height)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return contains_xy(geometry, x_grid, y_grid)


def _summarize_raster(
    raster: np.ndarray,
    aoi_mask: np.ndarray,
    *,
    resolution: int,
    water_threshold: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    raster = raster.astype(np.float32, copy=False)
    valid_mask = aoi_mask & np.isfinite(raster)
    aoi_pixels = int(aoi_mask.sum())

    if aoi_pixels == 0:
        raise ValueError("The selected AOI does not overlap the requested raster grid.")

    valid_fraction = float(valid_mask.sum() / aoi_pixels)
    if not valid_mask.any():
        return np.nan, 0.0, valid_fraction, valid_mask, np.zeros_like(valid_mask, dtype=bool)

    water_mask = valid_mask & (raster > water_threshold)
    mean_ndwi = float(np.nanmean(raster[valid_mask]))
    water_area_ha = float(water_mask.sum() * (resolution**2) / 10000)
    return mean_ndwi, water_area_ha, valid_fraction, valid_mask, water_mask


def analyze_time_series(
    *,
    geometry: BaseGeometry,
    collection_id: str,
    config: SHConfig,
    start_date: date,
    end_date: date,
    interval: IntervalName,
    resolution: int,
    water_threshold: float,
    minimum_valid_fraction: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AnalysisResult:
    bounds = tuple(float(value) for value in geometry.bounds)
    bbox = BBox(bbox=bounds, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    aoi_mask = _build_aoi_mask(geometry, bounds, size)

    water_counts = np.zeros((size[1], size[0]), dtype=np.uint16)
    valid_counts = np.zeros((size[1], size[0]), dtype=np.uint16)
    interval_rows: list[dict[str, Any]] = []
    interval_rasters: list[IntervalRaster] = []
    interval_specs = _iter_intervals(start_date, end_date, interval)
    total_intervals = len(interval_specs)

    if progress_callback is not None:
        progress_callback(0, total_intervals, "Initializing")

    for index, (interval_start, interval_end, label) in enumerate(interval_specs, start=1):
        raster = _request_raster(
            collection_id=collection_id,
            config=config,
            evalscript=MEAN_NDWI_EVALSCRIPT,
            bounds=bounds,
            size=size,
            start_date=interval_start,
            end_date=interval_end,
        )
        mean_ndwi, water_area_ha, valid_fraction, valid_mask, water_mask = _summarize_raster(
            raster,
            aoi_mask,
            resolution=resolution,
            water_threshold=water_threshold,
        )
        valid_counts += valid_mask.astype(np.uint16)
        water_counts += water_mask.astype(np.uint16)

        if valid_fraction < minimum_valid_fraction:
            if progress_callback is not None:
                progress_callback(index, total_intervals, label)
            continue

        interval_rows.append(
            {
                "label": label,
                "start": interval_start,
                "end": interval_end,
                "mean_ndwi": mean_ndwi,
                "water_area_ha": water_area_ha,
                "valid_fraction": valid_fraction,
            }
        )
        interval_rasters.append(
            IntervalRaster(
                label=label,
                start=interval_start,
                end=interval_end,
                raster=raster,
                mean_ndwi=mean_ndwi,
                water_area_ha=water_area_ha,
                valid_fraction=valid_fraction,
            )
        )

        if progress_callback is not None:
            progress_callback(index, total_intervals, label)

    if not interval_rows:
        raise ValueError(
            "No intervals met the valid-pixel threshold. Try a smaller AOI, a wider date range, or a lower minimum valid fraction."
        )

    water_frequency = np.full(valid_counts.shape, np.nan, dtype=np.float32)
    np.divide(water_counts, valid_counts, out=water_frequency, where=valid_counts > 0)
    water_frequency[~aoi_mask] = np.nan

    series = pd.DataFrame(interval_rows)
    return AnalysisResult(
        geometry=geometry,
        bounds=bounds,
        size=size,
        interval_rasters=interval_rasters,
        series=series,
        water_frequency=water_frequency,
    )


def fetch_true_color_composite(
    *,
    collection_id: str,
    config: SHConfig,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
    start_date: date,
    end_date: date,
) -> np.ndarray:
    raster = _request_raster(
        collection_id=collection_id,
        config=config,
        evalscript=TRUE_COLOR_EVALSCRIPT,
        bounds=bounds,
        size=size,
        start_date=start_date,
        end_date=end_date,
    )
    clipped = np.clip(raster, 0, 255).astype(np.uint8)
    if clipped.shape[-1] == 3:
        alpha = np.full(clipped.shape[:2] + (1,), 255, dtype=np.uint8)
        clipped = np.concatenate([clipped, alpha], axis=-1)
    return clipped


def fetch_false_color_composite(
    *,
    collection_id: str,
    config: SHConfig,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
    start_date: date,
    end_date: date,
) -> np.ndarray:
    raster = _request_raster(
        collection_id=collection_id,
        config=config,
        evalscript=FALSE_COLOR_EVALSCRIPT,
        bounds=bounds,
        size=size,
        start_date=start_date,
        end_date=end_date,
    )
    clipped = np.clip(raster, 0, 255).astype(np.uint8)
    if clipped.shape[-1] == 3:
        alpha = np.full(clipped.shape[:2] + (1,), 255, dtype=np.uint8)
        clipped = np.concatenate([clipped, alpha], axis=-1)
    return clipped


def _water_overlay(ndwi_raster: np.ndarray, water_threshold: float) -> np.ndarray:
    overlay = np.zeros(ndwi_raster.shape + (4,), dtype=np.uint8)
    water_mask = np.isfinite(ndwi_raster) & (ndwi_raster > water_threshold)
    overlay[..., 0] = np.where(water_mask, 16, 0)
    overlay[..., 1] = np.where(water_mask, 96, 0)
    overlay[..., 2] = np.where(water_mask, 224, 0)
    overlay[..., 3] = np.where(water_mask, 185, 0)
    return overlay


def _frequency_overlay(water_frequency: np.ndarray) -> np.ndarray:
    intensity = np.nan_to_num(water_frequency, nan=0.0)
    overlay = np.zeros(water_frequency.shape + (4,), dtype=np.uint8)
    overlay[..., 0] = (intensity * 30).astype(np.uint8)
    overlay[..., 1] = (intensity * 140).astype(np.uint8)
    overlay[..., 2] = (80 + intensity * 175).clip(0, 255).astype(np.uint8)
    overlay[..., 3] = np.where(np.isnan(water_frequency), 0, 150).astype(np.uint8)
    return overlay


def build_web_map(
    *,
    geometry: BaseGeometry,
    bounds: tuple[float, float, float, float],
    ndwi_raster: np.ndarray,
    water_frequency: np.ndarray,
    true_color: np.ndarray | None,
    false_color: np.ndarray | None,
    water_threshold: float,
    show_water_mask: bool,
    show_water_frequency: bool,
    show_aoi: bool,
    true_color_opacity: float,
    false_color_opacity: float,
    water_mask_opacity: float,
    water_frequency_opacity: float,
    aoi_fill_opacity: float,
) -> folium.Map:
    min_x, min_y, max_x, max_y = bounds
    center = [(min_y + max_y) / 2, (min_x + max_x) / 2]
    folium_map = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    if true_color is not None:
        folium.raster_layers.ImageOverlay(
            image=true_color,
            bounds=[[min_y, min_x], [max_y, max_x]],
            name="True color composite",
            opacity=true_color_opacity,
            interactive=True,
            show=True,
        ).add_to(folium_map)

    if false_color is not None:
        folium.raster_layers.ImageOverlay(
            image=false_color,
            bounds=[[min_y, min_x], [max_y, max_x]],
            name="False color composite",
            opacity=false_color_opacity,
            interactive=True,
            show=True,
        ).add_to(folium_map)

    folium.raster_layers.ImageOverlay(
        image=_water_overlay(ndwi_raster, water_threshold),
        bounds=[[min_y, min_x], [max_y, max_x]],
        name="Detected water mask",
        opacity=water_mask_opacity,
        interactive=True,
        show=show_water_mask,
    ).add_to(folium_map)

    folium.raster_layers.ImageOverlay(
        image=_frequency_overlay(water_frequency),
        bounds=[[min_y, min_x], [max_y, max_x]],
        name="Water frequency",
        opacity=water_frequency_opacity,
        interactive=True,
        show=show_water_frequency,
    ).add_to(folium_map)

    folium.GeoJson(
        mapping(geometry),
        name="AOI",
        show=show_aoi,
        style_function=lambda _: {
            "color": "#ef4444",
            "weight": 2,
            "fillColor": "#ef4444",
            "fillOpacity": aoi_fill_opacity,
        },
    ).add_to(folium_map)

    folium.LayerControl(collapsed=False).add_to(folium_map)
    folium_map.fit_bounds([[min_y, min_x], [max_y, max_x]])
    return folium_map

