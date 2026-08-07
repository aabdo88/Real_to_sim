# -*- coding: utf-8 -*-
"""
Build an Optimized OSM Map (one-time step per map location)
============================================================

Fetches OpenStreetMap data for a whole nuScenes map location and
OPTIMIZES it to match the nuScenes map:

    1) Map-wide alignment against the nuScenes lane graph
       (translation fit — reuses the reconstruction pipeline).
    2) Road refinement against the drivable-area mask:
         - each road's WIDTH is MEASURED from the true pavement
           (perpendicular probing), replacing OSM tag guesses
         - each centerline is SNAPPED to the pavement center
       Roads outside nuScenes map coverage keep their tag widths.
    3) Saves the result to  osm_optimized/<map_name>.pkl  and a
       side-by-side preview PNG (nuScenes map vs optimized OSM).

The reconstruction script (nuscenes_scene_reconstruction.py)
auto-loads the optimized map when the file exists — no fetching
or aligning at runtime.

Run this once per map location:
    python build_osm_map.py

Requires nuscenes_scene_reconstruction.py in the same folder.

Created on Thu Jul 30 2026
@author: abdoah1
"""

import pickle
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from nuscenes.map_expansion.map_api import NuScenesMap

# Reuse the reconstruction pipeline's settings and functions so
# both scripts always agree on coordinates and file formats.
from nuscenes_scene_reconstruction import (
    DATAROOT,
    OPTIMIZED_MAP_DIR,
    auto_align_osm_geometries,
    fetch_osm_geometries,
    generate_lanes_from_roads,
    sample_lane_reference_points,
)


# ============================================================
# SETTINGS
# ============================================================

# Which map location to build. The four nuScenes locations are:
# singapore-onenorth, singapore-hollandvillage,
# singapore-queenstown, boston-seaport.
MAP_NAME = "singapore-onenorth"

# Resolution of the drivable-area mask used for refinement and
# for the saved road surface. Higher = smoother edges.
MASK_PIXELS_PER_METER = 3

# Road probing: sample the centerline every PROBE_SPACING meters
# and walk sideways in PROBE_STEP increments, up to
# MAX_HALF_WIDTH meters per side.
PROBE_SPACING_METERS = 5.0
PROBE_STEP_METERS = 0.5
MAX_HALF_WIDTH_METERS = 15.0

# A road is refined only if at least this fraction of its probes
# land on the nuScenes drivable area; otherwise (roads outside
# map coverage) it keeps its OSM tag width unchanged.
MIN_COVERAGE = 0.3

# Centerline points are never shifted more than this.
MAX_CENTER_SHIFT_METERS = 4.0

# After carving, pavement (per the nuScenes mask) within this
# distance of painted road is claimed as road too. This fills
# the unclaimed strips between diverging carriageways and inside
# junctions — true medians (non-drivable in the mask) stay
# white. Raise it if white wedges remain between roads.
GAP_FILL_METERS = 6.0


# ============================================================
# DRIVABLE MASK OVER THE WHOLE CANVAS
# ============================================================


def build_drivable_mask(nusc_map):
    """
    Boolean drivable-area mask covering the whole map canvas,
    plus its extent [x_min, x_max, y_min, y_max].
    """
    canvas_width, canvas_height = (
        float(nusc_map.canvas_edge[0]),
        float(nusc_map.canvas_edge[1]),
    )

    patch_box = (
        canvas_width / 2.0,
        canvas_height / 2.0,
        canvas_height,
        canvas_width,
    )

    canvas_size = (
        int(canvas_height * MASK_PIXELS_PER_METER),
        int(canvas_width * MASK_PIXELS_PER_METER),
    )

    map_mask = nusc_map.get_map_mask(
        patch_box=patch_box,
        patch_angle=0,
        layer_names=["drivable_area"],
        canvas_size=canvas_size,
    )

    mask = map_mask[0].astype(bool)

    extent = [0.0, canvas_width, 0.0, canvas_height]

    return mask, extent


def make_mask_lookup(mask, extent):
    """Fast (x, y) -> on-drivable boolean lookup."""
    height, width = mask.shape

    x_min, x_max, y_min, y_max = extent

    def on_drivable(x, y):
        column = int((x - x_min) / (x_max - x_min) * width)
        row = int((y - y_min) / (y_max - y_min) * height)

        if 0 <= row < height and 0 <= column < width:
            return bool(mask[row, column])

        return False

    return on_drivable


# ============================================================
# ROAD REFINEMENT
# ============================================================


def resample_polyline(points, spacing):
    """Resample a polyline at roughly even spacing (keeps ends)."""
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return points

    step_lengths = np.hypot(
        *np.diff(points, axis=0).T
    )

    cumulative = np.concatenate([[0.0], np.cumsum(step_lengths)])

    total_length = float(cumulative[-1])

    if total_length < spacing:
        return points

    sample_distances = np.arange(0.0, total_length, spacing)

    sample_distances = np.append(sample_distances, total_length)

    resampled_x = np.interp(
        sample_distances, cumulative, points[:, 0]
    )

    resampled_y = np.interp(
        sample_distances, cumulative, points[:, 1]
    )

    return np.column_stack([resampled_x, resampled_y])


def refine_road_against_mask(road, on_drivable):
    """
    Refine ONE road against the drivable mask.

    At each resampled centerline point, walk perpendicular in
    both directions while still on pavement. This gives:
        measured width  = left extent + right extent
        center shift    = (left - right) / 2   (snaps to center)

    Returns a new road dictionary with a "coverage" field
    (fraction of probes that landed on pavement). Roads with
    coverage below MIN_COVERAGE are returned unchanged.
    """
    points = resample_polyline(
        road["points"], PROBE_SPACING_METERS
    )

    if len(points) < 2:
        return {**road, "coverage": 0.0}

    widths = []
    shifted_points = []
    hits = 0

    for index, point in enumerate(points):
        # Local direction from neighbors.
        previous_point = points[max(0, index - 1)]
        next_point = points[min(len(points) - 1, index + 1)]

        direction = next_point - previous_point
        norm = float(np.hypot(*direction))

        if norm < 1e-6:
            shifted_points.append(point)
            continue

        direction = direction / norm

        left_normal = np.array([-direction[1], direction[0]])

        if not on_drivable(point[0], point[1]):
            shifted_points.append(point)
            continue

        hits += 1

        # Walk outward on both sides while still on pavement.
        def side_extent(sign):
            extent = 0.0

            steps = np.arange(
                PROBE_STEP_METERS,
                MAX_HALF_WIDTH_METERS + PROBE_STEP_METERS / 2,
                PROBE_STEP_METERS,
            )

            for step in steps:
                probe = point + sign * step * left_normal

                if not on_drivable(probe[0], probe[1]):
                    break

                extent = step

            return extent

        left_extent = side_extent(+1.0)
        right_extent = side_extent(-1.0)

        widths.append(left_extent + right_extent)

        center_shift = float(
            np.clip(
                (left_extent - right_extent) / 2.0,
                -MAX_CENTER_SHIFT_METERS,
                MAX_CENTER_SHIFT_METERS,
            )
        )

        shifted_points.append(point + center_shift * left_normal)

    coverage = hits / len(points)

    if coverage < MIN_COVERAGE or not widths:
        return {**road, "coverage": round(coverage, 2)}

    # Light smoothing so the snapped centerline does not zigzag.
    shifted_points = np.asarray(shifted_points)

    smoothed = shifted_points.copy()

    if len(shifted_points) > 2:
        smoothed[1:-1] = (
            shifted_points[:-2]
            + shifted_points[1:-1]
            + shifted_points[2:]
        ) / 3.0

    measured_width = float(
        np.clip(np.median(widths), 3.0, 30.0)
    )

    return {
        "points": smoothed,
        "width_m": round(measured_width, 1),
        "coverage": round(coverage, 2),
    }


def refine_roads(roads, on_drivable):
    """Refine every road; print summary statistics."""
    refined = []
    refined_count = 0

    widths_before = []
    widths_after = []

    for road in roads:
        result = refine_road_against_mask(road, on_drivable)

        refined.append(result)

        if result["coverage"] >= MIN_COVERAGE:
            refined_count += 1
            widths_before.append(road["width_m"])
            widths_after.append(result["width_m"])

    print(
        f"Roads refined against nuScenes pavement: "
        f"{refined_count} / {len(roads)}"
    )

    if widths_before:
        print(
            "Median width (tag-based -> measured): "
            f"{np.median(widths_before):.1f} m -> "
            f"{np.median(widths_after):.1f} m"
        )

    return refined


# ============================================================
# SURFACE RASTERIZATION
# ============================================================
# Thick centerline strokes can never look like the real map —
# medians, edges, and junction shapes are POLYGON features.
# So the optimized map is stored as a rasterized road SURFACE:
# OSM says where roads exist (painted at their measured widths),
# and the nuScenes drivable mask carves the exact shape (edges,
# medians, junction fans). Roads outside nuScenes coverage keep
# their painted shape unchanged.
# ============================================================


def rasterize_optimized_map(roads, buildings, mask, extent):
    """
    Label raster over the whole canvas (same grid as the mask):
        0 = background, 1 = building, 2 = road surface.
    """
    height_px, width_px = mask.shape
    ppm = MASK_PIXELS_PER_METER

    covered_paint = PILImage.new("L", (width_px, height_px), 0)
    uncovered_paint = PILImage.new("L", (width_px, height_px), 0)
    building_paint = PILImage.new("L", (width_px, height_px), 0)

    draw_covered = ImageDraw.Draw(covered_paint)
    draw_uncovered = ImageDraw.Draw(uncovered_paint)
    draw_buildings = ImageDraw.Draw(building_paint)

    # Buildings as filled footprints.
    for footprint in buildings:
        pixels = [
            (point[0] * ppm, point[1] * ppm)
            for point in footprint
        ]

        if len(pixels) >= 3:
            draw_buildings.polygon(pixels, fill=1)

    # Roads painted at their measured (or tag) widths, with round
    # caps so segments join cleanly.
    for road in roads:
        target = (
            draw_covered
            if road["coverage"] >= MIN_COVERAGE
            else draw_uncovered
        )

        stroke_width = max(
            1, int(round(road["width_m"] * ppm))
        )

        pixels = [
            (point[0] * ppm, point[1] * ppm)
            for point in road["points"]
        ]

        if len(pixels) < 2:
            continue

        target.line(
            pixels, fill=1, width=stroke_width, joint="curve"
        )

        radius = stroke_width / 2.0

        for end in (pixels[0], pixels[-1]):
            target.ellipse(
                [
                    end[0] - radius,
                    end[1] - radius,
                    end[0] + radius,
                    end[1] + radius,
                ],
                fill=1,
            )

    covered_array = np.array(covered_paint, dtype=bool)
    uncovered_array = np.array(uncovered_paint, dtype=bool)
    building_array = np.array(building_paint, dtype=bool)

    # The key step: within nuScenes coverage, the painted OSM
    # roads are CARVED by the true pavement — exact edges,
    # medians, and junction shapes come from the mask.
    road_surface = (covered_array & mask) | uncovered_array

    # Cosmetic cleanup: the raster intersection leaves white
    # speckles inside roads and ragged edges. Closing fills the
    # small holes; opening removes isolated road specks. Both
    # operate well below median width (~2 m), so real medians
    # and gaps survive.
    try:
        from scipy import ndimage

        kernel = np.ones((3, 3), dtype=bool)

        road_surface = ndimage.binary_closing(
            road_surface, structure=kernel, iterations=2
        )

        road_surface = ndimage.binary_opening(
            road_surface, structure=kernel, iterations=1
        )

        # Gap bridging: OSM centerline strokes leave true
        # pavement unclaimed between diverging carriageways and
        # inside junctions. Claim mask-pavement within
        # GAP_FILL_METERS of painted road; true medians are
        # non-drivable in the mask and therefore stay white —
        # matching what white means on the real map.
        bridge_iterations = max(
            1, int(round(GAP_FILL_METERS * MASK_PIXELS_PER_METER))
        )

        reach = ndimage.binary_dilation(
            road_surface,
            structure=kernel,
            iterations=bridge_iterations,
        )

        road_surface = road_surface | (reach & mask)

    except ImportError:
        print(
            "scipy not available — skipping surface smoothing."
        )

    label = np.zeros(mask.shape, dtype=np.uint8)
    label[building_array] = 1
    label[road_surface] = 2

    return label





def save_preview(mask, extent, surface, output_path):
    """Side-by-side preview: nuScenes pavement vs optimized OSM."""
    from matplotlib.colors import ListedColormap

    figure, (left_axis, right_axis) = plt.subplots(
        1, 2, figsize=(16, 9)
    )

    left_axis.imshow(
        mask,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
    )

    left_axis.set_title("nuScenes drivable area (ground truth)")

    right_axis.imshow(
        surface,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["white", "0.82", "0.45"]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )

    right_axis.set_title(
        "Optimized OSM surface (carved by true pavement)"
    )

    for axis in (left_axis, right_axis):
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")

    figure.suptitle(f"Optimized OSM map — {MAP_NAME}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=110)
    plt.close(figure)

    print("Preview image written to:", output_path)


# ============================================================
# MAIN
# ============================================================


def main():
    start_time = time.time()

    print("Building optimized OSM map for:", MAP_NAME)

    nusc_map = NuScenesMap(
        dataroot=str(DATAROOT),
        map_name=MAP_NAME,
    )

    canvas_width, canvas_height = (
        float(nusc_map.canvas_edge[0]),
        float(nusc_map.canvas_edge[1]),
    )

    fetch_extent = [0.0, canvas_width, 0.0, canvas_height]

    # ---- Step 1: fetch (cached) and align map-wide.
    geometries = fetch_osm_geometries(MAP_NAME, fetch_extent)

    reference_points = sample_lane_reference_points(nusc_map)

    if reference_points is None:
        raise RuntimeError("Could not sample the lane graph.")

    geometries, (shift_x, shift_y), fit = (
        auto_align_osm_geometries(
            geometries,
            reference_points,
            max_reference_points=150,
        )
    )

    print(
        f"Alignment: dx = {shift_x:+.1f} m, "
        f"dy = {shift_y:+.1f} m (fit {fit:.2f} m)"
    )

    # ---- Step 2: refine roads against the drivable mask.
    mask, extent = build_drivable_mask(nusc_map)

    on_drivable = make_mask_lookup(mask, extent)

    optimized_roads = refine_roads(
        geometries["roads"], on_drivable
    )

    # ---- Step 3: rasterize the road surface, carved by the
    # ---- true pavement, so the map LOOKS like the real one.
    surface = rasterize_optimized_map(
        optimized_roads,
        geometries["buildings"],
        mask,
        extent,
    )

    road_pixels = int((surface == 2).sum())

    print(
        "Surface rasterized: "
        f"{road_pixels / (MASK_PIXELS_PER_METER ** 2):.0f} m² "
        "of road."
    )

    # ---- Step 4: divide the refined roads into lanes, clipped
    # ---- to the carved surface so lanes stop at junctions and
    # ---- never cross medians.
    def on_road_surface(x, y):
        row = int(y * MASK_PIXELS_PER_METER)
        column = int(x * MASK_PIXELS_PER_METER)

        return (
            0 <= row < surface.shape[0]
            and 0 <= column < surface.shape[1]
            and surface[row, column] == 2
        )

    lanes = generate_lanes_from_roads(
        optimized_roads, keep_point=on_road_surface
    )

    print(f"Lane structure: {len(lanes)} lanes generated.")

    # ---- Step 5: save the optimized map + preview.
    OPTIMIZED_MAP_DIR.mkdir(exist_ok=True)

    payload = {
        "version": 3,
        "map_name": MAP_NAME,
        "created": datetime.now().isoformat(timespec="seconds"),
        "alignment": (round(shift_x, 2), round(shift_y, 2)),
        "alignment_fit_m": round(fit, 2),
        "surface": surface,
        "surface_extent": extent,
        "surface_ppm": MASK_PIXELS_PER_METER,
        "roads": optimized_roads,
        "lanes": lanes,
        "buildings": geometries["buildings"],
    }

    output_path = (
        OPTIMIZED_MAP_DIR / f"osm_optimized_{MAP_NAME}.pkl"
    )

    output_path.write_bytes(pickle.dumps(payload))

    print("Optimized map written to:", output_path)

    save_preview(
        mask,
        extent,
        surface,
        OPTIMIZED_MAP_DIR / f"preview_{MAP_NAME}.png",
    )

    print(f"Done in {time.time() - start_time:.1f} s.")
    print(
        "The reconstruction script will now load this map "
        "automatically."
    )


if __name__ == "__main__":
    main()