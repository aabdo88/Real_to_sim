# -*- coding: utf-8 -*-
"""
nuScenes Scene Reconstruction Pipeline (V2 architecture)
========================================================

Reconstructs a driving scene from a five-second front-camera
clip and compares it, side by side, against the ground truth:

    Layer 0 : Ego pose ESTIMATED FROM THE VIDEO — ground-plane
              visual odometry + lane-marking detection, anchored
              to the (OSM-derived) road network from a coarse
              GPS-like prior. No ground-truth pose is used on
              the reconstructed side.
    Layer 1 : Lane-relative localization from the HD map —
              kept as the REFERENCE the video estimate is
              compared against (CLRNet/UFLD would replace it).
    Layer 2 : Dynamic agents — still nuScenes 3D annotations,
              expressed relative to the ego (FCOS3D/StreamPETR
              is the drop-in replacement).
    Layer 3 : Static structure — HD-map probing (metric depth +
              segmentation is the replacement).
    Layer 4 : Fused BEV scene description (JSON) -> Scenic.

Honest accounting of inputs on the RECONSTRUCTED side:
    from the video   : ego trajectory, heading, lane offset
    from calibration : camera intrinsics + mounting pose
                       (sensor properties, not scene truth)
    from OSM         : the optimized map + anchoring network
    from a prior     : simulated GPS start fix (~10 m error)
    still borrowed   : agent boxes relative to the ego
The real-map panel always uses the true pose, so the two bottom
panels display the estimation error directly, and the console
prints mean/max ego error per run.

Display: four synchronized panels
    [ CAM_FRONT + 3D boxes | ego-centric BEV ]
    [ real map (truth)     | reconstructed map (estimated) ]

Outputs (written to OUTPUT_DIR):
    scene_description_<scene>.json   BEV description per keyframe
                                     (incl. ego_estimated block)
    scenario_<scene>.scenic          Scenic program, first keyframe
    reconstruction_<scene>.mp4       the four-panel playback

Created on Wed Jul 22 2026
@author: abdoah1
"""

import json
import math
import pickle
import time
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# SETTINGS
# ============================================================

DATAROOT = Path(r"C:\datasets\nuscenes\v1.0-mini")
VERSION = "v1.0-mini"

SCENE_INDEX = 0
CAMERA_CHANNEL = "CAM_FRONT"

VIDEO_DURATION_SECONDS = 5
PLAYBACK_FPS = 12

# Save the full four-panel playback as an MP4 into OUTPUT_DIR
# (reconstruction_<scene>.mp4). Uses OpenCV, which the nuScenes
# devkit already installs — no ffmpeg needed. The video contains
# exactly what plays on screen, at PLAYBACK_FPS.
SAVE_VIDEO = True

# ------------------------------------------------------------
# EGO ESTIMATION FROM VIDEO (no ground-truth pose).
# When enabled, the reconstructed panel places the vehicle using
# ONLY: (a) the camera video, (b) the camera calibration
# (intrinsics + mounting pose — a property of the sensor, not
# of the scene), and (c) a coarse GPS-like prior for the start
# position. Pipeline: ground-plane visual odometry for the
# trajectory, lane-marking detection + inverse perspective
# mapping for lateral position, and map matching (rotation +
# translation search against the road network) for anchoring.
# The real-map panel keeps the true pose, so left-vs-right
# shows the estimation error honestly.
# ------------------------------------------------------------
ESTIMATE_EGO_FROM_VIDEO = True

# Simulated GPS prior on the START position: the true start,
# offset by this many meters in a seeded random direction —
# what a consumer GNSS fix would give.
GPS_PRIOR_NOISE_METERS = 10.0
GPS_PRIOR_SEED = 7

# Anchoring search radius around the GPS prior.
ANCHOR_SEARCH_RADIUS_METERS = 25.0

# Visual odometry: image scale (speed), features per frame, and
# the image-row band (fractions of height) treated as road
# surface for ground-plane tracking.
VO_IMAGE_SCALE = 0.5
VO_MAX_FEATURES = 400
VO_GROUND_ROWS = (0.55, 0.95)

# Per-frame motion sanity limits at ~12 fps.
VO_MAX_FORWARD_STEP_M = 3.0
VO_MAX_YAW_STEP_DEG = 8.0

# Extra area around the vehicle route, in meters. Both bottom
# panels use this window — raise it (e.g. 250) to zoom out and
# judge the OSM alignment against more road structure.
MAP_MARGIN_METERS = 25

# Background images are capped at this many pixels per side, so
# zooming far out cannot exhaust memory (resolution adapts).
MAX_BACKGROUND_PIXELS = 3500

# Ego-centric BEV panel range, in meters.
BEV_FORWARD_METERS = 45
BEV_LATERAL_METERS = 20

# How far to search for the closest lane, in meters.
LANE_SEARCH_RADIUS_METERS = 5.0

# Where JSON / Scenic outputs are written.
OUTPUT_DIR = Path("reconstruction_output")

# ------------------------------------------------------------
# Perception-noise simulation for the reconstructed map panel.
# Mimics monocular perception error so the real-vs-reconstructed
# comparison is meaningful before the learned models are plugged
# in. Noise is seeded per agent instance, so each agent keeps the
# SAME error for the whole clip (no jitter between frames).
# Set to False to render the reconstruction with perfect poses.
# ------------------------------------------------------------
SIMULATE_PERCEPTION_NOISE = False

# Draw faint dashed ground-truth outlines on the reconstructed
# map panel (only useful when perception noise is ON, to spot
# the shifts; set False for a clean look).
SHOW_GROUND_TRUTH_OVERLAY = False

# Overlay the nuScenes drivable-area boundary (dotted blue) on
# the reconstructed panel, showing exactly which pavement OSM
# does not know about. Redundant with an optimized (carved)
# map, so off by default. Vehicles outside the true drivable
# area still get a red edge and a count in the title.
SHOW_DRIVABLE_OUTLINE = False

# ------------------------------------------------------------
# OpenStreetMap background for the reconstructed panel.
# Roads and buildings are fetched from OSM (via osmnx), converted
# into the nuScenes global frame, rendered once to an image, and
# cached on disk — so only the FIRST run per scene needs internet.
# If osmnx is missing or the fetch fails, the panel falls back to
# the nuScenes map background automatically.
# ------------------------------------------------------------
USE_OSM_BACKGROUND = True

OSM_CACHE_DIR = Path("osm_cache")

# Optimized OSM maps produced by build_osm_map.py (aligned and
# refined against the nuScenes map). When a file for the current
# map location exists here, it is loaded directly — no fetching
# or aligning at runtime.
OPTIMIZED_MAP_DIR = Path("osm_optimized")

# Fallback visual road width when OSM has no usable tags. Roads
# with a "lanes" tag are drawn at lanes x OSM_LANE_WIDTH_METERS;
# otherwise a per-highway-type default is used (see
# estimate_road_width). This is what keeps wide carriageways from
# rendering as skinny bands that leave agents in the "white".
OSM_ROAD_WIDTH_METERS = 7.0
OSM_LANE_WIDTH_METERS = 3.5

# Fine alignment nudge, in meters. The nuScenes origin references
# and OSM geometry can disagree by a few meters; adjust these if
# the roads sit visibly offset from the driven route.
OSM_ALIGN_OFFSET_X = 0.0
OSM_ALIGN_OFFSET_Y = 0.0

# Automatically estimate the residual offset by snapping the OSM
# road network onto the driven route (translation-only search).
# The applied shift is printed — copy it into the manual offsets
# above if you want it fixed for repeated runs.
OSM_AUTO_ALIGN = True

# Geographic origin (latitude, longitude) of each nuScenes map's
# global (0, 0) corner, as published with the nuScenes devkit.
REFERENCE_COORDINATES = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}

EARTH_RADIUS_METERS = 6378137.0

DEPTH_NOISE_FRACTION = 0.05     # range error, ~5 % of distance
LATERAL_NOISE_METERS = 0.25     # sideways position error
YAW_NOISE_DEGREES = 4.0         # heading error
DROPOUT_START_METERS = 25.0     # beyond this, agents may be missed

# Colors per agent class for the BEV panel.
CLASS_COLORS = {
    "car": "#1f77b4",
    "truck": "#9467bd",
    "bus": "#8c564b",
    "motorcycle": "#e377c2",
    "bicycle": "#2ca02c",
    "pedestrian": "#d62728",
}

# Approximate ego vehicle footprint (Renault Zoe), meters.
EGO_LENGTH = 4.08
EGO_WIDTH = 1.73


# ============================================================
# SMALL HELPERS
# ============================================================


def quaternion_yaw(rotation):
    """Yaw angle (radians) of a nuScenes quaternion [w, x, y, z]."""
    return Quaternion(rotation).yaw_pitch_roll[0]


def wrap_angle(angle):
    """Wrap an angle to the range [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def simplify_category(category_name):
    """
    Map a nuScenes category name to a simple class label.
    Returns None for categories we do not reconstruct
    (traffic cones, debris, and so on).
    """
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    return None


def rotated_rectangle_corners(center_x, center_y, length, width, yaw):
    """
    Corner points of a rectangle centered at (center_x, center_y),
    rotated by yaw. Returns an array of shape (4, 2).
    """
    half_l = length / 2.0
    half_w = width / 2.0

    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    rotation = np.array(
        [
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ]
    )

    return corners @ rotation.T + np.array([center_x, center_y])


def ego_to_global(frame, x_forward, y_left):
    """
    Transform a point from the ego frame (x forward, y left)
    into global map coordinates using this frame's ego pose.
    """
    cos_yaw = math.cos(frame["yaw"])
    sin_yaw = math.sin(frame["yaw"])

    global_x = frame["x"] + x_forward * cos_yaw - y_left * sin_yaw
    global_y = frame["y"] + x_forward * sin_yaw + y_left * cos_yaw

    return global_x, global_y


def strip_private_keys(record):
    """
    Remove visualization-only keys (those starting with "_")
    before a record is written to the JSON scene description.
    """
    if record is None:
        return None

    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def point_on_drivable(drivable_mask, map_extent, x, y):
    """
    O(1) check whether a global point lies on the nuScenes
    drivable area, via the precomputed boolean mask.
    """
    height, width = drivable_mask.shape

    column = int(
        (x - map_extent[0])
        / (map_extent[1] - map_extent[0])
        * width
    )

    row = int(
        (y - map_extent[2])
        / (map_extent[3] - map_extent[2])
        * height
    )

    if 0 <= row < height and 0 <= column < width:
        return bool(drivable_mask[row, column])

    return False


def get_drivable_area_outlines(nusc_map, map_extent):
    """
    Boundary polylines of the nuScenes drivable area inside the
    view window. Overlaid on the OSM panel, they show exactly
    which pavement exists in ground truth but not in OSM.
    """
    try:
        patch = (
            map_extent[0],
            map_extent[2],
            map_extent[1],
            map_extent[3],
        )

        records = nusc_map.get_records_in_patch(
            patch,
            ["drivable_area"],
            mode="intersect",
        )

        outlines = []

        for token in records["drivable_area"]:
            record = nusc_map.get("drivable_area", token)

            for polygon_token in record["polygon_tokens"]:
                polygon = nusc_map.extract_polygon(polygon_token)

                rings = [polygon.exterior] + list(
                    polygon.interiors
                )

                for ring in rings:
                    coordinates = np.array(ring.coords)

                    if len(coordinates) >= 3:
                        outlines.append(coordinates)

        return outlines

    except Exception as error:
        print("Drivable-area outline unavailable:", error)
        return []


# ============================================================
# FRAME COLLECTION (extended from the playback script)
# ============================================================


def collect_camera_frames(nusc, scene):
    """
    Collect five seconds of camera frames with synchronized ego poses.
    Each frame now also keeps the tokens and the full ego pose needed
    by the reconstruction layers.
    """
    first_sample = nusc.get("sample", scene["first_sample_token"])
    camera_token = first_sample["data"][CAMERA_CHANNEL]

    first_camera_data = nusc.get("sample_data", camera_token)
    start_timestamp = first_camera_data["timestamp"]

    frames = []

    while camera_token:
        camera_data = nusc.get("sample_data", camera_token)

        elapsed_seconds = (
            camera_data["timestamp"] - start_timestamp
        ) / 1_000_000.0

        if elapsed_seconds > VIDEO_DURATION_SECONDS:
            break

        ego_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])

        image_path = DATAROOT / camera_data["filename"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Camera image not found:\n{image_path}"
            )

        frames.append(
            {
                "camera_token": camera_token,
                "sample_token": camera_data["sample_token"],
                "is_key_frame": camera_data["is_key_frame"],
                "image_path": image_path,
                "elapsed_seconds": elapsed_seconds,
                "x": ego_pose["translation"][0],
                "y": ego_pose["translation"][1],
                "yaw": quaternion_yaw(ego_pose["rotation"]),
                "ego_pose": ego_pose,
            }
        )

        camera_token = camera_data["next"]

    if not frames:
        raise RuntimeError("No camera frames were found.")

    return frames


# ============================================================
# LAYER 0 — EGO ESTIMATION FROM VIDEO
# ============================================================
# Everything in this section uses ONLY the camera images, the
# camera calibration, and a coarse GPS prior. No ground-truth
# poses, no HD-map queries.
# ============================================================


def get_camera_geometry(nusc, camera_token):
    """
    Camera calibration (a property of the SENSOR, not the
    scene): intrinsics, and the camera's mounting rotation and
    position in the ego frame.
    """
    sample_data = nusc.get("sample_data", camera_token)

    calibration = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"],
    )

    return {
        "K": np.array(calibration["camera_intrinsic"], dtype=float),
        "R_cam_to_ego": Quaternion(
            calibration["rotation"]
        ).rotation_matrix,
        "t_cam_in_ego": np.array(
            calibration["translation"], dtype=float
        ),
    }


def pixel_to_ego_ground(geometry, u, v, image_scale=1.0):
    """
    Cast a pixel ray onto the flat ground plane (z = 0 in the
    ego frame). Returns (x_forward, y_left) in meters, or None
    for pixels above the horizon.
    """
    K = geometry["K"] * image_scale
    K[2, 2] = 1.0

    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])

    direction = geometry["R_cam_to_ego"] @ ray_cam
    origin = geometry["t_cam_in_ego"]

    if direction[2] > -1e-6:
        return None

    scale = -origin[2] / direction[2]

    if scale <= 0 or scale > 120.0:
        return None

    point = origin + scale * direction

    return float(point[0]), float(point[1])


def estimate_motion_between_frames(
    previous_gray, current_gray, geometry
):
    """
    Ground-plane visual odometry for one frame pair:

        1. track road-surface features (KLT optical flow),
        2. back-project both endpoints onto the ground plane,
        3. fit a rigid 2D transform between the point sets.

    Returns (forward_m, left_m, dyaw_rad) or None when tracking
    is unreliable (the caller then coasts on the last motion).
    """
    height, width = previous_gray.shape

    # Only the road band — tracking building facades or moving
    # vehicles would corrupt the ground-plane assumption.
    row_start = int(VO_GROUND_ROWS[0] * height)
    row_end = int(VO_GROUND_ROWS[1] * height)

    mask = np.zeros_like(previous_gray)
    mask[row_start:row_end, :] = 255

    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=VO_MAX_FEATURES,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask,
    )

    if corners is None or len(corners) < 12:
        return None

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )

    previous_points = []
    current_points = []

    for corner, point, ok in zip(corners, tracked, status):
        if not ok:
            continue

        ground_prev = pixel_to_ego_ground(
            geometry, corner[0][0], corner[0][1], VO_IMAGE_SCALE
        )

        ground_cur = pixel_to_ego_ground(
            geometry, point[0][0], point[0][1], VO_IMAGE_SCALE
        )

        if ground_prev is None or ground_cur is None:
            continue

        # Keep near-field ground points (stable back-projection).
        if not (2.0 < ground_prev[0] < 30.0):
            continue

        previous_points.append(ground_prev)
        current_points.append(ground_cur)

    if len(previous_points) < 10:
        return None

    previous_points = np.array(previous_points, dtype=np.float32)
    current_points = np.array(current_points, dtype=np.float32)

    # Static world point seen from a moving vehicle:
    #   p_prev = R(dyaw) @ p_cur + motion
    # so the similarity transform current -> previous carries the
    # vehicle's yaw change and its motion in the previous frame.
    transform, inliers = cv2.estimateAffinePartial2D(
        current_points,
        previous_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.15,
        maxIters=2000,
    )

    if transform is None or inliers is None:
        return None

    if int(inliers.sum()) < 8:
        return None

    scale = float(np.hypot(transform[0, 0], transform[1, 0]))

    # A similarity fit on a rigid planar scene must come back
    # with scale ~1; anything else means bad correspondences.
    if not 0.9 < scale < 1.1:
        return None

    # Refit a PURE rigid transform (Kabsch) on the RANSAC
    # inliers — removes the small bias the similarity
    # parameterization introduces.
    inlier_mask = inliers.ravel().astype(bool)

    current_inliers = current_points[inlier_mask].astype(float)
    previous_inliers = previous_points[inlier_mask].astype(float)

    current_centroid = current_inliers.mean(axis=0)
    previous_centroid = previous_inliers.mean(axis=0)

    covariance = (
        (current_inliers - current_centroid).T
        @ (previous_inliers - previous_centroid)
    )

    u_matrix, _, vt_matrix = np.linalg.svd(covariance)

    rotation = vt_matrix.T @ u_matrix.T

    if np.linalg.det(rotation) < 0:
        vt_matrix[1, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T

    translation = (
        previous_centroid - rotation @ current_centroid
    )

    dyaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    forward = float(translation[0])
    left = float(translation[1])

    # Far-field yaw: distant features (upper image) move almost
    # purely with rotation — their horizontal flow divided by
    # the focal length is a direct, translation-free yaw-rate
    # measurement. Far more reliable than the ground-plane
    # estimate, and it works with or without lane markings.
    far_mask = np.zeros_like(previous_gray)
    far_mask[int(0.05 * height):int(0.45 * height), :] = 255

    far_corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=7,
        mask=far_mask,
    )

    if far_corners is not None and len(far_corners) >= 20:
        far_tracked, far_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            far_corners,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )

        horizontal_flow = np.array([
            point[0][0] - corner[0][0]
            for corner, point, ok in zip(
                far_corners, far_tracked, far_status
            )
            if ok
        ])

        if len(horizontal_flow) >= 25:
            median_flow = float(np.median(horizontal_flow))

            spread = float(np.median(
                np.abs(horizontal_flow - median_flow)
            ))

            inliers_far = horizontal_flow[
                np.abs(horizontal_flow - median_flow)
                < 3.0 * spread + 1.0
            ]

            if len(inliers_far) >= 25:
                focal = geometry["K"][0, 0] * VO_IMAGE_SCALE

                # Yawing left moves distant features RIGHT in
                # the image: dyaw = +du / f.
                dyaw_far = float(
                    np.median(inliers_far) / focal
                )

                if abs(
                    math.degrees(dyaw_far)
                ) <= VO_MAX_YAW_STEP_DEG:
                    dyaw = dyaw_far

    # Physical sanity at ~12 fps.
    if abs(forward) > VO_MAX_FORWARD_STEP_M:
        return None

    if abs(math.degrees(dyaw)) > VO_MAX_YAW_STEP_DEG:
        return None

    return forward, left, dyaw


def estimate_lane_offset(gray, geometry):
    """
    Lateral position within the lane, from the image alone:
    threshold bright lane markings, back-project them to the
    ground plane, histogram their lateral positions, and take
    the midpoint of the marking lines bracketing the vehicle.
    Returns (offset_m, lane_width_m) with positive = left of
    lane center, or None when markings are not found.
    """
    height, width = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Two complementary marking detectors: adaptive threshold
    # (paint brighter than local road) and morphological top-hat
    # (thin bright structures) — their union survives shadows
    # and worn paint better than either alone.
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=-12,
    )

    tophat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )

    markings = cv2.bitwise_or(
        adaptive,
        (tophat > 25).astype(np.uint8) * 255,
    )

    row_start = int(0.55 * height)
    rows, columns = np.nonzero(markings[row_start:, :])
    rows = rows + row_start

    if len(rows) < 30:
        return None

    # Subsample for speed.
    if len(rows) > 1500:
        stride = len(rows) // 1500 + 1
        rows = rows[::stride]
        columns = columns[::stride]

    lateral_positions = []
    ground_points = []

    for v, u in zip(rows, columns):
        ground = pixel_to_ego_ground(
            geometry, float(u), float(v), VO_IMAGE_SCALE
        )

        if ground is None:
            continue

        # A forward window where markings are usable.
        if 4.0 < ground[0] < 25.0 and abs(ground[1]) < 6.0:
            lateral_positions.append(ground[1])
            ground_points.append(ground)

    if len(lateral_positions) < 20:
        return None

    # Histogram of lateral positions -> marking lines are peaks.
    histogram, edges = np.histogram(
        lateral_positions,
        bins=np.arange(-6.0, 6.01, 0.2),
    )

    centers = (edges[:-1] + edges[1:]) / 2.0

    threshold = max(5, int(0.25 * histogram.max()))

    peaks = []

    for index in range(1, len(histogram) - 1):
        if (
            histogram[index] >= threshold
            and histogram[index] >= histogram[index - 1]
            and histogram[index] >= histogram[index + 1]
        ):
            if peaks and centers[index] - peaks[-1][0] < 1.5:
                # Merge close peaks, keep the stronger one.
                if histogram[index] > peaks[-1][1]:
                    peaks[-1] = (centers[index], histogram[index])
                continue

            peaks.append((centers[index], histogram[index]))

    left_candidates = [p for p, _ in peaks if p > 0.3]
    right_candidates = [p for p, _ in peaks if p < -0.3]

    if not left_candidates and not right_candidates:
        return None

    left_line = min(left_candidates) if left_candidates else None
    right_line = max(right_candidates) if right_candidates else None

    # Offset needs BOTH bracketing lines and a plausible width;
    # the heading measurement below works from a single line.
    lateral_offset = None
    lane_width = None

    if left_line is not None and right_line is not None:
        lane_width = left_line - right_line

        if 2.2 < lane_width < 5.5:
            lane_center = (left_line + right_line) / 2.0

            # Ego sits at y = 0; if the lane center is at +0.4
            # (left), the ego is 0.4 m RIGHT of center.
            lateral_offset = -lane_center
        else:
            lane_width = None

    # Direction of the marking line(s) (PCA of each line's
    # points) -> the ego's yaw RELATIVE to the lane. One line
    # is enough — its direction IS the lane direction.
    ground_points = np.asarray(ground_points)

    line_angles = []

    for line_position in (left_line, right_line):
        if line_position is None:
            continue

        line_points = ground_points[
            np.abs(ground_points[:, 1] - line_position) < 0.35
        ]

        if len(line_points) < 8:
            continue

        mean_point = line_points.mean(axis=0)

        centered = line_points - mean_point

        _, singular_values, principal = np.linalg.svd(
            centered, full_matrices=False
        )

        direction = principal[0]

        if direction[0] < 0:
            direction = -direction

        angle = math.atan2(direction[1], direction[0])

        # ---- line-paint filters: real lane markings are LONG,
        # ---- THIN stripes running along the lane. Painted
        # ---- arrows, symbols, and text are wide blobs or
        # ---- angled strokes — reject them.
        along = centered @ direction
        across = centered @ principal[1]

        length = float(along.max() - along.min())

        thinness = float(np.std(across))

        elongation = float(
            singular_values[0]
            / max(singular_values[1], 1e-6)
        )

        if length < 3.5:
            continue

        if thinness > 0.22:
            continue

        if elongation < 6.0:
            continue

        if abs(math.degrees(angle)) > 20.0:
            continue

        line_angles.append(angle)

    yaw_relative = None

    if line_angles:
        # The lane tilting left in the ego frame means the ego
        # is yawed RIGHT of the lane direction.
        yaw_relative = -float(np.mean(line_angles))

        if abs(math.degrees(yaw_relative)) > 15.0:
            yaw_relative = None

    if lateral_offset is None and yaw_relative is None:
        return None

    return lateral_offset, lane_width, yaw_relative


def estimate_local_trajectory(frames, geometry):
    """
    Integrate frame-to-frame ground-plane odometry into a local
    trajectory, expressed in the FIRST frame's ego frame:
    x forward, y left, yaw counterclockwise. Also estimates the
    lane offset per frame. Uses images only.
    """
    trajectory = [(0.0, 0.0, 0.0)]
    lane_offsets = [None] * len(frames)
    lane_yaws = [None] * len(frames)
    motion_steps = []

    previous_gray = None
    last_motion = (0.0, 0.0, 0.0)
    coasted = 0

    for index, frame in enumerate(frames):
        image = cv2.imread(str(frame["image_path"]))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=VO_IMAGE_SCALE,
            fy=VO_IMAGE_SCALE,
            interpolation=cv2.INTER_AREA,
        )

        lane_estimate = estimate_lane_offset(gray, geometry)

        if lane_estimate is not None:
            if lane_estimate[0] is not None:
                lane_offsets[index] = round(lane_estimate[0], 2)

            lane_yaws[index] = lane_estimate[2]

        if previous_gray is not None:
            motion = estimate_motion_between_frames(
                previous_gray, gray, geometry
            )

            if motion is None:
                # Coast: reuse the last good motion briefly.
                motion = last_motion
                coasted += 1
            else:
                last_motion = motion

            motion_steps.append(motion)

            x, y, yaw = trajectory[-1]
            forward, left, dyaw = motion

            x += forward * math.cos(yaw) - left * math.sin(yaw)
            y += forward * math.sin(yaw) + left * math.cos(yaw)
            yaw = wrap_angle(yaw + dyaw)

            trajectory.append((x, y, yaw))

        previous_gray = gray

    path_length = sum(
        math.hypot(
            trajectory[i + 1][0] - trajectory[i][0],
            trajectory[i + 1][1] - trajectory[i][1],
        )
        for i in range(len(trajectory) - 1)
    )

    print(
        f"Visual odometry: {len(trajectory)} poses, "
        f"path {path_length:.1f} m, coasted {coasted} frames; "
        "lane offset on "
        f"{sum(1 for o in lane_offsets if o is not None)}, "
        "lane heading on "
        f"{sum(1 for o in lane_yaws if o is not None)} "
        f"of {len(frames)} frames."
    )

    return trajectory, lane_offsets, lane_yaws, motion_steps


def anchor_trajectory_to_map(
    local_trajectory, road_points, prior_xy
):
    """
    Place the local VO trajectory on the map: search over start
    position (within ANCHOR_SEARCH_RADIUS of the GPS prior) and
    heading (full circle) for the rigid transform that best lays
    the trajectory onto the road network. Same clipped-mean
    scoring as the OSM alignment.

    Returns (start_x, start_y, heading_rad, fit_m).
    """
    trajectory = np.array(
        [(p[0], p[1]) for p in local_trajectory]
    )

    if len(trajectory) > 25:
        trajectory = trajectory[:: len(trajectory) // 25 + 1]

    roads = np.asarray(road_points, dtype=float)

    # Only roads near the prior matter.
    near = (
        np.hypot(*(roads - np.array(prior_xy)).T)
        < ANCHOR_SEARCH_RADIUS_METERS + 120.0
    )

    if near.any():
        roads = roads[near]

    if len(roads) > 1500:
        roads = roads[:: len(roads) // 1500 + 1]

    def score(start_x, start_y, heading):
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        rotated_x = (
            trajectory[:, 0] * cos_h - trajectory[:, 1] * sin_h
        ) + start_x

        rotated_y = (
            trajectory[:, 0] * sin_h + trajectory[:, 1] * cos_h
        ) + start_y

        points = np.column_stack([rotated_x, rotated_y])

        deltas = points[:, None, :] - roads[None, :, :]

        nearest = np.sqrt(
            (deltas ** 2).sum(axis=2)
        ).min(axis=1)

        return float(np.mean(np.minimum(nearest, 10.0)))

    best = (prior_xy[0], prior_xy[1], 0.0)
    best_score = float("inf")

    # Coarse: full heading circle, position grid around prior.
    radius = ANCHOR_SEARCH_RADIUS_METERS

    for heading_deg in range(0, 360, 10):
        heading = math.radians(heading_deg)

        for dx in np.arange(-radius, radius + 2.0, 6.0):
            for dy in np.arange(-radius, radius + 2.0, 6.0):
                value = score(
                    prior_xy[0] + dx, prior_xy[1] + dy, heading
                )

                if value < best_score:
                    best_score = value
                    best = (
                        prior_xy[0] + dx,
                        prior_xy[1] + dy,
                        heading,
                    )

    # Fine refinement around the coarse winner.
    center = best

    for heading_deg in np.arange(-10.0, 10.5, 2.0):
        heading = wrap_angle(
            center[2] + math.radians(heading_deg)
        )

        for dx in np.arange(-5.0, 5.5, 1.0):
            for dy in np.arange(-5.0, 5.5, 1.0):
                value = score(
                    center[0] + dx, center[1] + dy, heading
                )

                if value < best_score:
                    best_score = value
                    best = (
                        center[0] + dx,
                        center[1] + dy,
                        heading,
                    )

    return best[0], best[1], best[2], best_score


def refine_trajectory_on_map(
    poses, motion_steps, lane_yaws, lane_offsets, network, prior_xy
):
    """
    Map-aided trajectory refinement (standard practice, no
    shortcuts): jointly optimize every per-frame pose with
    least squares over four physically honest constraints —

        odometry   each consecutive pose pair must reproduce
                   the measured VO step (the trajectory's shape
                   is evidence);
        on-road    a HINGE penalty on the distance beyond the
                   road's half-width — exactly zero anywhere ON
                   the road, so there is no pull toward the
                   centerline, only the plain fact that the
                   vehicle drives on pavement;
        heading    where the video measured the ego's yaw
                   relative to the lane markings, the global
                   yaw must equal the local road tangent plus
                   that relative yaw;
        lateral    consecutive lane-offset measurements say how
                   far the vehicle moved SIDEWAYS within its
                   lane between frames — their differences pin
                   the lateral motion, which is exactly the
                   component monocular VO drifts in;

    plus a loose GPS prior on the start. This is what removes
    the residual VO drift a single rigid anchor cannot.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("scipy not available — skipping refinement.")
        return poses

    road_points = network["points"]
    road_directions = network["directions"]
    half_widths = network["half_widths"]

    # Only the roads near the trajectory matter.
    center = np.mean([p[:2] for p in poses], axis=0)

    near = np.hypot(*(road_points - center).T) < 200.0

    if near.any():
        road_points = road_points[near]
        road_directions = road_directions[near]
        half_widths = half_widths[near]

    if len(road_points) > 800:
        stride = len(road_points) // 800 + 1
        road_points = road_points[::stride]
        road_directions = road_directions[::stride]
        half_widths = half_widths[::stride]

    motion = np.asarray(motion_steps, dtype=float)
    initial = np.asarray(poses, dtype=float).ravel()
    count = len(poses)

    # Noise levels per channel, matching the VO accuracy we
    # measured on synthetic data: forward is near-exact, while
    # lateral and yaw carry the coupled monocular uncertainty.
    sigma_forward = 0.08       # m
    sigma_left = 0.4           # m (weak — kinematics knows better)
    sigma_yaw_step = math.radians(1.0)
    sigma_road = 0.4           # m, off-road hinge
    sigma_heading = math.radians(2.5)
    sigma_lateral = 0.15       # m, lane-offset difference noise
    sigma_anchor = 3.0         # m, anchoring's own uncertainty
    sigma_yaw_rate = math.radians(0.6)  # steering smoothness

    # Lane-LEVEL network (centerline per lane, e.g. the lane
    # graph): the measured lane offset is then an ABSOLUTE
    # lateral measurement, not just a differential one.
    lane_level_map = bool(np.median(half_widths) < 2.6)

    lane_indices = [
        k for k, value in enumerate(lane_yaws)
        if value is not None
    ]

    offset_indices = [
        k for k, value in enumerate(lane_offsets)
        if value is not None
    ]

    # Chain lateral constraints between CONSECUTIVE MEASURED
    # frames — this pins the cumulative sideways motion, so
    # VO's lateral bias cannot accumulate through frames where
    # markings were missed, even across several seconds. The
    # correctness guard is the plausibility gate, not the gap
    # length: a small measured net change IS the evidence that
    # the vehicle stayed in its lane across the gap, while a
    # lane change (~3.5 m) or the detector latching onto the
    # neighboring lane's markings fails the gate and the pair
    # is simply dropped.
    lateral_pairs = [
        (offset_indices[i], offset_indices[i + 1])
        for i in range(len(offset_indices) - 1)
        if offset_indices[i + 1] - offset_indices[i] <= 40
        and abs(
            lane_offsets[offset_indices[i + 1]]
            - lane_offsets[offset_indices[i]]
        ) < 1.5
    ]

    def residuals(state):
        p = state.reshape(count, 3)

        cos_yaw = np.cos(p[:-1, 2])
        sin_yaw = np.sin(p[:-1, 2])

        dx = p[1:, 0] - p[:-1, 0]
        dy = p[1:, 1] - p[:-1, 1]

        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy

        dyaw = np.arctan2(
            np.sin(p[1:, 2] - p[:-1, 2]),
            np.cos(p[1:, 2] - p[:-1, 2]),
        )

        odometry_residuals = np.concatenate([
            (forward - motion[:, 0]) / sigma_forward,
            (left - motion[:, 1]) / sigma_left,
            (dyaw - motion[:, 2]) / sigma_yaw_step,
        ])

        # Nonholonomic prior: a car cannot slide sideways. The
        # lateral component of each pose increment must be near
        # zero, with an allowance proportional to how much the
        # vehicle is turning during the step (Ackermann motion
        # sweeps sideways while yawing).
        lateral_allowance = 0.05 + 0.5 * np.abs(dyaw) * np.maximum(
            forward, 0.0
        )

        nonholonomic_residuals = left / lateral_allowance

        # Steering smoothness: yaw rate cannot jump between
        # consecutive 80 ms steps — the steering wheel moves
        # continuously.
        yaw_smooth_residuals = (
            dyaw[1:] - dyaw[:-1]
        ) / sigma_yaw_rate

        deltas = p[:, None, :2] - road_points[None, :, :]
        distances = np.sqrt((deltas ** 2).sum(axis=2))

        nearest_index = distances.argmin(axis=1)
        nearest_distance = distances.min(axis=1)

        off_road = np.maximum(
            0.0,
            nearest_distance - half_widths[nearest_index],
        ) / sigma_road

        # Signed lateral coordinate relative to the local road
        # direction: positive = left of the nearest road point.
        tangents = road_directions[nearest_index]

        offsets_to_road = (
            p[:, :2] - road_points[nearest_index]
        )

        signed_lateral = (
            -tangents[:, 1] * offsets_to_road[:, 0]
            + tangents[:, 0] * offsets_to_road[:, 1]
        )

        heading_residuals = []

        for k in lane_indices:
            tangent = road_directions[nearest_index[k]]

            tangent_yaw = math.atan2(tangent[1], tangent[0])

            # A polyline's direction is sign-ambiguous relative
            # to travel; align it with the current heading.
            if math.cos(p[k, 2] - tangent_yaw) < 0:
                tangent_yaw = wrap_angle(tangent_yaw + math.pi)

            heading_residuals.append(
                wrap_angle(
                    p[k, 2] - (tangent_yaw + lane_yaws[k])
                ) / sigma_heading
            )

        # Lateral motion within the lane, as the video measured
        # it: the change in signed lateral position must match
        # the change in the measured lane offset.
        lateral_residuals = []

        for k1, k2 in lateral_pairs:
            tangent = road_directions[nearest_index[k1]]

            # Sign of "left" flips with travel direction along
            # the polyline; align like the heading residual.
            sign = 1.0

            if math.cos(
                p[k1, 2] - math.atan2(tangent[1], tangent[0])
            ) < 0:
                sign = -1.0

            measured_change = (
                lane_offsets[k2] - lane_offsets[k1]
            )

            lateral_residuals.append(
                (
                    sign
                    * (signed_lateral[k2] - signed_lateral[k1])
                    - measured_change
                ) / sigma_lateral
            )

        # On a lane-level map, the offset itself is absolute:
        # signed distance to the nearest lane centerline must
        # equal the measured offset from the lane center.
        if lane_level_map:
            for k in offset_indices:
                tangent = road_directions[nearest_index[k]]

                sign = 1.0

                if math.cos(
                    p[k, 2]
                    - math.atan2(tangent[1], tangent[0])
                ) < 0:
                    sign = -1.0

                lateral_residuals.append(
                    (
                        sign * signed_lateral[k]
                        - lane_offsets[k]
                    ) / 0.3
                )

        # The GLOBAL position information was already consumed
        # by the anchoring stage — re-applying the raw GPS
        # prior here would drag the path through the otherwise
        # unobservable directions. Anchor loosely to the
        # anchored start instead.
        prior_residuals = [
            (p[0, 0] - prior_xy[0]) / sigma_anchor,
            (p[0, 1] - prior_xy[1]) / sigma_anchor,
        ]

        return np.concatenate([
            odometry_residuals,
            nonholonomic_residuals,
            yaw_smooth_residuals,
            off_road,
            np.asarray(heading_residuals),
            np.asarray(lateral_residuals),
            np.asarray(prior_residuals),
        ])

    start_time = time.time()

    initial_cost = float(
        np.sum(residuals(initial) ** 2)
    )

    solution = least_squares(
        residuals,
        initial,
        method="trf",
        xtol=1e-6,
        ftol=1e-6,
        max_nfev=6000,
    )

    print(
        f"Refinement: cost {initial_cost:.0f} -> "
        f"{2.0 * solution.cost:.0f} "
        f"in {time.time() - start_time:.1f} s."
    )

    refined = solution.x.reshape(count, 3)

    return [
        (float(x), float(y), float(wrap_angle(yaw)))
        for x, y, yaw in refined
    ]


def apply_ego_estimation(nusc, frames, road_network):
    """
    Run the full Layer 0 pipeline and write the estimated pose
    into every frame as x_est / y_est / yaw_est (plus the lane
    offset estimate). Prints the honest error against ground
    truth. Ground-truth fields are left untouched — the real-map
    panel keeps using them.
    """
    if cv2 is None:
        print("Ego estimation needs opencv (cv2); skipping.")
        return False

    geometry = get_camera_geometry(
        nusc, frames[0]["camera_token"]
    )

    (
        local_trajectory,
        lane_offsets,
        lane_yaws,
        motion_steps,
    ) = estimate_local_trajectory(frames, geometry)

    # GPS-like prior: true start + seeded offset.
    rng = np.random.default_rng(GPS_PRIOR_SEED)

    prior_angle = rng.uniform(0.0, 2.0 * math.pi)

    prior_xy = (
        frames[0]["x"]
        + GPS_PRIOR_NOISE_METERS * math.cos(prior_angle),
        frames[0]["y"]
        + GPS_PRIOR_NOISE_METERS * math.sin(prior_angle),
    )

    print(
        f"GPS prior: ({prior_xy[0]:.1f}, {prior_xy[1]:.1f}), "
        f"{GPS_PRIOR_NOISE_METERS:.0f} m from the true start."
    )

    start_x, start_y, heading, fit = anchor_trajectory_to_map(
        local_trajectory, road_network["points"], prior_xy
    )

    print(
        f"Anchoring: heading {math.degrees(heading):.0f} deg, "
        f"road fit {fit:.2f} m."
    )

    cos_h, sin_h = math.cos(heading), math.sin(heading)

    anchored_poses = [
        (
            start_x + lx * cos_h - ly * sin_h,
            start_y + lx * sin_h + ly * cos_h,
            wrap_angle(heading + lyaw),
        )
        for lx, ly, lyaw in local_trajectory
    ]

    def mean_error(poses):
        return float(np.mean([
            math.hypot(pose[0] - frame["x"], pose[1] - frame["y"])
            for pose, frame in zip(poses, frames)
        ]))

    anchored_error = mean_error(anchored_poses)

    # Map-aided refinement: odometry + on-road hinge + lane
    # heading, solved jointly over all frames.
    refined_poses = refine_trajectory_on_map(
        anchored_poses,
        motion_steps,
        lane_yaws,
        lane_offsets,
        road_network,
        (anchored_poses[0][0], anchored_poses[0][1]),
    )

    refined_error = mean_error(refined_poses)

    for frame, pose, lane_offset in zip(
        frames, refined_poses, lane_offsets
    ):
        frame["x_est"] = pose[0]
        frame["y_est"] = pose[1]
        frame["yaw_est"] = pose[2]
        frame["lane_offset_est"] = lane_offset

    max_error = max(
        math.hypot(
            frame["x_est"] - frame["x"],
            frame["y_est"] - frame["y"],
        )
        for frame in frames
    )

    print(
        "Ego estimation error vs ground truth: "
        f"rigid anchor {anchored_error:.1f} m -> "
        f"refined {refined_error:.1f} m mean "
        f"(max {max_error:.1f} m)."
    )

    return True


def get_anchor_road_network(map_name, nusc_map):
    """
    Road network for anchoring and refinement: points, unit
    tangent directions, and half-widths. The optimized OSM map
    is preferred (then the whole placement chain is video + OSM
    only); the nuScenes lane graph is the fallback.
    """
    optimized_path = (
        OPTIMIZED_MAP_DIR / f"osm_optimized_{map_name}.pkl"
    )

    points = []
    directions = []
    half_widths = []

    if optimized_path.exists():
        try:
            payload = pickle.loads(optimized_path.read_bytes())

            for road in payload["roads"]:
                road_points = np.asarray(road["points"])

                if len(road_points) < 2:
                    continue

                stride = max(1, len(road_points) // 20)
                sampled = road_points[::stride]

                tangents = np.gradient(sampled, axis=0)

                norms = np.hypot(*tangents.T)
                norms[norms < 1e-9] = 1.0
                tangents = tangents / norms[:, None]

                points.append(sampled)
                directions.append(tangents)
                half_widths.append(
                    np.full(len(sampled), road["width_m"] / 2.0)
                )

            network = {
                "points": np.vstack(points),
                "directions": np.vstack(directions),
                "half_widths": np.concatenate(half_widths),
            }

            print(
                "Road network from optimized OSM "
                f"({len(network['points'])} points)."
            )

            return network

        except Exception as error:
            print(
                "Optimized map unreadable for anchoring:", error
            )

    lane_points = sample_lane_reference_points(
        nusc_map, max_points=3000
    )

    tangents = np.gradient(lane_points, axis=0)
    norms = np.hypot(*tangents.T)
    norms[norms < 1e-9] = 1.0

    network = {
        "points": lane_points,
        "directions": tangents / norms[:, None],
        # Lane centerlines: a lane's half-width.
        "half_widths": np.full(len(lane_points), 2.0),
    }

    print(
        "Road network from nuScenes lane graph "
        f"({len(lane_points)} points)."
    )

    return network


# ============================================================
# LANE MODEL — lanes derived from the optimized OSM roads
# ============================================================
# Each road is divided into parallel lanes (road width divided
# by a nominal lane width), giving per-lane centerlines. All
# vehicle placement below is expressed in this lane structure.
# ============================================================

NOMINAL_LANE_WIDTH = 3.5


def generate_lanes_from_roads(roads, keep_point=None):
    """
    Divide every road into lanes: lane_count = width / 3.5 m
    (rounded, at least 1), each lane a centerline polyline
    offset from the road centerline.

    keep_point(x, y) -> bool, when given, CLIPS lanes to the
    actual road surface: lane points off the carved pavement
    (junction interiors, medians, beyond road ends) are dropped
    and the lane splits into separate segments. This is what
    keeps lanes from crisscrossing junctions.
    """
    lanes = []

    for road_index, road in enumerate(roads):
        points = np.asarray(road["points"], dtype=float)

        if len(points) < 2:
            continue

        width = float(road.get("width_m", NOMINAL_LANE_WIDTH))

        lane_count = max(
            1, int(round(width / NOMINAL_LANE_WIDTH))
        )

        lane_width = width / lane_count

        tangents = np.gradient(points, axis=0)
        norms = np.hypot(*tangents.T)
        norms[norms < 1e-9] = 1.0
        tangents = tangents / norms[:, None]

        left_normals = np.column_stack(
            [-tangents[:, 1], tangents[:, 0]]
        )

        for lane_index in range(lane_count):
            offset = (
                lane_index - (lane_count - 1) / 2.0
            ) * lane_width

            lane_points = points + offset * left_normals

            if keep_point is None:
                segments = [lane_points]
            else:
                keep = np.array([
                    bool(keep_point(p[0], p[1]))
                    for p in lane_points
                ])

                segments = []
                run_start = None

                for i, kept in enumerate(keep):
                    if kept and run_start is None:
                        run_start = i
                    elif not kept and run_start is not None:
                        segments.append(
                            lane_points[run_start:i]
                        )
                        run_start = None

                if run_start is not None:
                    segments.append(lane_points[run_start:])

            for segment in segments:
                # Segments shorter than ~8 m are junction
                # slivers — not usable lanes.
                if len(segment) < 2:
                    continue

                length = float(np.sum(np.hypot(
                    *np.diff(segment, axis=0).T
                )))

                if length < 8.0:
                    continue

                lanes.append({
                    "lane_id": len(lanes),
                    "road_index": road_index,
                    "lane_index": lane_index,
                    "lane_width": round(lane_width, 2),
                    "points": segment,
                })

    return lanes


class LaneMap:
    """
    Queryable lane structure: which lane a (position, heading)
    belongs to, the signed lateral offset within it, the
    longitudinal arclength, and interpolation back from lane
    coordinates to map coordinates. A lane serves both travel
    directions; "direction" (+1 / -1) records which way along
    the polyline the vehicle travels, and lateral sign is
    always LEFT-positive in the travel direction.
    """

    def __init__(self, lanes):
        self.lanes = {}

        all_points = []
        all_tangents = []
        point_lane_ids = []

        for lane in lanes:
            points = np.asarray(lane["points"], dtype=float)

            tangents = np.gradient(points, axis=0)
            norms = np.hypot(*tangents.T)
            norms[norms < 1e-9] = 1.0
            tangents = tangents / norms[:, None]

            steps = np.hypot(*np.diff(points, axis=0).T)

            arclength = np.concatenate(
                [[0.0], np.cumsum(steps)]
            )

            self.lanes[lane["lane_id"]] = {
                **lane,
                "points": points,
                "tangents": tangents,
                "arclength": arclength,
            }

            all_points.append(points)
            all_tangents.append(tangents)
            point_lane_ids.extend(
                [lane["lane_id"]] * len(points)
            )

        self.all_points = np.vstack(all_points)
        self.all_tangents = np.vstack(all_tangents)
        self.point_lane_ids = np.asarray(point_lane_ids)

    def locate(self, x, y, heading, max_distance=6.0):
        """
        Lane assignment for a pose. Returns a dictionary or
        None when the position is not on the lane network
        (junction interiors, parking areas).
        """
        deltas = self.all_points - np.array([x, y])
        distances = np.hypot(deltas[:, 0], deltas[:, 1])

        # Only lane points whose direction is compatible with
        # the heading (in either polyline sense).
        tangent_yaws = np.arctan2(
            self.all_tangents[:, 1], self.all_tangents[:, 0]
        )

        alignment = np.cos(heading - tangent_yaws)

        compatible = np.abs(alignment) > 0.5

        if not compatible.any():
            return None

        candidate_distances = np.where(
            compatible, distances, np.inf
        )

        best = int(np.argmin(candidate_distances))

        if candidate_distances[best] > max_distance:
            return None

        lane_id = int(self.point_lane_ids[best])
        direction = 1 if alignment[best] > 0 else -1

        projection = self.project_to_lane(
            lane_id, direction, x, y
        )

        # Beyond the segment's end: the vehicle is in a gap
        # (junction), not in this lane.
        if projection["overshoot"] > 2.5:
            return None

        return projection

    def project_to_lane(self, lane_id, direction, x, y):
        """Lane-frame coordinates of a point in a GIVEN lane."""
        lane = self.lanes[lane_id]

        deltas = lane["points"] - np.array([x, y])
        distances = np.hypot(deltas[:, 0], deltas[:, 1])

        index = int(np.argmin(distances))

        tangent = lane["tangents"][index] * direction

        left_normal = np.array([-tangent[1], tangent[0]])

        offset = (
            np.array([x, y]) - lane["points"][index]
        )

        # Longitudinal overshoot beyond the segment ends: the
        # projection clamps there, so a large overshoot means
        # the vehicle is NOT actually beside this segment.
        # Longitudinal overshoot beyond the segment ends, in the
        # polyline's STORED orientation (endpoints are endpoints
        # in both travel directions).
        along_stored = float(lane["tangents"][index] @ offset)

        overshoot = 0.0

        if index == 0 and along_stored < 0:
            overshoot = -along_stored
        elif (
            index == len(lane["points"]) - 1
            and along_stored > 0
        ):
            overshoot = along_stored

        return {
            "lane_id": lane_id,
            "direction": direction,
            "lane_index": lane["lane_index"],
            "lane_width": lane["lane_width"],
            "lateral": float(left_normal @ offset),
            "s": float(lane["arclength"][index]),
            "overshoot": overshoot,
            "tangent_yaw": float(
                math.atan2(tangent[1], tangent[0])
            ),
        }

    def lane_pose(self, lane_id, direction, s, lateral):
        """Map position + travel yaw for lane coordinates."""
        lane = self.lanes[lane_id]

        arclength = lane["arclength"]

        s = float(np.clip(s, 0.0, arclength[-1]))

        index = int(np.searchsorted(arclength, s))
        index = min(max(index, 1), len(arclength) - 1)

        span = arclength[index] - arclength[index - 1]
        fraction = 0.0 if span < 1e-9 else (
            (s - arclength[index - 1]) / span
        )

        point = (
            lane["points"][index - 1] * (1 - fraction)
            + lane["points"][index] * fraction
        )

        tangent = (
            lane["tangents"][index - 1] * (1 - fraction)
            + lane["tangents"][index] * fraction
        ) * direction

        norm = float(np.hypot(*tangent))
        tangent = tangent / (norm if norm > 1e-9 else 1.0)

        left_normal = np.array([-tangent[1], tangent[0]])

        position = point + lateral * left_normal

        return (
            float(position[0]),
            float(position[1]),
            float(math.atan2(tangent[1], tangent[0])),
        )


def load_lane_map(map_name, road_network):
    """
    Lane model for the current map: from the optimized OSM map
    when it carries lanes (or its roads), otherwise from the
    anchoring network's polylines (one lane each, 3.5 m).
    """
    optimized_path = (
        OPTIMIZED_MAP_DIR / f"osm_optimized_{map_name}.pkl"
    )

    lanes = None

    if optimized_path.exists():
        try:
            payload = pickle.loads(optimized_path.read_bytes())

            keep_point = None

            if "surface" in payload:
                surface = payload["surface"]
                ppm = payload.get("surface_ppm", 3)

                def keep_point(x, y):
                    row = int(y * ppm)
                    column = int(x * ppm)

                    return (
                        0 <= row < surface.shape[0]
                        and 0 <= column < surface.shape[1]
                        and surface[row, column] == 2
                    )

            if "lanes" in payload:
                lanes = payload["lanes"]
            else:
                lanes = generate_lanes_from_roads(
                    payload["roads"], keep_point=keep_point
                )

        except Exception as error:
            print("Optimized map unreadable for lanes:", error)

    if lanes is None:
        pseudo_roads = [
            {"points": road_network["points"], "width_m": 3.5}
        ]

        lanes = generate_lanes_from_roads(pseudo_roads)

    print(f"Lane model: {len(lanes)} lanes.")

    return LaneMap(lanes)


# ============================================================
# VEHICLE TRACKER — persistent identities, lane-level states
# ============================================================
# Consumes per-frame DETECTIONS (class, ego-frame position,
# size — from the perception stand-in today, YOLO+depth or
# FCOS3D later) and maintains its OWN tracks: association by
# predicted position, alpha-beta filtering for speed, lane
# assignment, and a lane-change state machine whose rendered
# motion is rate-limited — vehicles sit near their lane center
# in normal driving and slide smoothly across the boundary
# during a change; they never jump.
# ============================================================

# Rendered lateral motion limits (per frame at ~12 fps).
DISPLAY_LATERAL_STEP_M = 0.18
# Soft snap: in-lane offsets are compressed toward the center.
LANE_KEEP_MAX_OFFSET_FRACTION = 0.3

LANE_CHANGE_STAGES = [
    (0.15, "in lane center"),
    (0.35, "moving toward boundary"),
    (0.60, "over the lane line"),
    (0.85, "entering target lane"),
    (10.0, "settling in target lane"),
]


class VehicleTracker:

    def __init__(self, lane_map, fps):
        self.lane_map = lane_map
        self.dt = 1.0 / fps
        self.tracks = {}
        self.next_id = 1

    # ---- association ----------------------------------------

    def update(self, ego_pose, detections):
        """
        One tracker step. ego_pose = (x, y, yaw) of the ego on
        the map; detections = ego-frame vehicle detections.
        Returns the list of vehicle state dictionaries.
        """
        ego_x, ego_y, ego_yaw = ego_pose

        cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)

        measurements = []

        for detection in detections:
            global_x = (
                ego_x
                + detection["x"] * cos_yaw
                - detection["y"] * sin_yaw
            )

            global_y = (
                ego_y
                + detection["x"] * sin_yaw
                + detection["y"] * cos_yaw
            )

            measurements.append({
                "position": np.array([global_x, global_y]),
                "heading": wrap_angle(
                    ego_yaw + math.radians(detection["yaw_deg"])
                ),
                "ego_distance": math.hypot(
                    detection["x"], detection["y"]
                ),
                "detection": detection,
            })

        # Predict all tracks forward.
        for track in self.tracks.values():
            track["predicted"] = (
                track["position"] + track["velocity"] * self.dt
            )

        # Greedy nearest association within a speed-aware gate.
        unmatched = list(range(len(measurements)))
        pairs = []

        for track_id, track in self.tracks.items():
            best_index = None
            best_distance = None

            gate = 3.0 + track["speed"] * self.dt * 2.0

            for index in unmatched:
                measurement = measurements[index]

                if (
                    measurement["detection"]["class"]
                    != track["class"]
                ):
                    continue

                distance = float(np.hypot(
                    *(measurement["position"]
                      - track["predicted"])
                ))

                if distance > gate:
                    continue

                if best_distance is None or (
                    distance < best_distance
                ):
                    best_distance = distance
                    best_index = index

            if best_index is not None:
                pairs.append((track_id, best_index))
                unmatched.remove(best_index)

        matched_ids = set()

        for track_id, index in pairs:
            self._update_track(
                self.tracks[track_id], measurements[index]
            )
            matched_ids.add(track_id)

        # Coast unmatched tracks briefly, then retire them.
        for track_id in list(self.tracks.keys()):
            if track_id in matched_ids:
                continue

            track = self.tracks[track_id]
            track["missed"] += 1
            track["position"] = track["predicted"]

            track["ego_distance"] = float(math.hypot(
                track["position"][0] - ego_x,
                track["position"][1] - ego_y,
            ))

            if track["missed"] > 6:
                del self.tracks[track_id]
            else:
                self._update_lane_state(track)

        # Births for unmatched detections.
        for index in unmatched:
            self._create_track(measurements[index])

        return [
            self._track_state(track)
            for track in self.tracks.values()
        ]

    # ---- filtering ------------------------------------------

    def _create_track(self, measurement):
        detection = measurement["detection"]

        track = {
            "id": self.next_id,
            "class": detection["class"],
            "length": detection["length_m"],
            "width": detection["width_m"],
            "position": measurement["position"].copy(),
            "velocity": np.zeros(2),
            "speed": 0.0,
            "heading": measurement["heading"],
            "ego_distance": measurement["ego_distance"],
            "missed": 0,
            "age": 1,
            "lane": None,          # (lane_id, direction)
            "display_lat": 0.0,
            "display_s": None,
            "display_pose": None,
            "change": None,        # lane-change state
            "change_votes": 0,
            "status": "new track",
        }

        self.next_id += 1
        self.tracks[track["id"]] = track

        self._update_lane_state(track)

    def _update_track(self, track, measurement):
        residual = measurement["position"] - track["predicted"]

        # Alpha-beta filter: smooth position, derive velocity.
        track["position"] = track["predicted"] + 0.5 * residual
        track["velocity"] = (
            track["velocity"] + 0.4 * residual / self.dt
        )

        speed = float(np.hypot(*track["velocity"]))

        if speed > 30.0:
            track["velocity"] *= 30.0 / speed
            speed = 30.0

        # Speed from displacement over a ~0.7 s window: position
        # jitter divides down over the window, so parked
        # vehicles read ~0 while true motion reads accurately.
        # The alpha-beta velocity above is kept ONLY for
        # association prediction.
        history = track.setdefault("history", [])
        history.append(track["position"].copy())

        if len(history) > 9:
            history.pop(0)

        if len(history) >= 5:
            span = (len(history) - 1) * self.dt

            raw_speed = float(np.hypot(
                *(history[-1] - history[0])
            )) / span
        else:
            raw_speed = float(np.hypot(*track["velocity"]))

        if raw_speed < 1.0:
            raw_speed = 0.0

        track["speed"] = (
            0.7 * track["speed"] + 0.3 * min(raw_speed, 30.0)
        )

        # Heading comes from the MEASURED box orientation with a
        # circular low-pass. Velocity direction is deliberately
        # NOT used: detections are ego-relative, so ego-pose
        # error creates apparent velocity that would visibly
        # rotate parked vehicles.
        heading_delta = wrap_angle(
            measurement["heading"] - track["heading"]
        )

        track["heading"] = wrap_angle(
            track["heading"] + 0.3 * heading_delta
        )

        track["ego_distance"] = measurement["ego_distance"]
        track["missed"] = 0
        track["age"] += 1

        self._update_lane_state(track)

    # ---- lane logic -----------------------------------------

    def _update_lane_state(self, track):
        located = self.lane_map.locate(
            track["position"][0],
            track["position"][1],
            track["heading"],
        )

        if located is None:
            self._display_off_network(track)
            return

        raw_lane = (located["lane_id"], located["direction"])

        if track["lane"] is None:
            track["lane"] = raw_lane
            track["display_lat"] = self._soft_snap(
                located["lateral"], located["lane_width"]
            )

        # Lane frame relative to the CURRENT lane.
        current = self.lane_map.project_to_lane(
            track["lane"][0],
            track["lane"][1],
            track["position"][0],
            track["position"][1],
        )

        # Drove past the current segment's end (junction gap):
        # hand off to off-network mode — the display limiter
        # keeps the motion continuous, and re-entry into the
        # next segment is likewise continuous.
        if current["overshoot"] > 2.5:
            track["lane"] = None
            track["change"] = None
            self._display_off_network(track)
            return

        lane_width = current["lane_width"]

        # Segment handover: clipped lanes split at junctions, so
        # a vehicle driving straight crosses from one segment to
        # the next. If the raw lane is COLLINEAR with the current
        # one (nearly the same lateral), hand over seamlessly —
        # this is continuation, not a lane change.
        if (
            raw_lane != track["lane"]
            and track["change"] is None
            and abs(located["lateral"] - current["lateral"]) < 0.8
        ):
            track["lane"] = raw_lane
            current = located
            lane_width = current["lane_width"]

        # ---- lane-change state machine
        if track["change"] is None:
            crossing = (
                abs(current["lateral"]) > 0.35 * lane_width
                or raw_lane != track["lane"]
            )

            if crossing:
                track["change_votes"] = (
                    track.get("change_votes", 0) + 1
                )
            else:
                track["change_votes"] = 0

            # Require 3 consistent frames to declare a change.
            if track.get("change_votes", 0) >= 3:
                side = 1 if current["lateral"] > 0 else -1

                track["change"] = {
                    "side": side,
                    "gap": side * lane_width,
                    "switched": False,
                }

                track["change_votes"] = 0
        else:
            change = track["change"]

            if not change["switched"] and (
                current["lateral"] / change["gap"] > 0.5
            ):
                # Crossed the midline: re-express everything
                # relative to the TARGET lane — continuously.
                target = self.lane_map.locate(
                    track["position"][0],
                    track["position"][1],
                    track["heading"],
                )

                if target is not None and (
                    target["lane_id"],
                    target["direction"],
                ) != track["lane"]:
                    track["lane"] = (
                        target["lane_id"], target["direction"]
                    )
                    track["display_lat"] -= change["gap"]
                    current = target
                    change["gap"] = -change["gap"]
                    change["switched"] = True

            if abs(current["lateral"]) < 0.2 * lane_width:
                track["change"] = None

        # ---- status text
        if track["change"] is not None:
            change = track["change"]

            raw_progress = current["lateral"] / change["gap"]

            if change["switched"]:
                stage_progress = 1.0 - raw_progress
            else:
                stage_progress = raw_progress

            stage_progress = float(
                np.clip(stage_progress, 0.0, 1.0)
            )

            for threshold, label in LANE_CHANGE_STAGES:
                if stage_progress <= threshold:
                    track["status"] = f"lane change: {label}"
                    break
        elif abs(current["lateral"]) > 0.3 * lane_width:
            track["status"] = "drifting toward lane boundary"
        else:
            track["status"] = "lane keeping"

        # ---- rendered pose: rate-limited lane-frame motion
        if track["change"] is not None:
            target_lat = current["lateral"]
        else:
            target_lat = self._soft_snap(
                current["lateral"], lane_width
            )

        step = float(np.clip(
            target_lat - track["display_lat"],
            -DISPLAY_LATERAL_STEP_M,
            DISPLAY_LATERAL_STEP_M,
        ))

        track["display_lat"] += step

        track["display_s"] = current["s"]

        pose = self.lane_map.lane_pose(
            track["lane"][0],
            track["lane"][1],
            current["s"],
            track["display_lat"],
        )

        # Rendered heading: lane direction plus a bounded slice
        # of the measured deviation (visualizes the maneuver
        # without letting measurement noise rotate the box).
        deviation = wrap_angle(track["heading"] - pose[2])

        deviation = float(np.clip(
            deviation, -math.radians(12), math.radians(12)
        ))

        self._limit_display(track, (
            pose[0], pose[1], wrap_angle(pose[2] + deviation)
        ))

        track["lane_info"] = current

    def _display_off_network(self, track):
        """Junction interior / parking / segment gaps."""
        track["status"] = "off lane network"
        track["lane_info"] = None

        self._limit_display(track, (
            float(track["position"][0]),
            float(track["position"][1]),
            track["heading"],
        ))

    def _limit_display(self, track, target):
        """
        THE no-teleport invariant: every rendered pose, from
        every code path, moves by at most a speed-aware step
        per frame. Lane adoptions, segment handoffs, and gap
        re-entries all become continuous motion.
        """
        previous = track.get("display_pose")

        if previous is None:
            track["display_pose"] = target
            return

        step_cap = max(0.6, track["speed"] * self.dt * 1.6)

        step_x = float(np.clip(
            target[0] - previous[0], -step_cap, step_cap
        ))

        step_y = float(np.clip(
            target[1] - previous[1], -step_cap, step_cap
        ))

        step_yaw = float(np.clip(
            wrap_angle(target[2] - previous[2]),
            -math.radians(6), math.radians(6),
        ))

        track["display_pose"] = (
            previous[0] + step_x,
            previous[1] + step_y,
            wrap_angle(previous[2] + step_yaw),
        )

    @staticmethod
    def _soft_snap(lateral, lane_width):
        """In-lane rendering sits near the lane center."""
        limit = LANE_KEEP_MAX_OFFSET_FRACTION * lane_width

        return float(np.clip(0.5 * lateral, -limit, limit))

    # ---- output ---------------------------------------------

    def _track_state(self, track):
        lane_info = track.get("lane_info")

        return {
            "id": int(track["id"]),
            "class": track["class"],
            "position": [
                round(float(track["position"][0]), 2),
                round(float(track["position"][1]), 2),
            ],
            "speed_mps": round(track["speed"], 2),
            "heading_deg": round(
                math.degrees(track["heading"]), 1
            ),
            "lane_id": (
                None if track["lane"] is None
                else int(track["lane"][0])
            ),
            "lane_direction": (
                None if track["lane"] is None
                else int(track["lane"][1])
            ),
            "lane_lateral_m": (
                None if lane_info is None
                else round(lane_info["lateral"], 2)
            ),
            "lane_s_m": (
                None if lane_info is None
                else round(lane_info["s"], 1)
            ),
            "distance_from_ego_m": round(
                track["ego_distance"], 1
            ),
            "lane_change": track["status"],
            "coasting": track["missed"] > 0,
            "display_pose": [
                round(track["display_pose"][0], 2),
                round(track["display_pose"][1], 2),
                round(
                    math.degrees(track["display_pose"][2]), 1
                ),
            ],
            "length_m": track["length"],
            "width_m": track["width"],
        }


# ============================================================
# GLOBAL AGENTS FROM ANNOTATIONS + LANE STATES
# ============================================================
# Objects on the reconstructed map are laid down directly from
# nusc.get_boxes() at their GLOBAL annotated poses, with speeds
# from box_velocity() — placement is independent of ego-pose
# estimation error. The lane model is used only to compute each
# vehicle's lane assignment and lane-change status.
# ============================================================


def get_agents_global(nusc, frame, max_range=60.0):
    """
    Annotated boxes in GLOBAL map coordinates for one frame:
    position, yaw, size, class, speed (box_velocity), and the
    persistent instance identity.
    """
    boxes = nusc.get_boxes(frame["camera_token"])

    agents = []

    for box in boxes:
        agent_class = simplify_category(box.name)

        if agent_class is None:
            continue

        distance = math.hypot(
            box.center[0] - frame["x"],
            box.center[1] - frame["y"],
        )

        if distance > max_range:
            continue

        velocity = nusc.box_velocity(box.token)

        speed = 0.0

        if np.isfinite(velocity[:2]).all():
            speed = float(np.hypot(velocity[0], velocity[1]))

        try:
            annotation = nusc.get("sample_annotation", box.token)
            instance = annotation["instance_token"]
        except Exception:
            instance = box.token

        agents.append({
            "instance": instance,
            "class": agent_class,
            "x": float(box.center[0]),
            "y": float(box.center[1]),
            "yaw": quaternion_yaw(box.orientation),
            "length_m": round(float(box.wlh[1]), 2),
            "width_m": round(float(box.wlh[0]), 2),
            "speed_mps": round(speed, 2),
        })

    return agents


class AnnotationLaneTracker:
    """
    Lane assignment + lane-change status for annotation-placed
    vehicles. Identity comes from the instance token (positions
    are already smooth and exact, so no filtering or display
    smoothing is needed); the lane state machine matches the
    VehicleTracker's: adoption, collinear segment handover,
    overshoot handling, vote-debounced changes, staged status.
    """

    def __init__(self, lane_map):
        self.lane_map = lane_map
        self.states = {}
        self.next_id = 1

    def update(self, agents, ego_xy):
        output = []

        for agent in agents:
            if agent["class"] == "pedestrian":
                continue

            state = self.states.get(agent["instance"])

            if state is None:
                state = {
                    "id": self.next_id,
                    "lane": None,
                    "change": None,
                    "votes": 0,
                }
                self.next_id += 1
                self.states[agent["instance"]] = state

            status = self._lane_status(state, agent)

            lane_info = state.get("lane_info")

            output.append({
                "id": state["id"],
                "class": agent["class"],
                "position": [
                    round(agent["x"], 2), round(agent["y"], 2)
                ],
                "speed_mps": agent["speed_mps"],
                "heading_deg": round(
                    math.degrees(agent["yaw"]), 1
                ),
                "lane_id": (
                    None if state["lane"] is None
                    else int(state["lane"][0])
                ),
                "lane_direction": (
                    None if state["lane"] is None
                    else int(state["lane"][1])
                ),
                "lane_lateral_m": (
                    None if lane_info is None
                    else round(lane_info["lateral"], 2)
                ),
                "lane_s_m": (
                    None if lane_info is None
                    else round(lane_info["s"], 1)
                ),
                "distance_from_ego_m": round(math.hypot(
                    agent["x"] - ego_xy[0],
                    agent["y"] - ego_xy[1],
                ), 1),
                "lane_change": status,
                "length_m": agent["length_m"],
                "width_m": agent["width_m"],
            })

        return output

    def _lane_status(self, state, agent):
        located = self.lane_map.locate(
            agent["x"], agent["y"], agent["yaw"]
        )

        if located is None:
            state["lane"] = None
            state["change"] = None
            state["votes"] = 0
            state["lane_info"] = None
            return "off lane network"

        raw_lane = (located["lane_id"], located["direction"])

        if state["lane"] is None:
            state["lane"] = raw_lane

        current = self.lane_map.project_to_lane(
            state["lane"][0], state["lane"][1],
            agent["x"], agent["y"],
        )

        if current["overshoot"] > 2.5:
            state["lane"] = None
            state["change"] = None
            state["votes"] = 0
            state["lane_info"] = None
            return "off lane network"

        # Collinear segment handover - continuation, not change.
        if (
            raw_lane != state["lane"]
            and state["change"] is None
            and abs(
                located["lateral"] - current["lateral"]
            ) < 0.8
        ):
            state["lane"] = raw_lane
            current = located

        lane_width = current["lane_width"]

        if state["change"] is None:
            crossing = (
                abs(current["lateral"]) > 0.35 * lane_width
                or raw_lane != state["lane"]
            )

            state["votes"] = (
                state["votes"] + 1 if crossing else 0
            )

            if state["votes"] >= 3:
                side = 1 if current["lateral"] > 0 else -1

                state["change"] = {
                    "gap": side * lane_width,
                    "switched": False,
                }

                state["votes"] = 0
        else:
            change = state["change"]

            if not change["switched"] and (
                current["lateral"] / change["gap"] > 0.5
            ):
                target = self.lane_map.locate(
                    agent["x"], agent["y"], agent["yaw"]
                )

                if target is not None and (
                    target["lane_id"], target["direction"]
                ) != state["lane"]:
                    state["lane"] = (
                        target["lane_id"], target["direction"]
                    )
                    current = target
                    change["gap"] = -change["gap"]
                    change["switched"] = True

            if abs(current["lateral"]) < 0.2 * lane_width:
                state["change"] = None

        state["lane_info"] = current

        if state["change"] is not None:
            change = state["change"]

            raw_progress = current["lateral"] / change["gap"]

            stage_progress = (
                1.0 - raw_progress if change["switched"]
                else raw_progress
            )

            stage_progress = float(
                np.clip(stage_progress, 0.0, 1.0)
            )

            for threshold, label in LANE_CHANGE_STAGES:
                if stage_progress <= threshold:
                    return f"lane change: {label}"

        if abs(current["lateral"]) > 0.3 * lane_width:
            return "drifting toward lane boundary"

        return "lane keeping"


# ============================================================
# LAYER 1 — EGO POSITION ON THE ROAD
# ============================================================
# REFERENCE from the HD map + true pose — the value the video
# estimate (Layer 0) is compared against. A lane detector
# (CLRNet / UFLD v2) + IPM would produce the same dictionary
# from the camera image alone.
# ============================================================


def get_ego_lane_info(nusc_map, ego_x, ego_y, ego_yaw):
    """
    Lane-relative localization of the ego vehicle.

    Returns a dictionary with:
        lane_token        which lane the ego is in
        lateral_offset_m  signed distance from the lane centerline
                          (positive = left of centerline)
        heading_error_deg ego heading minus lane direction
    Returns None if no lane is found nearby (e.g. parking lot).
    """
    lane_token = nusc_map.get_closest_lane(
        ego_x,
        ego_y,
        radius=LANE_SEARCH_RADIUS_METERS,
    )

    if not lane_token:
        return None

    lane_record = nusc_map.get_arcline_path(lane_token)

    centerline = np.array(
        arcline_path_utils.discretize_lane(
            lane_record,
            resolution_meters=0.5,
        )
    )

    # Each centerline point is (x, y, yaw).
    distances = np.hypot(
        centerline[:, 0] - ego_x,
        centerline[:, 1] - ego_y,
    )

    nearest_index = int(np.argmin(distances))
    nearest_point = centerline[nearest_index]

    lane_yaw = nearest_point[2]

    # Signed lateral offset: project the ego position onto the
    # lane's left-pointing normal vector.
    offset_x = ego_x - nearest_point[0]
    offset_y = ego_y - nearest_point[1]

    left_normal_x = -math.sin(lane_yaw)
    left_normal_y = math.cos(lane_yaw)

    lateral_offset = (
        offset_x * left_normal_x + offset_y * left_normal_y
    )

    heading_error = wrap_angle(ego_yaw - lane_yaw)

    # A window of centerline points around the ego, kept only for
    # the reconstructed-map panel (stripped before JSON export).
    window_start = max(0, nearest_index - 40)
    centerline_window = centerline[
        window_start: nearest_index + 40, :2
    ]

    return {
        "lane_token": lane_token,
        "lateral_offset_m": round(float(lateral_offset), 2),
        "heading_error_deg": round(math.degrees(heading_error), 1),
        "_centerline": centerline_window,
    }


# ============================================================
# LAYER 2 — DYNAMIC AGENTS IN THE EGO FRAME
# ============================================================
# Ground truth today: annotated 3D boxes (interpolated between
# keyframes by the devkit, so sweeps get smooth boxes too).
# Model later: replace the body of this function with FCOS3D /
# StreamPETR output. Keep the same return format and nothing
# downstream changes.
# ============================================================


def get_agents_in_ego_frame(nusc, frame):
    """
    All annotated agents, expressed in the ego vehicle frame:
        x forward, y left, z up (nuScenes ego convention).

    Returns a list of dictionaries, one per agent.
    """
    ego_pose = frame["ego_pose"]

    ego_translation = np.array(ego_pose["translation"])
    ego_rotation_inverse = Quaternion(ego_pose["rotation"]).inverse
    ego_yaw = frame["yaw"]

    # Boxes in the GLOBAL frame, interpolated for this timestamp.
    boxes = nusc.get_boxes(frame["camera_token"])

    agents = []

    for box in boxes:
        simple_class = simplify_category(box.name)

        if simple_class is None:
            continue

        # Global velocity of this annotation (may be NaN).
        velocity = nusc.box_velocity(box.token)

        # Transform the box into the ego frame.
        box.translate(-ego_translation)
        box.rotate(ego_rotation_inverse)

        x_forward = float(box.center[0])
        y_left = float(box.center[1])

        # Keep only agents inside the BEV window, in front-ish.
        if not (-5.0 < x_forward < BEV_FORWARD_METERS):
            continue
        if abs(y_left) > BEV_LATERAL_METERS:
            continue

        # After the rotate() above, box.orientation is already in the
        # ego frame, so its yaw is the yaw relative to the ego heading.
        relative_yaw = wrap_angle(
            quaternion_yaw(box.orientation)
        )

        speed = None

        if velocity is not None and not np.isnan(velocity).any():
            speed = round(float(np.linalg.norm(velocity[:2])), 2)

        # Stable identity of this agent across the whole clip —
        # used to seed its perception noise and, later, to match
        # predictions to ground truth during evaluation.
        annotation = nusc.get("sample_annotation", box.token)
        instance_id = annotation["instance_token"][:8]

        # nuScenes box size is (width, length, height).
        agents.append(
            {
                "instance": instance_id,
                "class": simple_class,
                "x": round(x_forward, 2),
                "y": round(y_left, 2),
                "yaw_deg": round(math.degrees(relative_yaw), 1),
                "length_m": round(float(box.wlh[1]), 2),
                "width_m": round(float(box.wlh[0]), 2),
                "v_mps": speed,
            }
        )

    # Closest agents first — nice for reading the JSON.
    agents.sort(key=lambda agent: agent["x"])

    return agents


# ============================================================
# SIMULATED PERCEPTION (predicted agents)
# ============================================================
# Applies a monocular-style error model to the ground-truth
# agents: range-proportional depth noise, lateral noise, yaw
# noise, and dropout of distant agents. Each agent's noise is
# seeded by its instance id, so the error is CONSTANT over the
# clip — the reconstructed panel shows a steady shift, not
# jitter. When a real detector (FCOS3D, StreamPETR) replaces
# Layer 2, delete this function and feed the detector output
# straight into the same downstream code.
# ============================================================


def apply_perception_noise(agents):
    """Return the 'predicted' version of the agent list."""
    if not SIMULATE_PERCEPTION_NOISE:
        return [dict(agent) for agent in agents]

    predicted = []

    for agent in agents:
        rng = np.random.default_rng(
            zlib.crc32(agent["instance"].encode())
        )

        distance = math.hypot(agent["x"], agent["y"])

        # Distant agents are sometimes missed entirely
        # ("not all objects are read").
        if distance > DROPOUT_START_METERS:
            miss_probability = min(
                0.6,
                (distance - DROPOUT_START_METERS) / 40.0,
            )

            if rng.random() < miss_probability:
                continue

        # Depth error grows with range ("positions are shifted").
        range_scale = 1.0 + rng.normal(0.0, DEPTH_NOISE_FRACTION)

        noisy_agent = dict(agent)

        noisy_agent["x"] = round(agent["x"] * range_scale, 2)

        noisy_agent["y"] = round(
            agent["y"] * range_scale
            + rng.normal(0.0, LATERAL_NOISE_METERS),
            2,
        )

        noisy_agent["yaw_deg"] = round(
            agent["yaw_deg"] + rng.normal(0.0, YAW_NOISE_DEGREES),
            1,
        )

        predicted.append(noisy_agent)

    return predicted


# ============================================================
# LAYER 3 — STATIC STRUCTURE (map-derived)
# ============================================================
# Ground truth today: HD map layers around the route.
# Model later: metric depth + segmentation -> semantic point
# cloud, squashed to the same kind of BEV summary.
# ============================================================


def get_static_structure(nusc_map, ego_x, ego_y, ego_yaw):
    """
    A compact static-layout summary around the ego:
    distance to the edge of the drivable area on the left and
    right, and whether a pedestrian crossing is within 25 m ahead.
    """

    def distance_to_nondrivable(direction_sign):
        """Walk sideways until leaving the drivable area.
        Returns (distance, (x, y)) or (None, None)."""
        left_normal_x = -math.sin(ego_yaw)
        left_normal_y = math.cos(ego_yaw)

        for step in np.arange(1.0, 20.0, 0.5):
            probe_x = ego_x + direction_sign * step * left_normal_x
            probe_y = ego_y + direction_sign * step * left_normal_y

            layers = nusc_map.layers_on_point(probe_x, probe_y)

            if not layers.get("drivable_area"):
                return (
                    round(float(step), 1),
                    (float(probe_x), float(probe_y)),
                )

        return None, None

    def crossing_ahead():
        forward_x = math.cos(ego_yaw)
        forward_y = math.sin(ego_yaw)

        for step in np.arange(2.0, 25.0, 1.0):
            probe_x = ego_x + step * forward_x
            probe_y = ego_y + step * forward_y

            layers = nusc_map.layers_on_point(probe_x, probe_y)

            if layers.get("ped_crossing"):
                return round(float(step), 1)

        return None

    left_distance, left_point = distance_to_nondrivable(+1)
    right_distance, right_point = distance_to_nondrivable(-1)

    return {
        "road_edge_left_m": left_distance,
        "road_edge_right_m": right_distance,
        "ped_crossing_ahead_m": crossing_ahead(),
        "_edge_points": {
            "left": left_point,
            "right": right_point,
        },
    }


# ============================================================
# LAYER 4 — FUSED BEV SCENE DESCRIPTION
# ============================================================


def build_scene_description(
    frame, lane_info, agents, static_info, predicted_agents=None
):
    """
    The single ego-centric scene description that everything
    downstream (Scenic, CARLA, evaluation) consumes.

    "agents" holds the ground truth; "agents_predicted" holds
    the (simulated or, later, model-produced) perception output,
    so every keyframe is a ready-made evaluation pair.

    Convention: x forward, y left, angles in degrees,
    yaw relative to the ego heading.
    """
    description = {
        "time_s": round(frame["elapsed_seconds"], 2),
        "sample_token": frame["sample_token"],
        "ego": {
            "lane": strip_private_keys(lane_info),
            "global_x": round(frame["x"], 2),
            "global_y": round(frame["y"], 2),
            "global_yaw_deg": round(math.degrees(frame["yaw"]), 1),
        },
        "agents": agents,
        "static": strip_private_keys(static_info),
    }

    if predicted_agents is not None:
        description["agents_predicted"] = predicted_agents

    return description


# ============================================================
# SCENIC EMITTER
# ============================================================

# Scenic's "offset by (dx, dy)" is in the ego's local frame with
# dy pointing forward. If the generated scene comes out mirrored
# left/right in CARLA, flip this sign — lateral conventions are
# the classic real-to-sim bug.
SCENIC_LATERAL_SIGN = -1.0  # nuScenes y_left -> Scenic dx


def write_scenic_file(scene_description, scene_name, output_path):
    """
    Emit a Scenic (CARLA model) program that reproduces the
    agents of one scene description relative to the ego.
    """
    lines = [
        "# Auto-generated from nuScenes scene "
        f"{scene_name}, t = {scene_description['time_s']} s",
        "param map = localPath('../maps/Town05.xodr')",
        "param carla_map = 'Town05'",
        "model scenic.simulators.carla.model",
        "",
        "ego = new Car with blueprint 'vehicle.tesla.model3'",
        "",
    ]

    lane_info = scene_description["ego"]["lane"]

    if lane_info is not None:
        lines.insert(
            5,
            "# Ego lateral offset from lane center: "
            f"{lane_info['lateral_offset_m']} m",
        )

    scenic_class = {
        "car": "Car",
        "truck": "Truck",
        "bus": "Truck",
        "motorcycle": "Motorcycle",
        "bicycle": "Bicycle",
        "pedestrian": "Pedestrian",
    }

    # Spawn what the pipeline PRODUCED (the predicted agents);
    # fall back to ground truth if no prediction is present.
    agents_to_spawn = scene_description.get(
        "agents_predicted",
        scene_description["agents"],
    )

    for index, agent in enumerate(agents_to_spawn):
        dx = SCENIC_LATERAL_SIGN * agent["y"]
        dy = agent["x"]

        lines.append(
            f"# {agent['class']}, {agent['x']} m ahead, "
            f"{agent['y']} m left, "
            f"speed {agent['v_mps']} m/s"
        )

        lines.append(
            f"agent_{index} = new {scenic_class[agent['class']]} "
            f"at ego offset by ({dx:.2f}, {dy:.2f}), "
            f"facing {agent['yaw_deg']:.1f} deg relative to ego.heading"
        )

        if agent["v_mps"] is not None and agent["v_mps"] > 0.3:
            lines.append(
                f"agent_{index}.speed = {agent['v_mps']:.2f}"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAP BACKGROUND (unchanged from the playback script)
# ============================================================


def create_map_background(nusc_map, frames):
    """Create a semantic map image around the vehicle route."""
    x_positions = np.array([frame["x"] for frame in frames])
    y_positions = np.array([frame["y"] for frame in frames])

    center_x = (x_positions.min() + x_positions.max()) / 2.0
    center_y = (y_positions.min() + y_positions.max()) / 2.0

    map_width = max(
        50.0,
        x_positions.max() - x_positions.min() + 2 * MAP_MARGIN_METERS,
    )

    map_height = max(
        50.0,
        y_positions.max() - y_positions.min() + 2 * MAP_MARGIN_METERS,
    )

    patch_box = (center_x, center_y, map_height, map_width)

    # Adapt resolution so large windows stay within memory.
    pixels_per_meter = 6

    largest_side = max(map_width, map_height)

    if largest_side * pixels_per_meter > MAX_BACKGROUND_PIXELS:
        pixels_per_meter = max(
            1, int(MAX_BACKGROUND_PIXELS / largest_side)
        )

    canvas_size = (
        max(400, int(map_height * pixels_per_meter)),
        max(400, int(map_width * pixels_per_meter)),
    )

    map_layers = [
        "drivable_area",
        "road_segment",
        "lane",
        "ped_crossing",
        "walkway",
    ]

    map_mask = nusc_map.get_map_mask(
        patch_box=patch_box,
        patch_angle=0,
        layer_names=map_layers,
        canvas_size=canvas_size,
    )

    map_background = np.zeros(map_mask.shape[1:], dtype=np.float32)

    layer_strengths = np.linspace(0.25, 1.0, len(map_layers))

    for layer_index, strength in enumerate(layer_strengths):
        map_background = np.maximum(
            map_background,
            map_mask[layer_index].astype(np.float32) * strength,
        )

    map_extent = [
        center_x - map_width / 2.0,
        center_x + map_width / 2.0,
        center_y - map_height / 2.0,
        center_y + map_height / 2.0,
    ]

    # Drivable-area-only mask (layer 0) — used for the fast
    # per-agent on-pavement check.
    drivable_mask = map_mask[0].astype(bool)

    return map_background, map_extent, drivable_mask


# ============================================================
# OPENSTREETMAP BACKGROUND (reconstructed panel)
# ============================================================
# The reconstructed side of the pipeline should not depend on
# the nuScenes HD map — in the real system, OSM is the map
# source that exists everywhere. These functions fetch OSM
# roads and buildings for the scene area, convert them into
# the nuScenes global frame, and render them to a background
# image that drops into the panel exactly like the HD map did.
# ============================================================


def nuscenes_xy_to_latlon(map_name, x, y):
    """nuScenes global meters -> (latitude, longitude)."""
    ref_lat, ref_lon = REFERENCE_COORDINATES[map_name]

    latitude = ref_lat + math.degrees(y / EARTH_RADIUS_METERS)

    longitude = ref_lon + math.degrees(
        x / (EARTH_RADIUS_METERS * math.cos(math.radians(ref_lat)))
    )

    return latitude, longitude


def latlon_to_nuscenes_xy(map_name, latitude, longitude):
    """(latitude, longitude) -> nuScenes global meters."""
    ref_lat, ref_lon = REFERENCE_COORDINATES[map_name]

    y = math.radians(latitude - ref_lat) * EARTH_RADIUS_METERS

    x = (
        math.radians(longitude - ref_lon)
        * EARTH_RADIUS_METERS
        * math.cos(math.radians(ref_lat))
    )

    return x + OSM_ALIGN_OFFSET_X, y + OSM_ALIGN_OFFSET_Y


def geometry_to_xy_arrays(map_name, geometry):
    """
    Convert one shapely geometry (LineString, Polygon, or their
    Multi- variants) from lat/lon into a list of Nx2 arrays of
    nuScenes coordinates.
    """
    parts = getattr(geometry, "geoms", [geometry])

    arrays = []

    for part in parts:
        if part.geom_type == "LineString":
            coordinates = part.coords
        elif part.geom_type == "Polygon":
            coordinates = part.exterior.coords
        else:
            continue

        points = [
            latlon_to_nuscenes_xy(map_name, lat, lon)
            for lon, lat in coordinates  # shapely stores (x=lon, y=lat)
        ]

        if len(points) >= 2:
            arrays.append(np.array(points))

    return arrays


def estimate_road_width(highway_tag, lanes_tag):
    """
    Visual width for one OSM road, in meters. Prefers the tagged
    lane count (lanes x 3.5 m); otherwise falls back to typical
    widths per highway type. osmnx sometimes merges ways, giving
    list-valued tags — take the first entry.
    """
    if isinstance(highway_tag, (list, tuple)) and highway_tag:
        highway_tag = highway_tag[0]

    if isinstance(lanes_tag, (list, tuple)) and lanes_tag:
        lanes_tag = lanes_tag[0]

    try:
        lane_count = float(str(lanes_tag))

        if lane_count > 0:
            return float(
                np.clip(
                    lane_count * OSM_LANE_WIDTH_METERS,
                    3.0,
                    25.0,
                )
            )

    except (TypeError, ValueError):
        pass

    type_widths = {
        "motorway": 12.0,
        "motorway_link": 8.0,
        "trunk": 11.0,
        "trunk_link": 8.0,
        "primary": 10.5,
        "primary_link": 7.5,
        "secondary": 9.0,
        "secondary_link": 7.0,
        "tertiary": 8.0,
        "residential": 6.5,
        "unclassified": 6.5,
        "service": 4.0,
    }

    return type_widths.get(highway_tag, OSM_ROAD_WIDTH_METERS)


def fetch_osm_geometries(map_name, map_extent):
    """
    Roads and buildings around the scene, in nuScenes coordinates.
    Cached on disk so only the first run per scene needs internet.
    """
    OSM_CACHE_DIR.mkdir(exist_ok=True)

    cache_key = "_".join(
        str(int(round(value))) for value in map_extent
    )

    cache_path = OSM_CACHE_DIR / f"osm_v2_{map_name}_{cache_key}.pkl"

    if cache_path.exists():
        print("OSM background: using cached data:", cache_path)
        return pickle.loads(cache_path.read_bytes())

    import osmnx as ox

    print(
        "OSM background: fetching from OpenStreetMap "
        f"(osmnx {ox.__version__}) ..."
    )

    # Small buffer so roads right at the window edge are included.
    buffer_meters = 30.0

    south, west = nuscenes_xy_to_latlon(
        map_name,
        map_extent[0] - buffer_meters,
        map_extent[2] - buffer_meters,
    )

    north, east = nuscenes_xy_to_latlon(
        map_name,
        map_extent[1] + buffer_meters,
        map_extent[3] + buffer_meters,
    )

    # Paste these into openstreetmap.org to verify the area.
    print(
        "OSM background: bbox "
        f"lat {south:.6f}..{north:.6f}, "
        f"lon {west:.6f}..{east:.6f}"
    )

    osmnx_major = int(ox.__version__.split(".")[0])

    def bbox_call(function, **kwargs):
        """
        Call an osmnx *_from_bbox function across API versions.
        OSMnx >= 2.0 expects bbox = (west, south, east, north);
        OSMnx 1.x expects (north, south, east, west) — passing
        the 2.x order into 1.x silently queries a nonsense box
        and returns an empty graph.
        """
        if osmnx_major >= 2:
            return function(
                bbox=(west, south, east, north), **kwargs
            )

        try:
            return function(
                bbox=(north, south, east, west), **kwargs
            )
        except TypeError:
            # Very old 1.x: positional arguments only.
            return function(north, south, east, west, **kwargs)

    # ---- road network: prefer drivable roads, fall back to all
    # ---- road types (some areas map service roads only).
    graph = None

    for network_type in ("drive", "all"):
        try:
            graph = bbox_call(
                ox.graph_from_bbox,
                network_type=network_type,
                retain_all=True,
            )
            break
        except Exception as error:
            print(
                f"OSM background: no '{network_type}' roads "
                f"({error}); trying next option."
            )

    if graph is None:
        raise RuntimeError("no OSM roads found in this area")

    road_edges = ox.graph_to_gdfs(graph, nodes=False)

    # Each road keeps its own visual width, taken from OSM tags.
    roads = []

    for record in road_edges.to_dict("records"):
        width = estimate_road_width(
            record.get("highway"),
            record.get("lanes"),
        )

        for points in geometry_to_xy_arrays(
            map_name, record["geometry"]
        ):
            roads.append(
                {"points": points, "width_m": width}
            )

    # ---- building footprints (optional layer; keep going if empty)
    buildings = []

    try:
        features_function = getattr(
            ox, "features_from_bbox", None
        )

        if features_function is None:
            features_function = ox.geometries_from_bbox

        building_features = bbox_call(
            features_function,
            tags={"building": True},
        )

        for geometry in building_features.geometry:
            buildings.extend(
                geometry_to_xy_arrays(map_name, geometry)
            )

    except Exception as error:
        print("OSM background: no buildings layer:", error)

    geometries = {"roads": roads, "buildings": buildings}

    cache_path.write_bytes(pickle.dumps(geometries))

    print(
        f"OSM background: {len(roads)} road segments, "
        f"{len(buildings)} building footprints (cached)."
    )

    return geometries


def sample_lane_reference_points(nusc_map, max_points=150):
    """
    Discretized centerline points from EVERY lane in the nuScenes
    map — a map-wide reference set for aligning OSM against the
    nuScenes frame. Far better constrained than the short driven
    route: it covers many road directions across the whole city
    area, so both offset components are always observable.
    """
    points = []

    lane_records = list(nusc_map.lane) + list(
        getattr(nusc_map, "lane_connector", [])
    )

    for lane_record in lane_records:
        try:
            lane_path = nusc_map.get_arcline_path(
                lane_record["token"]
            )

            centerline = arcline_path_utils.discretize_lane(
                lane_path,
                resolution_meters=4.0,
            )

            points.extend(
                (point[0], point[1]) for point in centerline
            )

        except Exception:
            continue

    if not points:
        return None

    points = np.array(points)

    if len(points) > max_points:
        stride = len(points) // max_points + 1
        points = points[::stride]

    return points


def auto_align_osm_geometries(
    geometries, route_xy, max_reference_points=40
):
    """
    Estimate the translation that best snaps the OSM road network
    onto the driven route, and shift ALL OSM geometry by it.

    The nuScenes reference origins and OSM geometry can disagree
    by ten or more meters; the vehicle, however, is guaranteed to
    have driven ON a road. A two-stage grid search finds the
    (dx, dy) that minimizes the clipped mean distance from the route
    points to the nearest OSM road point (clipped at 10 m per point so a
    few unmapped route sections cannot dominate the fit).

    Returns (geometries, (dx, dy), median_residual_m).
    """
    if not geometries["roads"] or route_xy is None:
        return geometries, (0.0, 0.0), None

    route = np.asarray(route_xy, dtype=float)

    if len(route) == 0:
        return geometries, (0.0, 0.0), None

    # Subsample the reference set — it describes the alignment
    # target plenty well at this density.
    if len(route) > max_reference_points:
        route = route[:: len(route) // max_reference_points + 1]

    # Resample road polylines into a dense point set (~1 m).
    road_points = []

    for segment in geometries["roads"]:
        segment_points = segment["points"]

        for start, end in zip(
            segment_points[:-1], segment_points[1:]
        ):
            length = float(np.hypot(*(end - start)))
            steps = max(2, int(length))
            road_points.append(np.linspace(start, end, steps))

    road_points = np.vstack(road_points)

    # Keep only road points near the reference set (speed). The
    # radius adapts to the reference spread, so this works for a
    # short route and for map-wide lane references alike.
    route_center = route.mean(axis=0)

    reference_spread = float(
        np.hypot(*(route.max(axis=0) - route.min(axis=0)))
    )

    near_radius = max(250.0, reference_spread / 2.0 + 50.0)

    near_mask = (
        np.hypot(*(road_points - route_center).T) < near_radius
    )

    if near_mask.any():
        road_points = road_points[near_mask]

    if len(road_points) > 3000:
        stride = len(road_points) // 3000 + 1
        road_points = road_points[::stride]

    def route_distance_score(dx, dy):
        """
        Mean route-to-road distance with each point clipped at
        10 m. The clip keeps unmapped route sections from
        dominating; the mean (unlike a median) still feels every
        misaligned point — a median lets a whole misaligned road
        leg hide behind the majority.
        """
        shifted = road_points + np.array([dx, dy])

        deltas = route[:, None, :] - shifted[None, :, :]

        nearest = np.sqrt(
            (deltas ** 2).sum(axis=2)
        ).min(axis=1)

        return float(np.mean(np.minimum(nearest, 10.0)))

    best_dx, best_dy = 0.0, 0.0

    # Among near-optimal candidates, prefer the SMALLEST shift.
    # For a straight route on a straight road, sliding along the
    # road direction is unobservable (every slide scores the
    # same); this tie-break keeps the unconstrained component at
    # zero instead of letting it wander.
    score_tolerance = 0.25

    def grid_search(center_dx, center_dy, span, step):
        candidates = np.arange(-span, span + step / 2.0, step)

        results = []

        for dx in candidates:
            for dy in candidates:
                offset_x = center_dx + dx
                offset_y = center_dy + dy

                results.append(
                    (
                        route_distance_score(offset_x, offset_y),
                        (offset_x, offset_y),
                    )
                )

        best_score = min(score for score, _ in results)

        near_optimal = [
            offset
            for score, offset in results
            if score <= best_score + score_tolerance
        ]

        return min(
            near_optimal,
            key=lambda off: off[0] ** 2 + off[1] ** 2,
        )

    # Coarse pass (±24 m, 3 m grid), then fine (±3 m, 0.5 m grid).
    best_dx, best_dy = grid_search(0.0, 0.0, 24.0, 3.0)
    best_dx, best_dy = grid_search(best_dx, best_dy, 3.0, 0.5)

    best_score = route_distance_score(best_dx, best_dy)

    shift = np.array([best_dx, best_dy])

    geometries["roads"] = [
        {**segment, "points": segment["points"] + shift}
        for segment in geometries["roads"]
    ]

    geometries["buildings"] = [
        footprint + shift
        for footprint in geometries["buildings"]
    ]

    return geometries, (best_dx, best_dy), best_score


def crop_optimized_surface(payload, view_extent):
    """
    Cut the view window out of an optimized map's surface raster
    and convert it to the panel's RGB style (white background,
    light-gray buildings, dark-gray road). Areas outside the
    stored surface are padded white. Returned image is flipped
    for origin="upper", matching the OSM panel convention.
    """
    surface = payload["surface"]
    surface_extent = payload["surface_extent"]

    height, width = surface.shape

    def to_column(x):
        return int(
            round(
                (x - surface_extent[0])
                / (surface_extent[1] - surface_extent[0])
                * width
            )
        )

    def to_row(y):
        return int(
            round(
                (y - surface_extent[2])
                / (surface_extent[3] - surface_extent[2])
                * height
            )
        )

    column_start = to_column(view_extent[0])
    column_end = to_column(view_extent[1])
    row_start = to_row(view_extent[2])
    row_end = to_row(view_extent[3])

    out_height = max(1, row_end - row_start)
    out_width = max(1, column_end - column_start)

    window = np.zeros((out_height, out_width), dtype=np.uint8)

    source_row_start = max(0, row_start)
    source_row_end = min(height, row_end)
    source_column_start = max(0, column_start)
    source_column_end = min(width, column_end)

    if (
        source_row_end > source_row_start
        and source_column_end > source_column_start
    ):
        window[
            source_row_start - row_start:
            source_row_start - row_start
            + (source_row_end - source_row_start),
            source_column_start - column_start:
            source_column_start - column_start
            + (source_column_end - source_column_start),
        ] = surface[
            source_row_start:source_row_end,
            source_column_start:source_column_end,
        ]

    gray = np.full(window.shape, 255, dtype=np.uint8)
    gray[window == 1] = 209   # buildings
    gray[window == 2] = 115   # road surface

    rgb = np.stack([gray] * 3, axis=-1)

    # Surface row 0 is y_min; the panel draws OSM backgrounds
    # with origin="upper", so flip vertically.
    return np.flipud(rgb).copy()


def render_osm_background(
    map_name, view_extent, reference_xy=None, fetch_extent=None
):
    """
    Render the OSM geometries into an RGB image covering exactly
    view_extent.

    fetch_extent controls how much OSM data is downloaded and
    cached — pass the WHOLE nuScenes map canvas so one download
    per city serves every scene and the alignment can use the
    full map. reference_xy is the point set OSM is aligned to
    (the map-wide nuScenes lane graph, or the driven route as a
    fallback). Returns None if OSM is unavailable, so the caller
    can fall back to the nuScenes map background.
    """
    if map_name not in REFERENCE_COORDINATES:
        print("OSM background: unknown map location:", map_name)
        return None

    if fetch_extent is None:
        fetch_extent = view_extent

    # An optimized map built by build_osm_map.py takes priority:
    # it is already aligned and its road widths were measured
    # against the nuScenes pavement.
    geometries = None
    already_optimized = False

    optimized_path = (
        OPTIMIZED_MAP_DIR / f"osm_optimized_{map_name}.pkl"
    )

    if optimized_path.exists():
        try:
            payload = pickle.loads(optimized_path.read_bytes())

            print(
                "OSM background: using optimized map "
                f"{optimized_path} "
                f"(built {payload.get('created', '?')}, "
                f"alignment {payload.get('alignment')})"
            )

            # Version 2 maps carry the rasterized road surface —
            # crop the view window and use it directly.
            if "surface" in payload:
                return crop_optimized_surface(
                    payload, view_extent
                )

            # Version 1 maps: vector roads, already aligned.
            geometries = {
                "roads": payload["roads"],
                "buildings": payload["buildings"],
            }

            already_optimized = True

        except Exception as error:
            print(
                "OSM background: optimized map unreadable "
                f"({error}); fetching instead."
            )
            geometries = None

    if geometries is None:
        try:
            geometries = fetch_osm_geometries(
                map_name, fetch_extent
            )
        except Exception as error:
            print(
                "OSM background unavailable "
                f"({error}); falling back to the nuScenes map."
            )
            return None

    if OSM_AUTO_ALIGN and not already_optimized:
        geometries, (shift_x, shift_y), residual = (
            auto_align_osm_geometries(
                geometries,
                reference_xy,
                max_reference_points=150,
            )
        )

        if residual is not None:
            print(
                "OSM background: auto-align shift "
                f"dx = {shift_x:+.1f} m, dy = {shift_y:+.1f} m "
                f"(route-to-road fit {residual:.2f} m). "
                "Copy into OSM_ALIGN_OFFSET_X/Y to make permanent."
            )

    # Keep only geometry near the view window (rendering speed —
    # the whole-map fetch can contain thousands of segments).
    def near_view(segment):
        margin = 30.0

        return not (
            segment[:, 0].max() < view_extent[0] - margin
            or segment[:, 0].min() > view_extent[1] + margin
            or segment[:, 1].max() < view_extent[2] - margin
            or segment[:, 1].min() > view_extent[3] + margin
        )

    visible_roads = [
        segment
        for segment in geometries["roads"]
        if near_view(segment["points"])
    ]

    visible_buildings = [
        footprint
        for footprint in geometries["buildings"]
        if near_view(footprint)
    ]

    width_meters = view_extent[1] - view_extent[0]
    height_meters = view_extent[3] - view_extent[2]

    # Adapt resolution so large windows stay within memory.
    pixels_per_meter = 6

    largest_side = max(width_meters, height_meters)

    if largest_side * pixels_per_meter > MAX_BACKGROUND_PIXELS:
        pixels_per_meter = max(
            1, int(MAX_BACKGROUND_PIXELS / largest_side)
        )

    dpi = 100

    figure = plt.figure(
        figsize=(
            width_meters * pixels_per_meter / dpi,
            height_meters * pixels_per_meter / dpi,
        ),
        dpi=dpi,
    )

    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(view_extent[0], view_extent[1])
    axis.set_ylim(view_extent[2], view_extent[3])
    axis.axis("off")

    # Buildings first (under the roads).
    for footprint in visible_buildings:
        axis.add_patch(
            plt.Polygon(
                footprint,
                closed=True,
                facecolor="0.82",
                edgecolor="0.65",
                linewidth=0.4,
            )
        )

    # Roads: each drawn at its own tag-derived width.
    for segment in visible_roads:
        road_linewidth_points = (
            segment["width_m"] * pixels_per_meter * 72.0 / dpi
        )

        axis.plot(
            segment["points"][:, 0],
            segment["points"][:, 1],
            color="0.45",
            linewidth=road_linewidth_points,
            solid_capstyle="round",
            zorder=2,
        )

    figure.canvas.draw()

    rgb_image = np.asarray(
        figure.canvas.buffer_rgba()
    )[..., :3].copy()

    plt.close(figure)

    return rgb_image


# ============================================================
# DRAWING
# ============================================================


def draw_camera_panel(nusc, video_axis, frame):
    """Camera image with the annotated 3D boxes projected onto it."""
    video_axis.clear()

    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        frame["camera_token"],
        box_vis_level=BoxVisibility.ANY,
    )

    with Image.open(image_path) as image:
        camera_image = image.convert("RGB").copy()

    video_axis.imshow(camera_image)

    for box in boxes:
        simple_class = simplify_category(box.name)

        if simple_class is None:
            continue

        color = CLASS_COLORS[simple_class]

        box.render(
            video_axis,
            view=camera_intrinsic,
            normalize=True,
            colors=(color, color, color),
        )

    video_axis.set_xlim(0, camera_image.size[0])
    video_axis.set_ylim(camera_image.size[1], 0)
    video_axis.axis("off")

    video_axis.set_title(
        f"{CAMERA_CHANNEL} — "
        f"{frame['elapsed_seconds']:.2f} s "
        f"{'(keyframe)' if frame['is_key_frame'] else ''}"
    )


def draw_bev_panel(bev_axis, agents, lane_info, lane_offset_est=None):
    """Ego-centric bird's-eye view: ego at origin, x forward (up)."""
    bev_axis.clear()

    # Ego footprint.
    ego_corners = rotated_rectangle_corners(
        0.0, 0.0, EGO_LENGTH, EGO_WIDTH, 0.0
    )

    bev_axis.add_patch(
        plt.Polygon(
            # Plot as (y_left mirrored to screen-x, x_forward as screen-y).
            np.column_stack([-ego_corners[:, 1], ego_corners[:, 0]]),
            closed=True,
            facecolor="black",
        )
    )

    bev_axis.arrow(
        0.0, EGO_LENGTH / 2.0, 0.0, 2.0,
        head_width=0.8, facecolor="black", edgecolor="black",
    )

    # Agents.
    for agent in agents:
        corners = rotated_rectangle_corners(
            agent["x"],
            agent["y"],
            agent["length_m"],
            agent["width_m"],
            math.radians(agent["yaw_deg"]),
        )

        color = CLASS_COLORS[agent["class"]]

        bev_axis.add_patch(
            plt.Polygon(
                np.column_stack([-corners[:, 1], corners[:, 0]]),
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )
        )

        label = agent["class"]

        if agent["v_mps"] is not None:
            label += f"\n{agent['v_mps']:.1f} m/s"

        label_y = agent["x"] + agent["length_m"] / 2.0 + 0.7

        # Skip labels that would spill past the top of the panel.
        if label_y < BEV_FORWARD_METERS - 3.0:
            bev_axis.text(
                -agent["y"],
                label_y,
                label,
                fontsize=7,
                ha="center",
            )

    title = "Ego-centric BEV (Layer 2)"

    if lane_info is not None:
        title += (
            f"\nlateral offset {lane_info['lateral_offset_m']:+.2f} m, "
            f"heading error {lane_info['heading_error_deg']:+.1f}°"
        )

    if lane_offset_est is not None:
        title += f"  |  video estimate {lane_offset_est:+.2f} m"

    bev_axis.set_title(title, fontsize=10)
    bev_axis.set_xlim(-BEV_LATERAL_METERS, BEV_LATERAL_METERS)
    bev_axis.set_ylim(-5, BEV_FORWARD_METERS)
    bev_axis.set_xlabel("left  ←  meters  →  right")
    bev_axis.set_ylabel("forward (meters)")
    bev_axis.set_aspect("equal", adjustable="box")
    bev_axis.grid(True, alpha=0.25)


def draw_reconstructed_map_panel(
    recon_axis,
    frame,
    predicted_agents,
    ground_truth_agents,
    background,
    drivable_outlines,
    map_extent,
    full_route_x,
    full_route_y,
    traveled_x,
    traveled_y,
    pose_label="ground-truth pose",
    global_agents=None,
):
    """
    Twin of the real map panel, but the map background comes from
    the RECONSTRUCTION side (OpenStreetMap when available), and
    the ego location and surrounding objects are drawn from the
    PIPELINE output (predicted agents), not from ground truth.
    Visually comparing the two bottom panels shows exactly what
    the reconstruction got right and wrong.

    background: {"image": ndarray, "origin": str, "source": str}
    """
    recon_axis.clear()

    recon_axis.imshow(
        background["image"],
        origin=background["origin"],
        extent=map_extent,
        cmap="gray_r",
        interpolation="bilinear",
    )

    # Ground-truth pavement boundary — dotted blue. White areas
    # inside this boundary are pavement that OSM does not map.
    for index, outline in enumerate(drivable_outlines):
        recon_axis.plot(
            outline[:, 0],
            outline[:, 1],
            color="steelblue",
            linestyle=":",
            linewidth=1.0,
            alpha=0.85,
            zorder=3,
            label=(
                "True pavement edge" if index == 0 else None
            ),
        )

    # Route context, mirroring the real panel.
    recon_axis.plot(
        full_route_x, full_route_y,
        linestyle="--", linewidth=1.5,
        label="Full 5-second route",
    )

    recon_axis.plot(
        traveled_x, traveled_y,
        linewidth=3,
        label="Traveled route",
    )

    # Current ego: marker for visibility at map scale, plus the
    # true-scale footprint and a heading arrow on the lane.
    recon_axis.plot(
        [frame["x"]], [frame["y"]],
        marker="o", markersize=12, linestyle="None",
        label="Ego (reconstructed)",
    )

    ego_corners = rotated_rectangle_corners(
        frame["x"], frame["y"],
        EGO_LENGTH, EGO_WIDTH,
        frame["yaw"],
    )

    recon_axis.add_patch(
        plt.Polygon(
            ego_corners, closed=True,
            facecolor="black", zorder=4,
        )
    )

    recon_axis.arrow(
        frame["x"],
        frame["y"],
        6.0 * math.cos(frame["yaw"]),
        6.0 * math.sin(frame["yaw"]),
        head_width=1.2,
        facecolor="black",
        edgecolor="black",
        zorder=4,
    )

    # Ground-truth outlines (optional) — faint dashed reference.
    if SHOW_GROUND_TRUTH_OVERLAY:
        for agent in ground_truth_agents:
            global_x, global_y = ego_to_global(
                frame, agent["x"], agent["y"]
            )

            corners = rotated_rectangle_corners(
                global_x, global_y,
                agent["length_m"], agent["width_m"],
                frame["yaw"] + math.radians(agent["yaw_deg"]),
            )

            recon_axis.add_patch(
                plt.Polygon(
                    corners, closed=True,
                    facecolor="none", edgecolor="0.45",
                    linewidth=1.0, linestyle="--",
                    zorder=3,
                )
            )

    # Surrounding objects. With a tracker: vehicles are drawn
    # at their LANE-SNAPPED, rate-limited display poses with
    # persistent IDs; pedestrians (not lane-bound) come from
    # the raw detections. Without a tracker: legacy behavior.
    off_pavement_count = 0
    off_pavement_labeled = False

    def draw_agent_box(
        x, y, yaw, length, width, color, flagged, label
    ):
        nonlocal off_pavement_labeled

        corners = rotated_rectangle_corners(
            x, y, length, width, yaw
        )

        patch_label = None

        if flagged and not off_pavement_labeled:
            patch_label = "Vehicle off true pavement"
            off_pavement_labeled = True

        recon_axis.add_patch(
            plt.Polygon(
                corners, closed=True,
                facecolor=color,
                edgecolor="red" if flagged else "black",
                linewidth=1.4 if flagged else 0.5,
                alpha=0.9,
                zorder=4,
                label=patch_label,
            )
        )

        recon_axis.plot(
            [x], [y], marker=".", markersize=4,
            color=color, zorder=5,
        )

        if label:
            recon_axis.text(
                x, y + length / 2.0 + 0.8, label,
                fontsize=6, ha="center", zorder=6,
            )

    if global_agents is not None:
        for agent in global_agents:
            is_vehicle = agent["class"] != "pedestrian"

            flagged = (
                is_vehicle
                and agent.get("on_drivable") is False
            )

            if flagged:
                off_pavement_count += 1

            label = None

            if is_vehicle:
                label = (
                    f"#{agent.get('track_id', '?')}  "
                    f"{agent['speed_mps']:.1f} m/s"
                )

            draw_agent_box(
                agent["x"], agent["y"], agent["yaw"],
                agent["length_m"], agent["width_m"],
                CLASS_COLORS[agent["class"]],
                flagged,
                label,
            )
    else:
        for agent in predicted_agents:
            global_x, global_y = ego_to_global(
                frame, agent["x"], agent["y"]
            )

            flagged = (
                agent["class"] != "pedestrian"
                and agent.get("on_drivable") is False
            )

            if flagged:
                off_pavement_count += 1

            draw_agent_box(
                global_x, global_y,
                frame["yaw"] + math.radians(agent["yaw_deg"]),
                agent["length_m"], agent["width_m"],
                CLASS_COLORS[agent["class"]],
                flagged,
                None,
            )

    missed_count = len(ground_truth_agents) - len(predicted_agents)

    noise_note = (
        "perception noise ON" if SIMULATE_PERCEPTION_NOISE
        else "perfect perception"
    )

    recon_axis.set_title(
        f"Reconstructed Map — {background['source']} — "
        f"{pose_label}\n"
        f"({noise_note}) — missed: {missed_count}, "
        f"off-pavement vehicles: {off_pavement_count}",
        fontsize=10,
    )

    recon_axis.legend(loc="upper right", fontsize=7)

    # Same window as the real map panel -> direct comparison.
    recon_axis.set_xlim(map_extent[0], map_extent[1])
    recon_axis.set_ylim(map_extent[2], map_extent[3])
    recon_axis.set_xlabel("Global X (m)")
    recon_axis.set_ylabel("Global Y (m)")
    recon_axis.set_aspect("equal", adjustable="box")
    recon_axis.grid(True, alpha=0.25)


# ============================================================
# VIDEO RECORDING
# ============================================================


def grab_figure_frame(figure):
    """
    Render the figure and return it as an RGB array, trimmed to
    even dimensions (some video codecs require them).
    """
    figure.canvas.draw()

    frame = np.asarray(
        figure.canvas.buffer_rgba()
    )[:, :, :3]

    even_height = frame.shape[0] // 2 * 2
    even_width = frame.shape[1] // 2 * 2

    return frame[:even_height, :even_width]


class VideoRecorder:
    """
    Writes figure frames to an MP4. The writer opens lazily on
    the first frame (that's when the size is known); later frames
    are resized if the window was resized mid-playback.
    """

    def __init__(self, output_path, fps):
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        self.frame_size = None
        self.frame_count = 0
        self.failed = False

    def add_frame(self, figure):
        if cv2 is None or self.failed:
            return

        try:
            frame = grab_figure_frame(figure)

            if self.writer is None:
                self.frame_size = (
                    frame.shape[1], frame.shape[0]
                )

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    self.fps,
                    self.frame_size,
                )

            if (
                frame.shape[1], frame.shape[0]
            ) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)

            # OpenCV expects BGR.
            self.writer.write(frame[:, :, ::-1])

            self.frame_count += 1

        except Exception as error:
            print("Video recording failed:", error)
            self.failed = True

    def close(self):
        if self.writer is not None:
            self.writer.release()

            print(
                f"Video written to: {self.output_path} "
                f"({self.frame_count} frames at {self.fps} fps)"
            )

        elif SAVE_VIDEO and cv2 is None:
            print(
                "Video not saved: opencv (cv2) is not installed."
            )


# ============================================================
# MAIN
# ============================================================


def main():
    nusc = NuScenes(
        version=VERSION,
        dataroot=str(DATAROOT),
        verbose=True,
    )

    if not 0 <= SCENE_INDEX < len(nusc.scene):
        raise ValueError(
            f"SCENE_INDEX must be between 0 and {len(nusc.scene) - 1}."
        )

    scene = nusc.scene[SCENE_INDEX]

    print("\nSelected scene")
    print("Name:", scene["name"])
    print("Description:", scene["description"])

    log_record = nusc.get("log", scene["log_token"])
    map_name = log_record["location"]

    print("Map:", map_name)

    nusc_map = NuScenesMap(
        dataroot=str(DATAROOT),
        map_name=map_name,
    )

    frames = collect_camera_frames(nusc, scene)

    print("Frames collected:", len(frames))

    map_background, map_extent, drivable_mask = (
        create_map_background(nusc_map, frames)
    )

    drivable_outlines = []

    if SHOW_DRIVABLE_OUTLINE:
        drivable_outlines = get_drivable_area_outlines(
            nusc_map, map_extent
        )

        print(
            "Drivable-area outline segments:",
            len(drivable_outlines),
        )

    # Background for the reconstructed panel: OpenStreetMap when
    # available, nuScenes map as the fallback.
    recon_background = {
        "image": map_background,
        "origin": "lower",
        "source": "nuScenes map (fallback)",
    }

    if USE_OSM_BACKGROUND:
        # Fetch OSM for the WHOLE nuScenes map canvas — cached
        # once per city, reused by every scene on this map.
        canvas_edge = getattr(nusc_map, "canvas_edge", None)

        if canvas_edge is not None:
            fetch_extent = [
                0.0, float(canvas_edge[0]),
                0.0, float(canvas_edge[1]),
            ]
        else:
            fetch_extent = map_extent

        # Align OSM map-wide against the nuScenes lane graph
        # (fall back to the driven route if that fails).
        reference_points = sample_lane_reference_points(nusc_map)

        if reference_points is None:
            reference_points = np.array(
                [[frame["x"], frame["y"]] for frame in frames]
            )

        print(
            "OSM background: aligning with "
            f"{len(reference_points)} map-wide lane points."
        )

        osm_image = render_osm_background(
            map_name, map_extent, reference_points, fetch_extent
        )

        if osm_image is not None:
            recon_background = {
                "image": osm_image,
                "origin": "upper",
                "source": "OpenStreetMap",
            }

    all_x = np.array([frame["x"] for frame in frames])
    all_y = np.array([frame["y"] for frame in frames])

    # ---- Layer 0: estimate the ego pose from the video itself.
    pose_label = "ground-truth pose"

    road_network = get_anchor_road_network(map_name, nusc_map)

    if ESTIMATE_EGO_FROM_VIDEO:
        if apply_ego_estimation(nusc, frames, road_network):
            pose_label = "video-estimated pose"

    # ---- Lane model (for lane assignment / status only — the
    # ---- DRAWN lane lines are the observed camera segments).
    lane_map = load_lane_map(map_name, road_network)

    annotation_tracker = AnnotationLaneTracker(lane_map)

    # Route as the RECONSTRUCTION believes it (falls back to
    # ground truth when estimation is off).
    est_x = np.array(
        [frame.get("x_est", frame["x"]) for frame in frames]
    )

    est_y = np.array(
        [frame.get("y_est", frame["y"]) for frame in frames]
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    keyframe_descriptions = []

    video_recorder = None

    if SAVE_VIDEO:
        video_recorder = VideoRecorder(
            OUTPUT_DIR / f"reconstruction_{scene['name']}.mp4",
            PLAYBACK_FPS,
        )

    # ========================================================
    # FOUR-PANEL WINDOW
    #   top:    camera + boxes   |  ego-centric BEV
    #   bottom: real map         |  reconstructed map
    # ========================================================

    plt.ion()

    figure, panel_axes = plt.subplots(
        2, 2,
        figsize=(16, 12),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    video_axis = panel_axes[0][0]
    bev_axis = panel_axes[0][1]
    map_axis = panel_axes[1][0]
    recon_axis = panel_axes[1][1]

    figure.suptitle(
        f"nuScenes {scene['name']} — Scene Reconstruction Pipeline",
        fontsize=14,
    )

    # ---- right panel: global map (static content drawn once)
    map_axis.imshow(
        map_background,
        origin="lower",
        extent=map_extent,
        cmap="gray_r",
        interpolation="nearest",
    )

    map_axis.plot(
        all_x, all_y,
        linestyle="--", linewidth=1.5,
        label="Full 5-second route",
    )

    traveled_line, = map_axis.plot(
        [all_x[0]], [all_y[0]],
        linewidth=3, label="Traveled route",
    )

    vehicle_marker, = map_axis.plot(
        [all_x[0]], [all_y[0]],
        marker="o", markersize=12, linestyle="None",
        label="Current vehicle location",
    )

    map_axis.plot(
        [all_x[0]], [all_y[0]],
        marker="s", markersize=8, linestyle="None",
        label="Start",
    )

    location_text = map_axis.text(
        0.02, 0.02, "",
        transform=map_axis.transAxes,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "alpha": 0.8},
    )

    map_axis.set_title(f"Real Map (ground truth) — {map_name}")
    map_axis.set_xlabel("Global X (m)")
    map_axis.set_ylabel("Global Y (m)")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.legend(loc="upper right", fontsize=7)
    map_axis.grid(True, alpha=0.25)

    plt.tight_layout()

    # ========================================================
    # PLAYBACK + PER-FRAME RECONSTRUCTION
    # ========================================================

    for frame_index, frame in enumerate(frames):
        if not plt.fignum_exists(figure.number):
            break

        # ---- Layer 1: ego lane-relative localization
        lane_info = get_ego_lane_info(
            nusc_map, frame["x"], frame["y"], frame["yaw"]
        )

        # ---- Layer 2: agents in the ego frame
        agents = get_agents_in_ego_frame(nusc, frame)

        # Pose used on the RECONSTRUCTED side: the video
        # estimate when available, ground truth otherwise. The
        # real-map panel always keeps the true pose.
        recon_frame = frame

        if "x_est" in frame:
            recon_frame = {
                **frame,
                "x": frame["x_est"],
                "y": frame["y_est"],
                "yaw": frame["yaw_est"],
            }

        # On-pavement check, evaluated where the RECONSTRUCTION
        # places each agent (also lands in the JSON).
        for agent in agents:
            global_x, global_y = ego_to_global(
                recon_frame, agent["x"], agent["y"]
            )

            agent["on_drivable"] = point_on_drivable(
                drivable_mask, map_extent, global_x, global_y
            )

        # ---- Simulated perception output ("predicted" agents)
        predicted_agents = apply_perception_noise(agents)

        # Predicted positions differ from ground truth, so their
        # on-pavement flag must be recomputed.
        for agent in predicted_agents:
            global_x, global_y = ego_to_global(
                recon_frame, agent["x"], agent["y"]
            )

            agent["on_drivable"] = point_on_drivable(
                drivable_mask, map_extent, global_x, global_y
            )

        # ---- Objects at their GLOBAL annotated poses, with
        # ---- box_velocity speeds; lane states from the model.
        agents_global = get_agents_global(nusc, frame)

        for agent in agents_global:
            agent["on_drivable"] = point_on_drivable(
                drivable_mask, map_extent,
                agent["x"], agent["y"],
            )

        lane_states = annotation_tracker.update(
            agents_global,
            (recon_frame["x"], recon_frame["y"]),
        )

        vehicles_global = [
            agent for agent in agents_global
            if agent["class"] != "pedestrian"
        ]

        for agent, state in zip(vehicles_global, lane_states):
            agent["track_id"] = state["id"]
            agent["status"] = state["lane_change"]

        # ---- Layers 3 + 4 only on keyframes (annotations live there;
        # ---- static probing is also the slowest step)
        if frame["is_key_frame"]:
            static_info = get_static_structure(
                nusc_map, frame["x"], frame["y"], frame["yaw"]
            )

            description = build_scene_description(
                frame, lane_info, agents, static_info,
                predicted_agents=predicted_agents,
            )

            if "x_est" in frame:
                ego_lane = lane_map.locate(
                    recon_frame["x"],
                    recon_frame["y"],
                    recon_frame["yaw"],
                )

                description["ego_estimated"] = {
                    "x": round(frame["x_est"], 2),
                    "y": round(frame["y_est"], 2),
                    "yaw_deg": round(
                        math.degrees(frame["yaw_est"]), 1
                    ),
                    "lane_offset_m": frame.get(
                        "lane_offset_est"
                    ),
                    "lane_id": (
                        None if ego_lane is None
                        else int(ego_lane["lane_id"])
                    ),
                    "lane_lateral_m": (
                        None if ego_lane is None
                        else round(ego_lane["lateral"], 2)
                    ),
                }

            description["tracks"] = lane_states

            keyframe_descriptions.append(description)

        # ---- draw all four panels
        draw_camera_panel(nusc, video_axis, frame)

        draw_bev_panel(
            bev_axis, agents, lane_info,
            lane_offset_est=frame.get("lane_offset_est"),
        )

        draw_reconstructed_map_panel(
            recon_axis,
            recon_frame,
            predicted_agents,
            agents,
            recon_background,
            drivable_outlines,
            map_extent,
            est_x,
            est_y,
            est_x[: frame_index + 1],
            est_y[: frame_index + 1],
            pose_label=pose_label,
            global_agents=agents_global,
        )

        vehicle_marker.set_data([frame["x"]], [frame["y"]])

        traveled_line.set_data(
            all_x[: frame_index + 1],
            all_y[: frame_index + 1],
        )

        location_text.set_text(
            f"Time: {frame['elapsed_seconds']:.2f} s\n"
            f"X: {frame['x']:.2f} m\n"
            f"Y: {frame['y']:.2f} m\n"
            f"Agents in view: {len(agents)}"
        )

        # Recording forces a full draw; otherwise draw lazily.
        if video_recorder is not None:
            video_recorder.add_frame(figure)
        else:
            figure.canvas.draw_idle()

        figure.canvas.flush_events()

        plt.pause(1.0 / PLAYBACK_FPS)

    if video_recorder is not None:
        video_recorder.close()

    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    json_path = OUTPUT_DIR / f"scene_description_{scene['name']}.json"

    json_path.write_text(
        json.dumps(keyframe_descriptions, indent=2),
        encoding="utf-8",
    )

    print("\nScene descriptions written to:", json_path)

    if keyframe_descriptions:
        scenic_path = OUTPUT_DIR / f"scenario_{scene['name']}.scenic"

        write_scenic_file(
            keyframe_descriptions[0],
            scene["name"],
            scenic_path,
        )

        print("Scenic scenario written to:", scenic_path)

    plt.ioff()
    plt.show()

    print("\nPlayback complete.")
    print("Displayed frames:", len(frames))
    print("Keyframes reconstructed:", len(keyframe_descriptions))


if __name__ == "__main__":
    main()