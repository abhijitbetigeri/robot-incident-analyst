"""Shared alpine presentation helpers for the fixed-line videos."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageOps


ALPINE_BACKDROP = (
    Path(__file__).resolve().parent
    / "assets"
    / "environments"
    / "fixed_line_himalaya.png"
)


def add_braided_rope_visual(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    flex_name: str = "fixed_rope",
) -> int:
    """Draw a raised, interwoven kernmantle sheath over the physical rope.

    MuJoCo's one-dimensional flex is the collision-bearing rope core.  Flexes
    do not generate useful longitudinal texture coordinates, so a checker
    material alone reads as a smooth cable.  These short capsule yarns follow
    the live flex vertices and make the rendered sheath visibly braided while
    remaining visual-only; mass, contacts, stretch, and reaction forces still
    come exclusively from the flex.
    """
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
    if flex_id < 0:
        return 0
    vertex_address = int(model.flex_vertadr[flex_id])
    vertex_count = int(model.flex_vertnum[flex_id])
    vertices = np.asarray(
        data.flexvert_xpos[vertex_address : vertex_address + vertex_count]
    )
    if len(vertices) < 2:
        return 0

    edge_vectors = np.diff(vertices, axis=0)
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-8:
        return 0

    def surface_point(
        distance: float, phase_offset: float, twist_direction: float
    ) -> np.ndarray:
        distance = float(np.clip(distance, 0.0, total_length))
        edge = int(
            np.clip(
                np.searchsorted(cumulative, distance, side="right") - 1,
                0,
                len(edge_lengths) - 1,
            )
        )
        length = max(float(edge_lengths[edge]), 1e-9)
        weight = (distance - float(cumulative[edge])) / length
        center = (1.0 - weight) * vertices[edge] + weight * vertices[edge + 1]
        tangent = edge_vectors[edge] / length
        # The route runs in X/Z, making world Y a stable first radial axis.
        radial_a = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        radial_a -= float(np.dot(radial_a, tangent)) * tangent
        radial_norm = float(np.linalg.norm(radial_a))
        if radial_norm < 1e-6:
            radial_a = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            radial_a -= float(np.dot(radial_a, tangent)) * tangent
            radial_norm = float(np.linalg.norm(radial_a))
        radial_a /= max(radial_norm, 1e-9)
        radial_b = np.cross(tangent, radial_a)
        # A short 80 mm pitch makes the crossing yarns resolve as a textile
        # diamond braid at demo-camera distance instead of a cable stripe.
        phase = twist_direction * 2.0 * np.pi * distance / 0.080 + phase_offset
        # Keep the visual braid inside the physical flex's 7 mm collision
        # envelope: 4.55 mm center radius + 2.25 mm yarn = 6.80 mm.
        return center + 0.00455 * (
            np.cos(phase) * radial_a + np.sin(phase) * radial_b
        )

    scene = renderer.scene
    added = 0
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
        focus_vertex = int(
            np.argmin(np.linalg.norm(vertices - data.xpos[pelvis_id], axis=1))
        )
        focus_distance = float(cumulative[focus_vertex])
        visible_start = max(0.006, focus_distance - 3.0)
        visible_end = min(total_length - 0.006, focus_distance + 3.0)
    else:
        visible_start, visible_end = 0.006, total_length - 0.006
    # Two broad yarn bundles in each direction form crossing chevrons. Their
    # lobed silhouette and high-contrast over/under breaks remain readable in
    # a 720p wide shot, unlike a sub-pixel surface texture on a smooth flex.
    strand_specs = (
        (0.0, 1.0, [1.00, 0.82, 0.12, 1.0]),
        (np.pi, 1.0, [1.00, 0.94, 0.62, 1.0]),
        (0.5 * np.pi, -1.0, [0.92, 0.20, 0.025, 1.0]),
        (1.5 * np.pi, -1.0, [0.42, 0.018, 0.008, 1.0]),
    )
    for phase_offset, twist_direction, rgba in strand_specs:
        color = np.asarray(rgba, dtype=np.float32)
        phase_shift = 0.015 if twist_direction < 0.0 else 0.0
        for start in np.arange(visible_start + phase_shift, visible_end, 0.030):
            end = min(float(start + 0.027), total_length - 0.002)
            samples = np.linspace(float(start), end, 5)
            points = [
                surface_point(sample, phase_offset, twist_direction)
                for sample in samples
            ]
            for start_point, end_point in zip(points[:-1], points[1:]):
                if scene.ngeom >= scene.maxgeom:
                    return added
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.eye(3, dtype=np.float64).ravel(),
                    color,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.00225,
                    start_point,
                    end_point,
                )
                geom.specular = 0.015
                geom.shininess = 0.005
                scene.ngeom += 1
                added += 1
    return added


@lru_cache(maxsize=8)
def alpine_backdrop(width: int, height: int) -> np.ndarray:
    """Load and crop the distant mountain plate for a render resolution."""
    with Image.open(ALPINE_BACKDROP) as source:
        fitted = ImageOps.fit(
            source.convert("RGB"),
            (int(width), int(height)),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.48),
        )
        return np.asarray(fitted).copy()


def render_with_alpine_backdrop(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    backdrop: np.ndarray,
) -> np.ndarray:
    """Render the 3-D scene while replacing only the synthetic skybox.

    The physical slope, its moving texture, contact shadows, the robot, and
    all climbing equipment stay native MuJoCo pixels.  Only pixels at the
    renderer's far plane are replaced, so the mountain range reads as a
    distant environment instead of a foreground photograph glued to the feet.
    """
    renderer.update_scene(data, camera=camera)
    add_braided_rope_visual(renderer, model, data)
    frame = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    frame[depth > 100.0] = backdrop[depth > 100.0]
    return frame
