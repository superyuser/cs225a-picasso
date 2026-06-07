"""Geometric model of the paint tool station.

The model describes the location of each tag and its associated cylindrical
volume (paint cup or water cup) in a single "station" coordinate frame whose
origin is the top-left corner of the tag with id 10. The model is
self-contained -- it does not require a camera or any robot connection -- and
can be exported as JSON so downstream scripts can read its coordinates without
re-deriving them.

Station-frame conventions:

- Origin: top-left corner of tag id 10 (a 35 mm black square).
- +X: along the row of tags, in the direction of decreasing tag id (i.e.
  tag 11 sits at station X = -60 mm).
- +Y: away from the tag row, across the table, in the direction the
  containers extend (so each paint cup sits at a positive station Y).
- +Z: out of the tag face (toward the camera when the wrist camera is
  pointed at the station). The containers extend in -Z (z_min_mm < 0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MM_TO_M = 1.0e-3


@dataclass
class CylinderVolume:
    """An upright cylinder permitted for the brush in station-frame mm."""

    name: str
    center_mm: np.ndarray   # shape (3,)
    radius_mm: float
    z_min_mm: float
    z_max_mm: float
    init_hover_z_offset_mm: float = 50.0  # how far above z_max the "INIT" pose sits

    def contains_station_point(self, p_mm: np.ndarray) -> bool:
        dx = p_mm[0] - self.center_mm[0]
        dy = p_mm[1] - self.center_mm[1]
        dz = p_mm[2]

        inside_xy = dx ** 2 + dy ** 2 <= self.radius_mm ** 2
        inside_z = self.z_min_mm <= dz <= self.z_max_mm
        return inside_xy and inside_z

    def init_hover_point_mm(self) -> np.ndarray:
        """Position centred above this cylinder, used as the "P*_INIT"
        cartesian waypoint. Stays clear of the rim by init_hover_z_offset_mm.
        """
        return np.array(
            [
                float(self.center_mm[0]),
                float(self.center_mm[1]),
                float(self.z_max_mm + self.init_hover_z_offset_mm),
            ],
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "center_mm": [float(v) for v in self.center_mm],
            "radius_mm": float(self.radius_mm),
            "z_min_mm": float(self.z_min_mm),
            "z_max_mm": float(self.z_max_mm),
            "init_hover_z_offset_mm": float(self.init_hover_z_offset_mm),
            "init_hover_point_station_mm": [
                float(v) for v in self.init_hover_point_mm()
            ],
        }


# Names exposed to downstream visit scripts. The list order is the order in
# Visit order used by demo-day/robot/visit_tool_station.py after calibration.
VOLUME_NAME_TO_INIT_LABEL = {
    "paint_1": "P1_INIT",
    "paint_2": "P2_INIT",
    "paint_3": "P3_INIT",
    "water": "WATER_INIT",
}
INIT_VISIT_ORDER = ("paint_1", "paint_2", "paint_3", "water")


R_STATION_TO_TAG_LOCAL = np.diag([-1.0, -1.0, 1.0])

# Preference order when picking which visible tag anchors the station frame.
REFERENCE_TAG_PREFERENCE = (10, 11, 12, 13)


def pick_reference_tag_id(visible_tag_ids: set[int] | list[int]) -> int | None:
    """Choose the reference tag from whatever is currently visible."""
    visible = set(int(tid) for tid in visible_tag_ids)
    for tag_id in REFERENCE_TAG_PREFERENCE:
        if tag_id in visible:
            return tag_id
    return None


def station_origin_in_reference_tag_local_m(
    tag_size_m: float,
    tag_tl_station_m: np.ndarray,
) -> np.ndarray:
    """Station-frame origin expressed in the chosen reference tag's local frame."""
    s = tag_size_m / 2.0
    tag_tl_corner = np.array([-s, s, 0.0], dtype=float)
    return tag_tl_corner + R_STATION_TO_TAG_LOCAL @ (-np.asarray(tag_tl_station_m, dtype=float))


def station_point_in_reference_tag_local_m(
    p_station_m: np.ndarray,
    *,
    tag_size_m: float,
    tag_tl_station_m: np.ndarray,
) -> np.ndarray:
    origin = station_origin_in_reference_tag_local_m(tag_size_m, tag_tl_station_m)
    return origin + R_STATION_TO_TAG_LOCAL @ np.asarray(p_station_m, dtype=float)


class PaintToolStationModel:
    def __init__(
        self,
        *,
        paint_init_hover_z_offset_mm: float = 50.0,
        water_init_hover_z_offset_mm: float = 50.0,
    ) -> None:
        self.tag_size_mm = 35.0
        self.margin_mm = 5.0

        self.paint_container_size_mm = 50.0
        self.paint_permitted_diameter_mm = 35.0
        self.paint_depth_mm = 40.0

        self.water_container_diameter_mm = 85.0
        self.water_permitted_diameter_mm = 75.0
        self.water_depth_mm = 50.0

        self.paint_init_hover_z_offset_mm = paint_init_hover_z_offset_mm
        self.water_init_hover_z_offset_mm = water_init_hover_z_offset_mm

        # Top-left black-square corners of each tag, in the station frame.
        self.tag_tl_mm: dict[int, np.ndarray] = {
            10: np.array([0.0,    0.0, 0.0]),
            11: np.array([-60.0,  0.0, 0.0]),
            12: np.array([-120.0, 0.0, 0.0]),
            13: np.array([-200.0, 0.0, 0.0]),
        }

        # Tag id whose pose defines the station -> camera transform at runtime.
        self.reference_tag_id = 10

        self.volumes: list[CylinderVolume] = self._build_volumes()

    def _tag_center_xy(self, tag_id: int) -> np.ndarray:
        tl = self.tag_tl_mm[tag_id]
        return np.array(
            [
                tl[0] - self.tag_size_mm / 2.0,
                tl[1] + self.tag_size_mm / 2.0,
            ]
        )

    def _build_volumes(self) -> list[CylinderVolume]:
        volumes: list[CylinderVolume] = []

        # Paint containers under ids 10, 11, 12.
        for tag_id, name in zip([10, 11, 12], ["paint_1", "paint_2", "paint_3"]):
            tag_center_xy = self._tag_center_xy(tag_id)
            center_mm = np.array(
                [
                    tag_center_xy[0],
                    self.tag_size_mm + self.margin_mm + self.paint_container_size_mm / 2.0,
                    0.0,
                ]
            )
            volumes.append(
                CylinderVolume(
                    name=name,
                    center_mm=center_mm,
                    radius_mm=self.paint_permitted_diameter_mm / 2.0,
                    z_min_mm=-self.paint_depth_mm,
                    z_max_mm=0.0,
                    init_hover_z_offset_mm=self.paint_init_hover_z_offset_mm,
                )
            )

        # Water container under id 13.
        tag_center_xy = self._tag_center_xy(13)
        water_center_mm = np.array(
            [
                tag_center_xy[0],
                self.tag_size_mm + self.margin_mm + self.water_container_diameter_mm / 2.0,
                0.0,
            ]
        )
        volumes.append(
            CylinderVolume(
                name="water",
                center_mm=water_center_mm,
                radius_mm=self.water_permitted_diameter_mm / 2.0,
                z_min_mm=-self.water_depth_mm,
                z_max_mm=0.0,
                init_hover_z_offset_mm=self.water_init_hover_z_offset_mm,
            )
        )
        return volumes

    def get_tag_tl_station_m(self, tag_id: int) -> np.ndarray:
        if tag_id not in self.tag_tl_mm:
            raise KeyError(f"Unknown tag id {tag_id}")
        return self.tag_tl_mm[tag_id] * MM_TO_M

    def get_volume(self, name: str) -> CylinderVolume:
        for volume in self.volumes:
            if volume.name == name:
                return volume
        raise KeyError(f"No volume named {name!r}")

    def print_summary(self) -> None:
        for v in self.volumes:
            print(
                f"{v.name}: center_mm={v.center_mm.tolist()}, "
                f"radius_mm={v.radius_mm}, "
                f"z_range_mm=[{v.z_min_mm}, {v.z_max_mm}], "
                f"init_hover_point_station_mm={v.init_hover_point_mm().tolist()}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": "millimeters",
            "frame": (
                "Station frame. Origin: top-left corner of tag id "
                f"{self.reference_tag_id}. +X: along the row of tags toward "
                "decreasing tag id. +Y: across the table toward containers. "
                "+Z: out of the tag face."
            ),
            "tag_size_mm": self.tag_size_mm,
            "margin_mm": self.margin_mm,
            "reference_tag_id": self.reference_tag_id,
            "paint_container_size_mm": self.paint_container_size_mm,
            "paint_permitted_diameter_mm": self.paint_permitted_diameter_mm,
            "paint_depth_mm": self.paint_depth_mm,
            "water_container_diameter_mm": self.water_container_diameter_mm,
            "water_permitted_diameter_mm": self.water_permitted_diameter_mm,
            "water_depth_mm": self.water_depth_mm,
            "paint_init_hover_z_offset_mm": self.paint_init_hover_z_offset_mm,
            "water_init_hover_z_offset_mm": self.water_init_hover_z_offset_mm,
            "tag_tl_mm": {
                str(tag_id): [float(v) for v in tl]
                for tag_id, tl in self.tag_tl_mm.items()
            },
            "volumes": [volume.to_dict() for volume in self.volumes],
            "init_visit_order": list(INIT_VISIT_ORDER),
            "volume_name_to_init_label": dict(VOLUME_NAME_TO_INIT_LABEL),
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def main() -> None:
    station = PaintToolStationModel()
    station.print_summary()


if __name__ == "__main__":
    main()
