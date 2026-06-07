import numpy as np
from dataclasses import dataclass

MM_TO_M = 1e-3


@dataclass
class CylinderVolume:
    name: str
    center_mm: np.ndarray   # shape (3,)
    radius_mm: float
    z_min_mm: float
    z_max_mm: float

    def contains_station_point(self, p_mm: np.ndarray) -> bool:
        dx = p_mm[0] - self.center_mm[0]
        dy = p_mm[1] - self.center_mm[1]
        dz = p_mm[2]

        inside_xy = dx**2 + dy**2 <= self.radius_mm**2
        inside_z = self.z_min_mm <= dz <= self.z_max_mm
        return inside_xy and inside_z


class PaintToolStationModel:
    def __init__(self):
        self.tag_size_mm = 35.0
        self.margin_mm = 5.0

        self.paint_container_size_mm = 50.0
        self.paint_permitted_diameter_mm = 35.0
        self.paint_depth_mm = 40.0

        self.water_container_diameter_mm = 85.0
        self.water_permitted_diameter_mm = 75.0
        self.water_depth_mm = 50.0

        # Top-left black-square corners in station frame.
        self.tag_tl_mm = {
            10: np.array([0.0,    0.0, 0.0]),
            11: np.array([-60.0,  0.0, 0.0]),
            12: np.array([-120.0, 0.0, 0.0]),
            13: np.array([-200.0, 0.0, 0.0]),
        }

        self.volumes = self._build_volumes()

    def _tag_center_xy(self, tag_id: int):
        tl = self.tag_tl_mm[tag_id]
        return np.array([
            tl[0] - self.tag_size_mm / 2.0,
            tl[1] + self.tag_size_mm / 2.0,
        ])

    def _build_volumes(self):
        volumes = []

        # Paint containers under ids 10, 11, 12.
        for tag_id, name in zip([10, 11, 12], ["paint_1", "paint_2", "paint_3"]):
            tag_center_xy = self._tag_center_xy(tag_id)

            center_mm = np.array([
                tag_center_xy[0],
                self.tag_size_mm + self.margin_mm + self.paint_container_size_mm / 2.0,
                0.0,
            ])

            volumes.append(
                CylinderVolume(
                    name=name,
                    center_mm=center_mm,
                    radius_mm=self.paint_permitted_diameter_mm / 2.0,
                    z_min_mm=-self.paint_depth_mm,
                    z_max_mm=0.0,
                )
            )

        # Water container under id13.
        tag_center_xy = self._tag_center_xy(13)

        water_center_mm = np.array([
            tag_center_xy[0],
            self.tag_size_mm + self.margin_mm + self.water_container_diameter_mm / 2.0,
            0.0,
        ])

        volumes.append(
            CylinderVolume(
                name="water",
                center_mm=water_center_mm,
                radius_mm=self.water_permitted_diameter_mm / 2.0,
                z_min_mm=-self.water_depth_mm,
                z_max_mm=0.0,
            )
        )

        return volumes

    def print_summary(self):
        for v in self.volumes:
            print(
                f"{v.name}: center_mm={v.center_mm}, "
                f"radius_mm={v.radius_mm}, "
                f"z_range_mm=[{v.z_min_mm}, {v.z_max_mm}]"
            )


if __name__ == "__main__":
    station = PaintToolStationModel()
    station.print_summary()