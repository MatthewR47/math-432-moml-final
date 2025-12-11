import math
import numpy as np


def _los_vector(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)

    ux = math.cos(el) * math.sin(az)
    uy = math.cos(el) * math.cos(az)
    uz = math.sin(el)

    return np.array([ux, uy, uz])


class ComputeConditionNumber:
    def __init__(self) -> None:
        self.satellites: list[tuple[float, float]] = []

    def add_satellite(self, elevation_deg: float, azimuth_deg: float) -> None:
        self.satellites.append((azimuth_deg, elevation_deg))

    def _build_G(self) -> np.ndarray:
        rows = []
        for az_deg, el_deg in self.satellites:
            u = _los_vector(az_deg, el_deg)
            rows.append([u[0], u[1], u[2], 1.0])

        return np.vstack(rows)

    def compute_condition_number(self) -> float:
        G = self._build_G()
        _, S, _ = np.linalg.svd(G)

        sigma_max = float(S[0])
        sigma_min = float(S[-1])

        if sigma_min == 0.0:
            raise ValueError(
                "Geometry matrix is singular; condition number is infinite."
            )

        return sigma_max / sigma_min

    def compute_dops(self) -> tuple[float, float, float, float]:
        G = self._build_G()
        GTG = G.T @ G

        Q = np.linalg.inv(GTG)

        q_xx = float(Q[0, 0])
        q_yy = float(Q[1, 1])
        q_zz = float(Q[2, 2])
        q_tt = float(Q[3, 3])

        HDOP = math.sqrt(q_xx + q_yy)
        VDOP = math.sqrt(q_zz)
        PDOP = math.sqrt(q_xx + q_yy + q_zz)
        GDOP = math.sqrt(q_xx + q_yy + q_zz + q_tt)

        return HDOP, VDOP, PDOP, GDOP
