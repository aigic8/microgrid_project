from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

f32arr = npt.NDArray[np.float32]


@dataclass(kw_only=True, frozen=True)
class TurbineModel:
    cut_in_speed: float
    cut_out_speed: float
    rated_speed: float
    rated_power: float


def calc_wind_power(m: TurbineModel, wind_speeds: f32arr):
    res = wind_speeds.copy()
    res[res < m.cut_in_speed] = 0.0
    res[res > m.cut_out_speed] = 0.0
    res[res >= m.rated_speed] = m.rated_power
    cond = (res >= m.cut_in_speed) & (res < m.rated_speed)
    res[cond] = m.rated_power * (res[cond] ** 3 - m.cut_in_speed ** 3) / (m.rated_speed ** 3 - m.cut_in_speed ** 3)
    return res


def estimate_wind_speeds_in_height(wind_speeds: f32arr, *, h_ref: float, h_new: float):
    return wind_speeds * (h_new / h_ref) ** (1 / 7)
