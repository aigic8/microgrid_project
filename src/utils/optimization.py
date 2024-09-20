from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

f32arr = npt.NDArray[np.float32]

PV_ARRAYS_INDEX = 0
TURBINES_INDEX = 1
BATTERIES_INDEX = 2

PV_PANEL_LIFE_TIME = 25  # years
TURBINE_LIFE_TIME = 25  # years
BATTERIES_LIFE_TIME = 25  # years


@dataclass(kw_only=True, frozen=True)
class FitnessParameters:
    inflation_rate: float
    pv_array_cost: float
    pv_array_yearly_cost: float
    turbine_cost: float
    turbine_yearly_cost: float
    electricity_price_per_wh: float
    battery_price_per_wh: float
    min_pv_arrays: int
    max_pv_arrays: int
    min_turbines: int
    max_turbines: int
    min_batteries_cap_wh: int
    max_batteries_cap_wh: int
    batteries_cap_power_ratio: int
    pv_generation_per_array: f32arr
    turbine_generation_per_turbine: f32arr
    demand: f32arr
    years: int


def make_fitness_func(p: FitnessParameters):
    if len(p.pv_generation_per_array) != len(p.demand):
        raise ValueError("pv generation and demand length does not match")
    if len(p.turbine_generation_per_turbine) != len(p.demand):
        raise ValueError("pv generation and demand length does not match")

    sppw_arr = calc_sppw(p.inflation_rate, p.years)
    uspw = calc_uspw(p.inflation_rate, p.years)

    def fitness_func(_, s, __):
        res = calc_lcc(p, s, uspw, sppw_arr)

        # we want to minimize the LCC
        fitness = 1 / res.lcc
        return fitness

    return fitness_func


@dataclass(kw_only=True, frozen=True)
class LCCResult:
    c_pv_init: float
    c_pv_yearly: float
    c_turbine_init: float
    c_turbine_yearly: float
    delta_generation_consumption: f32arr
    grid_electricity: f32arr
    grid_electricity_per_year: f32arr
    grid_cost_per_year: f32arr
    lcc: float


def calc_lcc(p: FitnessParameters, s, uspw: float, sppw_arr: f32arr):
    pv_arrays_count = np.floor(
        p.min_pv_arrays + s[PV_ARRAYS_INDEX] * (p.max_pv_arrays - p.min_pv_arrays)
    )
    c_pv_yearly = p.pv_array_yearly_cost * uspw
    c_pv_init = (p.years / PV_PANEL_LIFE_TIME) * pv_arrays_count * p.pv_array_cost
    c_pv = c_pv_init + c_pv_yearly

    turbine_count = np.floor(
        p.min_turbines + s[TURBINES_INDEX] * (p.max_turbines - p.min_turbines)
    )
    c_turbine_yearly = p.turbine_yearly_cost * uspw
    c_turbine_init = (p.years / TURBINE_LIFE_TIME) * turbine_count * p.turbine_cost
    c_turbine = c_turbine_init + c_turbine_yearly

    batteries_cap = np.floor(
        p.min_batteries_cap_wh
        + s[BATTERIES_INDEX] * (p.max_batteries_cap_wh - p.min_batteries_cap_wh)
    )
    c_battery = (p.years / BATTERIES_LIFE_TIME) * batteries_cap * p.battery_price_per_wh

    delta_generation_consumption = (
            pv_arrays_count * p.pv_generation_per_array +
            turbine_count * p.turbine_generation_per_turbine -
            p.demand
    )
    batteries_delta_e, _ = calculate_batteries_effect(
        generation_consumption=delta_generation_consumption,
        batteries_cap=batteries_cap,
        batteries_max_power=float(p.max_batteries_cap_wh) / float(p.batteries_cap_power_ratio),
    )

    grid_electricity_consumption = -1 * np.minimum(delta_generation_consumption + batteries_delta_e, 0)
    grid_electricity_consumption_per_year = np.sum(np.array_split(grid_electricity_consumption, p.years), axis=1)
    grid_cost_per_year = grid_electricity_consumption_per_year * p.electricity_price_per_wh
    c_g = np.sum(grid_cost_per_year * sppw_arr)

    lcc = c_pv + c_battery + c_turbine + c_g
    return LCCResult(
        c_turbine_yearly=c_turbine_yearly,
        c_turbine_init=c_turbine_init,
        c_pv_yearly=c_pv_yearly,
        c_pv_init=c_pv_init,
        delta_generation_consumption=delta_generation_consumption,
        grid_electricity=grid_electricity_consumption,
        grid_electricity_per_year=grid_electricity_consumption_per_year,
        grid_cost_per_year=grid_cost_per_year,
        lcc=lcc,
    )


def calculate_batteries_effect(
        *,
        generation_consumption: np.ndarray,
        batteries_cap: float,
        batteries_max_power: float,
):
    delta_arr = np.zeros(len(generation_consumption))
    batteries_charge = 0.0
    charge_arr = np.zeros(len(generation_consumption))
    i = 0
    for item in generation_consumption:
        # charge
        if item > 0.0:
            delta = np.max([batteries_cap - batteries_charge, item])
            batteries_charge += delta
            delta_arr[i] = -1 * delta
            charge_arr[i] = batteries_charge
        # discharge
        elif item < 0.0:
            delta = np.min([batteries_charge, item, batteries_max_power])
            batteries_charge -= delta
            delta_arr[i] = delta
            charge_arr[i] = batteries_charge
        else:
            delta_arr[i] = 0.0
            charge_arr[i] = batteries_charge
        i += 1
    return delta_arr, charge_arr


def calc_sppw(inflation_rate: float, years: int):
    return np.array([(1 + inflation_rate) ** (i + 1) for i in range(0, years)])


def calc_uspw(inflation_rate: float, years: int):
    return ((1 + inflation_rate) ** years) / (inflation_rate * (1 + inflation_rate))
