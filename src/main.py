import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import pygad
from pvlib import pvsystem, location, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from models.demand import model_demand_data
from models.inverter_model import inverter_model2obj, model_inverter, InverterDataSheet
from models.panel_model import model2sdm_obj, model_panel, PanelDataSheet, PanelTech
from models.turbine_model import calc_wind_power, TurbineModel, estimate_wind_speeds_in_height
from utils.optimization import FitnessParameters, make_fitness_func, calc_lcc, calc_sppw, calc_uspw
from utils.weather import prepare_poa_data

f32arr = npt.NDArray[np.float32]

WIND_DATA_SOURCE = "data/wind/NASA_hourly_wind_50m.csv"
SOLAR_DATA_SOURCE = "data/solar/pgvis_hourly_2005_2016.csv"

RESULTS_PATH = "data/results"
POWER_RESULT_FILE = 'power.csv'

LEAP_YEARS = [2004, 2008, 2012, 2016, 2020]


def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # PANEL MODELING #########################################
    # Trina Solar Vertex Mono 500W
    # ref: https://pages.trinasolar.com/rs/567-KJK-096/images/DE18m%20Datasheet%20update.pdf
    panel_data_sheet = PanelDataSheet(
        tech=PanelTech.MONO_SI,
        v_mp=42.8,
        i_mp=11.69,
        v_oc=51.7,
        i_sc=12.28,
        alpha_sc=0.04,
        beta_voc=-0.25,
        cells_in_series=72,
        gamma=-0.34,
    )

    # Growatt Kaco Blueplanet 105 TLS 3
    # ref: https://www.enfsolar.com/pv/inverter-datasheet/14507
    inverter_data_sheet = InverterDataSheet(
        efficiency=0.985,
        i_dc_max=183,
        max_p_ac=157500,
        p_init=650,
        p_nt=5,
        v_ac=300,
        v_dc_max=1500,
        v_dc_mppt_high=1300,
        v_dc_mppt_low=580,
        v_dc_nominal=900,
    )

    panel_model = model_panel(panel_data_sheet)
    inverter_model = model_inverter(inverter_data_sheet)

    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]
    modules_per_string = 30
    strings_per_inverter = 10
    sys = pvsystem.PVSystem(
        module_parameters=model2sdm_obj(panel_model, panel_data_sheet),
        inverter_parameters=inverter_model2obj(inverter_model),
        temperature_model_parameters=temperature_model_parameters,
        modules_per_string=modules_per_string,
        strings_per_inverter=strings_per_inverter,
    )

    lat, long = 30.588, 56.514
    loc = location.Location(lat, long)

    # FIXME: aoi_model should be set a better value
    # FIXME: spectral_model should be set a better value
    mc = modelchain.ModelChain(
        system=sys, location=loc, aoi_model="no_loss", spectral_model="no_loss"
    )

    end_year = 2016
    start_year = 2005
    years = end_year - start_year + 1
    weather_df = pd.read_csv(SOLAR_DATA_SOURCE)
    year_remainder = len(weather_df) % years
    if year_remainder != 0:
        weather_df = weather_df[:-year_remainder]
    weather_df_poa = prepare_poa_data(weather_df)
    mc.run_model_from_poa(weather_df_poa)
    res = pd.DataFrame(weather_df)
    res["pv_power"] = mc.results.ac

    turbine_model = TurbineModel(
        cut_in_speed=3.5,
        cut_out_speed=25.0,
        rated_power=2.5e6,
        rated_speed=12.0,
    )
    ws = estimate_wind_speeds_in_height(weather_df['wind_speed'].to_numpy(dtype=np.float32), h_ref=10.0, h_new=85.0)
    res['wind_power'] = calc_wind_power(turbine_model, ws)

    home_demand = np.array([])
    for year in range(start_year, end_year + 1):
        if year in LEAP_YEARS:
            home_demand = np.concatenate((home_demand, model_demand_data(True)))
        else:
            home_demand = np.concatenate((home_demand, model_demand_data(False)))
    home_demand = home_demand[:len(res)]
    res['home_demand'] = home_demand

    bootia_demand = np.ones(shape=len(res)) * 100e6
    res['bootia_demand'] = bootia_demand

    power_res_path = os.path.join(RESULTS_PATH, POWER_RESULT_FILE)
    res.to_csv(power_res_path, float_format="{:.3f}".format, index=False)
    print(f'result file saved in: {power_res_path}')

    pv_generation_per_array = np.array(res["pv_power"])
    turbine_generation_per_turbine = np.array(res["wind_power"])
    pv_panel_price = 100
    inverter_price = 750
    pv_array_cost = pv_panel_price * modules_per_string * strings_per_inverter + 1 * inverter_price

    optimize_home_demand(home_demand, pv_array_cost, pv_generation_per_array, turbine_generation_per_turbine, years)
    optimize_bootia_demand(bootia_demand, pv_array_cost, pv_generation_per_array, turbine_generation_per_turbine, years)


def optimize_home_demand(home_demand: f32arr, pv_array_cost: float, pv_generation_per_array: f32arr,
                         turbine_generation_per_turbine: f32arr, years: int):
    batteries_min = int(0)
    batteries_max = int(10e3)
    pv_min = 0
    pv_max = 300
    turbines_min = 0
    turbines_max = 5
    fitness_parameters = FitnessParameters(
        electricity_price_per_wh=0.3e-3,
        battery_price_per_wh=100e-3,
        pv_generation_per_array=pv_generation_per_array,
        max_pv_arrays=pv_max,
        min_pv_arrays=pv_min,
        pv_array_cost=pv_array_cost,
        max_turbines=turbines_max,
        min_turbines=turbines_min,
        demand=home_demand,
        inflation_rate=0.05,
        min_batteries_cap_wh=batteries_min,
        max_batteries_cap_wh=batteries_max,
        years=years,
        turbine_cost=2.4e6,
        batteries_cap_power_ratio=5,
        turbine_generation_per_turbine=turbine_generation_per_turbine,
        pv_array_yearly_cost=2000,
        turbine_yearly_cost=80e3
    )
    num_generations = 100
    optimization_output = os.path.join(RESULTS_PATH,
                                       f'opti_home_pv({pv_min}-{pv_max})_tb({turbines_min}-{turbines_max})_bt({batteries_min}-{batteries_max}')
    grid_output = os.path.join(RESULTS_PATH, f'home_grid.csv')
    optimize(fitness_parameters, num_generations, optimization_output, grid_output)


def optimize_bootia_demand(bootia_demand: f32arr, pv_array_cost: float, pv_generation_per_array: f32arr,
                           turbine_generation_per_turbine: f32arr, years: int):
    batteries_min = int(0)
    batteries_max = int(10000e3)
    pv_min = 5000
    pv_max = 8000
    turbines_min = 0
    turbines_max = 5
    fitness_parameters = FitnessParameters(
        electricity_price_per_wh=calc_electricity_price_with_natural_gas(),
        battery_price_per_wh=100e-3,
        pv_generation_per_array=pv_generation_per_array,
        max_pv_arrays=pv_max,
        min_pv_arrays=pv_min,
        pv_array_cost=pv_array_cost,
        max_turbines=turbines_max,
        min_turbines=turbines_min,
        demand=bootia_demand,
        inflation_rate=0.05,
        min_batteries_cap_wh=batteries_min,
        max_batteries_cap_wh=batteries_max,
        years=years,
        turbine_cost=2.4e6,
        batteries_cap_power_ratio=5,
        turbine_generation_per_turbine=turbine_generation_per_turbine,
        pv_array_yearly_cost=2000,
        turbine_yearly_cost=80e3
    )
    num_generations = 50
    optimization_output = os.path.join(RESULTS_PATH,
                                       f'opti_bootia_pv({pv_min}-{pv_max})_tb({turbines_min}-{turbines_max})_bt({batteries_min}-{batteries_max})')
    grid_output = os.path.join(RESULTS_PATH, f'bootia_grid.csv')
    optimize(fitness_parameters, num_generations, optimization_output, grid_output)


def optimize(p: FitnessParameters, num_generations: int, optimization_output: str, grid_output: str):
    fitness_func = make_fitness_func(p)
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=8,
        num_genes=3,
        init_range_low=0.0,
        init_range_high=1.0,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=40,
        gene_space={"low": 0, "high": 1.0},
    )
    ga_instance.run()
    ga_instance.save(optimization_output)
    print(f'optimization data was saved on: {optimization_output}')
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    min_lcc = 1 / fitness_func(None, solution, None)
    res = calc_lcc(p, solution, calc_uspw(p.inflation_rate, p.years), calc_sppw(p.inflation_rate, p.years))
    df = pd.DataFrame({'grid': res.grid_electricity})
    df.to_csv(grid_output, float_format='{:.3f}'.format, index=False)
    print(f"Predicted output based on the best solution : {min_lcc}")


def calc_electricity_price_with_natural_gas():
    lcv_g = 35.8e6  # J/m3
    cost_g_per_m3 = 0.3  # $/m3
    plant_efficiency = 0.35
    energy_per_m3 = lcv_g * plant_efficiency
    return 1 * 3600 * cost_g_per_m3 / energy_per_m3  # Wh


if __name__ == "__main__":
    main()
