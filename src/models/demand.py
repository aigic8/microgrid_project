import numpy as np
import numpy.typing as npt

f32arr = npt.NDArray[np.float32]

# Modeling demand data based on:
# https://sci-hub.st/10.1080/15567240903330384

HOURLY_AVERAGE_LOAD = 1523.04e3
DAILY_AVERAGE_LOAD = 24 * HOURLY_AVERAGE_LOAD  # kWH

# from June
SUMMER_DAILY_LOAD_TREND = np.array([
    00.5, 00.5, 00.5, 00.5, 01.5, 01.5,
    00.5, 05.0, 05.0, 06.2, 06.2, 06.5,
    06.5, 11.0, 11.0, 05.0, 05.0, 05.0,
    04.0, 08.5, 08.5, 07.0, 01.5, 00.5,
], dtype=np.float32)

# from April
SPRING_DAILY_LOAD_TREND = np.array([
    00.5, 00.5, 00.5, 00.5, 01.5, 01.5,
    00.5, 01.8, 02.0, 04.0, 04.0, 03.5,
    04.0, 06.0, 05.8, 04.0, 04.0, 04.0,
    04.0, 04.0, 04.0, 02.0, 01.5, 00.5,
], dtype=np.float32)

# from April
AUTUMN_DAILY_LOAD_TREND = np.array([
    00.5, 00.5, 00.5, 00.5, 01.5, 01.5,
    00.5, 01.8, 02.0, 04.0, 04.0, 03.5,
    04.0, 06.0, 05.8, 04.0, 04.0, 04.0,
    04.0, 04.0, 04.0, 02.0, 01.5, 00.5,
], dtype=np.float32)

# from December
WINTER_DAILY_LOAD_TREND = np.array([
    00.5, 00.5, 00.5, 00.5, 01.5, 01.5,
    00.5, 00.5, 00.5, 02.0, 02.0, 01.0,
    01.0, 01.0, 00.5, 00.5, 00.5, 00.5,
    04.0, 05.0, 04.5, 03.5, 01.5, 00.5,
], dtype=np.float32)

SUMMER_INDEX = 0
SPRING_INDEX = 1
WINTER_INDEX = 2
AUTUMN_INDEX = 3
SEASONS_LOAD_TREND = np.array([1.0, 1.7, 2.5, 1.5], dtype=np.float32)


def model_demand_data(is_leap_year: bool = False):
    season_trends = normalize_mean(SEASONS_LOAD_TREND)
    summer_daily_load = normalize_sum(SUMMER_DAILY_LOAD_TREND) * DAILY_AVERAGE_LOAD * season_trends[SUMMER_INDEX]
    spring_daily_load = normalize_sum(SPRING_DAILY_LOAD_TREND) * DAILY_AVERAGE_LOAD * season_trends[SPRING_INDEX]
    winter_daily_load = normalize_sum(WINTER_DAILY_LOAD_TREND) * DAILY_AVERAGE_LOAD * season_trends[WINTER_INDEX]
    autumn_daily_load = normalize_sum(AUTUMN_DAILY_LOAD_TREND) * DAILY_AVERAGE_LOAD * season_trends[AUTUMN_INDEX]

    feb_days = 29 if is_leap_year else 28
    jan_feb_load = np.repeat(summer_daily_load, 31 + feb_days)
    spring_load = np.repeat(spring_daily_load, 31 + 30 + 31)
    winter_load = np.repeat(winter_daily_load, 30 + 31 + 31)
    autumn_load = np.repeat(autumn_daily_load, 30 + 31 + 30)
    dec_load = np.repeat(summer_daily_load, 31)

    yearly_load = np.concatenate((jan_feb_load, spring_load, winter_load, autumn_load, dec_load))

    hourly_randomness = 0.15
    h_random = np.random.uniform(low=1.0 - hourly_randomness, high=1.0 + hourly_randomness, size=len(yearly_load))
    return yearly_load * h_random


def normalize_sum(data: f32arr):
    c = 1 / np.sum(data)
    return c * data


def normalize_mean(data: f32arr):
    c = 1 / np.mean(data)
    return c * data
