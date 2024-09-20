import numpy as np
import pandas as pd

# TODO: add instructions on how to change the NASA csv file


WIND_DATA_SOURCE = "data/wind/NASA_hourly_wind_50m_cleaned.csv"
WIND_DATA_OUTPUT = "data/wind/NASA_hourly_wind_50m.csv"
ROUGHNESS_LENGTH = 0.15


def main():
    df = pd.read_csv(WIND_DATA_SOURCE)
    res = pd.DataFrame(df[["PS", "T2M", "WS50M"]])
    res["roughness_length"] = np.ones(len(res)) * ROUGHNESS_LENGTH
    res.rename(
        columns={"PS": "pressure", "T2M": "temperature", "WS50M": "wind_speed"},
        inplace=True,
    )
    # convert celsius to kelvin
    res["temperature"] += 273.15

    years = df["YEAR"]
    months = df["MO"]
    days = df["DY"]
    hours = df["HR"]

    datetime_arr: list[str] = []
    for i in range(0, len(res)):
        datetime_arr.append(
            f"{years[i]}-{str(months[i]).zfill(2)}-{str(days[i]).zfill(2)} {str(hours[i]).zfill(2)}:00:00+03:30"
        )
    res["variable_name"] = datetime_arr
    heights_row = {
        "variable_name": "height",
        "wind_speed": 50,
        "pressure": 0,
        "temperature": 2,
        "roughness_length": 50,
    }
    res = pd.concat([pd.DataFrame([heights_row]), res])

    res.to_csv(WIND_DATA_OUTPUT, index=False, float_format="{:.2f}".format)
    print(f"exported file to: {WIND_DATA_OUTPUT}")


if __name__ == "__main__":
    main()
