import pandas as pd


def prepare_poa_data(df: pd.DataFrame) -> pd.DataFrame:
    """prepares PGVis weather data for the function ``run_model_from_poa``.
    For more info read: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.run_model_from_poa.html#pvlib.modelchain.ModelChain.run_model_from_poa"""

    # NOTE: It is unknown weather this is correct or not
    # It is not specified weather PVLib will calculate the ground diffuse itself or should I add it to diffuse
    res = pd.DataFrame(df[["wind_speed", "temp_air"]])
    res["poa_diffuse"] = df["poa_sky_diffuse"]
    res["poa_global"] = df["poa_direct"] + df["poa_sky_diffuse"]
    res["poa_direct"] = df["poa_direct"]
    res["t"] = df["time"]
    return res
