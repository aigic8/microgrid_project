from dataclasses import dataclass


@dataclass(kw_only=True, frozen=True)
class InverterModel:
    """data class representing inverter model
    based on https://pvpmc.sandia.gov/modeling-guide/dc-to-ac-conversion/sandia-inverter-model/"""

    p_ac0: float
    "maximum AC power output"
    p_dc0: float
    "maximum DC power input"
    p_c0: float
    "minimum power required for the inverter to start functioning"
    v_dc0: float
    "MPPT inverter DC voltage"
    v_ac: float
    "inverter AC output voltage"
    p_nt: float
    """inverter power usage in a time frame, 
    the time frame is dependant on the frequency of time provided in the df.
    For example, if the data is hourly, this value would be energy usage in an hour."""
    c_0: float
    "c_0 parameter"
    c_1: float
    "c_1 parameter"
    c_2: float
    "c_2 parameter"
    c_3: float
    "c_3 parameter"
    v_dc_max: float
    "max DC input voltage"
    i_dc_max: float
    "max DC input current"
    mppt_low: float
    "min MPPT voltage range"
    mppt_high: float
    "max MPPT voltage range"


@dataclass(kw_only=True, frozen=True)
class InverterDataSheet:
    """data class representing inverter data sheet"""

    max_p_ac: float
    "maximum power output in AC"
    v_ac: float
    "rated voltage for AC"
    v_dc_nominal: float
    "nominal voltage for DC"
    v_dc_max: float
    "maximum voltage for DC"
    v_dc_mppt_low: float
    "minimum MPPT voltage"
    v_dc_mppt_high: float
    "maximum MPPT voltage"
    i_dc_max: float
    "maximum DC current"
    p_init: float
    "minimum power required for the inverter to start working"
    p_nt: float
    """inverter power usage in a time frame, 
    the time frame is dependant on the frequency of time provided in the df.
    For example, if the data is hourly, this value would be energy usage in an hour."""
    efficiency: float
    "inverter efficiency"


def model_inverter(ds: InverterDataSheet) -> InverterModel:
    # FIXME: find a way to calculate c values
    return InverterModel(
        v_ac=ds.v_ac,
        p_ac0=ds.max_p_ac,
        p_dc0=ds.max_p_ac / ds.efficiency,
        p_c0=ds.p_init,
        v_dc0=ds.v_dc_nominal,
        p_nt=ds.p_nt,
        c_0=0.0,
        c_1=0.0,
        c_2=0.0,
        c_3=0.0,
        v_dc_max=ds.v_dc_max,
        i_dc_max=ds.i_dc_max,
        mppt_low=ds.v_dc_mppt_low,
        mppt_high=ds.v_dc_mppt_high,
    )


def inverter_model2obj(m: InverterModel) -> dict[str, float]:
    return {
        "Vac": m.v_ac,
        "Pso": m.p_c0,
        "Paco": m.p_ac0,
        "Pdco": m.p_dc0,
        "Vdco": m.v_dc0,
        "C0": m.c_0,
        "C1": m.c_1,
        "C2": m.c_2,
        "C3": m.c_3,
        "Pnt": m.p_nt,
        "Vdcmax": m.v_dc_max,
        "Idcmax": m.i_dc_max,
        "Mppt_low": m.mppt_low,
        "Mppt_high": m.mppt_high,
    }
