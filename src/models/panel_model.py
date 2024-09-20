from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pvlib import ivtools


class PanelTech(StrEnum):
    MONO_SI = ("monoSi",)
    MULTI_SI = ("multiSi",)
    POLY_SI = ("polySi",)


@dataclass(kw_only=True, frozen=True)
class PanelDataSheet:
    tech: PanelTech
    "panel technology"
    v_mp: float
    "mpp voltage"
    i_mp: float
    "mpp current"
    v_oc: float
    "open circuit voltage"
    i_sc: float
    "short circuit current"
    alpha_sc: float
    "change in short circuit current due to temperature"
    beta_voc: float
    "change in open circuit voltage due to temperature"
    cells_in_series: int
    "number of cells in series for the panel"
    gamma: float
    "change in power due to temperature change"
    t_ref: int = field(default=25)
    "reference temperature in celsius (25 default)"


@dataclass(kw_only=True, frozen=True)
class PanelModel:
    i_l: float
    "photo-current current, usually equal to short circuit current"
    i_o: float
    "reverse diode saturation current"
    r_s: float
    "series resistance"
    r_sh: float
    "shunt resistance"
    n_ns_vth: float
    "``number of cells in series`` x ``diode ideality factor`` x ``cells thermal voltage``"
    adjust: float


def model_panel(cd: PanelDataSheet) -> PanelModel:
    a = ivtools.sdm.fit_cec_sam(
        celltype=cd.tech,
        gamma_pmp=cd.gamma,
        v_mp=cd.v_mp,
        i_mp=cd.i_mp,
        v_oc=cd.v_oc,
        i_sc=cd.i_sc,
        alpha_sc=cd.alpha_sc,
        beta_voc=cd.beta_voc,
        cells_in_series=cd.cells_in_series,
        temp_ref=cd.t_ref,
    )
    return PanelModel(
        i_l=a[0], i_o=a[1], r_s=a[2], r_sh=a[3], n_ns_vth=a[4], adjust=a[5]
    )


def model2sdm_obj(model: PanelModel, ds: PanelDataSheet) -> dict[str, Any]:
    panel_tech_sdm_str_dict: dict[PanelTech, str] = {
        PanelTech.MONO_SI: "Mono-c-Si",
        PanelTech.MULTI_SI: "Multi-c-Si",
    }
    return {
        "Technology": panel_tech_sdm_str_dict[ds.tech],
        "photocurrent": model.i_l,
        "alpha_sc": ds.alpha_sc,
        "a_ref": model.n_ns_vth,
        "I_L_ref": model.i_l,
        "I_o_ref": model.i_o,
        "R_sh_ref": model.r_sh,
        "R_s": model.r_s,
        "saturation_current": model.i_o,
        "resistance_series": model.r_s,
        "resistance_shunt": model.r_sh,
        "nNsVth": model.n_ns_vth,
        "Adjust": model.adjust,
    }
