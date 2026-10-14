"""SCI profiles: ISNCSCI motor scores / AIS grade -> per-muscle voluntary capacity, atrophy,
passive stiffness and spasticity, plus ready-made presets.

The mapping is deliberately explicit so you can replace it with your participants'
exam sheets (motor_scores per key-muscle group and side) and measured MAS.
"""
from __future__ import annotations

import copy
import re

import mujoco
import numpy as np

from ..config import (GROUP_SEGMENT, MOTOR_SCORE_TO_CAPACITY, MUSCLE_GROUPS, SpasticityParams,
                      SubjectParams)
from .muscles import MuscleModel

_AIS_DEFAULT_SCORE = {"A": 0, "B": 0, "C": 2, "D": 4, "E": 5}


def _nli_to_lumbosacral_index(nli: str) -> int:
    """Return the last *normal* lumbosacral root index (L1=1 ... S1=6). Cervical/thoracic -> 0."""
    if not nli or nli.lower() == "none":
        return 99
    mt = re.match(r"([CTLS])(\d+)", nli.upper())
    if not mt:
        raise ValueError(f"bad NLI '{nli}'")
    reg, k = mt.group(1), int(mt.group(2))
    if reg in ("C", "T"):
        return 0
    if reg == "L":
        return k
    return 5 + k  # S1 -> 6


def group_capacity(subject: SubjectParams) -> dict:
    """{(group, side): capacity in [0,1]} = max voluntary activation."""
    out = {}
    for side in ("l", "r"):
        for g in MUSCLE_GROUPS:
            if not subject.sci:
                out[(g, side)] = 1.0
                continue
            ms = subject.motor_scores.get(side, {}).get(g)
            if ms is None:
                last_normal = _nli_to_lumbosacral_index(subject.nli)
                ms = 5 if GROUP_SEGMENT[g] <= last_normal else _AIS_DEFAULT_SCORE[subject.ais.upper()]
            out[(g, side)] = MOTOR_SCORE_TO_CAPACITY[int(round(ms))]
    return out


def muscle_capacity(mm: MuscleModel, subject: SubjectParams) -> np.ndarray:
    gc = group_capacity(subject)
    cap = np.ones(mm.n)
    for i in range(mm.n):
        gs = mm.group_of.get(i, [])
        if gs:
            cap[i] = max(gc[(g, mm.side[i])] for g in gs)
    return cap


def apply_to_model(m: mujoco.MjModel, mm: MuscleModel, subject: SubjectParams, cap: np.ndarray):
    """In-place: disuse atrophy (Fmax) and passive-stiffness changes on the compiled model."""
    for k, i in enumerate(mm.ids):
        scale = 1.0
        if subject.sci:
            # atrophy grows as voluntary capacity falls (complete paralysis -> subject.atrophy)
            scale = subject.atrophy + (1 - subject.atrophy) * cap[k]
        for arr in (m.actuator_gainprm, m.actuator_biasprm):
            arr[i, 2] = mm.F0[k] * scale
        ps = 1.0
        for g in mm.group_of.get(k, []):
            ps = max(ps, subject.passive_stiffness_scale.get(g, 1.0))
        if ps != 1.0:
            m.actuator_biasprm[i, 7] *= ps
            m.actuator_gainprm[i, 7] *= ps
    mm.F0 = m.actuator_gainprm[mm.ids, 2].copy()


# ------------------------------------------------------------------------------------------
PRESETS = {}


def _preset(fn):
    PRESETS[fn.__name__] = fn
    return fn


@_preset
def healthy():
    return SubjectParams(name="healthy")


@_preset
def sci_t10_aisA():
    """Complete thoracic SCI (typical Indego user): no leg volition, moderate spasticity."""
    return SubjectParams(
        name="sci_t10_aisA", sci=True, nli="T10", ais="A", atrophy=0.6, trunk_strength=0.6,
        passive_stiffness_scale={"ankle_plantarflexors": 1.3, "knee_flexors": 1.2, "hip_flexors": 1.2},
        spasticity=SpasticityParams(mas={"knee_extensors": 1.5, "ankle_plantarflexors": 2.0, "hip_adductors": 1.5,
                                         "hip_flexors": 1.0, "knee_flexors": 1.0},
                                    clonus_gain_boost=1.2, spasm_rate_per_min=0.5, spasm_stretch_trigger=0.2))


@_preset
def sci_t5_aisA():
    """Higher complete lesion (lower edge of Indego indication): weak trunk, stronger spasticity."""
    s = sci_t10_aisA()
    s.name, s.nli, s.trunk_strength = "sci_t5_aisA", "T5", 0.3
    s.spasticity.mas = {"knee_extensors": 2.0, "ankle_plantarflexors": 2.5, "hip_adductors": 2.0,
                        "hip_flexors": 1.5, "knee_flexors": 1.5, "hip_extensors": 1.0}
    s.spasticity.spasm_rate_per_min = 1.0
    return s


@_preset
def sci_l2_aisC():
    """Incomplete: some hip flexion / knee extension, weak distal muscles, spastic plantarflexors."""
    ms = {"hip_flexors": 3, "hip_adductors": 2, "knee_extensors": 2, "ankle_dorsiflexors": 1,
          "hip_abductors": 1, "hip_extensors": 1, "knee_flexors": 1, "ankle_plantarflexors": 1}
    return SubjectParams(
        name="sci_l2_aisC", sci=True, nli="L2", ais="C", atrophy=0.75, trunk_strength=0.9,
        motor_scores={"l": dict(ms), "r": dict(ms, hip_flexors=4, knee_extensors=3)},
        passive_stiffness_scale={"ankle_plantarflexors": 1.5},
        spasticity=SpasticityParams(mas={"ankle_plantarflexors": 2.0, "knee_extensors": 1.5, "hip_adductors": 1.0},
                                    clonus_gain_boost=1.8, spasm_rate_per_min=0.2))


@_preset
def sci_t12_aisD():
    """Incomplete ambulatory: anti-gravity strength in most groups, mild spasticity."""
    ms = {g: 4 for g in MUSCLE_GROUPS}
    ms.update({"ankle_dorsiflexors": 3, "ankle_plantarflexors": 3, "hip_abductors": 3})
    return SubjectParams(
        name="sci_t12_aisD", sci=True, nli="T12", ais="D", atrophy=0.85, trunk_strength=0.9,
        motor_scores={"l": dict(ms), "r": dict(ms)},
        spasticity=SpasticityParams(mas={"ankle_plantarflexors": 1.5, "knee_extensors": 1.0}, clonus_gain_boost=1.3))


def get_preset(name: str) -> SubjectParams:
    return copy.deepcopy(PRESETS[name]())
