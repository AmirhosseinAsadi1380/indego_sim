"""Shared setup for the walking diagnostics. A config is a JSON dict:
    {"subj": "sci_t10_aisA", "subject": {SubjectParams fields to override, e.g. "motor_scores"}, "walk_at": 2.0,
     "fsm": {IndegoFSMParams fields}, "sup": {SupportParams fields, "therapist": {TherapistParams fields},
             "planarizer": {PlanarizerParams fields}},
     "vol": {VolitionParams fields}, "sens": {SensorParams fields}, "exo": {ExoParams fields},
     "sim": {SimConfig fields}, "ctrl": "fsm" | "dynamic"}. With "sim": {"planar": true} the FSM's lateral weight
     check is off unless set (there is no sideways shift to detect). "ctrl": "dynamic" = DynamicWalkController, its
     parameters in "fsm"."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from indego_sim.config import (ExoParams, PlanarizerParams, SensorParams, SimConfig, SupportParams,  # noqa: E402
                               TherapistParams, VolitionParams, load_yaml)
from indego_sim.controllers.indego_fsm import IndegoFSMController  # noqa: E402
from indego_sim.human.sci import get_preset  # noqa: E402
from indego_sim.sim import HumanExoSim  # noqa: E402


def make_sim(cfg: dict, seed: int = 0, log_rate_hz: float = 50.0):
    sup = SupportParams(kind=cfg.get("sup", {}).get("kind", "crutches"))
    for k, v in cfg.get("sup", {}).items():
        nested = {"therapist": TherapistParams, "planarizer": PlanarizerParams}
        setattr(sup, k, nested[k](**v) if k in nested else v)
    subject = load_yaml(get_preset(cfg.get("subj", "sci_t10_aisA")), cfg.get("subject", {}))
    sim = HumanExoSim(subject=subject, exo=ExoParams(**cfg.get("exo", {})),
                      support=sup, volition=load_yaml(VolitionParams(), cfg.get("vol", {})),
                      sensors=SensorParams(**cfg.get("sens", {})), sim=SimConfig(**cfg.get("sim", {})),
                      schedule=[(cfg.get("walk_at", 2.0), "walk")], seed=seed, log_muscles=False,
                      log_rate_hz=log_rate_hz)
    fsm_kw = dict(cfg.get("fsm", {}))
    if cfg.get("sim", {}).get("planar"):
        fsm_kw.setdefault("lat_dy", 0.0)
    ctrl = cfg.get("ctrl", "fsm")
    if ctrl == "dynamic":
        from indego_sim.controllers.dynamic_walk import DynamicWalkController
        fsm = DynamicWalkController.from_model(sim.m, **fsm_kw)
    elif ctrl in ("transparent", "aan", "zero"):
        # controllers for people who walk themselves; wrapped so the diagnostics can read an FSM-like state
        from indego_sim.controllers.transparent import AssistAsNeeded, TransparentController
        fsm = IndegoFSMController.from_model(sim.m, **fsm_kw)
        inner = {"transparent": TransparentController, "aan": AssistAsNeeded}.get(ctrl)
        fsm.inner = inner() if inner else None
        est_only = fsm.__call__

        def call(obs, _f=fsm, _e=est_only):
            _e(obs)                                   # keeps the estimator / state bookkeeping running
            return _f.inner(obs) if _f.inner is not None else obs["q"] * 0.0
        fsm.__call__ = call
        fsm = _Callable(fsm)
    else:
        fsm = IndegoFSMController.from_model(sim.m, **fsm_kw)
    obs = sim.reset(settle=1.5)
    return sim, fsm, obs


class _Callable:
    """Wraps a controller object whose __call__ was replaced on the instance (Python looks __call__ up on the type)."""

    def __init__(self, obj):
        self.__dict__["_obj"] = obj

    def __call__(self, obs):
        return self._obj.__dict__["__call__"](obs)

    def __getattr__(self, k):
        return getattr(self._obj, k)

    def __setattr__(self, k, v):
        setattr(self._obj, k, v)
