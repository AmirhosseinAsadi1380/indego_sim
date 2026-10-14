"""Grey-box identification of an Indego joint drive from bench data.

Model identified (joint side, i.e. what MuJoCo sees after the gear):

    J*qdd + b*qd + Fc*sgn(qd) [+ Stribeck] + mgl*sin(q) + c0 = N*K*i(t - tau_d)

with i the motor current (measured, or commanded current-loop output). It maps 1:1 onto
`MotorParams`: rotor_inertia = (J - J_link)/N^2, viscous = b/N^2, coulomb = Fc/N,
static/stribeck_vel from the friction map, cmd_delay = tau_d, te from the current loop,
thermal_R/C from a heating run.

Typical workflow (docs/CALIBRATION.md has the bench protocol):

    seg = ...  # your logged arrays at 200 Hz
    fr = friction_map(v_levels, tau_levels)              # constant-velocity sweeps
    rb = fit_rigid_body(t, q, tau_from_current, ...)     # PRBS / multisine, segment hanging
    dl = estimate_delay(cmd, current, fs)                # torque steps
    fr_frf = frf(u, qd, fs) -> fit_frf_mass_damper(...)  # cross-check in the frequency domain
    th = fit_thermal(t, current, T_winding, R)           # optional, if winding temperature is logged
    params = to_motor_params(rb, fr, dl, ...)            # -> MotorParams / YAML for the simulator
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal


# ---------------------------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------------------------
def derivatives(q: np.ndarray, fs: float, window: float = 0.07, poly: int = 3):
    """Zero-phase Savitzky-Golay position/velocity/acceleration (offline only)."""
    w = max(poly + 2, int(round(window * fs)) | 1)
    qs = signal.savgol_filter(q, w, poly)
    qd = signal.savgol_filter(q, w, poly, deriv=1, delta=1.0 / fs)
    qdd = signal.savgol_filter(q, w, poly, deriv=2, delta=1.0 / fs)
    return qs, qd, qdd


def estimate_delay(u: np.ndarray, y: np.ndarray, fs: float, max_delay: float = 0.05) -> float:
    """Pure delay between a command u and a response y that follows it with ~unit gain at high frequency
    (e.g. commanded vs measured current) from the cross-correlation of their increments, with parabolic
    sub-sample interpolation."""
    du, dy = np.diff(u), np.diff(y)
    du = du - du.mean()
    dy = dy - dy.mean()
    n = int(max_delay * fs) + 1
    lags = np.arange(0, n)
    c = np.array([np.dot(du[: len(du) - L], dy[L:]) for L in lags])
    k = int(np.argmax(c))
    if 0 < k < n - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        off = 0.5 * (c[k - 1] - c[k + 1]) / den if den != 0 else 0.0
    else:
        off = 0.0
    return float((k + off) / fs)


# ---------------------------------------------------------------------------------------------
# friction from constant-velocity sweeps
# ---------------------------------------------------------------------------------------------
@dataclass
class FrictionFit:
    coulomb: float
    static: float
    stribeck_vel: float
    viscous: float
    offset: float
    rmse: float


def steady_state_points(t, v_ref, tau, min_dur: float = 0.25, trim: float = 0.2, v_meas=None):
    """(velocity, torque) means over constant-velocity segments of a sweep.

    v_ref: the commanded velocity (piecewise constant, e.g. np.gradient of the position reference used with
    excitation.triangle_velocity). Segments are found on v_ref (robust to encoder quantisation) and the
    first/last `trim` fraction of each is discarded (acceleration transients). If v_meas is given, its
    segment mean is reported instead of v_ref."""
    t, v_ref, tau = map(np.asarray, (t, v_ref, tau))
    pts = []
    i, n = 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and abs(v_ref[j + 1] - v_ref[i]) <= 1e-6 + 1e-3 * abs(v_ref[i]):
            j += 1
        if t[j] - t[i] >= min_dur and abs(v_ref[i]) > 1e-3:
            k = int(trim * (j - i))
            a, b = i + k, j - k
            v = float(np.mean(v_meas[a:b])) if v_meas is not None else float(v_ref[i])
            pts.append((v, float(np.mean(tau[a:b]))))
        i = j + 1
    return np.array(pts)


def friction_map(v: np.ndarray, tau: np.ndarray, stribeck: bool = False, v_max: float = None) -> FrictionFit:
    """Fit tau_f(v) = sgn(v)*(Fc + (Fs-Fc)*exp(-(v/vs)^2)) + b*v + c0 to steady-state points.
    In the bench sweep, gravity must be removed from tau first (or sweep a horizontal/balanced link).
    Points above |v_max| are ignored (at high speed the position loop no longer reaches steady state).
    The Stribeck term is opt-in: it needs several well-tracked points below ~0.1 rad/s (joint side)."""
    v, tau = np.asarray(v, float), np.asarray(tau, float)
    if v_max is not None:
        keep = np.abs(v) <= v_max
        v, tau = v[keep], tau[keep]
    A = np.column_stack([np.sign(v), v, np.ones_like(v)])
    fc0, b0, c00 = np.linalg.lstsq(A, tau, rcond=None)[0]
    r0 = tau - A @ [fc0, b0, c00]
    lin = FrictionFit(float(fc0), float(fc0), 0.0, float(b0), float(c00), float(np.sqrt(np.mean(r0 ** 2))))
    if not stribeck or len(v) < 8:
        return lin

    def model(p, v):                                  # p = Fc, Fs-Fc (>=0), vs, b, c0
        fc, dfs, vs, b, c0 = p
        return np.sign(v) * (fc + dfs * np.exp(-(v / vs) ** 2)) + b * v + c0
    p0 = [max(fc0, 1e-3), 0.1 * abs(fc0) + 1e-3, 0.05, max(b0, 0.0), c00]
    sol = optimize.least_squares(lambda p: model(p, v) - tau, p0,
                                 bounds=([0, 0, 5e-3, 0, -np.inf], [np.inf, np.inf, 0.5, np.inf, np.inf]))
    fc, dfs, vs, b, c0 = sol.x
    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    if rmse > 0.9 * lin.rmse and dfs > 0:             # Stribeck term not supported by the data
        return lin
    return FrictionFit(float(fc), float(fc + dfs), float(vs), float(b), float(c0), rmse)


# ---------------------------------------------------------------------------------------------
# time-domain rigid-body + friction regression
# ---------------------------------------------------------------------------------------------
@dataclass
class RigidBodyFit:
    inertia: float
    viscous: float
    coulomb: float
    mgl: float
    offset: float
    delay: float
    r2: float
    rmse: float


def fit_rigid_body(t, q, tau, fs: float = None, gravity: bool = True, delays=None, v_eps: float = 0.02,
                   cutoff: float = 12.0, friction: "FrictionFit" = None, v_min: float = 0.3) -> RigidBodyFit:
    """Least squares on J*qdd + b*qd + Fc*tanh(qd/v_eps) + mgl*sin(q) + c0 = tau(t - d), scanning the delay d.

    tau: joint torque produced by the drive, best = N*K*i_measured (or the command if no current is
    logged, then the fitted delay also contains the current loop). q in rad (0 = hanging down).
    Derivatives are central differences; every regressor and tau then get the *same* zero-phase low-pass
    (`cutoff` Hz), so filtering does not bias the estimates. Pass `friction` (from friction_map) for the
    recommended two-stage fit (friction first, then inertia/gravity). Samples slower than `v_min` rad/s
    are excluded: in stick phases the friction torque is indeterminate (anything below breakaway)."""
    fs = fs or 1.0 / np.median(np.diff(t))
    q = np.asarray(q, float)
    tau = np.asarray(tau, float)
    qd = np.gradient(q, 1.0 / fs)
    qdd = np.gradient(qd, 1.0 / fs)
    bb, aa = signal.butter(4, min(cutoff / (0.5 * fs), 0.99))
    lp = lambda x: signal.filtfilt(bb, aa, x)
    qd_s = lp(qd)                                     # smooth velocity for the friction nonlinearity
    cols_all = {"qdd": lp(qdd), "qd": qd_s, "sgn": lp(np.tanh(qd_s / v_eps)), "sin": lp(np.sin(q))}
    tau = lp(tau)
    if friction is not None:
        fr = friction
        f_tau = np.tanh(qd_s / v_eps) * (fr.coulomb + (fr.static - fr.coulomb) *
                                         np.exp(-(qd_s / max(fr.stribeck_vel, 1e-3)) ** 2)) + fr.viscous * qd_s
        tau = tau - lp(f_tau)
    delays = np.arange(0, int(0.03 * fs) + 1) if delays is None else np.asarray(delays)
    edge = int(0.25 * fs)
    best = None
    names = ["qdd", "qd", "sgn"] if friction is None else ["qdd"]
    if gravity:
        names.append("sin")
    for dk in delays:
        sl = slice(edge + dk, len(q) - edge)
        ts = tau[edge: len(q) - edge - dk]
        A = np.column_stack([cols_all[n][sl] for n in names] + [np.ones(len(ts))])
        keep = np.abs(qd_s[sl]) >= v_min
        A, ts = A[keep], ts[keep]
        x, *_ = np.linalg.lstsq(A, ts, rcond=None)
        r = ts - A @ x
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, dk, x, r, ts)
    sse, dk, x, r, ts = best
    val = dict(zip(names, x[:-1]))
    J = val["qdd"]
    b = val.get("qd", friction.viscous if friction is not None else 0.0)
    fc = val.get("sgn", friction.coulomb if friction is not None else 0.0)
    mgl = val.get("sin", 0.0)
    c0 = x[-1]
    r2 = 1 - sse / float(np.sum((ts - ts.mean()) ** 2))
    return RigidBodyFit(float(J), float(b), float(fc), float(mgl), float(c0), float(dk / fs), float(r2),
                        float(np.sqrt(sse / len(ts))))


# ---------------------------------------------------------------------------------------------
# frequency domain
# ---------------------------------------------------------------------------------------------
def frf(u, y, fs: float, nperseg: int = None):
    """H1 estimate of the frequency response y/u with magnitude-squared coherence."""
    nperseg = nperseg or min(len(u), int(8 * fs))
    f, Puu = signal.welch(u, fs, nperseg=nperseg)
    _, Puy = signal.csd(u, y, fs, nperseg=nperseg)
    _, Pyy = signal.welch(y, fs, nperseg=nperseg)
    H = Puy / Puu
    coh = np.abs(Puy) ** 2 / (Puu * Pyy + 1e-30)
    return f, H, coh


def fit_frf_mass_damper(f, H, coh=None, f_band=(0.5, 12.0), output: str = "velocity", coh_min: float = 0.8):
    """Fit H(s) = s^k * exp(-s*tau) / (J s^2 + b s + k) to an FRF from torque to velocity (k=1) or
    angle (k=0). The stiffness k captures gravity (mgl) about the operating point. Returns dict."""
    f = np.asarray(f)
    m = (f >= f_band[0]) & (f <= f_band[1])
    if coh is not None:
        m &= coh >= coh_min
    w = 2 * np.pi * f[m]
    Hm = H[m]
    s = 1j * w
    num_pow = 1 if output == "velocity" else 0

    def model(p):
        J, b, k, tau = p
        return s ** num_pow * np.exp(-s * tau) / (J * s ** 2 + b * s + k)

    def res(p):
        e = np.log(model(p)) - np.log(Hm)            # log-error: weights magnitude and phase evenly
        e = np.where(np.isfinite(e), e, 0.0)
        return np.concatenate([e.real, np.angle(np.exp(1j * e.imag))])
    p0 = [0.2, 1.0, 5.0, 0.005]
    sol = optimize.least_squares(res, p0, bounds=([1e-4, 0, 0, 0], [10, 100, 500, 0.05]))
    J, b, k, tau = sol.x
    return dict(inertia=float(J), viscous=float(b), stiffness=float(k), delay=float(tau),
                rms_log_error=float(np.sqrt(np.mean(sol.fun ** 2))))


# ---------------------------------------------------------------------------------------------
# current loop and thermal
# ---------------------------------------------------------------------------------------------
def fit_current_loop(t, i_cmd, i_meas):
    """First-order lag + delay from commanded to measured current (locked-output steps)."""
    fs = 1.0 / np.median(np.diff(t))
    d = estimate_delay(i_cmd, i_meas, fs)

    def sim(tau_e):
        a = np.exp(-1.0 / (fs * max(tau_e, 1e-6)))
        y = signal.lfilter([1 - a], [1, -a], i_cmd)
        k = int(round(d * fs))
        return np.concatenate([np.full(k, y[0]), y[: len(y) - k]])
    sol = optimize.minimize_scalar(lambda te: float(np.mean((sim(te) - i_meas) ** 2)), bounds=(1e-5, 0.05),
                                   method="bounded")
    return dict(delay=d, time_constant=float(sol.x))


def fit_thermal(t, current, T_winding, R25: float, temp_coef: float = 0.0039, ambient: float = None):
    """dT/dt = (i^2 R(T) - (T - Ta)/Rth) / C  ->  thermal_R, thermal_C (least squares on the derivative)."""
    t, i, T = map(np.asarray, (t, current, T_winding))
    Ta = T[0] if ambient is None else ambient
    fs = 1.0 / np.median(np.diff(t))
    _, Td, _ = derivatives(T, fs, window=min(5.0, 0.1 * (t[-1] - t[0])), poly=2)
    P = i ** 2 * R25 * (1 + temp_coef * (T - 25.0))
    # Td = P/C - (T-Ta)/(Rth*C)  -> linear in [1/C, 1/(Rth C)]
    A = np.column_stack([P, -(T - Ta)])
    x, *_ = np.linalg.lstsq(A, Td, rcond=None)
    C = 1.0 / x[0]
    Rth = x[0] / x[1] if x[1] > 0 else np.inf
    return dict(thermal_C=float(C), thermal_R=float(Rth), tau=float(Rth * C))


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------
def to_motor_params(base, gear: float, rb: RigidBodyFit = None, fr: FrictionFit = None, link_inertia: float = 0.0,
                    delay: float = None, current_loop: dict = None, thermal: dict = None):
    """Write identified joint-side values back into a MotorParams (copy of `base`)."""
    import copy
    mp = copy.deepcopy(base)
    N = gear
    if rb is not None:
        mp.rotor_inertia = max(rb.inertia - link_inertia, 0.0) / N ** 2
        mp.viscous = max(rb.viscous, 0.0) / N ** 2
        mp.coulomb = max(rb.coulomb, 0.0) / N
    if fr is not None:
        mp.coulomb = fr.coulomb / N
        mp.static = fr.static / N
        mp.stribeck_vel = fr.stribeck_vel * N
        mp.viscous = fr.viscous / N ** 2
    if delay is not None:
        mp.cmd_delay = delay
    if current_loop is not None:
        mp.te = current_loop["time_constant"]
    if thermal is not None:
        mp.thermal_R, mp.thermal_C = thermal["thermal_R"], thermal["thermal_C"]
    return mp
