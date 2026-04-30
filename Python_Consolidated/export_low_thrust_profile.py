"""Export low-thrust thrust-vs-time and multibody propagation data.

Mission architecture enforced:
Impulsive Earth->Mars, powered flyby, chemical insertion to Aegina,
low-thrust transfer/capture to Beatrix, low-thrust transfer/capture to Vesta.
"""

import os
import numpy as np
import pandas as pd
import spiceypy
from scipy.integrate import solve_ivp

from core import (
    load_kernels,
    get_id_from_asteroid_name,
    get_state,
    get_mu,
)
from lowthrust import optimize_lt_leg, DEFAULT_THRUST_N, ISP_ELEC, G0


SECONDS_PER_DAY = 86400.0


def _segment_rows(et0, et1, throttles, thrust_n, label):
    nseg = len(throttles) // 3
    dt = (et1 - et0) / nseg
    rows = []
    for i in range(nseg):
        u = np.array(throttles[3 * i:3 * i + 3], dtype=float)
        t_mid = et0 + (i + 0.5) * dt
        t_vec = thrust_n * u
        mag = float(np.linalg.norm(t_vec))
        direction = t_vec / mag if mag > 0 else np.zeros(3)
        rows.append({
            "phase": label,
            "segment_idx": i,
            "et_seconds": t_mid,
            "utc": spiceypy.et2utc(t_mid, "C", 3),
            "thrust_mag_N": mag,
            "thrust_x_N": float(t_vec[0]),
            "thrust_y_N": float(t_vec[1]),
            "thrust_z_N": float(t_vec[2]),
            "dir_x": float(direction[0]),
            "dir_y": float(direction[1]),
            "dir_z": float(direction[2]),
        })
    return rows


def _build_thrust_schedule(et_a1_depart, et_bx_arrive, et_bx_depart, et_v_arrive, lt3, lt4):
    rows = []
    rows.extend(_segment_rows(et_a1_depart, et_bx_arrive, lt3["throttles"], lt3["thrust_N"], "LT_AEGINA_TO_BEATRIX_CAPTURE"))
    rows.extend(_segment_rows(et_bx_depart, et_v_arrive, lt4["throttles"], lt4["thrust_N"], "LT_BEATRIX_TO_VESTA_CAPTURE"))
    return rows


def _thrust_lookup(t, schedule, thrust_n):
    for seg in schedule:
        if seg["t0"] <= t < seg["t1"]:
            u = seg["u"]
            return thrust_n * u
    return np.zeros(3)


def _build_piecewise_schedule(et0, et1, throttles, phase):
    nseg = len(throttles) // 3
    dt = (et1 - et0) / nseg
    out = []
    for i in range(nseg):
        out.append({
            "phase": phase,
            "t0": et0 + i * dt,
            "t1": et0 + (i + 1) * dt,
            "u": np.array(throttles[3 * i:3 * i + 3], dtype=float),
        })
    return out


def _multibody_rhs(t, y, body_ids, schedule, thrust_n, isp_s):
    r = y[0:3]
    v = y[3:6]
    m = max(float(y[6]), 1e-6)

    mu_sun = get_mu(10)
    r_norm = max(np.linalg.norm(r), 1.0)
    a = -mu_sun * r / (r_norm ** 3)

    for body_id in body_ids:
        mu_b = get_mu(int(body_id))
        r_b, _ = get_state(str(body_id), t)
        dr = r_b - r
        dr_norm = max(np.linalg.norm(dr), 1.0)
        rb_norm = max(np.linalg.norm(r_b), 1.0)
        # Third-body perturbation in heliocentric frame
        a += mu_b * (dr / (dr_norm ** 3) - r_b / (rb_norm ** 3))

    t_vec = _thrust_lookup(t, schedule, thrust_n)
    t_mag = float(np.linalg.norm(t_vec))
    a_thrust = t_vec / m / 1000.0  # N/kg -> m/s^2 -> km/s^2
    mdot = -t_mag / (isp_s * G0)

    dydt = np.zeros(7)
    dydt[0:3] = v
    dydt[3:6] = a + a_thrust
    dydt[6] = mdot
    return dydt


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)

    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")

    csv_in = os.path.join(repo, "optimal_asteroid_paths", "csv", "results_science_priority_v2.csv")
    df = pd.read_csv(csv_in)
    row = df[
        (df["a1"] == "AEGINA")
        & (df["a2"] == "BEATRIX")
        & (df["a3"] == "VESTA")
        & (df["arch"].str.lower() == "mars")
    ].iloc[0]

    et_launch = spiceypy.str2et(row["date_launch"])
    et_flyby = spiceypy.str2et(row["date_flyby"])
    et_arrive_1 = spiceypy.str2et(row["date_arrive_A1"])
    et_stay_1 = spiceypy.str2et(row["date_leave_A1"])
    et_arrive_2 = spiceypy.str2et(row["date_arrive_A2"])
    et_stay_2 = spiceypy.str2et(row["date_leave_A2"])
    et_arrive_3 = spiceypy.str2et(row["date_arrive_A3"])

    a1_id = str(int(get_id_from_asteroid_name(asteroid_list, "AEGINA")))
    a2_id = str(int(get_id_from_asteroid_name(asteroid_list, "BEATRIX")))
    a3_id = str(int(get_id_from_asteroid_name(asteroid_list, "VESTA")))

    m0 = 1500.0
    thrust_n = DEFAULT_THRUST_N
    nseg = 30

    r1, v1 = get_state(a1_id, et_stay_1)
    r2, v2 = get_state(a2_id, et_arrive_2)
    lt3 = optimize_lt_leg(r1, v1, r2, v2, et_arrive_2 - et_stay_1, m_init_kg=m0, thrust_N=thrust_n, isp_s=ISP_ELEC, nseg=nseg)
    if not lt3["converged"]:
        raise RuntimeError(f"L3 low-thrust did not converge: {lt3['reason']}")

    r2b, v2b = get_state(a2_id, et_stay_2)
    r3, v3 = get_state(a3_id, et_arrive_3)
    lt4 = optimize_lt_leg(r2b, v2b, r3, v3, et_arrive_3 - et_stay_2, m_init_kg=lt3["m_final"], thrust_N=thrust_n, isp_s=ISP_ELEC, nseg=nseg)
    if not lt4["converged"]:
        raise RuntimeError(f"L4 low-thrust did not converge: {lt4['reason']}")

    thrust_rows = _build_thrust_schedule(et_stay_1, et_arrive_2, et_stay_2, et_arrive_3, lt3, lt4)
    thrust_df = pd.DataFrame(thrust_rows).sort_values("et_seconds")

    schedule = []
    schedule.extend(_build_piecewise_schedule(et_stay_1, et_arrive_2, lt3["throttles"], "LT_AEGINA_TO_BEATRIX_CAPTURE"))
    schedule.extend(_build_piecewise_schedule(et_stay_2, et_arrive_3, lt4["throttles"], "LT_BEATRIX_TO_VESTA_CAPTURE"))

    y0 = np.hstack([r1, v1, m0])
    t0 = et_stay_1
    tf = et_arrive_3
    t_eval = np.arange(t0, tf + SECONDS_PER_DAY, SECONDS_PER_DAY)
    body_ids = ["399", "499", a1_id, a2_id, a3_id]

    sol = solve_ivp(
        fun=lambda t, y: _multibody_rhs(t, y, body_ids, schedule, thrust_n, ISP_ELEC),
        t_span=(t0, tf),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"Multibody propagation failed: {sol.message}")

    prop_rows = []
    for k, et in enumerate(sol.t):
        prop_rows.append({
            "et_seconds": float(et),
            "utc": spiceypy.et2utc(float(et), "C", 3),
            "x_km": float(sol.y[0, k]),
            "y_km": float(sol.y[1, k]),
            "z_km": float(sol.y[2, k]),
            "vx_km_s": float(sol.y[3, k]),
            "vy_km_s": float(sol.y[4, k]),
            "vz_km_s": float(sol.y[5, k]),
            "mass_kg": float(sol.y[6, k]),
        })
    prop_df = pd.DataFrame(prop_rows)

    out_dir = os.path.join(repo, "optimal_asteroid_paths", "csv")
    os.makedirs(out_dir, exist_ok=True)
    thrust_out = os.path.join(out_dir, "aegina_beatrix_vesta_low_thrust_profile.csv")
    prop_out = os.path.join(out_dir, "aegina_beatrix_vesta_multibody_states.csv")
    summary_out = os.path.join(out_dir, "aegina_beatrix_vesta_low_thrust_summary.csv")

    thrust_df.to_csv(thrust_out, index=False)
    prop_df.to_csv(prop_out, index=False)
    pd.DataFrame([{
        "architecture": "Impulsive to Mars -> Powered Flyby -> Chemical Insertion to Aegina -> Low Thrust to Beatrix -> Low thrust capture -> Low thrust to Vesta",
        "launch_utc": spiceypy.et2utc(et_launch, "C", 0),
        "mars_flyby_utc": spiceypy.et2utc(et_flyby, "C", 0),
        "aegina_arrive_utc": spiceypy.et2utc(et_arrive_1, "C", 0),
        "aegina_depart_utc": spiceypy.et2utc(et_stay_1, "C", 0),
        "beatrix_arrive_utc": spiceypy.et2utc(et_arrive_2, "C", 0),
        "beatrix_depart_utc": spiceypy.et2utc(et_stay_2, "C", 0),
        "vesta_arrive_utc": spiceypy.et2utc(et_arrive_3, "C", 0),
        "lt_leg3_converged": lt3["converged"],
        "lt_leg3_dv_integral_km_s": lt3["dv_integral_kms"],
        "lt_leg3_end_mass_kg": lt3["m_final"],
        "lt_leg4_converged": lt4["converged"],
        "lt_leg4_dv_integral_km_s": lt4["dv_integral_kms"],
        "lt_leg4_end_mass_kg": lt4["m_final"],
    }]).to_csv(summary_out, index=False)

    print("Wrote:")
    print(f"  {thrust_out}")
    print(f"  {prop_out}")
    print(f"  {summary_out}")


if __name__ == "__main__":
    main()
