"""Export low-thrust thrust-vs-time and multibody propagation data.

By default, this script reads the latest constrained science CSV and uses rank=1.
Override with:
  --csv <path-to-results-csv>
  --rank <int>
"""

import os
import sys
import argparse
from pathlib import Path

_PC_ROOT = Path(__file__).resolve().parent.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))

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
from results_artifacts import latest_results_csv, optimal_asteroid_dir
from lowthrust import (
    optimize_lt_leg,
    DEFAULT_M_INIT_KG,
    DEFAULT_THRUST_N,
    ISP_ELEC,
    G0,
)


SECONDS_PER_DAY = 86400.0


def _linspace_eval_times(et0: float, et1: float, max_points: int = 1200, min_points: int = 48) -> np.ndarray:
    span = float(et1 - et0)
    if span <= 0:
        return np.array([et0])
    step = 12.0 * 3600.0
    n = int(max(min_points, min(max_points, span / step + 1)))
    return np.linspace(et0, et1, n)


def _ivp_solution_to_sc_df(sol, phase: str) -> pd.DataFrame:
    rows = []
    for k, et in enumerate(sol.t):
        rows.append(
            {
                "phase": phase,
                "et_seconds": float(et),
                "utc": spiceypy.et2utc(float(et), "C", 3),
                "sc_x_km": float(sol.y[0, k]),
                "sc_y_km": float(sol.y[1, k]),
                "sc_z_km": float(sol.y[2, k]),
                "sc_vx_km_s": float(sol.y[3, k]),
                "sc_vy_km_s": float(sol.y[4, k]),
                "sc_vz_km_s": float(sol.y[5, k]),
            }
        )
    return pd.DataFrame(rows)


def _sample_stay_sc(et0: float, et1: float, body_id: str, phase: str, n: int = 80) -> pd.DataFrame:
    ets = np.linspace(float(et0), float(et1), n)
    rows = []
    for et in ets:
        r, v = get_state(str(body_id), float(et))
        rows.append(
            {
                "phase": phase,
                "et_seconds": float(et),
                "utc": spiceypy.et2utc(float(et), "C", 3),
                "sc_x_km": float(r[0]),
                "sc_y_km": float(r[1]),
                "sc_z_km": float(r[2]),
                "sc_vx_km_s": float(v[0]),
                "sc_vy_km_s": float(v[1]),
                "sc_vz_km_s": float(v[2]),
            }
        )
    return pd.DataFrame(rows)


def integrated_lt_transfers_for_package(
    et_stay_1: float,
    et_arrive_2: float,
    et_stay_2: float,
    et_arrive_3: float,
    a1_id: str,
    a2_id: str,
    a3_id: str,
    *,
    m0_kg: float = DEFAULT_M_INIT_KG,
    thrust_n: float = DEFAULT_THRUST_N,
    nseg: int = 30,
) -> pd.DataFrame:
    """Sims–Flanagan LT legs + split multibody RK for A1→A2 and A2→A3 (matches standalone export).

    Leg 3: IVP from A1 departure to A2 arrival. Stay A2: SPICE co-orbit. Leg 4: IVP to A3 arrival.
    Returns a single DataFrame with ``phase`` suitable for ``time_position_velocity.csv``.
    """
    r1, v1 = get_state(a1_id, et_stay_1)
    r2_arr, v2_arr = get_state(a2_id, et_arrive_2)
    lt3 = optimize_lt_leg(
        r1,
        v1,
        r2_arr,
        v2_arr,
        et_arrive_2 - et_stay_1,
        m_init_kg=m0_kg,
        thrust_N=thrust_n,
        isp_s=ISP_ELEC,
        nseg=nseg,
    )
    if not lt3.get("converged"):
        raise RuntimeError(f"L3 low-thrust did not converge: {lt3.get('reason', '')}")

    r2b, v2b = get_state(a2_id, et_stay_2)
    r3, v3 = get_state(a3_id, et_arrive_3)
    lt4 = optimize_lt_leg(
        r2b,
        v2b,
        r3,
        v3,
        et_arrive_3 - et_stay_2,
        m_init_kg=float(lt3["m_final"]),
        thrust_N=thrust_n,
        isp_s=ISP_ELEC,
        nseg=nseg,
    )
    if not lt4.get("converged"):
        raise RuntimeError(f"L4 low-thrust did not converge: {lt4.get('reason', '')}")

    body_ids = ["399", "499", str(a1_id), str(a2_id), str(a3_id)]
    sched3 = _build_piecewise_schedule(et_stay_1, et_arrive_2, lt3["throttles"], "lt_a1_to_a2")
    y0_3 = np.hstack([r1, v1, float(m0_kg)])
    t_eval_3 = _linspace_eval_times(et_stay_1, et_arrive_2)
    sol3 = solve_ivp(
        fun=lambda t, y: _multibody_rhs(t, y, body_ids, sched3, thrust_n, ISP_ELEC),
        t_span=(float(et_stay_1), float(et_arrive_2)),
        y0=y0_3,
        t_eval=t_eval_3,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )
    if not sol3.success:
        raise RuntimeError(f"L3 propagation failed: {sol3.message}")

    m_after_l3 = float(sol3.y[6, -1])
    sched4 = _build_piecewise_schedule(et_stay_2, et_arrive_3, lt4["throttles"], "lt_a2_to_a3")
    y0_4 = np.hstack([r2b, v2b, m_after_l3])
    t_eval_4 = _linspace_eval_times(et_stay_2, et_arrive_3)
    sol4 = solve_ivp(
        fun=lambda t, y: _multibody_rhs(t, y, body_ids, sched4, thrust_n, ISP_ELEC),
        t_span=(float(et_stay_2), float(et_arrive_3)),
        y0=y0_4,
        t_eval=t_eval_4,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )
    if not sol4.success:
        raise RuntimeError(f"L4 propagation failed: {sol4.message}")

    df3 = _ivp_solution_to_sc_df(sol3, "transfer_a1_to_a2")
    stay_df = _sample_stay_sc(et_arrive_2, et_stay_2, a2_id, "stay_a2", n=80)
    if len(stay_df) > 1:
        stay_df = stay_df.iloc[1:].copy()
    df4 = _ivp_solution_to_sc_df(sol4, "transfer_a2_to_a3")
    if len(df4) > 0:
        df4 = df4.iloc[1:].copy()

    return pd.concat([df3, stay_df, df4], ignore_index=True)


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


def _resolve_name_column(df):
    for col in ["a1", "A1", "asteroid_1"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find asteroid name columns in input CSV.")


def _as_float(row, key_options):
    for key in key_options:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    raise ValueError(f"Missing required epoch field. Tried: {key_options}")


def main():
    parser = argparse.ArgumentParser(description="Export low-thrust profile from ranked mission CSV.")
    parser.add_argument("--csv", dest="csv_in", default=None, help="Input mission results CSV path.")
    parser.add_argument("--rank", dest="rank", type=int, default=1, help="Rank in CSV to export (default: 1).")
    args = parser.parse_args()

    repo = str(Path(__file__).resolve().parent.parent.parent)
    os.chdir(repo)

    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")

    csv_in = args.csv_in
    if csv_in is None:
        csv_in = latest_results_csv(repo)
    if not os.path.isabs(csv_in):
        csv_in = os.path.join(repo, csv_in)

    df = pd.read_csv(csv_in)
    if "rank" in df.columns:
        row = df[df["rank"] == args.rank].iloc[0]
    else:
        row = df.iloc[max(args.rank - 1, 0)]

    # Support both old-style and new constrained CSV schemas.
    et_launch = _as_float(row, ["launch_et", "et_launch"])
    if "flyby_et" in row and pd.notna(row["flyby_et"]):
        et_flyby = float(row["flyby_et"])
    elif "et_flyby" in row and pd.notna(row.get("et_flyby")):
        et_flyby = float(row["et_flyby"])
    else:
        et_flyby = np.nan
    et_arrive_1 = _as_float(row, ["arrive_a1_et", "et_arrive_1"])
    et_stay_1 = _as_float(row, ["leave_a1_et", "et_stay_1"])
    et_arrive_2 = _as_float(row, ["arrive_a2_et", "et_arrive_2"])
    et_stay_2 = _as_float(row, ["leave_a2_et", "et_stay_2"])
    et_arrive_3 = _as_float(row, ["arrive_a3_et", "et_arrive_3"])

    a1_name = str(row.get("a1", "")).upper()
    a2_name = str(row.get("a2", "")).upper()
    a3_name = str(row.get("a3", "")).upper()
    if not a1_name or not a2_name or not a3_name:
        raise ValueError("Input CSV must provide a1/a2/a3 columns.")

    a1_id = str(int(get_id_from_asteroid_name(asteroid_list, a1_name)))
    a2_id = str(int(get_id_from_asteroid_name(asteroid_list, a2_name)))
    a3_id = str(int(get_id_from_asteroid_name(asteroid_list, a3_name)))

    m0 = DEFAULT_M_INIT_KG
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

    out_dir = optimal_asteroid_dir(repo)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{a1_name.lower()}_{a2_name.lower()}_{a3_name.lower()}"
    thrust_out = os.path.join(out_dir, f"{stem}_low_thrust_profile.csv")
    prop_out = os.path.join(out_dir, f"{stem}_multibody_states.csv")
    summary_out = os.path.join(out_dir, f"{stem}_low_thrust_summary.csv")

    thrust_df.to_csv(thrust_out, index=False)
    prop_df.to_csv(prop_out, index=False)
    pd.DataFrame([{
        "architecture": str(row.get("architecture", "unknown")),
        "a1": a1_name,
        "a2": a2_name,
        "a3": a3_name,
        "source_csv": csv_in,
        "source_rank": int(args.rank),
        "launch_utc": spiceypy.et2utc(et_launch, "C", 0),
        "flyby_utc": spiceypy.et2utc(float(et_flyby), "C", 0) if np.isfinite(et_flyby) else "",
        "a1_arrive_utc": spiceypy.et2utc(et_arrive_1, "C", 0),
        "a1_depart_utc": spiceypy.et2utc(et_stay_1, "C", 0),
        "a2_arrive_utc": spiceypy.et2utc(et_arrive_2, "C", 0),
        "a2_depart_utc": spiceypy.et2utc(et_stay_2, "C", 0),
        "a3_arrive_utc": spiceypy.et2utc(et_arrive_3, "C", 0),
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
