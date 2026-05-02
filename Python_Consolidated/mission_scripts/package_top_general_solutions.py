"""Package top-N constrained missions into per-solution folders with full exports."""

import argparse
import os
import sys
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

_PC_ROOT = Path(__file__).resolve().parent.parent
if str(_PC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC_ROOT))
_MS_DIR = Path(__file__).resolve().parent
if str(_MS_DIR) not in sys.path:
    sys.path.insert(0, str(_MS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy

from core import (
    DAY,
    MU_SUN,
    audit_flyby_geometry,
    get_id_from_asteroid_name,
    get_state,
    load_kernels,
    solve_lambert,
    two_body_sim,
)
from visualization import flightpath_animation

from results_artifacts import latest_results_csv, latest_results_pkl

from lowthrust import DEFAULT_M_INIT_KG
from export_low_thrust_profile import integrated_lt_transfers_for_package

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


def _safe_slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _impulse_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=float)))


def _flyby_et(result: Dict) -> float:
    if result.get("et_flyby") is not None:
        try:
            return float(result["et_flyby"])
        except (TypeError, ValueError):
            pass
    if result.get("et_mars") is not None:
        try:
            return float(result["et_mars"])
        except (TypeError, ValueError):
            pass
    return float("nan")


def _build_burn_events(row: pd.Series, result: Dict) -> pd.DataFrame:
    arch = str(result.get("architecture", "direct"))
    if arch == "mars":
        launch_to = "Mars"
    elif arch == "moon":
        launch_to = "Moon"
    else:
        launch_to = row["a1"]
    events = [
        ("earth_launch", "Earth", launch_to, result["et_launch"], result["delta_v_launch"]),
    ]
    et_fb = _flyby_et(result)
    if arch in ("mars", "moon") and np.isfinite(et_fb):
        fb_name = "Mars" if arch == "mars" else "Moon"
        events.append((f"{arch}_gravity_assist_unpowered", fb_name, fb_name, et_fb, np.zeros(3)))
    events.extend(
        [
        ("a1_arrive", row["a1"], row["a1"], result["et_arrive_1"], result["delta_v_A1_arrive"]),
        ("a1_depart", row["a1"], row["a2"], result["et_stay_1"], result["delta_v_A1_leave"]),
        ("a2_arrive", row["a2"], row["a2"], result["et_arrive_2"], result["delta_v_A2_arrive"]),
        ("a2_depart", row["a2"], row["a3"], result["et_stay_2"], result["delta_v_A2_leave"]),
        ("a3_arrive", row["a3"], row["a3"], result["et_arrive_3"], result["delta_v_A3_arrive"]),
        ]
    )
    rows = []
    for event, from_body, to_body, et, dv_vec in events:
        dv_vec = np.asarray(dv_vec, dtype=float)
        dv_norm = _impulse_norm(dv_vec)
        d = dv_vec / (dv_norm + 1e-12)
        rows.append(
            {
                "event": event,
                "from_body": from_body,
                "to_body": to_body,
                "et_seconds": float(et),
                "utc": spiceypy.et2utc(float(et), "C", 3),
                "dv_km_s": dv_norm,
                "dv_x_km_s": float(dv_vec[0]),
                "dv_y_km_s": float(dv_vec[1]),
                "dv_z_km_s": float(dv_vec[2]),
                "dv_dir_x": float(d[0]),
                "dv_dir_y": float(d[1]),
                "dv_dir_z": float(d[2]),
            }
        )
    return pd.DataFrame(rows)


def _build_transfer_dv(row: pd.Series, result: Dict) -> pd.DataFrame:
    arch = str(result.get("architecture", "direct"))
    et_fb = _flyby_et(result)
    fb_id = str(result.get("flyby_body", ""))
    if arch == "mars":
        launch_dest = "Mars"
    elif arch == "moon":
        launch_dest = "Moon"
    else:
        launch_dest = row["a1"]

    launch_seg = {
        "segment": "earth_launch_reference",
        "from_body": "Earth",
        "to_body": launch_dest,
        "depart_et": float(result["et_launch"]),
        "arrive_et": float(result["et_launch"]),
        "depart_utc": spiceypy.et2utc(float(result["et_launch"]), "C", 3),
        "arrive_utc": spiceypy.et2utc(float(result["et_launch"]), "C", 3),
        "dv_km_s": _impulse_norm(result["delta_v_launch"]),
        "transfer_type": "impulsive",
        "propulsion": "launch_vehicle_reference",
    }

    a1_to_a2 = {
        "segment": "transfer_a1_to_a2",
        "from_body": row["a1"],
        "to_body": row["a2"],
        "depart_et": float(result["et_stay_1"]),
        "arrive_et": float(result["et_arrive_2"]),
        "depart_utc": spiceypy.et2utc(float(result["et_stay_1"]), "C", 3),
        "arrive_utc": spiceypy.et2utc(float(result["et_arrive_2"]), "C", 3),
        "dv_km_s": _impulse_norm(result["delta_v_A1_leave"]) + _impulse_norm(result["delta_v_A2_arrive"]),
        "transfer_type": "interplanetary_transfer",
        "propulsion": "electric_low_thrust_equivalent",
    }
    a2_to_a3 = {
        "segment": "transfer_a2_to_a3",
        "from_body": row["a2"],
        "to_body": row["a3"],
        "depart_et": float(result["et_stay_2"]),
        "arrive_et": float(result["et_arrive_3"]),
        "depart_utc": spiceypy.et2utc(float(result["et_stay_2"]), "C", 3),
        "arrive_utc": spiceypy.et2utc(float(result["et_arrive_3"]), "C", 3),
        "dv_km_s": _impulse_norm(result["delta_v_A2_leave"]) + _impulse_norm(result["delta_v_A3_arrive"]),
        "transfer_type": "interplanetary_transfer",
        "propulsion": "electric_low_thrust_equivalent",
    }

    if arch in ("mars", "moon") and np.isfinite(et_fb) and fb_id:
        fb_label = "Mars" if arch == "mars" else "Moon"
        coast = {
            "segment": f"ballistic_earth_to_{arch}_flyby",
            "from_body": "Earth",
            "to_body": fb_label,
            "depart_et": float(result["et_launch"]),
            "arrive_et": et_fb,
            "depart_utc": spiceypy.et2utc(float(result["et_launch"]), "C", 3),
            "arrive_utc": spiceypy.et2utc(et_fb, "C", 3),
            "dv_km_s": 0.0,
            "transfer_type": "ballistic_coast",
            "propulsion": "none",
        }
        assist = {
            "segment": f"{arch}_gravity_assist_unpowered",
            "from_body": fb_label,
            "to_body": fb_label,
            "depart_et": et_fb,
            "arrive_et": et_fb,
            "depart_utc": spiceypy.et2utc(et_fb, "C", 3),
            "arrive_utc": spiceypy.et2utc(et_fb, "C", 3),
            "dv_km_s": 0.0,
            "transfer_type": "gravity_assist",
            "propulsion": "unpowered",
        }
        mars_to_a1 = {
            "segment": f"transfer_post_{arch}_flyby_to_a1",
            "from_body": fb_label,
            "to_body": row["a1"],
            "depart_et": et_fb,
            "arrive_et": float(result["et_arrive_1"]),
            "depart_utc": spiceypy.et2utc(et_fb, "C", 3),
            "arrive_utc": spiceypy.et2utc(float(result["et_arrive_1"]), "C", 3),
            "dv_km_s": _impulse_norm(result["delta_v_A1_arrive"]),
            "transfer_type": "interplanetary_transfer",
            "propulsion": "electric_low_thrust_equivalent",
        }
        rows = [launch_seg, coast, assist, mars_to_a1, a1_to_a2, a2_to_a3]
    else:
        earth_to_a1 = {
            "segment": "transfer_earth_to_a1",
            "from_body": "Earth",
            "to_body": row["a1"],
            "depart_et": float(result["et_launch"]),
            "arrive_et": float(result["et_arrive_1"]),
            "depart_utc": spiceypy.et2utc(float(result["et_launch"]), "C", 3),
            "arrive_utc": spiceypy.et2utc(float(result["et_arrive_1"]), "C", 3),
            "dv_km_s": _impulse_norm(result["delta_v_A1_arrive"]),
            "transfer_type": "interplanetary_transfer",
            "propulsion": "electric_low_thrust_equivalent",
        }
        rows = [launch_seg, earth_to_a1, a1_to_a2, a2_to_a3]

    return pd.DataFrame(rows)


def _build_flyby_geometry(result: Dict, a1_id: str) -> pd.DataFrame:
    arch = str(result.get("architecture", "direct"))
    et_fb = _flyby_et(result)
    if arch not in ("mars", "moon") or not np.isfinite(et_fb):
        return pd.DataFrame([{"note": "no_flyby_in_architecture", "architecture": arch}])
    geom = audit_flyby_geometry(
        float(result["et_launch"]),
        et_fb,
        float(result["et_arrive_1"]),
        a1_id,
        arch,
    )
    if not geom.get("feasible", False):
        return pd.DataFrame([{"feasible": False, "reason": geom.get("reason", "unknown")}])
    vin = np.asarray(geom["v_inf_in_vec"], dtype=float)
    vout = np.asarray(geom["v_inf_out_vec"], dtype=float)
    return pd.DataFrame(
        [
            {
                "feasible": True,
                "turn_angle_deg": float(geom["turn_angle_deg"]),
                "turn_max_deg": float(geom["turn_max_deg"]),
                "periapsis_alt_km": float(geom["periapsis_alt_km"]),
                "safe_periapsis_alt_km": float(geom["safe_periapsis_alt_km"]),
                "energy_residual_kms": float(geom["energy_residual_kms"]),
                "v_inf_in_kms": float(geom["v_inf_in_kms"]),
                "v_inf_out_kms": float(geom["v_inf_out_kms"]),
                "v_inf_in_x": float(vin[0]),
                "v_inf_in_y": float(vin[1]),
                "v_inf_in_z": float(vin[2]),
                "v_inf_out_x": float(vout[0]),
                "v_inf_out_y": float(vout[1]),
                "v_inf_out_z": float(vout[2]),
            }
        ]
    )


def _sample_leg(et0: float, et1: float, body_id: str, dv_vec: np.ndarray, phase: str, n: int = 250) -> pd.DataFrame:
    r0, v0 = get_state(body_id, float(et0))
    x0 = np.concatenate([r0, v0 + np.asarray(dv_vec, dtype=float)])
    X, T = two_body_sim(float(et1 - et0), x0, MU_SUN)
    if len(X) > n:
        idx = np.linspace(0, len(X) - 1, n).astype(int)
        X = X[idx]
        T = T[idx]
    et = float(et0) + np.asarray(T, dtype=float)
    return pd.DataFrame(
        {
            "phase": phase,
            "et_seconds": et,
            "utc": [spiceypy.et2utc(float(t), "C", 3) for t in et],
            "sc_x_km": X[:, 0],
            "sc_y_km": X[:, 1],
            "sc_z_km": X[:, 2],
            "sc_vx_km_s": X[:, 3],
            "sc_vy_km_s": X[:, 4],
            "sc_vz_km_s": X[:, 5],
        }
    )


def _sample_leg_heliocentric(et0: float, et1: float, r0: np.ndarray, v0: np.ndarray, phase: str, n: int = 250) -> pd.DataFrame:
    r0 = np.asarray(r0, dtype=float).ravel()
    v0 = np.asarray(v0, dtype=float).ravel()
    x0 = np.concatenate([r0, v0])
    X, T = two_body_sim(float(et1 - et0), x0, MU_SUN)
    if len(X) > n:
        idx = np.linspace(0, len(X) - 1, n).astype(int)
        X = X[idx]
        T = T[idx]
    et = float(et0) + np.asarray(T, dtype=float)
    return pd.DataFrame(
        {
            "phase": phase,
            "et_seconds": et,
            "utc": [spiceypy.et2utc(float(t), "C", 3) for t in et],
            "sc_x_km": X[:, 0],
            "sc_y_km": X[:, 1],
            "sc_z_km": X[:, 2],
            "sc_vx_km_s": X[:, 3],
            "sc_vy_km_s": X[:, 4],
            "sc_vz_km_s": X[:, 5],
        }
    )


def _snap_sc_end_to_body(df: pd.DataFrame, body_id: str) -> pd.DataFrame:
    """Force last row SC state to match *body_id* at that row's ET (fixes CSV gaps vs stays)."""
    if df.empty:
        return df
    out = df.copy()
    il = len(out) - 1
    et = float(out["et_seconds"].iloc[il])
    r, v = get_state(str(body_id), et)
    out.iloc[il, out.columns.get_loc("sc_x_km")] = float(r[0])
    out.iloc[il, out.columns.get_loc("sc_y_km")] = float(r[1])
    out.iloc[il, out.columns.get_loc("sc_z_km")] = float(r[2])
    out.iloc[il, out.columns.get_loc("sc_vx_km_s")] = float(v[0])
    out.iloc[il, out.columns.get_loc("sc_vy_km_s")] = float(v[1])
    out.iloc[il, out.columns.get_loc("sc_vz_km_s")] = float(v[2])
    return out


def _sample_stay(et0: float, et1: float, body_id: str, phase: str, n: int = 80) -> pd.DataFrame:
    ets = np.linspace(float(et0), float(et1), n)
    rows = []
    for et in ets:
        r, v = get_state(body_id, float(et))
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


def _enrich_body_states(df: pd.DataFrame, body_ids: Dict[str, str], flyby_body_id: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    bodies = [("earth", "399"), ("a1", body_ids["a1"]), ("a2", body_ids["a2"]), ("a3", body_ids["a3"])]
    if flyby_body_id:
        bodies.append(("flyby_body", flyby_body_id))
    for body_name, body_id in bodies:
        pos = []
        vel = []
        for et in out["et_seconds"].to_numpy():
            r, v = get_state(body_id, float(et))
            pos.append(r)
            vel.append(v)
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        out[f"{body_name}_x_km"] = pos[:, 0]
        out[f"{body_name}_y_km"] = pos[:, 1]
        out[f"{body_name}_z_km"] = pos[:, 2]
        out[f"{body_name}_vx_km_s"] = vel[:, 0]
        out[f"{body_name}_vy_km_s"] = vel[:, 1]
        out[f"{body_name}_vz_km_s"] = vel[:, 2]
    return out


def _snap_join_to_body(df: pd.DataFrame, from_phase: str, to_phase: str, body_id: str) -> pd.DataFrame:
    """Anchor phase join by snapping the final state of from_phase to body state."""
    if df.empty:
        return df
    out = df.copy()
    phase = out["phase"].astype(str).to_numpy()
    for i in range(len(out) - 1):
        if phase[i] == from_phase and phase[i + 1] == to_phase:
            et = float(out.iloc[i]["et_seconds"])
            r, v = get_state(str(body_id), et)
            out.iloc[i, out.columns.get_loc("sc_x_km")] = float(r[0])
            out.iloc[i, out.columns.get_loc("sc_y_km")] = float(r[1])
            out.iloc[i, out.columns.get_loc("sc_z_km")] = float(r[2])
            out.iloc[i, out.columns.get_loc("sc_vx_km_s")] = float(v[0])
            out.iloc[i, out.columns.get_loc("sc_vy_km_s")] = float(v[1])
            out.iloc[i, out.columns.get_loc("sc_vz_km_s")] = float(v[2])
    return out


def _build_time_state_csv(result: Dict, body_ids: Dict[str, str], folder: Optional[str] = None) -> pd.DataFrame:
    arch = str(result.get("architecture", "direct"))
    et_fb = _flyby_et(result)
    if arch == "mars":
        fb_id = str(result.get("flyby_body") or "4")
    elif arch == "moon":
        fb_id = str(result.get("flyby_body") or "301")
    else:
        fb_id = str(result.get("flyby_body") or "")
    parts: List[pd.DataFrame] = []

    if arch in ("mars", "moon") and np.isfinite(et_fb) and fb_id:
        leg_ballistic = _sample_leg(
            float(result["et_launch"]),
            et_fb,
            "399",
            result["delta_v_launch"],
            f"ballistic_earth_to_{arch}_flyby",
        )
        parts.append(leg_ballistic)
        r_exit = leg_ballistic[["sc_x_km", "sc_y_km", "sc_z_km"]].iloc[-1].to_numpy(dtype=float)
        vout = result.get("v_sc_post_flyby")
        vo = np.asarray(vout, dtype=float).ravel() if vout is not None else np.zeros(0)
        if vo.size < 3:
            a1_r, _ = get_state(body_ids["a1"], float(result["et_arrive_1"]))
            rfb, _ = get_state(fb_id, et_fb)
            tof_d = (float(result["et_arrive_1"]) - et_fb) / DAY
            _, vo, ef = solve_lambert(rfb, a1_r, tof_d, 0, MU_SUN)
            if ef != 1:
                _, vo, _ = solve_lambert(rfb, a1_r, tof_d, 1, MU_SUN)
            vo = np.asarray(vo, dtype=float).ravel()
        leg_m2a1 = _sample_leg_heliocentric(
            et_fb,
            float(result["et_arrive_1"]),
            r_exit,
            vo,
            f"heliocentric_{arch}_exit_to_a1",
        )
        parts.append(_snap_sc_end_to_body(leg_m2a1, body_ids["a1"]))
    else:
        leg_ea1 = _sample_leg(
            float(result["et_launch"]),
            float(result["et_arrive_1"]),
            "399",
            result["delta_v_launch"],
            "transfer_earth_to_a1",
        )
        parts.append(_snap_sc_end_to_body(leg_ea1, body_ids["a1"]))

    parts.append(_sample_stay(float(result["et_arrive_1"]), float(result["et_stay_1"]), body_ids["a1"], "stay_a1"))

    lt_block = None
    try:
        lt_block = integrated_lt_transfers_for_package(
            float(result["et_stay_1"]),
            float(result["et_arrive_2"]),
            float(result["et_stay_2"]),
            float(result["et_arrive_3"]),
            body_ids["a1"],
            body_ids["a2"],
            body_ids["a3"],
            m0_kg=DEFAULT_M_INIT_KG,
        )
    except Exception as exc:
        warnings.warn(
            "integrated_lt_transfers_for_package failed; using two-body reference for A1→A2→A3: %s" % exc,
            UserWarning,
            stacklevel=2,
        )
        if folder:
            try:
                with open(os.path.join(folder, "time_state_lt_fallback.txt"), "w", encoding="utf-8") as fh:
                    fh.write(str(exc) + "\n")
            except OSError:
                pass

    if lt_block is not None and len(lt_block) > 0:
        parts.append(_snap_sc_end_to_body(lt_block, body_ids["a3"]))
    else:
        leg_a1a2 = _sample_leg(
            float(result["et_stay_1"]),
            float(result["et_arrive_2"]),
            body_ids["a1"],
            result["delta_v_A1_leave"],
            "transfer_a1_to_a2",
        )
        leg_a2a3 = _sample_leg(
            float(result["et_stay_2"]),
            float(result["et_arrive_3"]),
            body_ids["a2"],
            result["delta_v_A2_leave"],
            "transfer_a2_to_a3",
        )
        parts.extend(
            [
                _snap_sc_end_to_body(leg_a1a2, body_ids["a2"]),
                _sample_stay(float(result["et_arrive_2"]), float(result["et_stay_2"]), body_ids["a2"], "stay_a2"),
                _snap_sc_end_to_body(leg_a2a3, body_ids["a3"]),
            ]
        )
    df = pd.concat(parts, ignore_index=True)
    # Guardrail: enforce continuity exactly at phase joins where SC should be on a body.
    df = _snap_join_to_body(df, "transfer_earth_to_a1", "stay_a1", body_ids["a1"])
    df = _snap_join_to_body(df, "heliocentric_mars_exit_to_a1", "stay_a1", body_ids["a1"])
    df = _snap_join_to_body(df, "heliocentric_moon_exit_to_a1", "stay_a1", body_ids["a1"])
    df = _snap_join_to_body(df, "transfer_a1_to_a2", "stay_a2", body_ids["a2"])
    flyby_for_enrich = fb_id if arch in ("mars", "moon") and fb_id else None
    return _enrich_body_states(df, body_ids, flyby_for_enrich)


def _save_thrust_plots(folder: str, transfer_df: pd.DataFrame, burn_df: pd.DataFrame) -> None:
    launch_et = float(transfer_df["depart_et"].min())
    t_days = (transfer_df["arrive_et"] - launch_et) / SECONDS_PER_DAY
    plot_mask = transfer_df["segment"] != "earth_launch_reference"
    leg_dv = transfer_df.loc[plot_mask, "dv_km_s"].to_numpy(dtype=float)
    leg_t = t_days.loc[plot_mask].to_numpy(dtype=float)
    leg_names = transfer_df.loc[plot_mask, "segment"].astype(str).tolist()

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 4.5))
    plt.step(leg_t, leg_dv, where="post", label="Per-transfer DV [km/s]", lw=2.2, color="tab:orange")
    plt.scatter(leg_t, leg_dv, s=45, color="tab:red", zorder=5)
    y_top = max(float(np.max(leg_dv)), 1e-3)
    for td, name in zip(leg_t, leg_names):
        plt.axvline(td, color="gray", ls="--", lw=0.8, alpha=0.5)
        plt.text(td, y_top * 1.02, name.replace("transfer_", "").replace("ballistic_", "coast ").replace("_", " "),
                 rotation=90, va="bottom", ha="center", fontsize=8, color="dimgray")
    plt.xlabel("Mission Time Since Launch [days]")
    plt.ylabel("Transfer DV [km/s]")
    plt.title("Per-Transfer DV with Transfer Markers")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, y_top * 1.22)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "thrust_magnitude_vs_time.png"), dpi=220)
    plt.close()

    burn_t_days = (burn_df["et_seconds"] - launch_et) / SECONDS_PER_DAY
    plt.figure(figsize=(10, 4.5))
    plt.plot(burn_t_days, burn_df["dv_dir_x"], marker="o", label="x", lw=2.0)
    plt.plot(burn_t_days, burn_df["dv_dir_y"], marker="o", label="y", lw=2.0)
    plt.plot(burn_t_days, burn_df["dv_dir_z"], marker="o", label="z", lw=2.0)
    for td, name in zip(leg_t, leg_names):
        plt.axvline(td, color="gray", ls="--", lw=0.8, alpha=0.5)
        plt.text(td, 1.03, name.replace("transfer_", "").replace("ballistic_", "coast ").replace("_", " "),
                 rotation=90, va="bottom", ha="center", fontsize=8, color="dimgray")
    plt.xlabel("Mission Time Since Launch [days]")
    plt.ylabel("DV Direction Component [-]")
    plt.title("Burn Direction Components with Transfer Markers")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.05, 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "thrust_direction_components_vs_time.png"), dpi=220)
    plt.close()

    m0 = DEFAULT_M_INIT_KG
    cumulative = np.cumsum(np.clip(leg_dv, 0.0, None))
    prop_ref = m0 * (1.0 - np.exp(-cumulative / 20.0))
    mass_ref = m0 - prop_ref
    plt.figure(figsize=(10, 4.5))
    plt.step(leg_t, mass_ref, where="post", label="Reference mass [kg]", lw=2.2, color="tab:green")
    plt.scatter(leg_t, mass_ref, s=45, color="tab:olive", zorder=5)
    for td, name in zip(leg_t, leg_names):
        plt.axvline(td, color="gray", ls="--", lw=0.8, alpha=0.5)
        plt.text(td, mass_ref.max() * 1.005, name.replace("transfer_", "").replace("ballistic_", "coast ").replace("_", " "),
                 rotation=90, va="bottom", ha="center", fontsize=8, color="dimgray")
    plt.xlabel("Mission Time Since Launch [days]")
    plt.ylabel("Mass [kg]")
    plt.title("Reference Mass vs Time with Transfer Markers")
    plt.grid(True, alpha=0.3)
    plt.ylim(mass_ref.min() * 0.98, mass_ref.max() * 1.08)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "spacecraft_mass_vs_time.png"), dpi=220)
    plt.close()


def _write_report(folder: str, row: pd.Series, transfer_df: pd.DataFrame, flyby_df: pd.DataFrame) -> None:
    mission_years = (float(row["arrive_a3_et"]) - float(row["launch_et"])) / SECONDS_PER_YEAR
    lines: List[str] = []
    lines.append(f"# Rank {int(row['rank'])}: {row['a1']} -> {row['a2']} -> {row['a3']}")
    lines.append("")
    lines.append("## Mission Summary")
    lines.append("")
    lines.append(f"- Architecture: {row['architecture']} ({row['assist_type']})")
    lines.append(f"- Composition sequence: {row['comp_a1']} -> {row['comp_a2']} -> {row['comp_a3']}")
    lines.append(f"- Earth launch DV [km/s]: {float(row['dv_launch_km_s']):.6f}")
    lines.append(f"- Post-launch transfer DV [km/s]: {float(row['dv_after_launch_km_s']):.6f}")
    lines.append(f"- Mission duration [years]: {mission_years:.4f}")
    if "flyby_et" in row and pd.notna(row["flyby_et"]) and str(row.get("architecture", "")) in ("mars", "moon"):
        lines.append(
            f"- Mars/Moon flyby epoch (ET): {float(row['flyby_et']):.3f} — UTC: {row.get('flyby_utc', '')}"
        )
    lines.append("")
    lines.append("## Transfer DV Breakdown")
    lines.append("")
    lines.append(transfer_df[["segment", "from_body", "to_body", "depart_utc", "arrive_utc", "dv_km_s"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Flyby Geometry")
    lines.append("")
    lines.append(flyby_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `flyby_geometry.csv` includes flyby altitude, bending angles, and v-infinity vectors.")
    lines.append("- `time_position_velocity.csv` includes dense spacecraft/body time-state data.")
    lines.append("- `visualization_flightpath.mp4` is generated with `visualization.py`.")
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _index_from_name(asteroid_list: List[Dict], name: str) -> int:
    upper = str(name).upper()
    for i, ast in enumerate(asteroid_list):
        if str(ast["NAME"]).upper() == upper:
            return i
    raise ValueError(f"Could not find asteroid '{name}' in loaded kernel list.")


def _result_for_row(results: List, asteroid_list: List[Dict], row: pd.Series, rank: int) -> Dict:
    """Get PKL result matching CSV row triplet; fallback to rank only if needed."""
    target = (str(row["a1"]).upper(), str(row["a2"]).upper(), str(row["a3"]).upper())
    for entry in results:
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            continue
        i, j, k, res = entry[0], entry[1], entry[2], entry[3]
        try:
            names = (
                str(asteroid_list[int(i)]["NAME"]).upper(),
                str(asteroid_list[int(j)]["NAME"]).upper(),
                str(asteroid_list[int(k)]["NAME"]).upper(),
            )
        except Exception:
            continue
        if names == target:
            return res
    # Backward-compatible fallback for legacy PKLs that don't carry stable indices.
    return results[rank - 1][3]


def _run_visualization(folder: str, result: Dict, asteroid_list: List[Dict], row: pd.Series) -> None:
    output_mp4 = os.path.join(folder, "visualization_flightpath.mp4")
    try:
        flightpath_animation(
            result,
            asteroid_list,
            _index_from_name(asteroid_list, row["a1"]),
            _index_from_name(asteroid_list, row["a2"]),
            _index_from_name(asteroid_list, row["a3"]),
            t_duration=18.0,
            output_video_name=output_mp4,
            package_folder=folder,
        )
    except Exception as exc:
        with open(os.path.join(folder, "visualization_note.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"visualization.py flightpath_animation failed: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create top-N mission package folders from constrained results (results_<timestamp>.csv/pkl)."
    )
    parser.add_argument("--csv", default=None, help="Path to constrained DV-only CSV.")
    parser.add_argument("--pkl", default=None, help="Path to constrained DV-only PKL.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top missions to package.")
    parser.add_argument(
        "--out",
        default="optimal_asteroid_paths/reports/top_general_solution_packages",
        help="Output parent directory (default: under optimal_asteroid_paths/reports/).",
    )
    parser.add_argument("--clean-out", action="store_true", help="Delete existing output folder before generating.")
    parser.add_argument(
        "--report-dir",
        default="optimal_asteroid_paths/reports",
        help="Directory for top5_report.{md,csv} summary outputs.",
    )
    args = parser.parse_args()

    repo = str(Path(__file__).resolve().parent.parent.parent)
    os.chdir(repo)

    csv_path = args.csv or latest_results_csv(repo)
    pkl_path = args.pkl or latest_results_pkl(repo)
    out_parent = os.path.join(repo, args.out)
    if args.clean_out and os.path.isdir(out_parent):
        shutil.rmtree(out_parent)
    os.makedirs(out_parent, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("rank").head(args.top_n).reset_index(drop=True)
    with open(pkl_path, "rb") as fh:
        results = pickle.load(fh)

    asteroid_list = load_kernels("NOTABLE_ASTEROID_BSPs", "generic_kernels")
    summary_rows = []
    for _, row in df.iterrows():
        rank = int(row["rank"])
        result = _result_for_row(results, asteroid_list, row, rank)
        folder_name = f"rank_{rank:02d}_{_safe_slug(row['a1'])}_{_safe_slug(row['a2'])}_{_safe_slug(row['a3'])}"
        folder = os.path.join(out_parent, folder_name)
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame([row]).to_csv(os.path.join(folder, "solution_row.csv"), index=False)

        body_ids = {
            "a1": str(int(get_id_from_asteroid_name(asteroid_list, str(row["a1"]).upper()))),
            "a2": str(int(get_id_from_asteroid_name(asteroid_list, str(row["a2"]).upper()))),
            "a3": str(int(get_id_from_asteroid_name(asteroid_list, str(row["a3"]).upper()))),
        }
        transfer_df = _build_transfer_dv(row, result)
        transfer_df.to_csv(os.path.join(folder, "transfer_dv_breakdown.csv"), index=False)
        burn_df = _build_burn_events(row, result)
        burn_df.to_csv(os.path.join(folder, "burn_events.csv"), index=False)
        flyby_df = _build_flyby_geometry(result, body_ids["a1"])
        flyby_df.to_csv(os.path.join(folder, "flyby_geometry.csv"), index=False)
        state_df = _build_time_state_csv(result, body_ids, folder=folder)
        state_df.to_csv(os.path.join(folder, "time_position_velocity.csv"), index=False)
        _save_thrust_plots(folder, transfer_df, burn_df)
        _run_visualization(folder, result, asteroid_list, row)
        _write_report(folder, row, transfer_df, flyby_df)
        summary_rows.append(
            {
                "rank": rank,
                "a1": row["a1"],
                "a2": row["a2"],
                "a3": row["a3"],
                "earth_launch_dv_km_s": float(row["dv_launch_km_s"]),
                "post_launch_dv_km_s": float(row["dv_after_launch_km_s"]),
                "folder": folder_name,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("rank")

    report_dir = os.path.join(repo, args.report_dir)
    os.makedirs(report_dir, exist_ok=True)
    summary_md = os.path.join(report_dir, "top5_report.md")
    with open(summary_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Top {args.top_n} constrained solutions\n\n")
        fh.write(f"Source: `{csv_path}` / `{pkl_path}`\n\n")
        fh.write(summary_df.to_markdown(index=False))
        fh.write("\n")
    summary_df.to_csv(os.path.join(report_dir, "top5_report.csv"), index=False)

    print(f"Created package folder: {out_parent}")


if __name__ == "__main__":
    main()
