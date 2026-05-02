"""
visualization.py — Flightpath animation, asteroid orbit graphing, and static trajectory plots.
"""

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import spiceypy
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from core import DAY, HOUR, MU_SUN, WEEK, YEAR, get_state, solve_lambert, two_body_sim


AU_KM = 149597870.7

# Synoptic MP4 theme (matches mission “board” style)
C_BG = "#0b1020"
C_PANEL = "#12182a"
C_GRID = "#2a3555"
C_TEXT = "#e8edf7"
C_MUTED = "#8b9bb4"
C_SUN = "#f5d547"
C_EARTH = "#5cadff"
C_MARS = "#e85d4c"
C_A1, C_A2, C_A3 = "#c084fc", "#4ade80", "#fbbf24"
C_COAST = "#ff8c42"
C_ELEC = "#38bdf8"
C_STAY = "#94a3b8"


def _get_positions(body_id, et_array):
    """Get positions at many epochs; return (N, 3) array in km."""
    return np.array([get_state(str(body_id), float(et))[0] for et in et_array])


def _phase_color_csv(phase: str) -> str:
    ph = str(phase).lower()
    if "stay" in ph:
        return C_STAY
    if "ballistic" in ph or ("earth" in ph and "mars" in ph):
        return C_COAST
    if "heliocentric" in ph or "flyby" in ph:
        return C_ELEC
    if "transfer_earth" in ph:
        return C_COAST
    if "transfer_a" in ph:
        return C_ELEC
    return C_ELEC


def _bridge_xy_gaps(
    et: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    colors: List[str],
    gap_au: float = 0.06,
    max_seg: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Insert linear XY bridges where consecutive samples jump (e.g. legacy CSV)."""
    if len(et) < 2:
        return et, x, y, colors
    et_l: List[float] = []
    x_l: List[float] = []
    y_l: List[float] = []
    c_l: List[str] = []
    for i in range(len(et)):
        if i == 0:
            et_l.append(float(et[0]))
            x_l.append(float(x[0]))
            y_l.append(float(y[0]))
            c_l.append(colors[0])
            continue
        x0b, y0b, et0b = x_l[-1], y_l[-1], et_l[-1]
        dxy = float(np.hypot(float(x[i]) - x0b, float(y[i]) - y0b))
        dt = float(et[i] - et0b)
        if dxy > gap_au and dt > 1.0:
            n_ins = min(max_seg, max(2, int(np.ceil(dxy / gap_au))))
            x1b, y1b, et1b = float(x[i]), float(y[i]), float(et[i])
            for j in range(1, n_ins):
                s = j / float(n_ins)
                et_l.append(et0b + s * dt)
                x_l.append(x0b + s * (x1b - x0b))
                y_l.append(y0b + s * (y1b - y0b))
                c_l.append(c_l[-1])
        et_l.append(float(et[i]))
        x_l.append(float(x[i]))
        y_l.append(float(y[i]))
        c_l.append(colors[i])
    return np.asarray(et_l), np.asarray(x_l), np.asarray(y_l), c_l


def _dedupe_et_series(
    et: np.ndarray, x: np.ndarray, y: np.ndarray, colors: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Collapse duplicate ET rows (keep last) so ``np.interp`` has strictly increasing x."""
    et = np.asarray(et, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(et, kind="mergesort")
    et, x, y = et[order], x[order], y[order]
    colors = [colors[i] for i in order]
    et_o: List[float] = []
    x_o: List[float] = []
    y_o: List[float] = []
    c_o: List[str] = []
    for i in range(len(et)):
        if et_o and abs(et[i] - et_o[-1]) < 1e-6:
            x_o[-1], y_o[-1], c_o[-1] = float(x[i]), float(y[i]), colors[i]
        else:
            et_o.append(float(et[i]))
            x_o.append(float(x[i]))
            y_o.append(float(y[i]))
            c_o.append(colors[i])
    return np.asarray(et_o), np.asarray(x_o), np.asarray(y_o), c_o


def _read_state_csv(folder: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    path = os.path.join(folder, "time_position_velocity.csv")
    if not os.path.isfile(path):
        return None
    ets, xs, ys, phases = [], [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ets.append(float(row["et_seconds"]))
            xs.append(float(row["sc_x_km"]) / AU_KM)
            ys.append(float(row["sc_y_km"]) / AU_KM)
            phases.append(row.get("phase", ""))
    if not ets:
        return None
    return np.asarray(ets), np.asarray(xs), np.asarray(ys), phases


def _mission_years_sidebar(folder: str, pdv: Dict) -> Optional[str]:
    sr = os.path.join(folder, "solution_row.csv")
    if os.path.isfile(sr):
        with open(sr, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        r = rows[0] if rows else None
        if r and r.get("mission_years"):
            try:
                return f"Mission span · {float(r['mission_years']):.3f} yr"
            except (TypeError, ValueError):
                pass
    if pdv.get("mission_years") is not None:
        try:
            return f"Mission span · {float(pdv['mission_years']):.3f} yr"
        except (TypeError, ValueError):
            pass
    t0 = pdv.get("et_launch")
    t1 = pdv.get("et_arrive_3")
    if t0 is not None and t1 is not None:
        try:
            return f"Mission span · {(float(t1) - float(t0)) / YEAR:.3f} yr"
        except (TypeError, ValueError):
            pass
    return None


def _sidebar_from_package(folder: str, pdv: Dict, a1: str, a2: str, a3: str) -> str:
    lines: List[str] = []
    lines.append(f"{a1} → {a2} → {a3}")
    lines.append("")
    arch = str(pdv.get("architecture", "—"))
    lines.append(f"Arch · {arch.upper()}")
    my = _mission_years_sidebar(folder, pdv)
    if my:
        lines.append(my)
    dv_l = pdv.get("delta_v_launch")
    if dv_l is not None and len(np.asarray(dv_l).ravel()) >= 1:
        lines.append(f"Launch |Δv| … {float(np.linalg.norm(dv_l)):.3f} km/s")
    if pdv.get("delta_v_total") is not None:
        try:
            lines.append(f"Post-launch Σ|Δv| … {float(pdv['delta_v_total']):.3f} km/s")
        except (TypeError, ValueError):
            pass
    tr = os.path.join(folder, "transfer_dv_breakdown.csv")
    if os.path.isfile(tr):
        lines.append("")
        lines.append("Segments (ref)")
        lines.append("─" * 20)
        with open(tr, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seg = str(row.get("segment", ""))[:22]
                dv = float(row.get("dv_km_s", 0) or 0)
                dt = (float(row["arrive_et"]) - float(row["depart_et"])) / 86400.0
                if dt < 0.5:
                    lines.append(f"{seg:22s} {dv:5.2f}")
                else:
                    lines.append(f"{seg:22s} {dv:5.2f} {dt:6.0f}d")
    return "\n".join(lines)


def _build_path_from_lambert(
    pdv: Dict, a_id_1: str, a_id_2: str, a_id_3: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Heliocentric XY path in AU + phase label per sample (Lambert / two-body)."""
    et_list: List[float] = []
    x_list: List[float] = []
    y_list: List[float] = []
    col_list: List[str] = []

    def append_kepler(X, T, et0, color):
        for k in range(len(T)):
            et_list.append(float(et0 + T[k]))
            x_list.append(float(X[k, 0]) / AU_KM)
            y_list.append(float(X[k, 1]) / AU_KM)
            col_list.append(color)

    def append_stay(body_id: str, et_a: float, et_b: float, n: int, color: str):
        for et in np.linspace(et_a, et_b, max(2, n)):
            r, _ = get_state(body_id, float(et))
            et_list.append(float(et))
            x_list.append(float(r[0]) / AU_KM)
            y_list.append(float(r[1]) / AU_KM)
            col_list.append(color)

    mu_sun = MU_SUN
    arch = str(pdv.get("architecture", "direct"))
    et_fb = pdv.get("et_flyby", pdv.get("et_mars"))
    try:
        et_fb = float(et_fb)
    except (TypeError, ValueError):
        et_fb = float("nan")

    t_diff_1 = float(pdv["et_arrive_1"] - pdv["et_launch"])
    t_diff_2 = float(pdv["et_arrive_2"] - pdv["et_stay_1"])
    t_diff_3 = float(pdv["et_arrive_3"] - pdv["et_stay_2"])

    if arch in ("mars", "moon") and np.isfinite(et_fb):
        fb_id = str(pdv.get("flyby_body", "4" if arch == "mars" else "301"))
        t_coast = float(et_fb - pdv["et_launch"])
        t_to_a1 = float(pdv["et_arrive_1"] - et_fb)
        r_earth, v_earth = get_state("399", pdv["et_launch"])
        x0 = np.concatenate([r_earth, v_earth + pdv["delta_v_launch"]])
        X_1a, T_1a = two_body_sim(t_coast, x0, mu_sun)
        append_kepler(X_1a, T_1a, pdv["et_launch"], C_COAST)
        r_fb, _ = get_state(fb_id, et_fb)
        vo = pdv.get("v_sc_post_flyby")
        vout = np.asarray(vo, dtype=float).ravel() if vo is not None else np.zeros(0)
        if vout.size < 3:
            a1_r, _ = get_state(a_id_1, pdv["et_arrive_1"])
            tof_d = t_to_a1 / DAY
            _, vout, ef = solve_lambert(r_fb, a1_r, tof_d, 0, mu_sun)
            if ef != 1:
                _, vout, _ = solve_lambert(r_fb, a1_r, tof_d, 1, mu_sun)
            vout = np.asarray(vout, dtype=float).ravel()
        x0b = np.concatenate([r_fb, vout])
        X_1b, T_1b = two_body_sim(t_to_a1, x0b, mu_sun)
        if len(X_1b) > 1:
            append_kepler(X_1b[1:, :], T_1b[1:], et_fb, C_ELEC)
    else:
        r_earth, v_earth = get_state("399", pdv["et_launch"])
        x0 = np.concatenate([r_earth, v_earth + pdv["delta_v_launch"]])
        X_1, T_1 = two_body_sim(t_diff_1, x0, mu_sun)
        append_kepler(X_1, T_1, pdv["et_launch"], C_ELEC)

    append_stay(a_id_1, float(pdv["et_arrive_1"]), float(pdv["et_stay_1"]), 40, C_STAY)

    r_a1, v_a1 = get_state(a_id_1, pdv["et_stay_1"])
    x0 = np.concatenate([r_a1, v_a1 + pdv["delta_v_A1_leave"]])
    X_2, T_2 = two_body_sim(t_diff_2, x0, mu_sun)
    append_kepler(X_2, T_2, pdv["et_stay_1"], C_ELEC)

    append_stay(a_id_2, float(pdv["et_arrive_2"]), float(pdv["et_stay_2"]), 40, C_STAY)

    r_a2, v_a2 = get_state(a_id_2, pdv["et_stay_2"])
    x0 = np.concatenate([r_a2, v_a2 + pdv["delta_v_A2_leave"]])
    X_3, T_3 = two_body_sim(t_diff_3, x0, mu_sun)
    append_kepler(X_3, T_3, pdv["et_stay_2"], C_ELEC)

    return np.asarray(et_list), np.asarray(x_list), np.asarray(y_list), col_list


def _synoptic_legend_handles(a1: str, a2: str, a3: str, show_mars: bool) -> List[Line2D]:
    h = [
        Line2D([0], [0], color=C_SUN, marker="o", ls="", ms=7, label="Sun"),
        Line2D([0], [0], color=C_EARTH, marker="o", ls="", ms=6, label="Earth"),
    ]
    if show_mars:
        h.append(Line2D([0], [0], color=C_MARS, marker="D", ls="", ms=6, label="Mars"))
    h.extend(
        [
            Line2D([0], [0], color=C_A1, marker="o", ls="", ms=6, label=a1),
            Line2D([0], [0], color=C_A2, marker="o", ls="", ms=6, label=a2),
            Line2D([0], [0], color=C_A3, marker="o", ls="", ms=6, label=a3),
            Line2D([0], [0], color=C_COAST, lw=2.4, label="Coast / escape"),
            Line2D([0], [0], color=C_ELEC, lw=2.4, label="Powered (LT ref)"),
            Line2D([0], [0], color=C_STAY, lw=2.4, label="Stay at asteroid"),
        ]
    )
    return h


def _plot_colored_path(ax, x, y, colors: List[str], end_idx: int) -> None:
    end_idx = min(end_idx, len(x) - 1)
    i = 0
    while i <= end_idx:
        c = colors[i]
        j = i + 1
        while j <= end_idx and colors[j] == c:
            j += 1
        ax.plot(x[i:j], y[i:j], color=c, lw=2.8, solid_capstyle="round", zorder=8)
        i = j


def flightpath_animation(
    path_defined_vector,
    asteroid_list,
    a_index_1,
    a_index_2,
    a_index_3,
    t_duration,
    output_video_name,
    package_folder: Optional[str] = None,
):
    """Create a 2D heliocentric (ECLIPJ2000 XY) synoptic MP4 with Mars + sidebar.

    When *package_folder* points to a rank folder containing
    ``time_position_velocity.csv``, the video follows that dense path (recommended
    for packaged exports). Otherwise the path is rebuilt from Lambert / two-body
    segments in *path_defined_vector*.

    Requires FFmpeg. Uses SPICE ``get_state`` for body trails and Mars orbit.
    """
    a_id_1 = str(int(asteroid_list[a_index_1]["ID"]))
    a_id_2 = str(int(asteroid_list[a_index_2]["ID"]))
    a_id_3 = str(int(asteroid_list[a_index_3]["ID"]))
    a_name_1 = asteroid_list[a_index_1]["NAME"]
    a_name_2 = asteroid_list[a_index_2]["NAME"]
    a_name_3 = asteroid_list[a_index_3]["NAME"]
    pdv = path_defined_vector

    pkg = package_folder
    if pkg is None:
        cand = os.path.dirname(os.path.abspath(output_video_name))
        if os.path.isfile(os.path.join(cand, "time_position_velocity.csv")):
            pkg = cand

    csv_block = _read_state_csv(pkg) if pkg else None
    if csv_block is not None:
        et_raw, x_raw, y_raw, phase_list = csv_block
        colors_raw = [_phase_color_csv(p) for p in phase_list]
    else:
        et_raw, x_raw, y_raw, colors_raw = _build_path_from_lambert(pdv, a_id_1, a_id_2, a_id_3)

    et_raw, x_raw, y_raw, colors_raw = _dedupe_et_series(et_raw, x_raw, y_raw, colors_raw)
    et_raw, x_raw, y_raw, colors_raw = _bridge_xy_gaps(et_raw, x_raw, y_raw, colors_raw)

    et0 = float(et_raw[0])
    et1 = float(et_raw[-1])
    span = max(et1 - et0, 1.0)
    n_dense = int(min(5000, max(500, span / (12.0 * HOUR))))
    et_path = np.linspace(et0, et1, n_dense)
    x_au = np.interp(et_path, et_raw, x_raw)
    y_au = np.interp(et_path, et_raw, y_raw)
    ix = np.searchsorted(et_raw, et_path, side="right") - 1
    ix = np.clip(ix, 0, max(len(colors_raw) - 1, 0))
    colors = [colors_raw[j] for j in ix]

    mission_time = np.linspace(et0, et1, min(500, max(100, int((et1 - et0) / (14 * DAY)))))

    earth_xy = _get_positions("399", mission_time)[:, :2] / AU_KM
    a1_xy = _get_positions(a_id_1, mission_time)[:, :2] / AU_KM
    a2_xy = _get_positions(a_id_2, mission_time)[:, :2] / AU_KM
    a3_xy = _get_positions(a_id_3, mission_time)[:, :2] / AU_KM

    arch = str(pdv.get("architecture", "direct")).lower()
    mars_xy = None
    if arch == "mars":
        ets_m = np.linspace(et0, et1, 450)
        mx = []
        my = []
        for et in ets_m:
            r, _ = get_state("4", float(et))
            mx.append(r[0] / AU_KM)
            my.append(r[1] / AU_KM)
        mars_xy = (np.asarray(mx), np.asarray(my))

    sidebar = _sidebar_from_package(pkg or "", pdv, a_name_1, a_name_2, a_name_3)

    n_frames = int(max(48, min(900, t_duration * 30)))
    n_path = len(et_path)
    if n_frames <= 1:
        idx_frames = np.zeros(1, dtype=int)
    else:
        idx_frames = np.round(np.linspace(0, n_path - 1, n_frames)).astype(int)

    fig = plt.figure(figsize=(14.5, 9.0), facecolor=C_BG)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4.15, 1.0], wspace=0.04)
    ax_m = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_m.set_facecolor(C_BG)
    ax_s.set_facecolor(C_PANEL)
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.text(
        0.04,
        0.98,
        sidebar,
        transform=ax_s.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        color=C_TEXT,
        family="monospace",
        linespacing=1.32,
    )

    fps = n_frames / float(t_duration)
    writer = FFMpegWriter(fps=fps)

    all_x = np.concatenate([x_au, x_raw, earth_xy[:, 0], a1_xy[:, 0], a2_xy[:, 0], a3_xy[:, 0]])
    all_y = np.concatenate([y_au, y_raw, earth_xy[:, 1], a1_xy[:, 1], a2_xy[:, 1], a3_xy[:, 1]])
    if mars_xy is not None:
        all_x = np.concatenate([all_x, mars_xy[0]])
        all_y = np.concatenate([all_y, mars_xy[1]])
    pad = 0.32
    xr = float(np.ptp(all_x)) * 0.5 + pad
    yr = float(np.ptp(all_y)) * 0.5 + pad
    cx = float(0.5 * (all_x.min() + all_x.max()))
    cy = float(0.5 * (all_y.min() + all_y.max()))
    rlim = max(xr, yr)

    legend_handles = _synoptic_legend_handles(a_name_1, a_name_2, a_name_3, mars_xy is not None)

    with writer.saving(fig, output_video_name, dpi=160):
        for fi in tqdm(range(n_frames), desc="Synoptic flightpath MP4"):
            end_ix = int(idx_frames[fi])
            et_now = float(et_path[end_ix])

            ax_m.clear()
            ax_m.set_facecolor(C_BG)
            ax_m.add_patch(Circle((0.0, 0.0), 0.035, facecolor=C_SUN, edgecolor="#ffe9a6", zorder=20))

            ax_m.plot(earth_xy[:, 0], earth_xy[:, 1], "--", color=C_EARTH, lw=1.0, alpha=0.45, zorder=2)
            ax_m.plot(a1_xy[:, 0], a1_xy[:, 1], "--", color=C_A1, lw=1.0, alpha=0.45, zorder=2)
            ax_m.plot(a2_xy[:, 0], a2_xy[:, 1], "--", color=C_A2, lw=1.0, alpha=0.45, zorder=2)
            ax_m.plot(a3_xy[:, 0], a3_xy[:, 1], "--", color=C_A3, lw=1.0, alpha=0.45, zorder=2)
            if mars_xy is not None:
                ax_m.plot(mars_xy[0], mars_xy[1], "--", color=C_MARS, lw=1.05, alpha=0.5, zorder=3)

            _plot_colored_path(ax_m, x_au, y_au, colors, end_ix)
            ax_m.legend(
                handles=legend_handles,
                loc="lower left",
                fontsize=6.8,
                framealpha=0.94,
                facecolor=C_PANEL,
                edgecolor=C_GRID,
                labelcolor=C_TEXT,
                borderpad=0.35,
            )

            re, _ = get_state("399", et_now)
            r1, _ = get_state(a_id_1, et_now)
            r2, _ = get_state(a_id_2, et_now)
            r3, _ = get_state(a_id_3, et_now)
            ax_m.scatter([re[0] / AU_KM], [re[1] / AU_KM], s=28, c=C_EARTH, zorder=12, edgecolors="white", linewidths=0.3)
            ax_m.scatter([r1[0] / AU_KM], [r1[1] / AU_KM], s=26, c=C_A1, zorder=12, edgecolors="white", linewidths=0.3)
            ax_m.scatter([r2[0] / AU_KM], [r2[1] / AU_KM], s=26, c=C_A2, zorder=12, edgecolors="white", linewidths=0.3)
            ax_m.scatter([r3[0] / AU_KM], [r3[1] / AU_KM], s=26, c=C_A3, zorder=12, edgecolors="white", linewidths=0.3)
            if mars_xy is not None:
                rm, _ = get_state("4", et_now)
                ax_m.scatter([rm[0] / AU_KM], [rm[1] / AU_KM], s=24, c=C_MARS, zorder=11, marker="D", edgecolors="white", linewidths=0.3)

            scx, scy = float(x_au[end_ix]), float(y_au[end_ix])
            ax_m.scatter([scx], [scy], s=42, c="white", zorder=25, marker="o", edgecolors="#0ea5e9", linewidths=1.2)

            ax_m.set_xlim(cx - rlim, cx + rlim)
            ax_m.set_ylim(cy - rlim, cy + rlim)
            ax_m.set_aspect("equal", adjustable="box")
            ax_m.set_xlabel("X [AU] · ECLIPJ2000", color=C_MUTED, fontsize=9)
            ax_m.set_ylabel("Y [AU] · ECLIPJ2000", color=C_MUTED, fontsize=9)
            ax_m.tick_params(colors=C_MUTED, labelsize=8)
            ax_m.grid(True, color=C_GRID, ls=":", alpha=0.55)
            title = f"{a_name_1} → {a_name_2} → {a_name_3}  ·  {spiceypy.et2utc(et_now, 'C', 3)}"
            ax_m.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=10)

            writer.grab_frame()

    plt.close(fig)
    print(f"Video saved: {output_video_name}")


# =============================================================================
# ASTEROID ORBIT ANIMATION
# =============================================================================

def graph_asteroids(asteroid_list, t_duration, fps, start_date, end_date,
                    output_video_name):
    """Create an animated MP4 showing all asteroid orbits over time.

    Parameters
    ----------
    asteroid_list : list of dict
        Each dict has 'ID' and 'NAME' keys.
    t_duration : float
        Desired video duration in seconds.
    fps : int
        Frames per second.
    start_date : str
        UTC start date string (e.g. 'Jan 1 12:00:00 UTC 2027').
    end_date : str
        UTC end date string.
    output_video_name : str
        Output filename (e.g. "asteroids.mp4").
    """
    et0 = spiceypy.str2et(start_date)
    etf = spiceypy.str2et(end_date)

    t_range = np.arange(et0, etf, 4 * WEEK)

    N = int(fps * t_duration)
    K = len(t_range) / N

    num_asteroids = len(asteroid_list)

    # Pre-compute all asteroid positions: shape (num_asteroids, len(t_range), 3)
    asteroid_positions = np.zeros((num_asteroids, len(t_range), 3))
    for j, ast in enumerate(asteroid_list):
        a_id = str(int(ast["ID"]))
        for ti, et in enumerate(t_range):
            state, _ = spiceypy.spkezr(a_id, float(et), 'ECLIPJ2000', 'NONE', '0')
            asteroid_positions[j, ti, :] = state[0:3]

    colors = plt.cm.hsv(np.linspace(0, 1, num_asteroids))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_video_name, dpi=150):
        for i in tqdm(range(N), desc='Encoding frames'):
            ax.clear()
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

            idx = min(int(np.ceil((i + 1) * K)) - 1, len(t_range) - 1)

            position_list = np.zeros((num_asteroids, 3))

            for j, ast in enumerate(asteroid_list):
                ax.plot(asteroid_positions[j, :, 0],
                        asteroid_positions[j, :, 1],
                        asteroid_positions[j, :, 2],
                        color=colors[j], label=ast["NAME"])

                position_list[j, :] = asteroid_positions[j, idx, :]

            ax.scatter(position_list[:, 0], position_list[:, 1],
                       position_list[:, 2], s=60, c=colors)

            title_date = spiceypy.et2utc(float(t_range[idx]), 'C', 1)
            ax.set_title(f'Asteroid Trajectories at {title_date}')
            ax.set_xlabel('X Position (km)')
            ax.set_ylabel('Y Position (km)')
            ax.set_zlabel('Z Position (km)')
            ax.legend(fontsize='small')
            ax.view_init(elev=30, azim=-60)  # 3D view (view(3) equivalent)

            writer.grab_frame()

    plt.close(fig)
    print(f"Video saved: {output_video_name}")


# =============================================================================
# STATIC TRAJECTORY PLOT
# =============================================================================

def graph_asteroid_flightpath(path_defined_vector, a_id_1, a_id_2):
    """Plot the three trajectory legs in 3D.

    Parameters
    ----------
    path_defined_vector : dict
        Keys: et_launch, et_arrive_1, et_stay_1, et_arrive_2, et_stay_2,
              et_arrive_3, delta_v_launch, delta_v_A1_leave, delta_v_A2_leave.
    a_id_1 : str
        NAIF ID string for asteroid 1.
    a_id_2 : str
        NAIF ID string for asteroid 2.
    """
    pdv = path_defined_vector

    t_1 = pdv['et_arrive_1'] - pdv['et_launch']
    t_2 = pdv['et_arrive_2'] - pdv['et_stay_1']
    t_3 = pdv['et_arrive_3'] - pdv['et_stay_2']

    mu_sun = MU_SUN

    # --- Leg 1: Earth to Asteroid 1 ---
    r_earth, v_earth = get_state('399', pdv['et_launch'])
    x_0 = np.concatenate([r_earth, v_earth + pdv['delta_v_launch']])
    X_1, T_1 = two_body_sim(t_1, x_0, mu_sun)

    # --- Leg 2: Asteroid 1 to Asteroid 2 ---
    r_a1, v_a1 = get_state(a_id_1, pdv['et_stay_1'])
    x_0 = np.concatenate([r_a1, v_a1 + pdv['delta_v_A1_leave']])
    X_2, T_2 = two_body_sim(t_2, x_0, mu_sun)

    # --- Leg 3: Asteroid 2 to Asteroid 3 ---
    r_a2, v_a2 = get_state(a_id_2, pdv['et_stay_2'])
    x_0 = np.concatenate([r_a2, v_a2 + pdv['delta_v_A2_leave']])
    X_3, T_3 = two_body_sim(t_3, x_0, mu_sun)

    # --- Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    ax.plot(X_1[:, 0], X_1[:, 1], X_1[:, 2], 'r', label='Trajectory to A1')
    ax.plot(X_2[:, 0], X_2[:, 1], X_2[:, 2], 'g', label='Trajectory from A1 to A2')
    ax.plot(X_3[:, 0], X_3[:, 1], X_3[:, 2], 'b', label='Trajectory from A2')

    ax.legend()
    ax.view_init(elev=30, azim=-60)  # 3D view
    plt.show()
