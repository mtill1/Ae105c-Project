"""Render 2D GIFs for the top 3 hybrid LT missions.

Reads results_hybrid_lt.pkl (sorted by final delivered mass), looks up each
triplet's impulsive trajectory from results_69ast_ega_real.pkl, and animates
the full 4-leg mission with the LT scoring info in the title.
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import spiceypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (load_kernels, solve_lambert, two_body_sim, get_state,
                  MU_SUN, DAY, YEAR)

AU = 149_597_870.7

FLYBY_META = {
    '399': {'name': 'Earth', 'color': '#4FC3F7'},
    '301': {'name': 'Moon',  'color': '#BDBDBD'},
    '4':   {'name': 'Mars',  'color': '#FF8A65'},
}


def build_full_mission(res, a_ids, flyby_id, n_per_leg=300):
    """Build concatenated (xy, et, tag) for 4 legs + 2 stays."""
    et_l  = res['et_launch']
    et_fb = res['et_flyby']
    et_a1a = res['et_arrive_1']
    et_a1d = res['et_stay_1']
    et_a2a = res['et_arrive_2']
    et_a2d = res['et_stay_2']
    et_a3a = res['et_arrive_3']

    r_e_l, v_e_l = get_state('399', et_l)
    r_fb,   v_fb = get_state(flyby_id, et_fb)
    r_a1a, _ = get_state(a_ids[0], et_a1a)
    r_a1d, _ = get_state(a_ids[0], et_a1d)
    r_a2a, _ = get_state(a_ids[1], et_a2a)
    r_a2d, _ = get_state(a_ids[1], et_a2d)
    r_a3a, _ = get_state(a_ids[2], et_a3a)

    xs, ys, ets, tags = [], [], [], []

    def add_leg(r0, v_dep, dt_sec, et_start, tag, n):
        X, T = two_body_sim(dt_sec, np.concatenate([r0, v_dep]), MU_SUN, n_steps=n)
        for pt, t in zip(X, T):
            xs.append(pt[0]); ys.append(pt[1])
            ets.append(float(et_start + t)); tags.append(tag)

    # L1: Earth launch → flyby body.  Use stored delta_v_launch for EXACT reproduction.
    v_dep1 = v_e_l + np.asarray(res['delta_v_launch'])
    add_leg(r_e_l, v_dep1, et_fb - et_l, et_l, 'L1', n_per_leg)

    # L2: flyby → A1
    v_dep2, _, _ = solve_lambert(r_fb, r_a1a, (et_a1a - et_fb)/DAY, 0, MU_SUN)
    add_leg(r_fb, v_dep2, et_a1a - et_fb, et_fb, 'L2', n_per_leg)

    # Stay A1
    for tt in np.linspace(et_a1a, et_a1d, 20):
        r, _ = get_state(a_ids[0], float(tt))
        xs.append(r[0]); ys.append(r[1]); ets.append(float(tt)); tags.append('stay1')

    # L3: A1 → A2
    v_dep3, _, _ = solve_lambert(r_a1d, r_a2a, (et_a2a - et_a1d)/DAY, 0, MU_SUN)
    add_leg(r_a1d, v_dep3, et_a2a - et_a1d, et_a1d, 'L3', n_per_leg)

    # Stay A2
    for tt in np.linspace(et_a2a, et_a2d, 20):
        r, _ = get_state(a_ids[1], float(tt))
        xs.append(r[0]); ys.append(r[1]); ets.append(float(tt)); tags.append('stay2')

    # L4: A2 → A3
    v_dep4, _, _ = solve_lambert(r_a2d, r_a3a, (et_a3a - et_a2d)/DAY, 0, MU_SUN)
    add_leg(r_a2d, v_dep4, et_a3a - et_a2d, et_a2d, 'L4', n_per_leg)

    return np.array(xs), np.array(ys), np.array(ets), np.array(tags)


def resample_uniform_time(xs, ys, ets, tags, n):
    t_target = np.linspace(ets[0], ets[-1], n)
    xo = np.interp(t_target, ets, xs)
    yo = np.interp(t_target, ets, ys)
    idx = np.clip(np.searchsorted(ets, t_target), 0, len(tags) - 1)
    return xo, yo, t_target, tags[idx]


CHEM_COLOR = '#FF8A65'   # warm orange — chemical (impulsive)
LT_COLOR   = '#4FC3F7'   # cyan       — low-thrust (electric)


def leg_propulsion(tag, arch):
    """Return (propulsion_label, color) for a given leg.  L1 & L2 always chem."""
    if tag == 'L1': return ('CHEM', CHEM_COLOR)
    if tag == 'L2': return ('CHEM', CHEM_COLOR)
    if tag == 'L3':
        return ('LT', LT_COLOR) if arch[0]=='E' else ('CHEM', CHEM_COLOR)
    if tag == 'L4':
        return ('LT', LT_COLOR) if arch[1]=='E' else ('CHEM', CHEM_COLOR)
    return ('', '#FFFFFF')


def leg_color(tag, arch):
    return leg_propulsion(tag, arch)[1]


def body_orbit_xy(body_id, et0, etf, n=300):
    ts = np.linspace(et0, etf, n)
    pts = np.array([get_state(str(body_id), float(t))[0] for t in ts])
    return pts[:, 0]/AU, pts[:, 1]/AU


def make_gif(res, hyb, names, ids, flyby_id, out_path, fps=30, duration_s=30):
    arch = hyb['best_arch']
    xs, ys, ets, tags = build_full_mission(res, ids, flyby_id)
    n_frames = fps * duration_s
    x_u, y_u, et_u, tag_u = resample_uniform_time(xs, ys, ets, tags, n_frames)

    et0, etf = res['et_launch'], res['et_arrive_3']
    fb_name  = FLYBY_META.get(flyby_id,{}).get('name', '?')
    fb_color = FLYBY_META.get(flyby_id,{}).get('color','#BBBBBB')

    e_x, e_y   = body_orbit_xy('399', et0, etf)
    fb_x, fb_y = body_orbit_xy(flyby_id, et0, etf) if flyby_id != '399' else (e_x, e_y)
    a1_x, a1_y = body_orbit_xy(ids[0], et0, etf)
    a2_x, a2_y = body_orbit_xy(ids[1], et0, etf)
    a3_x, a3_y = body_orbit_xy(ids[2], et0, etf)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a1f')
    ax.set_facecolor('#0a0a1f')

    ax.plot(e_x, e_y, ':', color='#4FC3F7', lw=1, alpha=0.7, label='Earth')
    if flyby_id != '399':
        ax.plot(fb_x, fb_y, ':', color=fb_color, lw=1, alpha=0.7, label=fb_name)
    ax.plot(a1_x, a1_y, ':', color='#EF5350', lw=0.9, alpha=0.5, label=names[0])
    ax.plot(a2_x, a2_y, ':', color='#5BBD72', lw=0.9, alpha=0.5, label=names[1])
    ax.plot(a3_x, a3_y, ':', color='#9575CD', lw=0.9, alpha=0.5, label=names[2])
    ax.scatter(0, 0, s=260, c='#FFD54F', edgecolor='#F57F17', zorder=10, label='Sun')

    all_x = np.concatenate([e_x, fb_x, a1_x, a2_x, a3_x, xs/AU])
    all_y = np.concatenate([e_y, fb_y, a1_y, a2_y, a3_y, ys/AU])
    lim = 1.1 * max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#666')
    ax.grid(True, alpha=0.15, color='white')
    ax.set_xlabel('X (AU)', color='white')
    ax.set_ylabel('Y (AU)', color='white')
    ax.set_title(f"Mission: {' → '.join(names)}    [{arch} hybrid, {fb_name} GA]",
                 color='white', fontsize=12, fontweight='bold')

    info = (f"Arch: {arch}  (L3={'E' if arch[0]=='E' else 'C'}, "
            f"L4={'E' if arch[1]=='E' else 'C'})\n"
            f"Launch mass  : {hyb['m_init_kg']:.0f} kg\n"
            f"Final mass   : {hyb['m_best_kg']:.1f} kg   (vs {hyb['m_baseline_CC_kg']:.1f} kg all-chem)\n"
            f"Improvement  : +{hyb['improvement_kg']:.1f} kg   "
            f"(+{hyb['improvement_kg']/hyb['m_baseline_CC_kg']*100:.0f}%)\n"
            f"Impulsive Δv : {res['delta_v_total']:.2f} km/s\n"
            f"30s video → each sec ≈ {(etf-et0)/YEAR/duration_s:.2f} yr")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white',
            fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(facecolor='#1a1a3a', edgecolor='#666', alpha=0.9,
                      boxstyle='round,pad=0.5'))

    # Build propulsion legend with a single CHEM and single LT entry
    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0],[0], color=CHEM_COLOR, lw=3, label='Chemical burn (impulsive)'),
        Line2D([0],[0], color=LT_COLOR,   lw=3, label='Low-thrust (electric Isp=3100s)'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + [h.get_label() for h in extra_handles],
              loc='lower right', facecolor='#1a1a3a', edgecolor='#666',
              labelcolor='white', fontsize=8)

    # Static midpoint labels: "L1: CHEM", "L3: LT", etc.
    leg_tags = ['L1', 'L2', 'L3', 'L4']
    for t in leg_tags:
        mask = tags == t
        if not mask.any(): continue
        idx_mid = np.where(mask)[0][len(np.where(mask)[0])//2]
        prop, _ = leg_propulsion(t, arch)
        ax.annotate(f"{t}: {prop}",
                    xy=(xs[idx_mid]/AU, ys[idx_mid]/AU),
                    xytext=(6, 6), textcoords='offset points',
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=leg_color(t, arch), alpha=0.85,
                              edgecolor='white', boxstyle='round,pad=0.2'))

    # Dynamic trails — one per leg with per-leg color depending on arch
    trails = {t: ax.plot([], [], '-', color=leg_color(t, arch), lw=2.5, zorder=7)[0]
              for t in leg_tags}

    (sc_dot,)    = ax.plot([], [], 'o', color='white', ms=9,
                           markeredgecolor='#FFD54F', markeredgewidth=1.5, zorder=12)
    (earth_dot,) = ax.plot([], [], 'o', color='#4FC3F7', ms=8, zorder=11)
    (fb_dot,)    = ax.plot([], [], 'o', color=fb_color, ms=7, zorder=11) \
                   if flyby_id != '399' else (earth_dot,)
    (a1_dot,)    = ax.plot([], [], 'o', color='#EF5350', ms=7, zorder=11)
    (a2_dot,)    = ax.plot([], [], 'o', color='#5BBD72', ms=7, zorder=11)
    (a3_dot,)    = ax.plot([], [], 'o', color='#9575CD', ms=7, zorder=11)

    clock = ax.text(0.98, 0.02, '', transform=ax.transAxes, color='white',
                    fontsize=10, ha='right', va='bottom', family='monospace',
                    bbox=dict(facecolor='#1a1a3a', edgecolor='#666',
                              alpha=0.85, boxstyle='round,pad=0.3'))

    leg_masks = {t: (tag_u == t) for t in leg_tags}

    def init():
        for tr in trails.values(): tr.set_data([], [])
        return list(trails.values()) + [sc_dot, earth_dot, fb_dot, a1_dot, a2_dot, a3_dot, clock]

    def update(fi):
        for t in leg_tags:
            m = leg_masks[t] & (np.arange(len(tag_u)) <= fi)
            trails[t].set_data(x_u[m]/AU, y_u[m]/AU)
        sc_dot.set_data([x_u[fi]/AU], [y_u[fi]/AU])
        et = et_u[fi]
        for dot, bid in [(earth_dot,'399'), (fb_dot, flyby_id),
                         (a1_dot, ids[0]), (a2_dot, ids[1]), (a3_dot, ids[2])]:
            r, _ = get_state(str(bid), float(et))
            dot.set_data([r[0]/AU], [r[1]/AU])
        date = spiceypy.et2utc(float(et), 'C', 0)[:11].strip()
        elapsed = (et - et0)/YEAR
        phase = {'L1':f'Earth→{fb_name} GA',
                 'L2':f'{fb_name}→{names[0]}',
                 'stay1':f'at {names[0]}',
                 'L3':f'{names[0]}→{names[1]}  ({"LT" if arch[0]=="E" else "chem"})',
                 'stay2':f'at {names[1]}',
                 'L4':f'{names[1]}→{names[2]}  ({"LT" if arch[1]=="E" else "chem"})',
                 }.get(tag_u[fi], '')
        clock.set_text(f"{date}   +{elapsed:5.2f} yr\n{phase}")
        return list(trails.values()) + [sc_dot, earth_dot, fb_dot, a1_dot, a2_dot, a3_dot, clock]

    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=1000/fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ast = load_kernels(os.path.join(repo,'NOTABLE_ASTEROID_BSPs'),
                       '/Users/rebnoob/Documents/ae105/generic_kernels')

    with open(os.path.join(repo,'optimal_asteroid_paths/pkl/results_hybrid_lt.pkl'),'rb') as f:
        hyb_all = pickle.load(f)
    with open(os.path.join(repo,'optimal_asteroid_paths/pkl/results_69ast_ega_real.pkl'),'rb') as f:
        imp_all = pickle.load(f)
    imp_by_idx = {(r[0],r[1],r[2]): r[3] for r in imp_all}

    valid = [h for h in hyb_all if '_error' not in h]
    valid.sort(key=lambda h: -h['m_best_kg'])

    out_dir = os.path.join(repo, 'Renders/Asteroid_Plots/EGA_Analysis')
    os.makedirs(out_dir, exist_ok=True)

    for rank, hyb in enumerate(valid[:3], 1):
        idx = tuple(hyb['_triplet_idx'])
        res = imp_by_idx[idx]
        names = hyb['_names']
        ids   = [str(int(ast[x]['ID'])) for x in idx]
        flyby_id = str(res.get('flyby_body', '4'))
        out_path = os.path.join(out_dir,
            f"hybrid_top{rank}_{names[0]}_{names[1]}_{names[2]}_labeled.gif")
        print(f"Rank {rank}: {' → '.join(names)}  m={hyb['m_best_kg']:.1f} kg")
        make_gif(res, hyb, names, ids, flyby_id, out_path,
                 fps=10, duration_s=30)


if __name__ == '__main__':
    main()
