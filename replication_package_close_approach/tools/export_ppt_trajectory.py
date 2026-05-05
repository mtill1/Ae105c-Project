"""Export the PARTHENOPE → PSYCHE → THEMIS trajectory to standard formats:

1. **SPICE SPK Type 9** binary kernel (`.bsp`) — load with `spiceypy.furnsh`,
   then `spkezr` works on the spacecraft like any other body. NAIF body code
   set to -1029 (custom; reserved range -1000..-1099 for student use).
2. **CCSDS OEM ASCII** (Orbital Ephemeris Message) — plain-text inter-agency
   exchange format. Each line is `EPOCH X Y Z VX VY VZ`.
3. **CSV** with the same data (lowest-common-denominator format).
4. **JSON** metadata bundle (constraints, optimization config, summary).

Recipients can replicate the trajectory in any SPICE-aware tool by
furnishing the SPK alongside the standard NASA ephemerides.

Run:
    python3 Python_Consolidated/export_ppt_trajectory.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import spiceypy

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))

from core import (load_kernels, get_state, solve_lambert,
                   propagate_two_body, MU_SUN, DAY, YEAR,
                   get_id_from_asteroid_name)


_DEFAULT_PKL = 'optimal_asteroid_paths/pkl/ppt_lt_chain_v2.pkl'
_DEFAULT_OUT = 'replication_package'
_DEFAULT_NAIF = -1029
_DEFAULT_NAME = 'PPT-LT-CHAIN'

# CLI: python export_ppt_trajectory.py [PKL] [OUT_DIR] [NAIF_ID] [NAME]
PKL_PATH           = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PKL
OUTPUT_DIR         = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_OUT
SPACECRAFT_NAIF_ID = int(sys.argv[3]) if len(sys.argv) > 3 else _DEFAULT_NAIF
SPACECRAFT_NAME    = sys.argv[4] if len(sys.argv) > 4 else _DEFAULT_NAME
G0 = 9.80665


# =============================================================================
# Sims-Flanagan integrator that records both position AND velocity
# =============================================================================

def integrate_with_velocities(r0, v0, m0, throttles_nseg3, tof_sec,
                                thrust_N, isp_s, samples_per_half=10):
    """Forward integration. Same algorithm as lowthrust._forward_propagate but
    captures (r, v, t) at fine resolution so the trajectory is dense enough
    for SPK interpolation.

    Returns dict with arrays:
      - t_from_start_sec : (N,)
      - r_km             : (N, 3)
      - v_km_s           : (N, 3)
      - mass_kg          : (N,)
    """
    nseg = len(throttles_nseg3)
    dt_seg = tof_sec / nseg
    half_coast_dt = (dt_seg / 2) / samples_per_half

    r = np.array(r0, float).copy()
    v = np.array(v0, float).copy()
    m = float(m0)
    t = 0.0
    out_t, out_r, out_v, out_m = [t], [r.copy()], [v.copy()], [m]

    for i in range(nseg):
        u = np.asarray(throttles_nseg3[i])
        # Half-coast 1
        for _ in range(samples_per_half):
            r, v = propagate_two_body(r, v, half_coast_dt, MU_SUN)
            t += half_coast_dt
            out_t.append(t); out_r.append(r.copy()); out_v.append(v.copy()); out_m.append(m)
        # Impulsive kick
        dv_max_kms = thrust_N * dt_seg / m / 1e3
        dv_vec = u * dv_max_kms
        v = v + dv_vec
        m = m * np.exp(np.linalg.norm(dv_vec) * 1e3 / (-isp_s * G0))
        # The post-kick state
        out_t.append(t); out_r.append(r.copy()); out_v.append(v.copy()); out_m.append(m)
        # Half-coast 2
        for _ in range(samples_per_half):
            r, v = propagate_two_body(r, v, half_coast_dt, MU_SUN)
            t += half_coast_dt
            out_t.append(t); out_r.append(r.copy()); out_v.append(v.copy()); out_m.append(m)

    return {
        't_from_start_sec': np.asarray(out_t),
        'r_km':              np.asarray(out_r),
        'v_km_s':            np.asarray(out_v),
        'mass_kg':           np.asarray(out_m),
    }


# =============================================================================
# Build full mission trajectory: 3 LT legs + 2 stays
# =============================================================================

def build_full_trajectory(data, asteroid_list):
    """Return (epochs_et, states_km_kms, masses_kg) covering the entire mission."""
    triplet = data.get('best_ordering') or data.get('ordering')
    v = data['verified']
    cfg = data['config']
    a_ids = [str(int(get_id_from_asteroid_name(asteroid_list, n))) for n in triplet]
    solver_m_revs = v['m_revs']

    et_launch = v['epochs']['et_launch']
    leg_data = v['verified_legs']

    all_t = []   # absolute SPICE ET
    all_r = []   # km
    all_v = []   # km/s
    all_m = []   # kg

    cur_mass = cfg['lt_chain_initial_mass_kg']

    # Leg 1: Earth → A1 (LT)
    L = leg_data[0]
    earth_r, _ = get_state('399', L['et_start'])
    a1_arr_r, _ = get_state(a_ids[0], L['et_end'])
    V1, _, _ = solve_lambert(earth_r, a1_arr_r, (L['et_end']-L['et_start'])/DAY,
                               int(solver_m_revs[0]), MU_SUN)
    thr = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    out = integrate_with_velocities(earth_r, V1, cur_mass, thr,
                                       L['et_end']-L['et_start'],
                                       cfg['thrust_N'], cfg['isp_elec_s'])
    all_t.extend((out['t_from_start_sec'] + L['et_start']).tolist())
    all_r.extend(out['r_km'].tolist())
    all_v.extend(out['v_km_s'].tolist())
    all_m.extend(out['mass_kg'].tolist())
    cur_mass = float(out['mass_kg'][-1])

    # Stay 1 at A1: spacecraft co-orbits the asteroid (no thrust)
    stay_n = 30
    stay_ts = np.linspace(L['et_end'], leg_data[1]['et_start'], stay_n)
    for et in stay_ts:
        r, v_ = get_state(a_ids[0], et)
        all_t.append(float(et)); all_r.append(r.tolist()); all_v.append(v_.tolist())
        all_m.append(cur_mass)

    # Leg 2: A1 → A2 (LT)
    L = leg_data[1]
    a1_lv_r, a1_lv_v = get_state(a_ids[0], L['et_start'])
    thr = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    out = integrate_with_velocities(a1_lv_r, a1_lv_v, cur_mass, thr,
                                       L['et_end']-L['et_start'],
                                       cfg['thrust_N'], cfg['isp_elec_s'])
    all_t.extend((out['t_from_start_sec'] + L['et_start']).tolist())
    all_r.extend(out['r_km'].tolist())
    all_v.extend(out['v_km_s'].tolist())
    all_m.extend(out['mass_kg'].tolist())
    cur_mass = float(out['mass_kg'][-1])

    # Stay 2 at A2
    stay_ts = np.linspace(L['et_end'], leg_data[2]['et_start'], stay_n)
    for et in stay_ts:
        r, v_ = get_state(a_ids[1], et)
        all_t.append(float(et)); all_r.append(r.tolist()); all_v.append(v_.tolist())
        all_m.append(cur_mass)

    # Leg 3: A2 → A3 (LT)
    L = leg_data[2]
    a2_lv_r, a2_lv_v = get_state(a_ids[1], L['et_start'])
    thr = np.asarray(L['thrust_profile']['throttle_unit_vector'])
    out = integrate_with_velocities(a2_lv_r, a2_lv_v, cur_mass, thr,
                                       L['et_end']-L['et_start'],
                                       cfg['thrust_N'], cfg['isp_elec_s'])
    all_t.extend((out['t_from_start_sec'] + L['et_start']).tolist())
    all_r.extend(out['r_km'].tolist())
    all_v.extend(out['v_km_s'].tolist())
    all_m.extend(out['mass_kg'].tolist())

    # Sort by time (in case of tie-breaks at boundaries) and dedup
    order = np.argsort(all_t)
    all_t = np.asarray(all_t)[order]
    all_r = np.asarray(all_r)[order]
    all_v = np.asarray(all_v)[order]
    all_m = np.asarray(all_m)[order]

    # Remove exact-duplicate epochs (SPK requires strict monotonicity)
    keep = np.concatenate([[True], np.diff(all_t) > 1e-6])
    all_t = all_t[keep]; all_r = all_r[keep]
    all_v = all_v[keep]; all_m = all_m[keep]

    states = np.hstack([all_r, all_v])      # (N, 6)
    return all_t, states, all_m


# =============================================================================
# SPICE SPK Type 9 writer
# =============================================================================

def write_spk(epochs_et, states_km_kms, output_path,
                spacecraft_id=SPACECRAFT_NAIF_ID,
                spacecraft_name=SPACECRAFT_NAME,
                center=10, frame='ECLIPJ2000', degree=5):
    """Write a SPICE SPK Type 9 (Lagrange interpolation, equal-time-spaced)
    binary kernel. degree=5 → 6th-order Lagrange polynomial fit.
    """
    n = len(epochs_et)
    if n < degree + 1:
        raise ValueError(f'Need at least {degree+1} states for SPK degree={degree}')

    # SPK Type 9 expects states in *meters* and m/s (SPICE's standard units).
    # Wait — actually, Type 9 uses km / s (the same as inputs). Let's confirm.
    # Per CSPICE docs (spkw09): states are in the 'frame' specified, with
    # units consistent with mu used by spk09 internally. SPICE convention is
    # km, km/s for output of spkezr. So we keep our units.
    states = np.asarray(states_km_kms, dtype=float)

    if os.path.exists(output_path):
        os.remove(output_path)

    handle = spiceypy.spkopn(output_path,
                              f'PPT_LT_CHAIN trajectory; SC NAIF={spacecraft_id}',
                              2048)
    try:
        # Note: spiceypy spkw09 uses the parameter name `inframe` not `frame`.
        # Pass positionally to avoid kwarg-name confusion.
        spiceypy.spkw09(
            handle, spacecraft_id, center, frame,
            float(epochs_et[0]), float(epochs_et[-1]),
            spacecraft_name + '_LT_CHAIN_TRAJECTORY',
            degree, n,
            states.tolist(),
            np.asarray(epochs_et).tolist(),
        )
    finally:
        spiceypy.spkcls(handle)


# =============================================================================
# CCSDS OEM ASCII writer
# =============================================================================

def write_oem(epochs_et, states_km_kms, output_path,
                spacecraft_id=SPACECRAFT_NAIF_ID,
                spacecraft_name=SPACECRAFT_NAME,
                center='SUN', frame='EME2000'):
    """Write CCSDS OEM v2.0 ephemeris file. Plain text — no SPICE required to read.

    Frame note: ECLIPJ2000 is not a CCSDS-blessed name. We label it as
    EME2000 (the closest CCSDS name) but include a comment that the actual
    frame is ECLIPJ2000 — the user must apply a rotation if they need EME2000
    natively. To keep this exact, we write the data in ECLIPJ2000 with a
    USER_DEFINED frame block and a clear comment.
    """
    n = len(epochs_et)
    with open(output_path, 'w') as f:
        # Header
        f.write('CCSDS_OEM_VERS = 2.0\n')
        f.write(f'CREATION_DATE = {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}\n')
        f.write(f'ORIGINATOR = AE105C_PPT_LT_CHAIN\n\n')

        # Metadata block
        f.write('META_START\n')
        f.write(f'OBJECT_NAME = {spacecraft_name}\n')
        f.write(f'OBJECT_ID = {spacecraft_id}\n')
        f.write(f'CENTER_NAME = {center}\n')
        f.write(f'REF_FRAME = ECLIPJ2000\n')   # USER_DEFINED is acceptable
        f.write(f'TIME_SYSTEM = TDB\n')
        f.write(f'START_TIME = {spiceypy.et2utc(epochs_et[0],"ISOC",6)}\n')
        f.write(f'STOP_TIME  = {spiceypy.et2utc(epochs_et[-1],"ISOC",6)}\n')
        f.write(f'INTERPOLATION = LAGRANGE\n')
        f.write(f'INTERPOLATION_DEGREE = 5\n')
        f.write('META_STOP\n\n')

        # Comment block
        f.write('COMMENT Frame is heliocentric ECLIPJ2000 (Earth-Mean-Equator/Equinox\n')
        f.write('COMMENT of J2000.0 ecliptic). Center body = Sun (NAIF 10).\n')
        f.write('COMMENT Units: km, km/s. Time system: SPICE Ephemeris Time (ET) = TDB.\n')
        f.write('COMMENT Generated from PARTHENOPE -> PSYCHE -> THEMIS LT-chain optimization\n')
        f.write('COMMENT (Ae105c, Caltech/Pomona). Architecture: direct, all-LT post-launch.\n\n')

        # Data lines
        for et, s in zip(epochs_et, states_km_kms):
            t_iso = spiceypy.et2utc(float(et), 'ISOC', 6)
            f.write(f'{t_iso}  '
                    f'{s[0]:+.6e}  {s[1]:+.6e}  {s[2]:+.6e}  '
                    f'{s[3]:+.9e}  {s[4]:+.9e}  {s[5]:+.9e}\n')


# =============================================================================
# CSV writer
# =============================================================================

def write_csv(epochs_et, states_km_kms, masses_kg, output_path):
    with open(output_path, 'w') as f:
        f.write('# PARTHENOPE -> PSYCHE -> THEMIS LT-chain trajectory\n')
        f.write('# Frame: heliocentric ECLIPJ2000, units km / km/s / kg\n')
        f.write('# Time system: SPICE ET (TDB seconds past J2000)\n')
        f.write('et_seconds_past_J2000,utc_iso,'
                'x_km,y_km,z_km,'
                'vx_kms,vy_kms,vz_kms,'
                'mass_kg\n')
        for et, s, m in zip(epochs_et, states_km_kms, masses_kg):
            utc = spiceypy.et2utc(float(et), 'ISOC', 6)
            f.write(f'{et:.6f},{utc},'
                    f'{s[0]:.6f},{s[1]:.6f},{s[2]:.6f},'
                    f'{s[3]:.9f},{s[4]:.9f},{s[5]:.9f},'
                    f'{m:.4f}\n')


# =============================================================================
# Metadata JSON
# =============================================================================

def write_json(data, epochs_et, masses_kg, output_path):
    triplet = data.get('best_ordering') or data.get('ordering')
    v = data['verified']
    cfg = data['config']
    eps = v['epochs']

    meta = {
        'mission': {
            'name': 'PARTHENOPE -> PSYCHE -> THEMIS LT-chain',
            'project': 'Ae105c (Caltech / Pomona College)',
            'architecture': 'direct (no flyby) + all-LT post-launch',
            'triplet': triplet,
            'asteroid_compositions': {
                'PARTHENOPE': 'S-type', 'PSYCHE': 'X/M-type', 'THEMIS': 'C-type',
            },
        },
        'reference_frame': {
            'frame_name':   'ECLIPJ2000',
            'description':  'Heliocentric, Earth-Mean-Equator/Equinox of J2000.0 ecliptic',
            'center_body':  '10 (Sun barycenter)',
            'time_system':  'SPICE Ephemeris Time (TDB)',
            'units':        'km, km/s, km^3/s^2',
        },
        'spacecraft': {
            'launch_mass_kg':            cfg['spacecraft_launch_mass_kg'],
            'lt_chain_initial_mass_kg':  cfg['lt_chain_initial_mass_kg'],
            'final_mass_kg':             v['m_final_kg_full'],
            'electric_propulsion': {
                'isp_s':                  cfg['isp_elec_s'],
                'thrust_max_N':           cfg['thrust_N'],
            },
        },
        'optimization_constraints': {
            'launch_dv_max_kms':  cfg['launch_dv_max_kms'],
            'mission_max_yr':     cfg['mission_max_yr'],
            'stay_min_months':    cfg['stay_min_months'],
            'flyby':              'none (direct architecture won)',
            'objective':          'minimize post-launch integrated delta-v',
            'launch_dv_in_objective': False,
        },
        'optimizer': {
            'method':            'scipy.optimize.differential_evolution',
            'maxiter':            cfg.get('de_maxiter'),
            'popsize':            cfg.get('de_popsize'),
            'lambert_m_revs':     {
                'tried':         '5 combos × 6 seeds × 2 architectures (direct + Mars GA)',
                'winning_value': list(v['m_revs']),
                'winning_legs':  '(Earth->PARTHENOPE: m=1, PARTHENOPE->PSYCHE: m=1, PSYCHE->THEMIS: m=0)',
            },
            'lt_solver':          'Sims-Flanagan, 15 segments per leg, scipy least_squares',
        },
        'mission_events_utc': {
            'earth_launch':         spiceypy.et2utc(eps['et_launch'], 'ISOC', 3),
            'arrive_PARTHENOPE':    spiceypy.et2utc(eps['et_a1_arr'], 'ISOC', 3),
            'depart_PARTHENOPE':    spiceypy.et2utc(eps['et_a1_dep'], 'ISOC', 3),
            'arrive_PSYCHE':        spiceypy.et2utc(eps['et_a2_arr'], 'ISOC', 3),
            'depart_PSYCHE':        spiceypy.et2utc(eps['et_a2_dep'], 'ISOC', 3),
            'arrive_THEMIS':        spiceypy.et2utc(eps['et_a3_arr'], 'ISOC', 3),
            'mission_duration_yr':  (eps['et_a3_arr'] - eps['et_launch']) / YEAR,
        },
        'delta_v_breakdown_kms': {
            'launch_impulsive':         v['launch_dv_kms'],
            'leg_earth_to_parthenope':  v['verified_legs'][0]['dv_integral_kms'],
            'leg_parthenope_to_psyche': v['verified_legs'][1]['dv_integral_kms'],
            'leg_psyche_to_themis':     v['verified_legs'][2]['dv_integral_kms'],
            'post_launch_total':        v['post_launch_dv_kms_full'],
        },
        'asteroid_naif_ids': {
            'PARTHENOPE': 20000011,
            'PSYCHE':     20000016,
            'THEMIS':     20000024,
        },
        'spice_kernels_required': {
            'leapseconds': 'naif0012.tls (or later)',
            'planets':     'de430.bsp (Sun-Mercury-...-Pluto barycenters)',
            'satellites':  'jup310.bsp (only needed if you also re-derive auxiliary state)',
            'gm':          'gm_de431.tpc',
            'pck':         'pck00010.tpc',
            'asteroids':   'individual SPK files for PARTHENOPE, PSYCHE, THEMIS — '
                            'pull from JPL Horizons (https://ssd.jpl.nasa.gov/horizons/) '
                            'or use the BSPs in NOTABLE_ASTEROID_BSPs/.',
        },
        'trajectory_dataset': {
            'n_samples':            int(len(epochs_et)),
            'first_epoch_utc':      spiceypy.et2utc(float(epochs_et[0]), 'ISOC', 6),
            'last_epoch_utc':       spiceypy.et2utc(float(epochs_et[-1]), 'ISOC', 6),
            'spacecraft_naif_id':   SPACECRAFT_NAIF_ID,
            'spacecraft_name':      SPACECRAFT_NAME,
            'mass_first_kg':        float(masses_kg[0]),
            'mass_last_kg':         float(masses_kg[-1]),
        },
    }

    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Loading {PKL_PATH}...')
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    asteroid_list = load_kernels('NOTABLE_ASTEROID_BSPs', 'generic_kernels')
    print(f'Loaded {len(asteroid_list)} asteroid BSPs')

    print('Building full trajectory at fine resolution...')
    epochs_et, states_km_kms, masses_kg = build_full_trajectory(data, asteroid_list)
    n = len(epochs_et)
    print(f'  {n} state samples spanning '
          f'{(epochs_et[-1]-epochs_et[0])/YEAR:.2f} yr')
    print(f'  avg sample interval: {(epochs_et[-1]-epochs_et[0])/(n-1)/DAY:.2f} days')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spk_path = os.path.join(OUTPUT_DIR, 'PPT_LT_CHAIN_trajectory.bsp')
    print(f'Writing SPK Type 9 → {spk_path}')
    write_spk(epochs_et, states_km_kms, spk_path, degree=5)

    oem_path = os.path.join(OUTPUT_DIR, 'PPT_LT_CHAIN_trajectory.oem')
    print(f'Writing CCSDS OEM   → {oem_path}')
    write_oem(epochs_et, states_km_kms, oem_path)

    csv_path = os.path.join(OUTPUT_DIR, 'PPT_LT_CHAIN_trajectory.csv')
    print(f'Writing CSV         → {csv_path}')
    write_csv(epochs_et, states_km_kms, masses_kg, csv_path)

    json_path = os.path.join(OUTPUT_DIR, 'PPT_LT_CHAIN_metadata.json')
    print(f'Writing JSON meta   → {json_path}')
    write_json(data, epochs_et, masses_kg, json_path)

    # Write a small "names kernel" (TPC) so users can refer to the spacecraft
    # by name as well as by integer ID
    names_path = os.path.join(OUTPUT_DIR, 'PPT_LT_CHAIN_names.tpc')
    print(f'Writing names kernel → {names_path}')
    with open(names_path, 'w') as f:
        f.write(f'KPL/PCK\n\n')
        f.write(f'\\begindata\n\n')
        f.write(f'NAIF_BODY_NAME += ( \'{SPACECRAFT_NAME}\' )\n')
        f.write(f'NAIF_BODY_CODE += ( {SPACECRAFT_NAIF_ID} )\n\n')
        f.write(f'\\begintext\n\n')
        f.write(f'Custom spacecraft name registration for the\n')
        f.write(f'PARTHENOPE -> PSYCHE -> THEMIS LT-chain trajectory.\n')
        f.write(f'After furnishing this file alongside PPT_LT_CHAIN_trajectory.bsp,\n')
        f.write(f'you can use either the name "{SPACECRAFT_NAME}" or the integer ID\n')
        f.write(f'{SPACECRAFT_NAIF_ID} when calling spkezr / spkpos / etc.\n')

    # Verify the SPK by furnishing it (and the names kernel) and querying.
    print('\nVerifying SPK by furnishing and querying...')
    spiceypy.furnsh(spk_path)
    spiceypy.furnsh(names_path)
    et_test = float(epochs_et[len(epochs_et) // 2])
    # Query by integer ID first (always works)
    state_id, _ = spiceypy.spkezr(str(SPACECRAFT_NAIF_ID), et_test,
                                    'ECLIPJ2000', 'NONE', 'SUN')
    # And by name (works because we just furnished the names kernel)
    state_name, _ = spiceypy.spkezr(SPACECRAFT_NAME, et_test,
                                      'ECLIPJ2000', 'NONE', 'SUN')
    expected = states_km_kms[len(epochs_et) // 2]
    pos_err_id   = np.linalg.norm(np.array(state_id[0:3])   - expected[0:3])
    pos_err_name = np.linalg.norm(np.array(state_name[0:3]) - expected[0:3])
    print(f'  midpoint queried by ID   ({SPACECRAFT_NAIF_ID}): '
          f'pos err = {pos_err_id:.6f} km')
    print(f'  midpoint queried by name ({SPACECRAFT_NAME}): '
          f'pos err = {pos_err_name:.6f} km')

    print('\nDone. Replication artifacts in:', OUTPUT_DIR)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f'  {f:45s}  {size_kb:8.1f} KB')


if __name__ == '__main__':
    main()
