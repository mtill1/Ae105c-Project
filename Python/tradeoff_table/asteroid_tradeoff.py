"""Asteroid trade-off table generator (v1).

Reads sbdb_query_results.csv, scores asteroids on multiple criteria
using Chebyshev-spaced bin boundaries, and outputs asteroid_tradeoff.csv.

Scoring weights:
  Mass           13%    Radius        10%    Eccentricity  15%
  Inclination    12%    SMA            5%    Rotation       5%
  Science        10%    Delta-V       30%  (placeholder = 0)
"""

import numpy as np
import pandas as pd
import re


def cheb_score(values, direction):
    """Assign integer scores 1-10 using Chebyshev-spaced boundaries."""
    n = len(values)
    scores = np.full(n, 5.0)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return scores
    v = values[valid]
    vmin, vmax = v.min(), v.max()
    if vmax <= vmin:
        scores[valid] = 5.0
        return scores
    k = np.arange(11)
    bnd = vmin + (vmax - vmin) * (1 - np.cos(k * np.pi / 10)) / 2
    bins = np.ones(valid.sum())
    for b in range(1, 10):
        bins[v >= bnd[b]] = b + 1
    bins[v >= bnd[10]] = 10
    if direction > 0:
        scores[valid] = bins
    else:
        scores[valid] = 11 - bins
    return scores


def safe_num(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').values
    print(f'  [warning] column "{name}" not found - filling with NaN.')
    return np.full(len(df), np.nan)


def safe_str(df, name):
    if name in df.columns:
        return df[name].fillna('').astype(str).values
    return np.full(len(df), '', dtype=object)


def tax_composition(b, t):
    tax = ''
    b, t = b.strip(), t.strip()
    if b:
        tax = b[0].upper()
    elif t:
        tax = t[0].upper()
    comp_map = {
        'C': 'Primitive carbonaceous', 'B': 'Primitive carbonaceous',
        'G': 'Primitive carbonaceous', 'F': 'Primitive carbonaceous',
        'D': 'Primitive dark (D-type)', 'T': 'Primitive transitional',
        'P': 'Primitive dark (P-type)',
        'S': 'Silicaceous stony', 'Q': 'Silicaceous stony',
        'A': 'Olivine-dominated', 'V': 'Basaltic achondrite',
        'R': 'Pyroxene-olivine', 'L': 'Spinel-bearing stony',
        'K': 'Eos-family stony',
        'M': 'Metallic (M-type)', 'X': 'Metallic / enstatite',
        'E': 'Enstatite achondrite',
    }
    return comp_map.get(tax, 'Unclassified')


def run(csv_path='sbdb_query_results.csv', output_path='asteroid_tradeoff.csv'):
    print(f'Reading {csv_path} ...')
    try:
        T = pd.read_csv(csv_path)
        if T.shape[1] < 5:
            T = pd.read_csv(csv_path, delimiter='\t')
    except Exception as e:
        raise RuntimeError(f'Could not read {csv_path}.\nError: {e}')

    n = len(T)
    print(f'  {n} asteroids loaded.\n')

    full_name = safe_str(T, 'full_name')
    spec_B = safe_str(T, 'spec_B')
    spec_T = safe_str(T, 'spec_T')

    H_mag = safe_num(T, 'H')
    diameter = safe_num(T, 'diameter')
    albedo_v = safe_num(T, 'albedo')
    rot_per = safe_num(T, 'rot_per')
    BV = safe_num(T, 'BV')
    diam_sig = safe_num(T, 'diameter_sigma')
    ecc = safe_num(T, 'e')
    sma = safe_num(T, 'a')
    inc = safe_num(T, 'i')

    # Fill missing diameters from H-magnitude
    for k in range(n):
        if np.isnan(diameter[k]) and not np.isnan(H_mag[k]):
            alb = albedo_v[k] if not np.isnan(albedo_v[k]) else 0.15
            diameter[k] = 1329 / np.sqrt(alb) * 10 ** (-H_mag[k] / 5)
    radius = diameter / 2

    # Taxonomy -> density -> mass
    density = np.zeros(n)
    for k in range(n):
        tax = ''
        b, t = spec_B[k].strip(), spec_T[k].strip()
        if b:
            tax = b[0].upper()
        elif t:
            tax = t[0].upper()
        if tax in ('C', 'B', 'F', 'G', 'D', 'T'):
            density[k] = 1400
        elif tax in ('S', 'Q', 'A', 'V', 'R', 'L', 'K'):
            density[k] = 2700
        elif tax in ('M', 'X', 'E', 'P'):
            density[k] = 3500
        else:
            density[k] = 2000
    mass_kg = (4 / 3) * np.pi * (radius * 1e3) ** 3 * density

    # Science potential raw score
    sci_raw = np.zeros(n)
    for k in range(n):
        s = 0
        b, t = spec_B[k].strip(), spec_T[k].strip()
        if b or t:
            s += 3
        if not np.isnan(rot_per[k]):
            s += 2
        if not np.isnan(diameter[k]) and not np.isnan(diam_sig[k]) and diam_sig[k] < 5:
            s += 2
        elif not np.isnan(diameter[k]):
            s += 1
        if not np.isnan(albedo_v[k]):
            s += 1
        if not np.isnan(BV[k]):
            s += 2
        sci_raw[k] = min(s, 10)

    # Rotation period score
    rot_dist = np.full(n, np.nan)
    for k in range(n):
        if not np.isnan(rot_per[k]) and rot_per[k] > 0:
            rot_dist[k] = abs(np.log(rot_per[k] / 12))

    score_mass = cheb_score(mass_kg, +1)
    score_radius = cheb_score(radius, +1)
    score_ecc = cheb_score(ecc, -1)
    score_inc = cheb_score(inc, -1)
    score_sma = cheb_score(sma, -1)
    score_rot = cheb_score(rot_dist, -1)
    score_sci = cheb_score(sci_raw, +1)
    score_dv = np.zeros(n)

    W = {'mass': 0.13, 'radius': 0.10, 'ecc': 0.15, 'inc': 0.12,
         'sma': 0.05, 'rot': 0.05, 'sci': 0.10, 'dv': 0.30}

    total = (W['mass'] * score_mass + W['radius'] * score_radius +
             W['ecc'] * score_ecc + W['inc'] * score_inc +
             W['sma'] * score_sma + W['rot'] * score_rot +
             W['sci'] * score_sci + W['dv'] * score_dv)

    # Format output columns
    name_col = []
    class_col = []
    for k in range(n):
        nm = full_name[k].strip()
        nm = re.sub(r'\s*\([A-Z]\d{3,4}\s+[A-Z]{2}\)\s*$', '', nm).strip()
        if np.isnan(radius[k]):
            name_col.append(f'{nm} (r = N/A)')
        else:
            name_col.append(f'{nm} (r = {radius[k]:.1f} km)')

        b, t = spec_B[k].strip(), spec_T[k].strip()
        comp = tax_composition(b, t)
        if b:
            class_col.append(f'{b} ({comp}) [SMASSII]')
        elif t:
            class_col.append(f'{t} ({comp}) [Tholen]')
        else:
            class_col.append('Unknown (Unclassified)')

    out = pd.DataFrame({
        'Name_DecRadius': name_col,
        'Class_Composition_SMASSII': class_col,
        'Mass_kg': mass_kg,
        'Radius_km': radius,
        'SMA_AU': sma,
        'Eccentricity': ecc,
        'Inclination_deg': inc,
        'RotPeriod_hr': rot_per,
        'SciPotential_Score': np.round(score_sci, 2),
        'DeltaV_Score': np.round(score_dv, 2),
        'Total_WeightedScore': np.round(total, 4),
        'Subscr_Mass_Score': np.round(score_mass, 2),
        'Subscr_Radius_Score': np.round(score_radius, 2),
        'Subscr_Ecc_Score': np.round(score_ecc, 2),
        'Subscr_Inc_Score': np.round(score_inc, 2),
        'Subscr_SMA_Score': np.round(score_sma, 2),
        'Subscr_RotPer_Score': np.round(score_rot, 2),
    })

    out = out.sort_values('Total_WeightedScore', ascending=False).reset_index(drop=True)
    out.to_csv(output_path, index=False)

    N = 40
    print(f'Done.  {output_path} written ({len(out)} rows).\n')
    print(f'Top {N} candidates:')
    print(f'{"Name":42s}  {"Class":30s}  {"Score":>6s}')
    print('-' * 82)
    for k in range(min(N, len(out))):
        print(f'{out.iloc[k]["Name_DecRadius"]:42s}  '
              f'{out.iloc[k]["Class_Composition_SMASSII"]:30s}  '
              f'{out.iloc[k]["Total_WeightedScore"]:6.4f}')


if __name__ == '__main__':
    run()
