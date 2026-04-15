"""Asteroid trade-off table generator.

Reads sbdb_query_results.csv and scores asteroids on multiple criteria using
Chebyshev-spaced bin boundaries, then writes asteroid_tradeoff.csv.

Scoring weights:
    Mass                13%
    Radius              10%
    Eccentricity        15%
    Inclination         12%
    Semi-major axis      5%
    Rotation period      5%
    Science potential   10%
    Delta-V             30% (set to 0 pending trajectory analysis)
"""

import re
import numpy as np
import pandas as pd


# =============================================================================
#  LOCAL HELPER FUNCTIONS
# =============================================================================

def cheb_score(values, direction):
    """Assign integer scores 1-10 using Chebyshev-spaced boundaries.

    Chebyshev spacing places bin boundaries at
        bnd(k) = vmin + (vmax-vmin)/2 * (1 - cos(k*pi/10)),  k = 0..10
    which are denser near the extremes so that the best and worst objects
    are more finely discriminated than the middle of the distribution.

    Parameters
    ----------
    values : array-like
        Numeric values to score.
    direction : int
        +1: highest values get score 10.
        -1: lowest values get score 10.

    Returns
    -------
    scores : ndarray
        Scores 1-10, with NaN values receiving 5.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    scores = 5.0 * np.ones(n)

    valid = ~np.isnan(values) & ~np.isinf(values)
    v = values[valid]
    if np.sum(valid) < 2:
        return scores

    vmin, vmax = np.min(v), np.max(v)
    if vmax <= vmin:
        scores[valid] = 5.0
        return scores

    # 11 Chebyshev boundary points -> 10 bins
    k = np.arange(11)
    bnd = vmin + (vmax - vmin) * (1 - np.cos(k * np.pi / 10)) / 2

    # Assign bin index (1 = lowest range, 10 = highest range)
    bins = np.ones(np.sum(valid), dtype=int)
    for b in range(1, 10):
        bins[v >= bnd[b]] = b + 1
    bins[v >= bnd[10]] = 10

    if direction > 0:
        scores[valid] = bins.astype(float)
    else:
        scores[valid] = (11 - bins).astype(float)

    return scores


def safe_num(df, name):
    """Extract a named column as float; return NaN series if absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').values.astype(float)
    else:
        print(f'  [warning] column "{name}" not found -- filling with NaN.')
        return np.full(len(df), np.nan)


def safe_str(df, name):
    """Extract a named column as list of strings; return '' if absent."""
    if name in df.columns:
        col = df[name].fillna('').astype(str).tolist()
    else:
        col = [''] * len(df)
    # Replace 'nan' strings with empty
    col = ['' if s.strip().lower() == 'nan' else s for s in col]
    return col


def tax_composition(b, t):
    """Return a human-readable composition string from taxonomy letters."""
    tax = ''
    b = b.strip()
    t = t.strip()
    if b:
        tax = b[0].upper()
    elif t:
        tax = t[0].upper()

    comp_map = {
        'C': 'Primitive carbonaceous',
        'B': 'Primitive carbonaceous',
        'G': 'Primitive carbonaceous',
        'F': 'Primitive carbonaceous',
        'D': 'Primitive dark (D-type)',
        'T': 'Primitive transitional',
        'P': 'Primitive dark (P-type)',
        'S': 'Silicaceous stony',
        'Q': 'Silicaceous stony',
        'A': 'Olivine-dominated',
        'V': 'Basaltic achondrite',
        'R': 'Pyroxene-olivine',
        'L': 'Spinel-bearing stony',
        'K': 'Eos-family stony',
        'M': 'Metallic (M-type)',
        'X': 'Metallic / enstatite',
        'E': 'Enstatite achondrite',
    }
    return comp_map.get(tax, 'Unclassified')


# =============================================================================
#  MAIN SCRIPT
# =============================================================================

def main():
    print('Reading sbdb_query_results.csv ...')

    try:
        df = pd.read_csv('sbdb_query_results.csv')
        if df.shape[1] < 5:
            df = pd.read_csv('sbdb_query_results.csv', delimiter='\t')
    except Exception as e:
        raise RuntimeError(f'Could not read sbdb_query_results.csv.\nError: {e}')

    n = len(df)
    print(f'  {n} asteroids loaded.\n')

    # ---- Extract columns ----
    full_name = safe_str(df, 'full_name')
    spec_B = safe_str(df, 'spec_B')
    spec_T = safe_str(df, 'spec_T')

    H_mag = safe_num(df, 'H')
    diameter = safe_num(df, 'diameter')
    albedo_v = safe_num(df, 'albedo')
    rot_per = safe_num(df, 'rot_per')
    BV = safe_num(df, 'BV')
    diam_sig = safe_num(df, 'diameter_sigma')
    ecc = safe_num(df, 'e')
    sma = safe_num(df, 'a')
    inc = safe_num(df, 'i')

    # ---- Fill missing diameters from H and albedo ----
    for k in range(n):
        if np.isnan(diameter[k]) and not np.isnan(H_mag[k]):
            alb = albedo_v[k]
            if np.isnan(alb):
                alb = 0.15
            diameter[k] = 1329 / np.sqrt(alb) * 10 ** (-H_mag[k] / 5)
    radius = diameter / 2.0

    # ---- Taxonomy -> density -> mass ----
    density_kgm3 = np.zeros(n)
    for k in range(n):
        tax = ''
        b = spec_B[k].strip()
        t = spec_T[k].strip()
        if b:
            tax = b[0].upper()
        elif t:
            tax = t[0].upper()

        if tax in ('C', 'B', 'F', 'G', 'D', 'T'):
            density_kgm3[k] = 1400
        elif tax in ('S', 'Q', 'A', 'V', 'R', 'L', 'K'):
            density_kgm3[k] = 2700
        elif tax in ('M', 'X', 'E', 'P'):
            density_kgm3[k] = 3500
        else:
            density_kgm3[k] = 2000

    r_m = radius * 1e3  # km -> m
    mass_kg = (4.0 / 3.0) * np.pi * r_m ** 3 * density_kgm3

    # ---- Science potential raw score (0-10) ----
    sci_raw = np.zeros(n)
    for k in range(n):
        s = 0
        b = spec_B[k].strip()
        t = spec_T[k].strip()
        has_tax = bool(b) or bool(t)
        if has_tax:
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

    # ---- Rotation period score ----
    rot_dist = np.full(n, np.nan)
    for k in range(n):
        if not np.isnan(rot_per[k]) and rot_per[k] > 0:
            rot_dist[k] = abs(np.log(rot_per[k] / 12))
    score_rot = cheb_score(rot_dist, -1)

    # ---- Chebyshev score all primary criteria (1-10) ----
    score_mass = cheb_score(mass_kg, +1)
    score_radius = cheb_score(radius, +1)
    score_ecc = cheb_score(ecc, -1)
    score_inc = cheb_score(inc, -1)
    score_sma = cheb_score(sma, -1)
    score_sci = cheb_score(sci_raw, +1)
    score_dv = np.zeros(n)  # placeholder

    # ---- Weighted total score (out of 10) ----
    W_mass = 0.13
    W_radius = 0.10
    W_ecc = 0.15
    W_inc = 0.12  # 0.15 - 0.03
    W_sma = 0.05
    W_rot = 0.05
    W_sci = 0.10
    W_dv = 0.30

    total = (W_mass * score_mass + W_radius * score_radius
             + W_ecc * score_ecc + W_inc * score_inc
             + W_sma * score_sma + W_rot * score_rot
             + W_sci * score_sci + W_dv * score_dv)

    # ---- Build formatted output columns ----
    name_col = []
    class_col = []

    for k in range(n):
        nm = full_name[k].strip()
        # Strip trailing (YYYY XX) designation if present
        nm = re.sub(r'\s*\([A-Z]\d{3,4}\s+[A-Z]{2}\)\s*$', '', nm).strip()
        if np.isnan(radius[k]):
            name_col.append(f'{nm} (r = N/A)')
        else:
            name_col.append(f'{nm} (r = {radius[k]:.1f} km)')

        b = spec_B[k].strip()
        t = spec_T[k].strip()
        comp = tax_composition(b, t)
        if b:
            class_col.append(f'{b} ({comp}) [SMASSII]')
        elif t:
            class_col.append(f'{t} ({comp}) [Tholen]')
        else:
            class_col.append('Unknown (Unclassified)')

    # ---- Assemble output table ----
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

    # Sort by total score, best candidates first
    out = out.sort_values('Total_WeightedScore', ascending=False).reset_index(drop=True)

    out.to_csv('asteroid_tradeoff.csv', index=False)

    N = 40
    print(f'Done.  asteroid_tradeoff.csv written ({len(out)} rows).\n')
    print(f'Top {N} candidates:')
    print(f'{"Name":<42s}  {"Class":<30s}  {"Score":>6s}')
    print('-' * 82)
    for k in range(min(N, len(out))):
        print(f'{out.iloc[k]["Name_DecRadius"]:<42s}  '
              f'{out.iloc[k]["Class_Composition_SMASSII"]:<30s}  '
              f'{out.iloc[k]["Total_WeightedScore"]:6.4f}')


if __name__ == '__main__':
    main()
