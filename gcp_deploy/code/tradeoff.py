"""Asteroid trade-off table generators -- scoring and ranking asteroids by physical properties."""

import numpy as np
import pandas as pd
import re


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def safe_num(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').values
    print(f'  [warning] column "{name}" not found - filling with NaN.')
    return np.full(len(df), np.nan)


def safe_str(df, name):
    if name in df.columns:
        return df[name].fillna('').astype(str).values
    return np.full(len(df), '', dtype=object)


# ============================================================================
# VERSION 1: BASIC CHEBYSHEV SCORING
# ============================================================================

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


def tax_composition_v1(b, t):
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


def run_tradeoff_v1(csv_path='sbdb_query_results.csv', output_path='asteroid_tradeoff.csv'):
    """Version 1: basic Chebyshev scoring trade-off table."""
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
        comp = tax_composition_v1(b, t)
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


# ============================================================================
# VERSION 3: ENHANCED SCIENCE SCORING
# ============================================================================

# -- Manual Science Interest Flag Lists --

ACTIVE_LIST = {'133P', '176P', '238P', '259P', '288P', '313P', '324P', '358P', '433P'}
ICE_LIST = {'24', '65', '90', '153', '3200'}
AMBIG_M_LIST = {'21', '22', '69', '97', '110', '129', '135', '785'}
VISITED_LIST = {'1', '4', '433', '951', '243', '253', '25143', '101955',
                '162173', '16', '2867', '21', '132524', '5535', '9969', '2685', '52246'}
FLYBY_ONLY_LIST = {'21', '951', '243', '253', '2867', '132524'}


def get_tax(b, t):
    b, t = b.strip(), t.strip()
    if b:
        return b[0].upper()
    if t:
        return t[0].upper()
    return ''


def log_cheb_score(values, direction):
    """Chebyshev-spaced bins on log10-transformed values (1-10)."""
    n = len(values)
    scores = np.full(n, 5.0)
    pos = (values > 0) & np.isfinite(values)
    if pos.sum() < 2:
        return scores
    lv = np.log10(values[pos])
    lmin, lmax = lv.min(), lv.max()
    if lmax <= lmin:
        scores[pos] = 5.0
        return scores
    k = np.arange(11)
    bnd = lmin + (lmax - lmin) * (1 - np.cos(k * np.pi / 10)) / 2
    bins = np.ones(pos.sum())
    for b in range(1, 10):
        bins[lv >= bnd[b]] = b + 1
    bins[lv >= bnd[10]] = 10
    if direction > 0:
        scores[pos] = bins
    else:
        scores[pos] = 11 - bins
    return scores


def pct_score(values, direction):
    """Percentile-rank scoring mapped linearly to 1-10."""
    n = len(values)
    scores = np.full(n, 5.0)
    valid = np.isfinite(values)
    nv = valid.sum()
    if nv < 2:
        return scores
    v = values[valid]
    order = np.argsort(v)
    rnk = np.zeros(nv)
    rnk[order] = np.arange(1, nv + 1, dtype=float)
    # Average ties
    for u in np.unique(v):
        mask = v == u
        rnk[mask] = rnk[mask].mean()
    pct = (rnk - 1) / (nv - 1)
    if direction < 0:
        pct = 1 - pct
    scores[valid] = 1 + pct * 9
    return scores


def tax_composition_v3(b, t):
    tax = get_tax(b, t)
    comp_map = {
        'C': 'Primitive carbonaceous', 'G': 'Primitive carbonaceous',
        'F': 'Primitive carbonaceous',
        'B': 'Primitive blue (possible organics/ice)',
        'D': 'Primitive dark (organic-rich)', 'T': 'Primitive transitional',
        'P': 'Primitive dark (P-type)',
        'S': 'Silicaceous stony', 'Q': 'Silicaceous stony',
        'A': 'Olivine-dominated (mantle fragment?)',
        'V': 'Basaltic achondrite (differentiated crust)',
        'R': 'Pyroxene-olivine', 'L': 'Spinel-bearing stony',
        'K': 'Eos-family stony (intermediate)',
        'M': 'Metallic or enstatite (ambiguous)',
        'X': 'X-complex (metallic / enstatite)',
        'E': 'Enstatite achondrite',
    }
    return comp_map.get(tax, 'Unclassified -- unknown composition')


def run_tradeoff_v3(csv_path='sbdb_query_results.csv', output_path='asteroid_tradeoff.csv'):
    """Version 3: enhanced science scoring trade-off table."""
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
    pdes_col = safe_str(T, 'pdes')
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

    # Fill missing diameters
    for k in range(n):
        if np.isnan(diameter[k]) and not np.isnan(H_mag[k]):
            alb = albedo_v[k] if not np.isnan(albedo_v[k]) else 0.15
            diameter[k] = 1329 / np.sqrt(alb) * 10 ** (-H_mag[k] / 5)
    radius = diameter / 2

    # Taxonomy -> density -> mass
    density = np.zeros(n)
    for k in range(n):
        tax = get_tax(spec_B[k], spec_T[k])
        if tax in ('C', 'B', 'F', 'G', 'D', 'T'):
            density[k] = 1400
        elif tax in ('S', 'Q', 'A', 'V', 'R', 'L', 'K'):
            density[k] = 2700
        elif tax in ('M', 'X', 'E', 'P'):
            density[k] = 3500
        else:
            density[k] = 2000
    mass_kg = (4 / 3) * np.pi * (radius * 1e3) ** 3 * density

    # -- Science potential --
    sci_A = np.zeros(n)
    sci_B = np.zeros(n)

    for k in range(n):
        pdes = pdes_col[k].strip()
        b = spec_B[k].strip()
        t = spec_T[k].strip()
        tax1 = get_tax(b, t)

        # Component A: Characterisation depth
        a = 0
        if b:
            a += 2
        elif t:
            a += 1
        if not np.isnan(diameter[k]) and not np.isnan(diam_sig[k]) and diam_sig[k] < 5:
            a += 2
        elif not np.isnan(diameter[k]):
            a += 1
        if not np.isnan(rot_per[k]):
            a += 2
        if not np.isnan(albedo_v[k]):
            a += 2
        if not np.isnan(BV[k]):
            a += 1
        sci_A[k] = min(a, 10)

        # Component B: Intrinsic science interest
        pts = 0
        if len(b) >= 2 and b[1].upper() == 'H':
            pts += 2
        if tax1 == 'B' and pdes not in ICE_LIST:
            pts += 1
        if pdes in ACTIVE_LIST:
            pts += 3
        if pdes in ICE_LIST:
            pts += 3
        if pdes in AMBIG_M_LIST:
            pts += 2
        if pdes in VISITED_LIST:
            if pdes in FLYBY_ONLY_LIST:
                pts -= 1
            else:
                pts -= 3
        sci_B[k] = max(0, min(pts, 10))

    sci_combined = 0.40 * sci_A + 0.60 * sci_B

    # Rotation period log-distance
    rot_dist = np.full(n, np.nan)
    for k in range(n):
        if not np.isnan(rot_per[k]) and rot_per[k] > 0:
            rot_dist[k] = abs(np.log(rot_per[k] / 12))

    # Score all criteria
    score_mass = log_cheb_score(mass_kg, +1)
    score_radius = log_cheb_score(radius, +1)
    score_ecc = pct_score(ecc, -1)
    score_inc = pct_score(inc, -1)
    score_sma = pct_score(sma, -1)
    score_rot = pct_score(rot_dist, -1)
    score_sci = pct_score(sci_combined, +1)
    score_dv = np.zeros(n)

    # Weighted total
    W = {'mass': 0.12, 'radius': 0.12, 'ecc': 0.06, 'inc': 0.20,
         'sma': 0.02, 'rot': 0.04, 'sci': 0.14, 'dv': 0.30}

    total = (W['mass'] * score_mass + W['radius'] * score_radius +
             W['ecc'] * score_ecc + W['inc'] * score_inc +
             W['sma'] * score_sma + W['rot'] * score_rot +
             W['sci'] * score_sci + W['dv'] * score_dv)

    # -- Science flag summary column --
    sci_flags = []
    for k in range(n):
        pdes = pdes_col[k].strip()
        b = spec_B[k].strip()
        tax1 = get_tax(b, spec_T[k].strip())
        tags = []
        if len(b) >= 2 and b[1].upper() == 'H':
            tags.append('aqueous-alteration(+3)')
        if tax1 == 'B' and pdes not in ICE_LIST:
            tags.append('B-type(+1)')
        if pdes in ACTIVE_LIST:
            tags.append('active-MBC(+3)')
        if pdes in ICE_LIST:
            tags.append('surface-ice(+3)')
        if pdes in AMBIG_M_LIST:
            tags.append('radar-ambiguous-M(+2)')
        if pdes in VISITED_LIST:
            if pdes in FLYBY_ONLY_LIST:
                tags.append('flyby-visited(-1)')
            else:
                tags.append('orbit-visited(-3)')
        sci_flags.append(' | '.join(tags) if tags else 'none')

    # -- Format columns --
    name_col = []
    class_col = []
    for k in range(n):
        nm = full_name[k].strip()
        nm = re.sub(r'\s*\([A-Z]\d{3,4}\s+[A-Z]{2}\)\s*$', '', nm).strip()
        if np.isnan(radius[k]):
            name_col.append(f'{nm}  (r=N/A)')
        else:
            name_col.append(f'{nm}  (r={radius[k]:.1f}km)')

        b, t = spec_B[k].strip(), spec_T[k].strip()
        comp = tax_composition_v3(b, t)
        if b:
            class_col.append(f'{b} -- {comp} [SMASSII]')
        elif t:
            class_col.append(f'{t} -- {comp} [Tholen]')
        else:
            class_col.append('Unclassified')

    out = pd.DataFrame({
        'Name_DecRadius': name_col,
        'Class_Composition_SMASSII': class_col,
        'Mass_kg': mass_kg,
        'Radius_km': radius,
        'SMA_AU': sma,
        'Eccentricity': ecc,
        'Inclination_deg': inc,
        'RotPeriod_hr': rot_per,
        'SciScore_A_Characterisation': np.round(sci_A, 2),
        'SciScore_B_Interest': np.round(sci_B, 2),
        'SciScore_Combined': np.round(sci_combined, 2),
        'SciFlags_Applied': sci_flags,
        'SciPotential_Score_1to10': np.round(score_sci, 2),
        'DeltaV_Score': np.round(score_dv, 2),
        'Total_WeightedScore': np.round(total, 4),
        'Sub_Mass_Score': np.round(score_mass, 2),
        'Sub_Radius_Score': np.round(score_radius, 2),
        'Sub_Ecc_Score': np.round(score_ecc, 2),
        'Sub_Inc_Score': np.round(score_inc, 2),
        'Sub_SMA_Score': np.round(score_sma, 2),
        'Sub_RotPer_Score': np.round(score_rot, 2),
    })

    out = out.sort_values('Total_WeightedScore', ascending=False).reset_index(drop=True)
    out.to_csv(output_path, index=False)

    # Console report
    print(f'Done.  {output_path} written ({len(out)} rows).\n')
    print('Top 25 candidates:')
    hdr = f'{"Name":46s}  {"Class":24s}  {"Sci":>5s}  {"DV":>5s}  {"Total":>5s}'
    print(hdr)
    print('-' * len(hdr))
    for k in range(min(25, len(out))):
        row = out.iloc[k]
        print(f'{row["Name_DecRadius"]:46s}  {row["Class_Composition_SMASSII"]:24s}  '
              f'{row["SciPotential_Score_1to10"]:5.2f}  {row["DeltaV_Score"]:5.2f}  '
              f'{row["Total_WeightedScore"]:5.2f}')

    print('\nScience flag summary (top 30):')
    print(f'{"Name":36s}  Flags')
    print('-' * 80)
    for k in range(min(30, len(out))):
        if out.iloc[k]['SciFlags_Applied'] != 'none':
            print(f'{out.iloc[k]["Name_DecRadius"]:36s}  {out.iloc[k]["SciFlags_Applied"]}')

    print('\nScore diagnostics:')
    names_diag = ['Mass', 'Radius', 'Ecc', 'Inc', 'SMA', 'Rot', 'Science']
    scores_diag = [score_mass, score_radius, score_ecc, score_inc,
                   score_sma, score_rot, score_sci]
    for j, name in enumerate(names_diag):
        s = scores_diag[j]
        print(f'  {name:8s}  min={s.min():4.1f}  max={s.max():4.1f}  '
              f'mean={s.mean():4.2f}  std={s.std():4.2f}')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    run_tradeoff_v3()
