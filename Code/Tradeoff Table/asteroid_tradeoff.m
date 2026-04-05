%% =========================================================================
%  Input  : sbdb_query_results.csv  (JPL SBDB query export)
%  Output : asteroid_tradeoff.csv   
%
%  Scoring weights
%    Mass                10 %
%    Radius              10 %
%    Eccentricity        15 %
%    Inclination         15 %
%    Semi-major axis      5 %
%    Rotation period      5 %
%    Science potential   10 %
%    Delta-V             30 %  (set to 0 pending trajectory analysis)
%
%  Each criterion is scored 1-10 using Chebyshev-spaced bin boundaries,
%  then multiplied by its weight.  Total score is out of 10.
% =========================================================================

clear; clc;

%% =========================================================================
%  SECTION 1 — READ INPUT FILE
% =========================================================================
fprintf('Reading sbdb_query_results.csv ...\n');

try
    T = readtable('sbdb_query_results.csv', ...
        'VariableNamingRule', 'preserve');
    % If only one column was parsed, the file is tab-delimited
    if width(T) < 5
        T = readtable('sbdb_query_results.csv', ...
            'Delimiter', '\t', 'VariableNamingRule', 'preserve');
    end
catch ME
    error('Could not read sbdb_query_results.csv.\nError: %s', ME.message);
end

n = height(T);
colNames = T.Properties.VariableNames;
fprintf('  %d asteroids loaded.\n\n', n);

%% =========================================================================
%  SECTION 2 — EXTRACT COLUMNS
%  safeNum returns a double column (NaN if column missing or value blank).
%  safeStr returns a cell-string column ('' if missing).
% =========================================================================
full_name = safeStr(T, 'full_name',        colNames, n);
spec_B    = safeStr(T, 'spec_B',           colNames, n);   % Bus-DeMeo taxonomy
spec_T    = safeStr(T, 'spec_T',           colNames, n);   % Tholen taxonomy

H_mag     = safeNum(T, 'H',               colNames, n);   % absolute magnitude
diameter  = safeNum(T, 'diameter',         colNames, n);   % km
albedo_v  = safeNum(T, 'albedo',           colNames, n);   % geometric albedo
rot_per   = safeNum(T, 'rot_per',          colNames, n);   % hours
BV        = safeNum(T, 'BV',              colNames, n);   % B-V colour index
diam_sig  = safeNum(T, 'diameter_sigma',   colNames, n);   % km
ecc       = safeNum(T, 'e',               colNames, n);   % eccentricity
sma       = safeNum(T, 'a',               colNames, n);   % semi-major axis, AU
inc       = safeNum(T, 'i',               colNames, n);   % inclination, deg

%% =========================================================================
%  SECTION 3 — FILL MISSING DIAMETERS FROM H-MAGNITUDE AND ALBEDO
%  Formula: D (km) = 1329 / sqrt(p_v) × 10^(-H/5)
%  Default albedo 0.15 used when albedo is also missing.
% =========================================================================
for k = 1:n
    if isnan(diameter(k)) && ~isnan(H_mag(k))
        alb = albedo_v(k);
        if isnan(alb)
            alb = 0.15;   % conservative default
        end
        diameter(k) = 1329 / sqrt(alb) * 10^(-H_mag(k) / 5);
    end
end
radius = diameter / 2;    % km

%% =========================================================================
%  SECTION 4 — ESTIMATE BULK DENSITY FROM TAXONOMY → MASS
%  Density values follow Carry (2012) Planetary & Space Science averages:
%    C / B / F / G / D / T  (primitive, hydrated)  ~1,300–1,500 kg/m^3
%    S / Q / A / L / R / V  (silicaceous stony)     ~2,700 kg/m^3
%    M / X / E / P          (metallic / enstatite)  ~3,500 kg/m^3
%    Unknown                                          ~2,000 kg/m^3
%  Mass = (4/3) π r^3 ρ
% =========================================================================
density_kgm3 = zeros(n, 1);
for k = 1:n
    tax = '';
    b = strtrim(spec_B{k});
    t = strtrim(spec_T{k});
    if ~isempty(b);     tax = upper(b(1));
    elseif ~isempty(t); tax = upper(t(1));
    end
    switch tax
        case {'C','B','F','G','D','T'}
            density_kgm3(k) = 1400;
        case {'S','Q','A','V','R','L','K'}
            density_kgm3(k) = 2700;
        case {'M','X','E','P'}
            density_kgm3(k) = 3500;
        otherwise
            density_kgm3(k) = 2000;
    end
end

r_m      = radius * 1e3;                              % km  →  m
mass_kg  = (4/3) * pi * r_m.^3 .* density_kgm3;     % kg

%% =========================================================================
%  SECTION 5 — SCIENCE POTENTIAL RAW SCORE  (0 – 10)
%  Proxy for pre-mission characterisation depth and instrument synergy.
%  Points are assigned for: confirmed taxonomy, measured rotation period,
%  low-uncertainty measured diameter, known albedo, colour photometry.
% =========================================================================
sci_raw = zeros(n, 1);
for k = 1:n
    s = 0;
    b = strtrim(spec_B{k});
    t = strtrim(spec_T{k});
    hasTax = ~isempty(b) || ~isempty(t);
    if hasTax;                                           s = s + 3; end
    if ~isnan(rot_per(k));                               s = s + 2; end
    if ~isnan(diameter(k)) && ~isnan(diam_sig(k)) && diam_sig(k) < 5
                                                         s = s + 2;
    elseif ~isnan(diameter(k));                          s = s + 1; end
    if ~isnan(albedo_v(k));                              s = s + 1; end
    if ~isnan(BV(k));                                    s = s + 2; end
    sci_raw(k) = min(s, 10);
end

%% =========================================================================
%  SECTION 6 — ROTATION PERIOD SCORE
%  Optimal science/operations window: 6–24 hours.
%  Score is based on log-distance from 12-hour midpoint so that both
%  very fast (<2 h) and very slow (>200 h) rotators are penalised.
% =========================================================================
rot_dist = NaN(n, 1);
for k = 1:n
    if ~isnan(rot_per(k)) && rot_per(k) > 0
        rot_dist(k) = abs(log(rot_per(k) / 12));
    end
end
% Lower distance → better rotation → direction = -1
score_rot = chebScore(rot_dist, -1);

%% =========================================================================
%  SECTION 7 — CHEBYSHEV SCORE ALL PRIMARY CRITERIA  (1 – 10)
% =========================================================================
score_mass   = chebScore(mass_kg,  +1);   % more massive   → better
score_radius = chebScore(radius,   +1);   % larger radius  → better
score_ecc    = chebScore(ecc,      -1);   % lower ecc      → better (cheaper Δv)
score_inc    = chebScore(inc,      -1);   % lower inc      → better (cheaper plane change)
score_sma    = chebScore(sma,      -1);   % closer to Sun  → better (shorter transfer)
score_sci    = chebScore(sci_raw,  +1);   % more data      → better
score_dv     = zeros(n, 1);              % placeholder — set to 0, update after trajectory

%% =========================================================================
%  SECTION 8 — WEIGHTED TOTAL SCORE  (out of 10)
% =========================================================================
W_mass   = 0.10;
W_radius = 0.10;
W_ecc    = 0.15;
W_inc    = 0.15;
W_sma    = 0.05;
W_rot    = 0.05;
W_sci    = 0.10;
W_dv     = 0.30;

total = W_mass   * score_mass   + ...
        W_radius * score_radius + ...
        W_ecc    * score_ecc    + ...
        W_inc    * score_inc    + ...
        W_sma    * score_sma    + ...
        W_rot    * score_rot    + ...
        W_sci    * score_sci    + ...
        W_dv     * score_dv;

%% =========================================================================
%  SECTION 9 — BUILD FORMATTED OUTPUT COLUMNS
% =========================================================================
name_col  = cell(n, 1);
class_col = cell(n, 1);

for k = 1:n
    % --- Name with parenthetical radius ---
    nm = strtrim(full_name{k});
    % Strip trailing (YYYY XX) designation if present
    nm = regexprep(nm, '\s*\([A-Z]\d{3,4}\s+[A-Z]{2}\)\s*$', '');
    nm = strtrim(nm);
    if isnan(radius(k))
        name_col{k} = sprintf('%s (r = N/A)', nm);
    else
        name_col{k} = sprintf('%s (r = %.1f km)', nm, radius(k));
    end

    % --- Class (Composition) [taxonomy source] ---
    b    = strtrim(spec_B{k});
    t    = strtrim(spec_T{k});
    comp = taxComposition(b, t);
    if ~isempty(b)
        class_col{k} = sprintf('%s (%s) [SMASSII]', b, comp);
    elseif ~isempty(t)
        class_col{k} = sprintf('%s (%s) [Tholen]',  t, comp);
    else
        class_col{k} = 'Unknown (Unclassified)';
    end
end

%% =========================================================================
%  SECTION 10 — ASSEMBLE OUTPUT TABLE AND WRITE CSV
%  Columns follow the requested layout, with individual criterion scores
%  added as extra columns for full traceability.
% =========================================================================
out = table( ...
    name_col,                           ...
    class_col,                          ...
    mass_kg,                            ...
    radius,                             ...
    sma,                                ...
    ecc,                                ...
    inc,                                ...
    rot_per,                            ...
    round(score_sci,    2),             ...
    round(score_dv,     2),             ...
    round(total,        4),             ...
    round(score_mass,   2),             ...
    round(score_radius, 2),             ...
    round(score_ecc,    2),             ...
    round(score_inc,    2),             ...
    round(score_sma,    2),             ...
    round(score_rot,    2),             ...
    'VariableNames', { ...
        'Name_DecRadius',               ...  col 1
        'Class_Composition_SMASSII',    ...  col 2
        'Mass_kg',                      ...  col 3
        'Radius_km',                    ...  col 4
        'SMA_AU',                       ...  col 5
        'Eccentricity',                 ...  col 6
        'Inclination_deg',              ...  col 7
        'RotPeriod_hr',                 ...  col 8
        'SciPotential_Score',           ...  col 9  (weight 10%)
        'DeltaV_Score',                 ...  col 10 (weight 30%, currently 0)
        'Total_WeightedScore',          ...  col 11
        'Subscr_Mass_Score',            ...  col 12 — traceability
        'Subscr_Radius_Score',          ...  col 13
        'Subscr_Ecc_Score',             ...  col 14
        'Subscr_Inc_Score',             ...  col 15
        'Subscr_SMA_Score',             ...  col 16
        'Subscr_RotPer_Score'           ...  col 17
    });

% Sort by total score, best candidates first
out = sortrows(out, 'Total_WeightedScore', 'descend');

writetable(out, 'asteroid_tradeoff.csv');

fprintf('Done.  asteroid_tradeoff.csv written (%d rows).\n\n', height(out));
fprintf('Top 20 candidates:\n');
fprintf('%-42s  %-30s  %6s\n', 'Name', 'Class', 'Score');
fprintf('%s\n', repmat('-', 1, 82));
for k = 1 : min(20, height(out))
    fprintf('%-42s  %-30s  %6.4f\n', ...
        out.Name_DecRadius{k}, out.Class_Composition_SMASSII{k}, ...
        out.Total_WeightedScore(k));
end

%% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function scores = chebScore(values, direction)
% CHEBSCORE  Assigns integer scores 1–10 using Chebyshev-spaced boundaries.
%
%   Chebyshev spacing places bin boundaries at
%       bnd(k) = vmin + (vmax-vmin)/2 * (1 - cos(k*pi/10)),  k = 0..10
%   which are denser near the extremes so that the best and worst objects
%   are more finely discriminated than the middle of the distribution.
%
%   direction = +1 : highest values → score 10
%   direction = -1 : lowest  values → score 10
%   NaN values     : neutral score 5
%
    n      = length(values);
    scores = 5 * ones(n, 1);          % default for NaN

    valid  = ~isnan(values) & ~isinf(values);
    v      = values(valid);
    if sum(valid) < 2; return; end

    vmin = min(v);
    vmax = max(v);
    if vmax <= vmin
        scores(valid) = 5;
        return;
    end

    % 11 Chebyshev boundary points → 10 bins
    k   = 0:10;
    bnd = vmin + (vmax - vmin) * (1 - cos(k * pi / 10)) / 2;

    % Assign bin index (1 = lowest range, 10 = highest range)
    bin = ones(sum(valid), 1);
    for b = 1:9
        bin(v >= bnd(b + 1)) = b + 1;
    end
    bin(v >= bnd(11)) = 10;           % handle exact maximum

    if direction > 0
        scores(valid) = bin;          % high value → high score
    else
        scores(valid) = 11 - bin;     % low value  → high score
    end
end

% -------------------------------------------------------------------------
function v = safeNum(T, name, cols, n)
% SAFENUM  Extracts a named column as double; returns NaN column if absent.
    if any(strcmp(cols, name))
        col = T.(name);
        if iscell(col)
            v = cellfun(@(x) str2double(string(x)), col);
        elseif isstring(col) || ischar(col)
            v = str2double(col);
        else
            v = double(col);
        end
        v = v(:);
    else
        fprintf('  [warning] column "%s" not found — filling with NaN.\n', name);
        v = NaN(n, 1);
    end
end

% -------------------------------------------------------------------------
function c = safeStr(T, name, cols, n)
% SAFESTR  Extracts a named column as a cell-string; returns '' if absent.
    if any(strcmp(cols, name))
        col = T.(name);
        if iscell(col)
            c = col;
        elseif isstring(col)
            c = cellstr(col);
        else
            c = cellstr(string(col));
        end
        c = c(:);
    else
        c = repmat({''}, n, 1);
    end
    % Replace missing/NaN strings with empty
    for k = 1:n
        if isequal(c{k}, missing) || (ischar(c{k}) && strcmpi(strtrim(c{k}), 'NaN'))
            c{k} = '';
        end
    end
end

% -------------------------------------------------------------------------
function comp = taxComposition(b, t)
% TAXCOMPOSITION  Returns a human-readable composition string from taxonomy.
    tax = '';
    if ~isempty(b) && length(b) >= 1; tax = upper(b(1)); end
    if isempty(tax) && ~isempty(t) && length(t) >= 1; tax = upper(t(1)); end
    switch tax
        case {'C','B','G','F'}; comp = 'Primitive carbonaceous';
        case 'D';               comp = 'Primitive dark (D-type)';
        case 'T';               comp = 'Primitive transitional';
        case 'P';               comp = 'Primitive dark (P-type)';
        case {'S','Q'};         comp = 'Silicaceous stony';
        case 'A';               comp = 'Olivine-dominated';
        case 'V';               comp = 'Basaltic achondrite';
        case 'R';               comp = 'Pyroxene-olivine';
        case 'L';               comp = 'Spinel-bearing stony';
        case 'K';               comp = 'Eos-family stony';
        case 'M';               comp = 'Metallic (M-type)';
        case 'X';               comp = 'Metallic / enstatite';
        case 'E';               comp = 'Enstatite achondrite';
        otherwise;              comp = 'Unclassified';
    end
end
