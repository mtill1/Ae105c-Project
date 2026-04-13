%% =========================================================================
%  ASTEROID TRADE-OFF TABLE GENERATOR  v3
%  Input  : sbdb_query_results.csv
%  Output : asteroid_tradeoff.csv
%
%  WEIGHT SCHEME (recommended, preliminary stage)
%    Delta-V             30 %   placeholder = 0 until trajectory analysis
%    Inclination         15 %
%    Science potential   15 %   split: 40% characterisation + 60% interest
%    Mass                12 %
%    Eccentricity        7 %
%    Radius               13 %
%    SMA                  3 %
%    Rotation period      5 %
%
%  SCIENCE POTENTIAL MODEL
%    Component A — Characterisation depth (how well do we know it)
%      Scored 0-10 from SBDB columns automatically.
%
%    Component B — Intrinsic science interest (only "extra extra" features)
%      These are the features that make a target genuinely exceptional
%      beyond its basic physical parameters. Each flag is either
%      auto-detected from taxonomy or manually set in Section 2 below.
%
%      Flag                               Points   How set
%      ─────────────────────────────────  ──────   ──────────────────────
%      Ch/Cgh aqueous alteration          +3       auto from spec_B
%      Active asteroid / main-belt comet  +3       manual  (ACTIVE_LIST)
%      B-type with surface ice/organics   +3       manual  (ICE_LIST)
%      M-type ambiguous radar albedo      +2       manual  (AMBIG_M_LIST)
%      Confirmed binary system            +2       manual  (BINARY_LIST)
%      Isolated V-type (non-Vesta family) +2       manual  (ISOLATED_V_LIST)
%      Collisional family parent body     +1       manual  (PARENT_LIST)
%      Previously orbited by spacecraft   -3       auto from VISITED_LIST
%
%      Raw bonus is capped at 10 before combining with Component A.
%
%    Final: SciScore = 0.40 * A + 0.60 * B   (interest weighted higher)
% =========================================================================

clear; clc;

%% =========================================================================
%  SECTION 1 — READ INPUT FILE
% =========================================================================
fprintf('Reading sbdb_query_results.csv ...\n');
try
    T = readtable('sbdb_query_results.csv', 'VariableNamingRule', 'preserve');
    if width(T) < 5
        T = readtable('sbdb_query_results.csv', ...
            'Delimiter', '\t', 'VariableNamingRule', 'preserve');
    end
catch ME
    error('Could not read sbdb_query_results.csv.\nError: %s', ME.message);
end
n        = height(T);
colNames = T.Properties.VariableNames;
fprintf('  %d asteroids loaded.\n\n', n);

%% =========================================================================
%  SECTION 2 — MANUAL SCIENCE INTEREST FLAGS
%
%  Add asteroid NUMBER (pdes) as a string to each list below.
%  Sources:
%    ACTIVE_LIST    : Hsieh & Jewitt (2006) + subsequent MBC discoveries
%    ICE_LIST       : Campins et al. (2010), Rivkin & Emery (2010),
%                     Takir & Emery (2012) 3-micron survey
%    AMBIG_M_LIST   : Ockert-Bell et al. (2010) IRTF M-type survey,
%                     Shepard et al. (2015) radar albedo study
%
%  Numbers shown here are the most commonly discussed candidates —
%  extend the lists as your candidate pool grows.
% =========================================================================

% Active asteroids / main-belt comets (confirmed outgassing or dust)
ACTIVE_LIST = { ...
    '133P',  ...   % 133P/Elst-Pizarro — prototype MBC
    '176P',  ...   % 176P/LINEAR
    '238P',  ...   % 238P/Read
    '259P',  ...   % 259P/Garradd
    '288P',  ...   % 288P — also a binary
    '313P',  ...   % 313P/Gibbs
    '324P',  ...   % 324P/La Sagra
    '358P',  ...   % 358P/PANSTARRS
    '433P'   ...   % 433P/Iannini
};

% B-types / C-types with confirmed surface ice or 3-micron ice feature
ICE_LIST = { ...
    '24',    ...   % 24 Themis — Campins et al. 2010 (first detection)
    '65',    ...   % 65 Cybele — Licandro et al. 2011
    '90',    ...   % 90 Antiope (Themis family, binary)
    '153',   ...   % 153 Hilda
    '3200'   ...   % 3200 Phaethon — B-type, anomalous, active near perihelion
};

% M-types with ambiguous radar albedo (could be metallic OR primitive)
AMBIG_M_LIST = { ...
    '21',    ...   % 21 Lutetia — Rosetta revealed non-metallic surface
    '22',    ...   % 22 Kalliope — moderate radar, lower than pure metal
    '69',    ...   % 69 Hesperia
    '97',    ...   % 97 Klotho
    '110',   ...   % 110 Lydia
    '129',   ...   % 129 Antigone
    '135',   ...   % 135 Hertha
    '785'    ...   % 785 Zwetana
};


% Asteroids previously orbited (or landed on) by a dedicated spacecraft
% These receive a -3 penalty for science duplication
VISITED_LIST = { ...
    '1',     ...   % 1 Ceres — Dawn (orbit 2015-2018)
    '4',     ...   % 4 Vesta — Dawn (orbit 2011-2012)
    '433',   ...   % 433 Eros — NEAR Shoemaker (orbit + landing 2000-2001)
    '951',   ...   % 951 Gaspra — Galileo flyby 1991
    '243',   ...   % 243 Ida — Galileo flyby 1993
    '253',   ...   % 253 Mathilde — NEAR flyby 1997
    '25143', ...   % 25143 Itokawa — Hayabusa (sample return 2005)
    '101955',...   % 101955 Bennu — OSIRIS-REx (sample return 2020-2021)
    '162173',...   % 162173 Ryugu — Hayabusa2 (sample return 2019)
    '16',    ...   % 16 Psyche — NASA Psyche mission en route (launched 2023)
    '2867',  ...   % 2867 Steins — Rosetta flyby 2008
    '21',    ...   % 21 Lutetia — Rosetta flyby 2010 (flyby only → -1, not full -3)
    '132524',...   % 132524 APL — New Horizons flyby 2006
    '5535',  ...   % 5535 Annefrank — Stardust flyby 2002
    '9969',  ...   % 9969 Braille — Deep Space 1 flyby 1999
    '2685',  ...   % 2685 Masursky — Cassini flyby 2000
    '52246'  ...   % 52246 Donaldjohanson — Lucy target (upcoming)
};

% Rosetta only did a flyby of Lutetia (not a full orbit mission) —
% override its penalty to -1 to reflect that significant science remains
FLYBY_ONLY_LIST = { ...
    '21',    ...   % 21 Lutetia — Rosetta flyby, no orbit, science incomplete
    '951',   ...   % 951 Gaspra
    '243',   ...   % 243 Ida
    '253',   ...   % 253 Mathilde
    '2867',  ...   % 2867 Steins
    '132524' ...   % 132524 APL
};

%% =========================================================================
%  SECTION 3 — EXTRACT COLUMNS
% =========================================================================
full_name = safeStr(T, 'full_name',      colNames, n);
pdes_col  = safeStr(T, 'pdes',           colNames, n);
spec_B    = safeStr(T, 'spec_B',         colNames, n);
spec_T    = safeStr(T, 'spec_T',         colNames, n);

H_mag    = safeNum(T, 'H',              colNames, n);
diameter = safeNum(T, 'diameter',        colNames, n);
albedo_v = safeNum(T, 'albedo',          colNames, n);
rot_per  = safeNum(T, 'rot_per',         colNames, n);
BV       = safeNum(T, 'BV',             colNames, n);
diam_sig = safeNum(T, 'diameter_sigma',  colNames, n);
ecc      = safeNum(T, 'e',              colNames, n);
sma      = safeNum(T, 'a',              colNames, n);
inc      = safeNum(T, 'i',              colNames, n);

%% =========================================================================
%  SECTION 4 — FILL MISSING DIAMETERS
% =========================================================================
for k = 1:n
    if isnan(diameter(k)) && ~isnan(H_mag(k))
        alb = albedo_v(k);
        if isnan(alb); alb = 0.15; end
        diameter(k) = 1329 / sqrt(alb) * 10^(-H_mag(k) / 5);
    end
end
radius = diameter / 2;

%% =========================================================================
%  SECTION 5 — TAXONOMY → DENSITY → MASS
% =========================================================================
density_kgm3 = zeros(n, 1);
for k = 1:n
    tax = getTax(spec_B{k}, spec_T{k});
    switch tax
        case {'C','B','F','G','D','T'}; density_kgm3(k) = 1400;
        case {'S','Q','A','V','R','L','K'}; density_kgm3(k) = 2700;
        case {'M','X','E','P'};         density_kgm3(k) = 3500;
        otherwise;                      density_kgm3(k) = 2000;
    end
end
mass_kg = (4/3) * pi * (radius * 1e3).^3 .* density_kgm3;

%% =========================================================================
%  SECTION 6 — SCIENCE POTENTIAL
%
%  Component A: Characterisation depth (auto from SBDB columns)
%  Component B: Intrinsic science interest (auto + manual flags)
% =========================================================================

sci_A = zeros(n, 1);
sci_B = zeros(n, 1);

for k = 1:n
    pdes = strtrim(pdes_col{k});
    b    = strtrim(spec_B{k});
    t    = strtrim(spec_T{k});
    tax1 = getTax(b, t);               % primary class letter

    % ── Component A: Characterisation depth ──────────────────────────────
    a = 0;
    if ~isempty(b);                                          a = a + 2; % SMASS-II class
    elseif ~isempty(t);                                      a = a + 1; end % Tholen only
    if ~isnan(diameter(k)) && ~isnan(diam_sig(k)) && diam_sig(k) < 5
                                                             a = a + 2; % precise size
    elseif ~isnan(diameter(k));                              a = a + 1; end
    if ~isnan(rot_per(k));                                   a = a + 2; end % spin known
    if ~isnan(albedo_v(k));                                  a = a + 2; end % albedo measured
    if ~isnan(BV(k));                                        a = a + 1; end % colour photometry
    sci_A(k) = min(a, 10);

    % ── Component B: Intrinsic science interest ───────────────────────────
    pts = 0;

    % Auto-detect: Ch / Cgh aqueous alteration (water-bearing)
    % The 'h' suffix in Bus-DeMeo (Ch, Cgh, Xh) indicates the 0.7-μm
    % phyllosilicate band — unambiguous evidence of past liquid water.
    if length(b) >= 2 && upper(b(2)) == 'H'
        pts = pts + 2;   % phyllosilicate band = aqueous alteration confirmed
    end

    % Auto-detect: B-type (unusual blue slope, possible organics/ice)
    % B-types are spectroscopically distinct from C-types and may harbour
    % surface organics or hydrated silicates in a different form.
    if strcmp(tax1, 'B') && ~ismember(pdes, ICE_LIST)
        pts = pts + 1;   % base B-type bonus (ice list gives full bonus below)
    end


    % Manual flags — applied via the lists defined in Section 2
    if ismember(pdes, ACTIVE_LIST);       pts = pts + 3; end  % MBC / active
    if ismember(pdes, ICE_LIST);          pts = pts + 3; end  % confirmed ice
    if ismember(pdes, AMBIG_M_LIST);      pts = pts + 2; end  % radar ambiguity


    % Visited penalty — full orbit missions get -3, flyby-only gets -1
    if ismember(pdes, VISITED_LIST)
        if ismember(pdes, FLYBY_ONLY_LIST)
            pts = pts - 1;   % flyby: significant science remains, small penalty
        else
            pts = pts - 3;   % dedicated orbit/landing: major duplication
        end
    end

    sci_B(k) = max(0, min(pts, 10));   % clamp to [0, 10]
end

% Combined science score (interest weighted 60%, depth 40%)
sci_combined = 0.40 * sci_A + 0.60 * sci_B;

%% =========================================================================
%  SECTION 7 — ROTATION PERIOD LOG-DISTANCE FROM 12-HR OPTIMUM
% =========================================================================
rot_dist = NaN(n, 1);
for k = 1:n
    if ~isnan(rot_per(k)) && rot_per(k) > 0
        rot_dist(k) = abs(log(rot_per(k) / 12));
    end
end

%% =========================================================================
%  SECTION 8 — SCORE ALL CRITERIA
%
%  logChebScore : log10-transformed Chebyshev bins (mass, radius)
%  pctScore     : percentile rank → 1-10 (all others)
% =========================================================================
score_mass   = logChebScore(mass_kg,       +1);
score_radius = logChebScore(radius,        +1);
score_ecc    = pctScore(ecc,               -1);
score_inc    = pctScore(inc,               -1);
score_sma    = pctScore(sma,               -1);
score_rot    = pctScore(rot_dist,          -1);
score_sci    = pctScore(sci_combined,      +1);
score_dv     = zeros(n, 1);
% Replace line above with pctScore(delta_v_km_s, -1) after trajectory runs

%% ========================================================================
%  SECTION 9 — WEIGHTED TOTAL
% =========================================================================
W_mass   = 0.12;
W_radius = 0.12;
W_ecc    = 0.06;
W_inc    = 0.20;
W_sma    = 0.02;
W_rot    = 0.04;
W_sci    = 0.14;
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
%  SECTION 10 — BUILD SCIENCE FLAG SUMMARY COLUMN
%  Human-readable string listing which science bonuses applied.
% =========================================================================
sci_flags = cell(n, 1);
for k = 1:n
    pdes = strtrim(pdes_col{k});
    b    = strtrim(spec_B{k});
    tax1 = getTax(b, strtrim(spec_T{k}));
    tags = {};

    % Auto flags
    if length(b) >= 2 && upper(b(2)) == 'H'
        tags{end+1} = 'aqueous-alteration(+3)'; end
    if strcmp(tax1,'B') && ~ismember(pdes, ICE_LIST)
        tags{end+1} = 'B-type(+1)'; end


    % Manual flags
    if ismember(pdes, ACTIVE_LIST)
        tags{end+1} = 'active-MBC(+3)'; end
    if ismember(pdes, ICE_LIST)
        tags{end+1} = 'surface-ice(+3)'; end
    if ismember(pdes, AMBIG_M_LIST)
        tags{end+1} = 'radar-ambiguous-M(+2)'; end

    if ismember(pdes, VISITED_LIST)
        if ismember(pdes, FLYBY_ONLY_LIST)
            tags{end+1} = 'flyby-visited(-1)';
        else
            tags{end+1} = 'orbit-visited(-3)';
        end
    end

    if isempty(tags)
        sci_flags{k} = 'none';
    else
        sci_flags{k} = strjoin(tags, ' | ');
    end
end

%% =========================================================================
%  SECTION 11 — FORMAT NAME AND CLASS COLUMNS
% =========================================================================
name_col  = cell(n, 1);
class_col = cell(n, 1);

for k = 1:n
    nm = strtrim(full_name{k});
    nm = regexprep(nm, '\s*\([A-Z]\d{3,4}\s+[A-Z]{2}\)\s*$', '');
    nm = strtrim(nm);
    if isnan(radius(k))
        name_col{k} = sprintf('%s  (r=N/A)', nm);
    else
        name_col{k} = sprintf('%s  (r=%.1fkm)', nm, radius(k));
    end

    b = strtrim(spec_B{k});  t = strtrim(spec_T{k});
    comp = taxComposition(b, t);
    if ~isempty(b)
        class_col{k} = sprintf('%s — %s [SMASSII]', b, comp);
    elseif ~isempty(t)
        class_col{k} = sprintf('%s — %s [Tholen]',  t, comp);
    else
        class_col{k} = 'Unclassified';
    end
end

%% =========================================================================
%  SECTION 12 — ASSEMBLE AND WRITE OUTPUT CSV
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
    round(sci_A,         2),            ...
    round(sci_B,         2),            ...
    round(sci_combined,  2),            ...
    sci_flags,                          ...
    round(score_sci,     2),            ...
    round(score_dv,      2),            ...
    round(total,         4),            ...
    round(score_mass,    2),            ...
    round(score_radius,  2),            ...
    round(score_ecc,     2),            ...
    round(score_inc,     2),            ...
    round(score_sma,     2),            ...
    round(score_rot,     2),            ...
    'VariableNames', { ...
        'Name_DecRadius',               ...
        'Class_Composition_SMASSII',    ...
        'Mass_kg',                      ...
        'Radius_km',                    ...
        'SMA_AU',                       ...
        'Eccentricity',                 ...
        'Inclination_deg',              ...
        'RotPeriod_hr',                 ...
        'SciScore_A_Characterisation',  ...
        'SciScore_B_Interest',          ...
        'SciScore_Combined',            ...
        'SciFlags_Applied',             ...
        'SciPotential_Score_1to10',     ...
        'DeltaV_Score',                 ...
        'Total_WeightedScore',          ...
        'Sub_Mass_Score',               ...
        'Sub_Radius_Score',             ...
        'Sub_Ecc_Score',                ...
        'Sub_Inc_Score',                ...
        'Sub_SMA_Score',                ...
        'Sub_RotPer_Score'              ...
    });

out = sortrows(out, 'Total_WeightedScore', 'descend');
writetable(out, 'asteroid_tradeoff.csv');

%% =========================================================================
%  SECTION 13 — CONSOLE REPORT
% =========================================================================
fprintf('Done.  asteroid_tradeoff.csv written (%d rows).\n\n', height(out));

fprintf('Top 25 candidates:\n');
hdr = sprintf('%-46s  %-24s  %5s  %5s  %5s', ...
    'Name', 'Class', 'Sci', 'ΔV', 'Total');
fprintf('%s\n%s\n', hdr, repmat('-', 1, length(hdr)));
for k = 1:min(25, height(out))
    fprintf('%-46s  %-24s  %5.2f  %5.2f  %5.2f\n', ...
        out.Name_DecRadius{k}, ...
        out.Class_Composition_SMASSII{k}, ...
        out.SciPotential_Score_1to10(k), ...
        out.DeltaV_Score(k), ...
        out.Total_WeightedScore(k));
end

fprintf('\nScience flag summary (top 30):\n');
fprintf('%-36s  %s\n', 'Name', 'Flags');
fprintf('%s\n', repmat('-', 1, 80));
for k = 1:min(30, height(out))
    if ~strcmp(out.SciFlags_Applied{k}, 'none')
        fprintf('%-36s  %s\n', out.Name_DecRadius{k}, out.SciFlags_Applied{k});
    end
end

fprintf('\nScore diagnostics:\n');
names_diag = {'Mass','Radius','Ecc','Inc','SMA','Rot','Science'};
scores_diag = [score_mass, score_radius, score_ecc, score_inc, ...
               score_sma,  score_rot,    score_sci];
for j = 1:7
    s = scores_diag(:,j);
    fprintf('  %-8s  min=%4.1f  max=%4.1f  mean=%4.2f  std=%4.2f\n', ...
        names_diag{j}, min(s), max(s), mean(s), std(s));
end

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function tax = getTax(b, t)
%GETTAX  Returns single uppercase taxonomy letter from Bus-DeMeo or Tholen.
    tax = '';
    b = strtrim(b);  t = strtrim(t);
    if ~isempty(b); tax = upper(b(1)); return; end
    if ~isempty(t); tax = upper(t(1)); end
end

% -------------------------------------------------------------------------
function scores = logChebScore(values, direction)
%LOGCHEBSCORE  Chebyshev-spaced bins on log10-transformed values (1-10).
%  Prevents outliers like Ceres from collapsing all other objects into
%  the bottom bins by working on the logarithmic scale.
    n      = length(values);
    scores = 5 * ones(n, 1);
    pos    = values > 0 & ~isnan(values) & ~isinf(values);
    if sum(pos) < 2; return; end
    lv   = log10(values(pos));
    lmin = min(lv);  lmax = max(lv);
    if lmax <= lmin; scores(pos) = 5; return; end
    k   = (0:10)';
    bnd = lmin + (lmax - lmin) * (1 - cos(k * pi / 10)) / 2;
    bin = ones(sum(pos), 1);
    for b = 1:9; bin(lv >= bnd(b+1)) = b + 1; end
    bin(lv >= bnd(11)) = 10;
    if direction > 0; scores(pos) = bin;
    else;             scores(pos) = 11 - bin; end
end

% -------------------------------------------------------------------------
function scores = pctScore(values, direction)
%PCTSCORE  Percentile-rank scoring mapped linearly to 1-10.
%  Distribution-free — no outlier can distort the rest of the scoring.
    n      = length(values);
    scores = 5 * ones(n, 1);
    valid  = ~isnan(values) & ~isinf(values);
    nv     = sum(valid);
    if nv < 2; return; end
    v = values(valid);
    [sv, si] = sort(v, 'ascend');
    rnk = (1:nv)';
    u = unique(sv);
    for j = 1:length(u)
        idx = sv == u(j);  rnk(idx) = mean(rnk(idx));
    end
    pct = zeros(nv,1);  pct(si) = (rnk - 1) / (nv - 1);
    if direction < 0; pct = 1 - pct; end
    scores(valid) = 1 + pct * 9;
end

% -------------------------------------------------------------------------
function v = safeNum(T, name, cols, n)
    if any(strcmp(cols, name))
        col = T.(name);
        if iscell(col); v = cellfun(@(x) str2double(string(x)), col);
        elseif isstring(col) || ischar(col); v = str2double(col);
        else; v = double(col); end
        v = v(:);
    else
        fprintf('  [warn] column "%s" not found — NaN.\n', name);
        v = NaN(n, 1);
    end
end

% -------------------------------------------------------------------------
function c = safeStr(T, name, cols, n)
    if any(strcmp(cols, name))
        col = T.(name);
        if iscell(col); c = col;
        elseif isstring(col); c = cellstr(col);
        else; c = cellstr(string(col)); end
        c = c(:);
    else
        c = repmat({''}, n, 1);
    end
    for k = 1:n
        if isequal(c{k}, missing) || ...
           (ischar(c{k}) && strcmpi(strtrim(c{k}), 'NaN'))
            c{k} = '';
        end
    end
end

% -------------------------------------------------------------------------
function comp = taxComposition(b, t)
    tax = getTax(b, t);
    switch tax
        case {'C','G','F'}; comp = 'Primitive carbonaceous';
        case 'B';           comp = 'Primitive blue (possible organics/ice)';
        case 'D';           comp = 'Primitive dark (organic-rich)';
        case 'T';           comp = 'Primitive transitional';
        case 'P';           comp = 'Primitive dark (P-type)';
        case {'S','Q'};     comp = 'Silicaceous stony';
        case 'A';           comp = 'Olivine-dominated (mantle fragment?)';
        case 'V';           comp = 'Basaltic achondrite (differentiated crust)';
        case 'R';           comp = 'Pyroxene-olivine';
        case 'L';           comp = 'Spinel-bearing stony';
        case 'K';           comp = 'Eos-family stony (intermediate)';
        case 'M';           comp = 'Metallic or enstatite (ambiguous)';
        case 'X';           comp = 'X-complex (metallic / enstatite)';
        case 'E';           comp = 'Enstatite achondrite';
        otherwise;          comp = 'Unclassified — unknown composition';
    end
end
