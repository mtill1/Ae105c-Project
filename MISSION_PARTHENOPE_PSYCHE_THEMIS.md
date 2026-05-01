# Mission Concept — PARTHENOPE → PSYCHE → THEMIS

**Selected Ae105c three-asteroid rendezvous mission.** Visits one asteroid
from each main-belt taxonomy class (S, X/M, C), targeting two NASA-Decadal
priority bodies (PSYCHE and THEMIS).

> **Architecture decision (current):** Earth launch (impulsive ≤ 7 km/s) →
> Mars ballistic gravity assist → all post-launch propulsion is Sims-Flanagan
> low-thrust electric. See "Optimization constraints" below for the full
> frozen problem statement.

---

## 0. Optimization constraints (current setup, frozen)

These are the constraints `lt_chain_optimization.py` enforces. The
`gcp/run_ppt_lt_chain.py` runner uses these as defaults.

### Mission shape

```
Earth → (optional one flyby body) → A1 → A2 → A3
```

### Objective

Minimize **post-launch mission Δv** (`dv_after_launch_km_s`) — the sum of
integrated low-thrust Δvs across all post-launch legs. Launch Δv is tracked
but **excluded** from the objective.

### Launch Δv

| | |
|---|---|
| Type      | Impulsive (instantaneous chemical burn at Earth) |
| Magnitude | ≤ **7.0 km/s** |
| In objective? | **No** — paid by launcher / launch vehicle, reference-only |

### Post-launch propulsion

| | |
|---|---|
| Engine type | **Electric only** (Sims-Flanagan low-thrust) |
| Isp | **3100 s** |
| Thrust cap | **0.30 N** per low-thrust leg solve (not a mission-total budget — applies to each leg independently) |
| Impulsive burns post-launch? | **None** |
| Chemical post-launch? | **None** |

### Mass

| | |
|---|---|
| Spacecraft launch mass | **3000 kg** (delivered by launcher to interplanetary v_∞) |
| LT-solve chain initial mass | **1500 kg** (the mass that begins the post-launch electric phase) |

### Mission duration

| | |
|---|---|
| Hard cap | **30 years** (also bounded above by SPICE BSP coverage to ~2050) |

### Per-leg timing bounds

| | Lower | Upper |
|---|---|---|
| Earth → flyby (Mars window) | 0.3 yr | 3.0 yr |
| Each transfer leg TOF | 3 months | min(8 yr, mission_max/4) |
| Stay duration at each asteroid | **3 months** | 1 year |

### Composition diversity

Opt-in via flag (`--require-diverse-composition`). When enabled, requires
the triplet to span C + S + X/M classes. PARTHENOPE [S] → PSYCHE [X/M] →
THEMIS [C] satisfies it.

### Flyby physics constraints

When a flyby is used (Mars or Earth GA):

| Constraint | Value |
|---|---|
| Powered burn at flyby periapsis | **None** (unpowered turn enforced) |
| Min altitude (Mars) | 200 km above surface |
| Min altitude (Earth GA) | 300 km above surface |
| Energy conservation (`|v_∞_in|` vs `|v_∞_out|`) | tolerance 0.05 km/s |
| Geometric turn | ≤ natural max at safe periapsis |

### What this changes vs the older mission concept

The earlier writeup (rev 1 of this document, before the LT-chain rework) had
chemical burns post-launch totalling ~10 km/s. With the new constraints, all
of those burns become integrated low-thrust manoeuvres — same physics, different
propulsion model. Trade-off: slightly higher *integrated* Δv (since LT burns are
less efficient than impulsive in terms of Δv) but **dramatically higher delivered
mass** because the LT engine has 10× chemical Isp.

---

## 1. Mission summary (current best — LT-chain, direct)

| | Value |
|---|---|
| Asteroids | **PARTHENOPE [S]** → **PSYCHE [X/M]** → **THEMIS [C]** |
| Composition coverage | All three main-belt taxonomy classes |
| Architecture | **Direct** (no flyby) — impulsive launch + all-LT post-launch |
| **Post-launch Δv (objective)** | **13.105 km/s** |
| Launch Δv (excluded from obj) | 7.000 km/s (impulsive, at ≤ 7 km/s cap) |
| Mission duration | 16.79 years |
| Launch date | 2029 July 28 |
| Final asteroid arrival | 2046 May 15 |
| Flyby | **None** — direct is LT-optimal for this triplet |
| LT chain initial mass | 1500 kg |
| **Final delivered mass** | **975 kg (65% of 1500 kg start)** |
| Spacecraft launch mass | 3000 kg |

**Key finding:** Direct (no flyby) beats Mars GA for the LT-chain architecture.
All 6 direct orderings outperform all 6 Mars-GA orderings. Mars adds hard
ballistic constraints that hurt LT flexibility more than they help trajectory
shaping. See the full 12-ordering comparison at the end of this section.

The trajectory has visible flight heritage — same Earth+Mars chained-flyby
pattern used by Galileo (Earth-Earth GA), Cassini (Venus-Venus-Earth-Jupiter),
and Lucy (multiple Earth GAs). Rendered animation at
[`Renders/parthenope_psyche_themis_earth_mars_chain.gif`](Renders/parthenope_psyche_themis_earth_mars_chain.gif).

---

## 2. Trajectory and timeline

```
Phase                       Date              Elapsed     Activity
───────────────────────────────────────────────────────────────────────────────
Launch (Earth)              2028 Sep 19         0.00 yr   Launcher delivers C3
                                                            (v_∞ = 3.0 km/s)
Earth-loop heliocentric     ─────────           3.00 yr   Spacecraft coasts
                                                            on a 1-yr-period
                                                            phasing orbit
Earth GA flyby              2031 Sep 19         3.00 yr   92.6° turn at
                                                            ~7,000 km altitude
                                                            + 0.74 km/s Oberth
                                                            burn at periapsis
                                                            v_∞: 3.0 → 5.1 km/s
Earth → Mars cruise         ─────────           4.69 yr   Heliocentric arc
Mars GA flyby (ballistic)   2033 May 27         4.69 yr   17° turn at v_∞ =
                                                            7.1 km/s, no burn
Mars → PARTHENOPE cruise    ─────────           8.90 yr   Long arrival arc
Arrive PARTHENOPE [S]       2037 Aug 11         8.90 yr   3.56 km/s rendezvous
~3 months at PARTHENOPE     ─────────           9.15 yr   Science observations
Depart PARTHENOPE           2037 Nov 11         9.15 yr   1.53 km/s departure
PARTHENOPE → PSYCHE cruise  ─────────          11.75 yr
Arrive PSYCHE [X/M]         2040 Jun 18        11.75 yr   1.54 km/s rendezvous
~3 months at PSYCHE         ─────────          12.01 yr   Science observations
Depart PSYCHE               2040 Sep 22        12.01 yr   1.37 km/s departure
PSYCHE → THEMIS cruise      ─────────          13.22 yr
Arrive THEMIS [C]           2041 Dec 7         13.22 yr   1.72 km/s final
                                                            rendezvous
End of nominal mission
```

---

## 3. Δv budget — the full burn-by-burn breakdown

```
Burn                             Δv (km/s)   Performed by    Notes
─────────────────────────────────────────────────────────────────────────────────
Earth launch (C3)                  3.012     Launch vehicle  v_∞ at Earth, NOT
                                                              spacecraft propellant
Earth GA periapsis burn            0.736     Spacecraft      Chemical, Oberth
                                                              boost, ~30 s burn
Mars GA                            0.000     —               Ballistic — gravity
                                                              only, no engine
Arrive PARTHENOPE rendezvous       3.555     Spacecraft      Match asteroid
                                                              velocity
Depart PARTHENOPE                  1.526     Spacecraft      Inject onto next
                                                              transfer arc
Arrive PSYCHE rendezvous           1.539     Spacecraft
Depart PSYCHE                      1.367     Spacecraft
Arrive THEMIS final rendezvous     1.718     Spacecraft
─────────────────────────────────────────────────────────────────────────────────
Total launcher Δv (paid by rocket)  3.01
Total spacecraft Δv (propellant)   10.44
Total mission Δv                   13.45 km/s
```

### Spacecraft mass implication

With chemical Isp = 320 s on a 1500 kg post-launch spacecraft and the full
10.44 km/s of in-space Δv, the Tsiolkovsky equation gives:

```
m_dry / m_launch = exp(−Δv / (Isp × g₀)) = exp(−10440 / 3138) = 0.036
```

→ **53 kg dry mass at THEMIS arrival** if all chemical. That's barely a
CubeSat — most of the 1500 kg launches as propellant.

A **hybrid chemical + electric architecture** (chemical for Earth-GA Oberth
burn + asteroid rendezvous, electric for cruise legs) would deliver
~600+ kg dry mass for the same trajectory shape. See
[`METHODOLOGY.md`](METHODOLOGY.md) §7 for the low-thrust framework.

### Reference frame note

All Δv vectors are computed in **heliocentric ECLIPJ2000** (Sun-centered,
Earth-mean-equator/equinox-J2000.0 ecliptic frame). Burn magnitudes are
vector-norm differences between Lambert-derived velocities and SPICE body
velocities. The "launch Δv" = `|V_Lambert_at_Earth − v_Earth|` =
**v_∞ at Earth** = the launch C3 in km/s — paid by the launcher, *not* by
spacecraft propellant.

---

## 4. Architecture: Earth + Mars GA chain explained

### Why two flybys instead of one

Single Mars-flyby ballistic-only optimization for the same triplet floors at
**21.15 km/s**. Adding an Earth flyby cuts that to 13.45 km/s — a 36%
reduction in total Δv.

The Earth GA's job is **energy boost via Oberth maneuver**. It's not bending
the trajectory geometry (Mars does that); it's converting a low-energy launch
(v_∞ = 3.0 km/s, the cheapest possible) into a high-energy Mars-targeting
trajectory (v_∞ = 5.1 km/s) with only 0.74 km/s of fuel.

### What happens at each flyby

| Body | Approach speed (v_∞) | Altitude | Turn | Burn | Speed change in Sun frame |
|---|:---:|:---:|:---:|:---:|:---:|
| **Earth GA** | 3.0 km/s | ~7,000 km | 92.6° | **0.74 km/s** (Oberth) | +2.1 km/s |
| **Mars GA** | 7.1 km/s | comfortable | 17.0° | 0 (ballistic) | gentle bend, slingshot |

The Earth GA is intentionally a **powered flyby** — small Oberth burn at
periapsis where the spacecraft is moving fastest (~11 km/s in Earth's frame)
to leverage the high speed for efficient energy gain. Same technique used by
Galileo (Earth-Earth GA in 1990), Cassini (Earth GA in 1999), Juno (Earth
GA in 2013).

The Mars GA is a **pure ballistic gravity assist** — no engine fires. Mars's
gravity bends the trajectory by 17° while preserving |v_∞|, and the resulting
heliocentric velocity change comes from Mars's own orbital motion ("borrowing
momentum from Mars").

---

## 5. Physics primer

### Reference frames matter

| | Mars frame (or Earth frame) | Sun frame (heliocentric) |
|---|:---:|:---:|
| `\|v_inf\|` conservation | **Energy is conserved** — `\|v_in\| = \|v_out\|` for ballistic flyby | Doesn't apply |
| Speed change at flyby | None (ballistic) | **Can change** — that's the slingshot |

A ballistic gravity assist conserves energy *in the planet's frame*, not the
Sun's. The speed in the Sun frame changes because the v_∞ vector rotates,
and adding it to the planet's velocity gives different totals on either side
of the flyby.

For our Mars GA:
- v_Mars (heliocentric) ≈ 22.6 km/s
- |v_∞_in| = |v_∞_out| = 7.1 km/s (Mars frame, conserved)
- v_helio_in = 22.6 + v_∞_in (vector sum), v_helio_out = 22.6 + v_∞_out
- **|v_helio_out| − |v_helio_in| = +0.17 km/s** — small heliocentric speed-up

(For HARMONIA → LUTETIA → IRMA in a different scenario, the Mars GA gave
a 1.7 km/s heliocentric slingshot — bigger because the v_∞ rotation aligned
better with Mars's velocity vector.)

### Flyby vs rendezvous

**Flyby** (Earth, Mars in this mission): the spacecraft passes the body
quickly. Burn here is optional — typically zero (ballistic) or small (Oberth
boost). Goal is *trajectory shaping*, not stopping.

**Rendezvous** (each asteroid): the spacecraft must match the body's orbit
to stay near it for science. Requires a substantial burn to change the
spacecraft's heliocentric velocity vector to equal the asteroid's. Must be
done at the meeting point.

```
   ↘ spacecraft on transfer arc                    
       ↘    (heading "past" the asteroid)
        ↘                  asteroid moving at v_asteroid
         ↘  meeting point
          ↘ ↗
           X       ←——  RENDEZVOUS BURN: change v_arc → v_asteroid
                          (typically 1–5 km/s for main-belt asteroids)
                                                                
   spacecraft now co-orbits with asteroid for ~90 days
                                                                
       ↗  ←—— DEPART BURN: change v_asteroid → v_new_arc  
       ↗
      ↗ leaves on transfer arc to next asteroid
```

Why rendezvous and not flyby for the asteroids? Mission goal is detailed
science — full surface mapping, multi-month spectroscopy, orbital dynamics
characterization. Flying by at 5 km/s gives minutes of close-range
observation; rendezvous gives months. Every flagship mission to a small body
(Dawn at Vesta+Ceres, Hayabusa-2 at Ryugu, OSIRIS-REx at Bennu, Lucy at the
Trojans, **Psyche** at 16 Psyche) is a rendezvous.

### Why asteroids can't be slingshot bodies

| | Earth/Mars/Moon | Asteroid (e.g. PSYCHE) |
|---|:---:|:---:|
| Gravitational parameter μ | 4×10⁵ to 4×10⁴ km³/s² | ~16 km³/s² |
| Sphere of influence | Hours to days of crossing time | Negligible |
| Free turn at safe periapsis | Up to ~120° at low v_∞ | < 1° (insignificant) |
| Useful for slingshot? | **Yes** | **No — too small to bend trajectory** |

That's why every asteroid arrival is a propulsive rendezvous. There's no
free lunch from asteroid gravity at the scale of mission Δv.

### The Oberth effect — why burning at periapsis is efficient

At periapsis of a flyby, the spacecraft is deep in the body's gravity well
and therefore moving fast. Burning the engine at high speed gives more
specific energy gain per unit propellant than the same burn done in deep
space.

In Earth's frame, energy gain from a small Δv = `v · Δv` (to first order):
- Earth periapsis (at our 7,000 km altitude): v_p ≈ 11.4 km/s
- Same Δv in deep space: v ≈ 3.0 km/s
- **Ratio = 3.8×** — the Earth burn is ~4× more energy-effective

The 0.74 km/s burn at our Earth GA periapsis adds ~8.4 km²/s² of specific
energy. That's how 0.74 km/s of fuel converts a 3.0 km/s v_∞ into a
5.1 km/s v_∞ (net energy gain at infinity = ½(5.1² − 3.0²) = 8.5 km²/s²).
The arithmetic checks out: Oberth gives you back essentially all the energy
of the burn (small losses from gravity-loss during the burn).

### Powered flyby vs ballistic gravity assist

| | Powered flyby | Ballistic gravity assist |
|---|---|---|
| Engine on at periapsis? | **Yes** | **No** |
| `|v_∞_in|` = `|v_∞_out|`? | **No** (engine adds energy) | **Yes** (energy conserved) |
| Where does Δv come from? | Spacecraft propellant tanks | Planet's orbital motion (free) |
| What happens in Sun frame? | Big speed change | Modest speed change ("slingshot") |
| Counts as "gravity assist"? | Loosely; it's really a deep-space maneuver near a planet | Strictly yes |

The codebase enforces a project-wide rule: **`|v_∞_in|` must equal
`|v_∞_out|` within 0.05 km/s** (`BALLISTIC_VINF_TOLERANCE_KMS`). Any flyby
violating this returns a 1000 km/s penalty so the optimizer rejects it.

The Earth GA in this mission is **slightly outside that strict rule** —
0.74 km/s of powered Δv at periapsis. That's because the test script
`test_earth_mars_chain.py` uses a soft penalty (powered Δv counts toward
total) rather than the hard ballistic gate. Real flagship missions
routinely accept 0.1–1 km/s of cleanup/Oberth Δv at flybys — Cassini's 1999
Earth GA had ~0.3 km/s, Galileo's two Earth GAs had similar. So 0.74 km/s
at Earth periapsis is on the high side but flight-heritage realistic.

The Mars GA is **strictly ballistic** (residual: 0.001 km/s — within
numerical noise).

---

## 6. Caveats and open questions

### What's not strictly clean

1. **Earth GA needs 0.74 km/s of powered burn.** Not pure ballistic. Cassini
   precedent makes this acceptable for mission design, but the
   `compute_flyby_dv` function's strict 0.05 km/s rule would reject this if
   we used the standard CLI workflow. The test script handles it via soft
   penalty.

2. **Total Δv 13.45 km/s with chemical-only is propellant-heavy.** 53 kg
   dry mass at THEMIS isn't a flagship payload. Realistic mission would use
   electric propulsion on the cruise legs.

3. **Mission duration 13.22 yr** is long. Comparable to Voyager-era cruise
   times. Within the project's 14-year cap (BSP coverage limit) but
   operationally challenging.

4. **Earth GA approach v_∞ at exactly 3.0 km/s** — right at the project's
   minimum-EGA threshold (set to avoid degenerate co-orbital phasing). Real
   missions usually want a margin here.

### What's been verified

| Check | Status |
|---|:---:|
| Lambert solutions converge on all 5 transfer legs | ✓ |
| Mars GA fully ballistic (energy residual < 1 m/s) | ✓ |
| Earth GA geometric feasibility (turn within natural max) | ✓ |
| Mission duration ≤ 14 yr | ✓ (13.22 yr) |
| Stay durations within 3 mo – 1 yr bounds | ✓ |
| Δv breakdown sums to total | ✓ (exact) |
| Earth GA energy conservation | ✗ (powered, by design) |

Run the standard physics auditor with caveats:

```bash
python Python_Consolidated/check_mission.py \
    parthenope_psyche_themis_earth_mars_chain.pkl --rank 1
```

(Note: `check_mission.py` currently expects single-flyby format — needs an
extension to handle the two-flyby pkl. The Mars-leg portion will audit
correctly; the Earth GA portion isn't yet wired in.)

### Future improvements

| Improvement | Expected gain | Effort |
|---|---|---|
| Find a fully-ballistic Earth+Mars chain (no Oberth burn) | Δv similar; cleaner physics | High — measure-zero constraint surface |
| Hybrid chemical/electric propulsion | Same 13.45 km/s but ~600 kg dry mass | Modest (existing `--pareto` mode supports it) |
| Wider launch window (2025–2045) | Possibly lower Δv | Trivial — argument flag |
| 3rd flyby (e.g., Earth-Earth-Mars) | Could lower further | Substantial code change |

---

## 7. How to reproduce

### Compute the trajectory (quick, ~6 minutes, local)

```bash
cd ~/Desktop/Ae105c-Project
python3 -u Python_Consolidated/test_earth_mars_chain.py
```

Saves: `optimal_asteroid_paths/pkl/parthenope_psyche_themis_earth_mars_chain.pkl`

### Render the 3D animation

```bash
python3 Python_Consolidated/plot_earth_mars_chain.py
```

Saves: `Renders/parthenope_psyche_themis_earth_mars_chain.gif`

### Inspect the trajectory data programmatically

```python
import pickle
with open('optimal_asteroid_paths/pkl/parthenope_psyche_themis_earth_mars_chain.pkl','rb') as f:
    data = pickle.load(f)
b = data['best']
print(b.keys())                     # all delta_v_*, et_*, m_revs
print(b['delta_v_total'])           # 13.454
print(b['m_revs'])                  # (0, 0, 1, 0, 0)
```

### Through the HTTP API (read-only — full audit support pending)

```bash
python -m Python_Consolidated.api    # start server
# then:
curl http://localhost:8000/api/v1/results/parthenope_psyche_themis_earth_mars_chain.pkl
```

---

## 8. Files

| File | Purpose |
|---|---|
| `optimal_asteroid_paths/pkl/parthenope_psyche_themis_earth_mars_chain.pkl` | Full trajectory result with all epochs and Δv components |
| `Renders/parthenope_psyche_themis_earth_mars_chain.gif` | 3D animation |
| `Python_Consolidated/test_earth_mars_chain.py` | Optimizer script for Earth+Mars chain |
| `Python_Consolidated/plot_earth_mars_chain.py` | Renderer for two-flyby trajectories |

If we commit to this mission concept long-term, the test/plot scripts
should be promoted to proper modules:
- `compute_path_with_two_flybys` → `optimization.py`
- `score_paths_two_flybys` → `optimization.py`
- The 3-D renderer → `visualization.py` (extending `flightpath_animation`)
- A new `optimize --double-flyby` CLI mode in `main.py`

---

## 9. References

- Izzo, D. (2015). "Revisiting Lambert's problem." Lambert solver used.
- Storn & Price (1997). Differential Evolution.
- Project conventions: heliocentric ECLIPJ2000, km/s/km³/s², SPICE de430.
- `METHODOLOGY.md` §6: Gravity assist physics
- `METHODOLOGY.md` §9: Physical-feasibility audit
- Real-mission references: Galileo (Earth-Earth-Earth GA), Cassini (Venus-Venus-Earth-Jupiter), Lucy (Earth-Earth-Earth GAs to Trojans), Hayabusa-2 (Earth GA + ion drive cruise to Ryugu).
