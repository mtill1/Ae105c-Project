"""Create presentation-ready plots for the exported low-thrust mission data."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def add_event_lines(ax, events_days):
    for label, t_day in events_days:
        ax.axvline(t_day, linestyle="--", linewidth=0.9, alpha=0.7, color="gray")
        ax.text(
            t_day,
            0.98,
            label,
            rotation=90,
            va="top",
            ha="right",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="gray",
        )


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(repo, "optimal_asteroid_paths", "csv")
    out_dir = os.path.join(repo, "optimal_asteroid_paths", "plots")
    os.makedirs(out_dir, exist_ok=True)

    thrust_path = os.path.join(csv_dir, "aegina_beatrix_vesta_low_thrust_profile.csv")
    states_path = os.path.join(csv_dir, "aegina_beatrix_vesta_multibody_states.csv")
    summary_path = os.path.join(csv_dir, "aegina_beatrix_vesta_low_thrust_summary.csv")

    thrust = pd.read_csv(thrust_path)
    states = pd.read_csv(states_path)
    summary = pd.read_csv(summary_path).iloc[0]

    t0 = float(thrust["et_seconds"].iloc[0])
    thrust["t_days"] = (thrust["et_seconds"] - t0) / 86400.0
    states["t_days"] = (states["et_seconds"] - t0) / 86400.0

    # Mission event markers relative to low-thrust start (Aegina departure)
    event_utc = [
        ("Mars Flyby", summary["mars_flyby_utc"]),
        ("Aegina Arrive", summary["aegina_arrive_utc"]),
        ("Aegina Depart", summary["aegina_depart_utc"]),
        ("Beatrix Arrive", summary["beatrix_arrive_utc"]),
        ("Beatrix Depart", summary["beatrix_depart_utc"]),
        ("Vesta Arrive", summary["vesta_arrive_utc"]),
    ]
    t0_utc = pd.to_datetime(thrust["utc"].iloc[0], format="%Y %b %d %H:%M:%S.%f")
    events_days = []
    for label, utc in event_utc:
        event_ts = pd.to_datetime(str(utc), format="%Y %b %d %H:%M:%S")
        events_days.append((label, (event_ts - t0_utc).total_seconds() / 86400.0))

    # 1) Thrust magnitude vs time
    plt.figure(figsize=(11, 5))
    plt.plot(thrust["t_days"], thrust["thrust_mag_N"], linewidth=1.4)
    plt.xlabel("Mission Time Since LT Start [days]")
    plt.ylabel("Thrust Magnitude [N]")
    plt.title("Low-Thrust Magnitude vs Time (Aegina -> Beatrix -> Vesta)")
    plt.grid(True, alpha=0.35)
    add_event_lines(plt.gca(), events_days)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "thrust_magnitude_vs_time.png"), dpi=220)
    plt.close()

    # 2) Thrust direction components vs time
    plt.figure(figsize=(11, 5))
    plt.plot(thrust["t_days"], thrust["dir_x"], label="dir_x", linewidth=1.2)
    plt.plot(thrust["t_days"], thrust["dir_y"], label="dir_y", linewidth=1.2)
    plt.plot(thrust["t_days"], thrust["dir_z"], label="dir_z", linewidth=1.2)
    plt.xlabel("Mission Time Since LT Start [days]")
    plt.ylabel("Thrust Unit-Vector Component [-]")
    plt.title("Low-Thrust Direction Components vs Time")
    plt.grid(True, alpha=0.35)
    plt.legend()
    add_event_lines(plt.gca(), events_days)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "thrust_direction_components_vs_time.png"), dpi=220)
    plt.close()

    # 3) Spacecraft mass vs time
    plt.figure(figsize=(11, 5))
    plt.plot(states["t_days"], states["mass_kg"], linewidth=1.4)
    plt.xlabel("Mission Time Since LT Start [days]")
    plt.ylabel("Spacecraft Mass [kg]")
    plt.title("Spacecraft Mass Depletion During Low-Thrust Legs")
    plt.grid(True, alpha=0.35)
    add_event_lines(plt.gca(), events_days)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spacecraft_mass_vs_time.png"), dpi=220)
    plt.close()

    # 4) Heliocentric distance vs time
    r_km = (states["x_km"] ** 2 + states["y_km"] ** 2 + states["z_km"] ** 2) ** 0.5
    r_au = r_km / 149597870.7
    plt.figure(figsize=(11, 5))
    plt.plot(states["t_days"], r_au, linewidth=1.4)
    plt.xlabel("Mission Time Since LT Start [days]")
    plt.ylabel("Heliocentric Distance [AU]")
    plt.title("Heliocentric Distance During Multibody Propagation")
    plt.grid(True, alpha=0.35)
    add_event_lines(plt.gca(), events_days)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heliocentric_distance_vs_time.png"), dpi=220)
    plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
