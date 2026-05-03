#!/usr/bin/env python
"""
This script is used to analyse the ROMR stability test. Each trial, computes:
  - Mean commanded vs measured linear velocity (steady-state portion only)
  - Mean commanded vs measured angular velocity
  - Tracking error
  - Max IMU roll/pitch during the trial
  - Actual turning radius (computed from measured v/omega)

Usage:
  python3 analyze_stability.py romr_stability_20260422_143000.csv
  python3 analyze_stability.py romr_stability_*.csv --plot
"""

import argparse
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze(path, plot=False):
    df = pd.read_csv(path)
    ss = df[df["in_settle"] == 0].copy()

    # Per-trial summary
    grp = ss.groupby("trial_idx")
    summary = grp.agg(
        cmd_radius=("cmd_radius", "first"),
        cmd_lin=("cmd_lin", "first"),
        cmd_ang=("cmd_ang", "first"),
        payload_kg=("payload_kg", "first"),
        cog_m=("cog_m", "first"),
        meas_lin_mean=("odom_vx", "mean"),
        meas_lin_std=("odom_vx", "std"),
        meas_ang_mean=("odom_wz", "mean"),
        meas_ang_std=("odom_wz", "std"),
        wheel_L=("wheel_left", "mean"),
        wheel_R=("wheel_right", "mean"),
    ).reset_index()

    summary["lin_err"] = summary["meas_lin_mean"] - summary["cmd_lin"]
    summary["ang_err"] = summary["meas_ang_mean"] - summary["cmd_ang"]
    # Actual radius from measured v/w.
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["meas_radius"] = np.where(
            np.abs(summary["meas_ang_mean"]) > 1e-3,
            summary["meas_lin_mean"] / summary["meas_ang_mean"],
            np.inf,
        )

    # IMU stats
    if "imu_roll" in ss.columns and ss["imu_roll"].notna().any():
        imu = grp.agg(
            max_roll=("imu_roll", lambda s: s.abs().max()),
            max_pitch=("imu_pitch", lambda s: s.abs().max()),
        ).reset_index()
        summary = summary.merge(imu, on="trial_idx")

    print(summary.to_string(index=False, float_format="%.3f"))

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(summary["trial_idx"], summary["cmd_lin"],
                   "o-", label="commanded")
        ax[0].plot(summary["trial_idx"], summary["meas_lin_mean"],
                   "s-", label="measured")
        ax[0].fill_between(summary["trial_idx"],
                           summary["meas_lin_mean"] - summary["meas_lin_std"],
                           summary["meas_lin_mean"] + summary["meas_lin_std"],
                           alpha=0.2)
        ax[0].set_ylabel("linear vel (m/s)")
        ax[0].legend(); ax[0].grid(True)

        ax[1].plot(summary["trial_idx"], summary["cmd_ang"],
                   "o-", label="commanded")
        ax[1].plot(summary["trial_idx"], summary["meas_ang_mean"],
                   "s-", label="measured")
        ax[1].set_ylabel("angular vel (rad/s)")
        ax[1].set_xlabel("trial index")
        ax[1].legend(); ax[1].grid(True)
        plt.tight_layout()
        plt.savefig(path.replace(".csv", "_velocities.png"), dpi=120)
        print("Saved plot to", path.replace(".csv", "_velocities.png"))
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    analyze(args.csv, plot=args.plot)


if __name__ == "__main__":
    main()
