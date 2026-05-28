#!/usr/bin/env python3
"""
Build pseudo-population trajectory JSON from Mante 2013 .mat files.

Loads a directory of single-unit PFC recordings, condition-averages within
each unit, stacks across units, runs PCA, and emits a JSON of precomputed
trajectories ready for in-browser visualization.

Usage:
    python build_trajectory_json.py <input_dir> <output_json> [options]

Examples:
    # Default: signed motion coherence in motion context, correct trials only
    python build_trajectory_json.py PFC_data/ trajectories.json --sigma-ms 5

    # Use choice x context grouping (works with fewer trials per unit)
    python build_trajectory_json.py PFC_data/ trajectories.json --grouping choice_context

    # Custom smoothing and number of PCs
    python build_trajectory_json.py PFC_data/ trajectories.json --sigma-ms 50 --n-pcs 4

JSON output structure:
    {
        "metadata": { n_units, time_ms, var_explained, grouping, ... },
        "conditions": [
            { label, color, pc1[...], pc2[...], dpc1_dt[...], ddpc1_dt2[...], ... },
            ...
        ]
    }
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------------------------
# Grouping schemes
# --------------------------------------------------------------------------
# Each scheme returns a list of (label, color, mask_fn) tuples. mask_fn takes
# the unit's task_variable struct and returns a boolean trial mask.
#
# To add a new grouping scheme, define another function below and register it
# in GROUPING_SCHEMES at the bottom of this section.

MOTION_COHERENCES = [-0.5, -0.14, -0.04, 0.04, 0.14, 0.5]


def _coh_color(coh, max_coh=0.5):
    """Red for positive coherence, blue for negative, intensity = strength."""
    intensity = 0.4 + 0.6 * (abs(coh) / max_coh)
    if coh > 0:
        return f"rgba(220, 20, 60, {intensity:.3f})"  # crimson
    else:
        return f"rgba(70, 130, 180, {intensity:.3f})"  # steelblue


def grouping_motion_coh(tv):
    """6 signed motion coherence levels, motion context, correct trials only.

    This is the standard Mante Figure 2 grouping. Reveals the choice axis and
    rotational dynamics most clearly. Requires enough trials per coherence
    level per unit.
    """
    conditions = []
    for coh in MOTION_COHERENCES:
        label = f"coh = {coh:+.2f}"
        color = _coh_color(coh)
        mask_fn = lambda tv, c=coh: (
            np.isclose(tv.stim_dir, c) & (tv.context == -1) & (tv.correct == 1)
        )
        conditions.append((label, color, mask_fn, {"coherence": float(coh)}))
    return conditions


def grouping_motion_coh_both_contexts(tv):
    """12 conditions: 6 motion coh x 2 context, correct trials only.

    Useful for comparing how motion is represented when relevant (motion
    context) vs irrelevant (color context).
    """
    conditions = []
    for ctx, ctx_label in [(-1, "motCtx"), (1, "colCtx")]:
        for coh in MOTION_COHERENCES:
            label = f"coh={coh:+.2f} ({ctx_label})"
            base = _coh_color(coh)
            # dim the irrelevant-context trajectories visually
            color = base if ctx == -1 else base.replace("0.", "0.")
            mask_fn = lambda tv, c=coh, x=ctx: (
                np.isclose(tv.stim_dir, c) & (tv.context == x) & (tv.correct == 1)
            )
            conditions.append((label, color, mask_fn,
                               {"coherence": float(coh), "context": int(ctx)}))
    return conditions


def grouping_choice_context(tv):
    """4 conditions: 2 choices x 2 contexts. Survives with fewer trials/unit.

    Coarser but more units typically have all 4 conditions filled. Use this
    when working with a small number of units or short recordings.
    """
    conditions = []
    palette = {
        (-1, -1): ("#1f77b4", "choice-, motion ctx"),
        (-1, +1): ("#aec7e8", "choice-, color ctx"),
        (+1, -1): ("#d62728", "choice+, motion ctx"),
        (+1, +1): ("#ff9896", "choice+, color ctx"),
    }
    for (choice, ctx), (color, label) in palette.items():
        # In motion context, choice tracks targ_dir; in color, targ_col.
        def mask_fn(tv, ch=choice, x=ctx):
            choice_var = tv.targ_dir if x == -1 else tv.targ_col
            return (choice_var == ch) & (tv.context == x)
        conditions.append((label, color, mask_fn,
                           {"choice": int(choice), "context": int(ctx)}))
    return conditions


GROUPING_SCHEMES = {
    "motion_coh": grouping_motion_coh,
    "motion_coh_both_contexts": grouping_motion_coh_both_contexts,
    "choice_context": grouping_choice_context,
}


# --------------------------------------------------------------------------
# Loading and population building
# --------------------------------------------------------------------------

def find_mat_files(input_dir):
    """Recursively find all .mat files under input_dir."""
    path = Path(input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(path.rglob("*.mat"))
    if not files:
        raise RuntimeError(f"No .mat files found under {input_dir}")
    return [str(f) for f in files]


def load_unit_psths(filepath, conditions, sigma_ms, min_trials=3):
    """Load one unit and return condition-averaged PSTHs.

    Returns:
        psths: (n_conditions, n_time) array, with NaN rows for conditions
               that had fewer than min_trials trials.
        t: time array in seconds
        trial_counts: list of n trials per condition
        meta: dict with unit name, monkey, etc.
    """
    try:
        d = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"  ! Failed to load {filepath}: {e}", file=sys.stderr)
        return None

    u = d["unit"]
    resp = u.response.astype(np.float32)  # (n_trials, n_time), binary spikes
    t = np.asarray(u.time).flatten()
    tv = u.task_variable

    psths = np.full((len(conditions), len(t)), np.nan, dtype=np.float32)
    trial_counts = []
    for i, (_, _, mask_fn, _) in enumerate(conditions):
        try:
            mask = mask_fn(tv)
        except AttributeError:
            # unit is missing a task_variable field this grouping needs
            trial_counts.append(0)
            continue
        n = int(mask.sum())
        trial_counts.append(n)
        if n >= min_trials:
            mean_rate = resp[mask].mean(axis=0) * 1000.0  # Hz
            psths[i] = gaussian_filter1d(mean_rate, sigma=sigma_ms)

    name = str(u.name) if hasattr(u, "name") else os.path.basename(filepath)
    monkey = os.path.basename(filepath)[:2]
    meta = {
        "name": name,
        "monkey": monkey,
        "filepath": filepath,
        "n_total_trials": int(resp.shape[0]),
    }
    return psths, t, trial_counts, meta


def build_population(mat_files, conditions, sigma_ms, min_trials):
    """Load all units, return population tensor and metadata."""
    psths_list = []
    metas = []
    counts_per_unit = []

    for fp in mat_files:
        result = load_unit_psths(fp, conditions, sigma_ms, min_trials)
        if result is None:
            continue
        psths, t, counts, meta = result
        psths_list.append(psths)
        metas.append(meta)
        counts_per_unit.append(counts)

    pop = np.stack(psths_list, axis=0)  # (n_units, n_conditions, n_time)
    counts_array = np.array(counts_per_unit)

    # Drop units that have any condition completely missing
    has_all = ~np.any(np.isnan(pop).all(axis=2), axis=1)
    pop_kept = pop[has_all]
    metas_kept = [m for m, k in zip(metas, has_all) if k]
    print(f"  Loaded {len(metas)} units; kept {len(metas_kept)} with all "
          f"{len(conditions)} conditions filled (min_trials={min_trials})")

    return pop_kept, t, metas_kept, counts_array, has_all


# --------------------------------------------------------------------------
# Dimensionality reduction
# --------------------------------------------------------------------------

def run_pca(pop, n_pcs, demix=True):
    """PCA on the pseudo-population.

    Inputs:
        pop: (n_units, n_conditions, n_time) array of condition-averaged rates
        n_pcs: number of components to keep
        demix: if True, subtract the cross-condition mean at each (unit, time)
            before PCA. This removes the time-varying signal shared across
            conditions (the universal stimulus-evoked response) and lets PCA
            find the directions of condition-specific variation. Recommended
            for any dataset with enough units that the shared signal dominates
            raw PCA. Set False to reproduce vanilla PCA behaviour.

    Returns:
        trajectories: (n_conditions, n_time, n_pcs)
        var_explained: (n_total_pcs,) percent variance
    """
    # z-score each unit so high-firing units don't dominate
    means = pop.mean(axis=(1, 2), keepdims=True)
    stds = pop.std(axis=(1, 2), keepdims=True)
    pop_z = (pop - means) / (stds + 1e-6)

    if demix:
        # Cross-condition mean per (unit, time). Subtracting this removes the
        # signal that's shared across conditions, so the residual carries only
        # the condition-discriminating variation.
        cross_cond_mean = pop_z.mean(axis=1, keepdims=True)  # (n_units, 1, n_time)
        pop_z = pop_z - cross_cond_mean

    n_units, n_cond, n_time = pop_z.shape
    # X: (n_cond * n_time, n_units), each row is a population state vector
    X = pop_z.transpose(1, 2, 0).reshape(-1, n_units)
    X_centered = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_exp = (S ** 2 / (S ** 2).sum() * 100.0)

    k = min(n_pcs, Vt.shape[0])
    scores = X_centered @ Vt[:k].T
    trajectories = scores.reshape(n_cond, n_time, k)
    return trajectories, var_exp


# --------------------------------------------------------------------------
# Phase portrait derivatives
# --------------------------------------------------------------------------

def compute_derivatives(trajectories, dt, trim=60):
    """First and second derivatives along time, with edge trimming.

    Returns trimmed copies so the website doesn't have to handle NaN edges.
    """
    # trajectories: (n_cond, n_time, n_pcs)
    dx = np.gradient(trajectories, dt, axis=1)
    ddx = np.gradient(dx, dt, axis=1)

    if trim > 0:
        sl = slice(trim, -trim)
        x_trim = trajectories[:, sl, :]
        dx_trim = dx[:, sl, :]
        ddx_trim = ddx[:, sl, :]
    else:
        x_trim = trajectories
        dx_trim = dx
        ddx_trim = ddx
    return dx, ddx, x_trim, dx_trim, ddx_trim, trim


# --------------------------------------------------------------------------
# JSON serialization
# --------------------------------------------------------------------------

def to_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def build_output_dict(trajectories, var_exp, t, conditions, metas, args,
                      pop_shape, trim):
    """Assemble the final JSON-ready dictionary."""
    n_cond, n_time, n_pcs = trajectories.shape
    dt = float(t[1] - t[0])
    dx, ddx, x_trim, dx_trim, ddx_trim, _ = compute_derivatives(
        trajectories, dt, trim=trim
    )

    t_ms = (t * 1000.0).round(2)
    t_trim_ms = t_ms[trim:-trim] if trim > 0 else t_ms

    cond_blocks = []
    for i, (label, color, _, extra) in enumerate(conditions):
        block = {
            "index": i,
            "label": label,
            "color": color,
        }
        block.update(extra)
        # Full-length series
        for p in range(n_pcs):
            block[f"pc{p+1}"] = trajectories[i, :, p].round(4).tolist()
            block[f"dpc{p+1}_dt"] = dx[i, :, p].round(4).tolist()
        # Trimmed second-derivative series for the dx/dt vs d2x/dt2 plot
        for p in range(n_pcs):
            block[f"pc{p+1}_trim"] = x_trim[i, :, p].round(4).tolist()
            block[f"dpc{p+1}_dt_trim"] = dx_trim[i, :, p].round(4).tolist()
            block[f"ddpc{p+1}_dt2_trim"] = ddx_trim[i, :, p].round(4).tolist()
        cond_blocks.append(block)

    return {
        "metadata": {
            "reference": "Mante, Sussillo, Shenoy, Newsome 2013, Nature",
            "grouping": args.grouping,
            "n_units_loaded": int(pop_shape[0]),
            "n_conditions": int(n_cond),
            "n_time_bins": int(n_time),
            "n_pcs_stored": int(n_pcs),
            "smoothing_sigma_ms": float(args.sigma_ms),
            "demixed": bool(not args.no_demix),
            "min_trials_per_condition": int(args.min_trials),
            "trim_bins_for_derivatives": int(trim),
            "time_ms": t_ms.tolist(),
            "time_ms_trim": t_trim_ms.tolist(),
            "var_explained_pct": var_exp.round(3).tolist(),
            "var_explained_top3_pct": float(var_exp[:3].sum().round(2)),
            "monkeys": sorted(set(m["monkey"] for m in metas)),
            "unit_names": [m["name"] for m in metas],
        },
        "conditions": cond_blocks,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build pseudo-population trajectory JSON from Mante 2013 .mat files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input_dir", help="Directory containing .mat files (recursive)")
    p.add_argument("output_json", help="Path for output JSON file")
    p.add_argument(
        "--grouping",
        default="motion_coh",
        choices=sorted(GROUPING_SCHEMES.keys()),
        help="How to group trials into conditions (default: motion_coh)",
    )
    p.add_argument("--sigma-ms", type=float, default=20.0,
                   help="Gaussian smoothing kernel sigma in ms (default: 20). "
                        "Lower values preserve higher-frequency content in the "
                        "trajectories; raise to e.g. 40 for very smooth signals.")
    p.add_argument("--n-pcs", type=int, default=8,
                   help="Number of PCs to keep in output (default: 8). Higher "
                        "PCs carry progressively higher-frequency oscillatory "
                        "content, so storing several lets you sum them for a "
                        "Fourier-style reconstruction downstream.")
    p.add_argument("--min-trials", type=int, default=3,
                   help="Minimum trials per (unit, condition) (default: 3)")
    p.add_argument("--no-demix", action="store_true",
                   help="Disable cross-condition mean subtraction before PCA. "
                        "By default, the universal stimulus-evoked response is "
                        "subtracted so PC1 captures the choice axis rather "
                        "than the shared signal. Pass this flag for vanilla "
                        "PCA (useful only if you want to see the dominant "
                        "shared signal explicitly).")
    p.add_argument("--trim-bins", type=int, default=60,
                   help="Time bins to trim from each end for derivatives "
                        "(reduces gradient edge noise; default: 60)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Searching for .mat files under: {args.input_dir}")
    mat_files = find_mat_files(args.input_dir)
    print(f"  Found {len(mat_files)} file(s)")

    print(f"\nGrouping scheme: {args.grouping}")
    conditions = GROUPING_SCHEMES[args.grouping](None)
    print(f"  {len(conditions)} conditions defined")

    print(f"\nLoading and condition-averaging units "
          f"(sigma={args.sigma_ms}ms, min_trials={args.min_trials})...")
    pop, t, metas, counts, _ = build_population(
        mat_files, conditions, args.sigma_ms, args.min_trials
    )

    if pop.shape[0] < 2:
        print("ERROR: fewer than 2 units survived. Try --grouping choice_context "
              "or lower --min-trials.", file=sys.stderr)
        return 1

    print(f"\nRunning PCA on population: {pop.shape}")
    demix = not args.no_demix
    print(f"  Cross-condition demixing: {'on' if demix else 'off'}")
    trajectories, var_exp = run_pca(pop, args.n_pcs, demix=demix)
    print(f"  Variance explained (top 5): "
          f"{var_exp[:5].round(1).tolist()} %")
    print(f"  Top 3 PCs capture {var_exp[:3].sum():.1f} % of variance")

    print(f"\nBuilding output JSON...")
    output = build_output_dict(
        trajectories, var_exp, t, conditions, metas, args,
        pop.shape, args.trim_bins
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(to_serializable(output), f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024.0
    print(f"\nWrote {out_path} ({size_kb:.1f} KB)")
    print(f"  {output['metadata']['n_units_loaded']} units, "
          f"{output['metadata']['n_conditions']} conditions, "
          f"{output['metadata']['n_time_bins']} time bins, "
          f"{output['metadata']['n_pcs_stored']} PCs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
