"""Generate all paper figures from real results JSON files. No synthetic
or illustrative data; every figure is a direct plot of results already
saved under results/."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300,
})

COLORS = {
    "Quantum-Enhanced": "#7b3294", "Quantum-TreeEnsemble": "#008837",
    "Uncertainty Sampling": "#1f78b4", "Query by Committee": "#e66101",
    "Random Sampling": "#999999", "best_baseline": "#1f78b4",
}


def fig1_learning_curves():
    with open(os.path.join(RESULTS, "primary_benchmark.json")) as f:
        primary = json.load(f)
    task_data = primary["tasks"]["band_gap"]["methods"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name, style in [("Quantum-Enhanced", "-"), ("Uncertainty Sampling", "--"),
                         ("Query by Committee", "--"), ("Random Sampling", ":")]:
        trials = task_data[name]["trials"]
        curves = np.array([t["r2"] for t in trials if not t["error"]])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        iters = trials[0]["n_labeled"]
        color = COLORS.get(name, "gray")
        ax.plot(iters, mean, style, label=name, color=color, linewidth=1.6)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Labeled samples")
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title("Band gap: learning curves")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_learning_curves.pdf"))
    plt.close(fig)
    print("saved fig1_learning_curves.pdf")


def fig2_primary_comparison():
    with open(os.path.join(RESULTS, "primary_benchmark_summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(RESULTS, "v3_all_tasks.json")) as f:
        v3 = json.load(f)

    tasks = ["band_gap", "formation_energy", "bulk_modulus", "magnetic_moment", "dielectric_constant"]
    task_labels = ["Band gap", "Formation\nenergy", "Bulk\nmodulus", "Magnetic\nmoment", "Dielectric\nconst."]

    orig = [summary[t]["Quantum-Enhanced"]["final_r2_mean"] for t in tasks]
    orig_err = [summary[t]["Quantum-Enhanced"]["final_r2_std"] for t in tasks]
    best = [v3[t]["comparison_vs_best_baseline"]["best_baseline_r2"] for t in tasks]
    v3_full = [v3[t]["v3_full"]["final_r2_mean"] for t in tasks]
    v3_err = [v3[t]["v3_full"]["final_r2_std"] for t in tasks]

    # Best classical baseline has no std reported here (single number per
    # task from the primary summary's argmax); leave it as a point marker
    # with no error bar rather than fabricate one.
    x = np.arange(len(tasks))
    off = 0.14
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.errorbar(x - off, orig, yerr=orig_err, fmt="o", color="#7b3294", label="Quantum (original)",
                capsize=3, markersize=5, linewidth=1.2, elinewidth=1.0)
    ax.errorbar(x, best, fmt="s", color="#1f78b4", label="Best classical baseline",
                markersize=5, linewidth=1.2)
    ax.errorbar(x + off, v3_full, yerr=v3_err, fmt="^", color="#008837", label="Quantum-TreeEnsemble",
                capsize=3, markersize=5, linewidth=1.2, elinewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylabel(r"Final test $R^2$ (mean $\pm$ s.d.)")
    ax.set_title("Primary comparison across all five real tasks")
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_primary_comparison.pdf"))
    plt.close(fig)
    print("saved fig2_primary_comparison.pdf")


def fig3_ablation():
    with open(os.path.join(RESULTS, "ablation.json")) as f:
        abl = json.load(f)
    with open(os.path.join(RESULTS, "v3_all_tasks.json")) as f:
        v3 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

    variants = ["full_model", "no_covariance", "commuting_only", "real_only_coeff", "single_observable_K1"]
    vlabels = ["Full", "No\ncov.", "Comm.\nonly", "Real-only\ncoeff.", "Single\nobs."]
    means = [abl["variants"][v]["final_r2_mean"] for v in variants]
    stds = [abl["variants"][v]["final_r2_std"] for v in variants]
    xv = np.arange(len(variants))
    axes[0].errorbar(xv, means, yerr=stds, fmt="o", color="#7b3294", capsize=3,
                      markersize=5, linewidth=1.2, elinewidth=1.0)
    axes[0].set_xticks(xv)
    axes[0].set_xticklabels(vlabels, fontsize=6.5)
    axes[0].set_xlim(-0.5, len(variants) - 0.5)
    axes[0].set_ylabel(r"Final $R^2$ (band gap, mean $\pm$ s.d.)")
    axes[0].set_title("(a) Original formalism ablation", fontsize=8.5)
    axes[0].spines[["top", "right"]].set_visible(False)

    tasks = ["band_gap", "formation_energy", "bulk_modulus", "magnetic_moment", "dielectric_constant"]
    tlabels = ["Band\ngap", "Form.\nenergy", "Bulk\nmod.", "Mag.\nmom.", "Dielec."]
    full = [v3[t]["v3_full"]["final_r2_mean"] for t in tasks]
    full_err = [v3[t]["v3_full"]["final_r2_std"] for t in tasks]
    nocov = [v3[t]["v3_no_covariance"]["final_r2_mean"] for t in tasks]
    nocov_err = [v3[t]["v3_no_covariance"]["final_r2_std"] for t in tasks]
    single = [v3[t]["v3_single_group"]["final_r2_mean"] for t in tasks]
    single_err = [v3[t]["v3_single_group"]["final_r2_std"] for t in tasks]
    x = np.arange(len(tasks))
    off = 0.16
    axes[1].errorbar(x - off, full, yerr=full_err, fmt="o", color="#7b3294", label="Full (Cov+K=3)",
                      capsize=2.5, markersize=4.5, linewidth=1.0, elinewidth=0.9)
    axes[1].errorbar(x, nocov, yerr=nocov_err, fmt="s", color="#fdb863", label="No covariance",
                      capsize=2.5, markersize=4.5, linewidth=1.0, elinewidth=0.9)
    axes[1].errorbar(x + off, single, yerr=single_err, fmt="^", color="#008837", label="Single obs.",
                      capsize=2.5, markersize=4.5, linewidth=1.0, elinewidth=0.9)
    axes[1].axhline(0, color="black", linewidth=0.6, zorder=0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tlabels, fontsize=6.5)
    axes[1].set_xlim(-0.5, len(tasks) - 0.5)
    axes[1].set_ylabel(r"Final $R^2$")
    axes[1].set_title("(b) Residual-coupled variant ablation", fontsize=8.5)
    axes[1].legend(loc="lower left", frameon=False, fontsize=6)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIGDIR, "fig3_ablation.pdf"))
    plt.close(fig)
    print("saved fig3_ablation.pdf")


def fig4_shot_convergence():
    with open(os.path.join(RESULTS, "quantum_circuit_experiment.json")) as f:
        qc = json.load(f)

    shots = sorted([int(k) for k in qc["shot_sweep"].keys()])
    mae = [qc["shot_sweep"][str(s)]["mae_vs_exact"] for s in shots]
    tau = [qc["shot_sweep"][str(s)]["kendall_tau"] for s in shots]
    overlap = [qc["shot_sweep"][str(s)]["top_b_overlap_frac"] * 100 for s in shots]

    fig, ax1 = plt.subplots(figsize=(3.4, 2.6))
    ax1.plot(shots, tau, "o-", color="#1f78b4", label=r"Kendall $\tau$")
    ax1.plot(shots, [o / 100 for o in overlap], "s--", color="#008837", label="Top-15 overlap")
    ax1.set_xscale("log")
    ax1.set_xlabel("Shots")
    ax1.set_ylabel("Ranking fidelity")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Acquisition-ranking fidelity vs. shots")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_shot_convergence.pdf"))
    plt.close(fig)
    print("saved fig4_shot_convergence.pdf")


def fig5_measurement_grouping():
    with open(os.path.join(RESULTS, "measurement_grouping.json")) as f:
        mg = json.load(f)
    totals = mg["totals_K3"]

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    labels = ["Raw", "QWC\ngrouped", "Fully\ngrouped"]
    values = [totals["raw"], totals["qwc_groups"], totals["full_commuting_groups"]]
    colors = ["#d7191c", "#fdae61", "#1a9641"]
    ax.bar(labels, values, color=colors, width=0.6)
    for i, v in enumerate(values):
        ax.text(i, v + 100, str(v), ha="center", fontsize=8)
    ax.set_ylabel("Measurement settings\n(9 quantities, K=3)")
    ax.set_title("Measurement grouping")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_measurement_grouping.pdf"))
    plt.close(fig)
    print("saved fig5_measurement_grouping.pdf")


if __name__ == "__main__":
    fig1_learning_curves()
    fig2_primary_comparison()
    fig3_ablation()
    fig4_shot_convergence()
    fig5_measurement_grouping()
    print("\nAll figures saved to", FIGDIR)
