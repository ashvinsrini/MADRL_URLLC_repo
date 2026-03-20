import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI / no plt.show()
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ============================================================
# JSON paths
# Edit if needed
# ============================================================
ALL4_JSON_PATH = "./plot_json_bundle/all_4_figures_bundle.json"
SENS_JSON_PATH = "./plot_json_bundle/sensitivity_plots_bundle.json"


# ============================================================
# Helpers
# ============================================================
def arr(x):
    return np.asarray(x, dtype=float)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_results_dir_from_json(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_figure(fig, out_path, dpi=300, pad_inches=0.18):
    """
    Robust save helper to reduce truncation.
    """
    fig.canvas.draw()
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white"
    )
    plt.close(fig)


def logplot_mask(y, floor):
    out = np.asarray(y, dtype=float).copy()
    out[out < floor] = np.nan
    return out


def add_method_ci(ax, x, y, color, half_width=0.02, alpha=0.22, zorder=1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    y_lo = np.clip(y - half_width, 0.0, 1.0)
    y_hi = np.clip(y + half_width, 0.0, 1.0)

    ax.fill_between(
        x, y_lo, y_hi,
        color=color,
        alpha=alpha,
        linewidth=0.8,
        edgecolor=color,
        zorder=zorder
    )


# ============================================================
# Bundle 1: all_4_figures_bundle.json
# ============================================================
def plot_all4_bundle(json_path):
    bundle = load_json(json_path)
    figs = bundle["figures"]
    results_dir = ensure_results_dir_from_json(json_path)

    # --------------------------------------------------------
    # 1) SINR CDF + CI
    # --------------------------------------------------------
    sinr = figs["sinr_cdf_ci"]
    sset = sinr["settings"]
    scurves = sinr["curves"]

    plot_floor = float(sset["PLOT_FLOOR"])
    xmin = float(sset["XMIN"])
    xmax = float(sset["XMAX"])

    x_grid = arr(scurves["x_grid"])

    blue_x = arr(scurves["blue"]["line_x"])
    blue_y = arr(scurves["blue"]["line_y"])
    blue_lo = arr(scurves["blue"]["lower"])
    blue_hi = arr(scurves["blue"]["upper"])

    orange_x = arr(scurves["orange"]["line_x"])
    orange_y = arr(scurves["orange"]["line_y"])
    orange_lo = arr(scurves["orange"]["lower"])
    orange_hi = arr(scurves["orange"]["upper"])

    green_x = arr(scurves["green"]["line_x"])
    green_y = arr(scurves["green"]["line_y"])
    green_lo = arr(scurves["green"]["lower"])
    green_hi = arr(scurves["green"]["upper"])

    blue_y_plot = logplot_mask(blue_y, plot_floor)
    orange_y_plot = logplot_mask(orange_y, plot_floor)
    green_y_plot = logplot_mask(green_y, plot_floor)

    blue_lo_plot = np.maximum(blue_lo, plot_floor)
    blue_hi_plot = np.maximum(blue_hi, plot_floor)

    orange_lo_plot = np.maximum(orange_lo, plot_floor)
    orange_hi_plot = np.maximum(orange_hi, plot_floor)

    green_lo_plot = np.maximum(green_lo, plot_floor)
    green_hi_plot = np.maximum(green_hi, plot_floor)

    ymin_curve = np.nanmin([
        np.nanmin(blue_y_plot),
        np.nanmin(orange_y_plot),
        np.nanmin(green_y_plot),
    ])
    ymin_plot = 0.95 * ymin_curve

    blue_lo_clip = np.where(blue_lo_plot < ymin_plot, ymin_plot, blue_lo_plot)
    orange_lo_clip = np.where(orange_lo_plot < ymin_plot, ymin_plot, orange_lo_plot)
    green_lo_clip = np.where(green_lo_plot < ymin_plot, ymin_plot, green_lo_plot)

    blue_mask = blue_hi_plot >= ymin_plot
    orange_mask = orange_hi_plot >= ymin_plot
    green_mask = green_hi_plot >= ymin_plot

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    l1, = ax.plot(blue_x, blue_y_plot, lw=2.0, label=scurves["blue"]["label"])
    l2, = ax.plot(orange_x, orange_y_plot, lw=2.0, label=scurves["orange"]["label"])
    l3, = ax.plot(green_x, green_y_plot, lw=2.0, label=scurves["green"]["label"])

    ax.fill_between(x_grid, blue_lo_clip, blue_hi_plot, where=blue_mask, color=l1.get_color(), alpha=0.16)
    ax.fill_between(x_grid, orange_lo_clip, orange_hi_plot, where=orange_mask, color=l2.get_color(), alpha=0.16)
    ax.fill_between(x_grid, green_lo_clip, green_hi_plot, where=green_mask, color=l3.get_color(), alpha=0.16)

    ax.set_xlabel(sset["xlabel"], fontsize=14)
    ax.set_ylabel(sset["ylabel"], fontsize=14)
    ax.set_yscale("log")
    ax.set_ylim(ymin_plot, 1.0)
    ax.set_xlim(xmin, xmax)
    ax.grid(True, which="major", linestyle="-", alpha=0.45)
    ax.grid(True, which="minor", linestyle="-", alpha=0.18)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(results_dir, sset["filename_png"])
    save_figure(fig, out_path, dpi=300, pad_inches=0.16)

    # --------------------------------------------------------
    # 2) BLER CDF + CI
    # --------------------------------------------------------
    bler = figs["bler_cdf_ci"]
    bset = bler["settings"]
    methods = bler["methods"]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    plot_order = ["async", "sync", "async_nourllc", "sync_nourllc", "ctde", "greedy"]
    handles = []

    for key in plot_order:
        m = methods[key]
        h, = ax.plot(
            arr(m["x"]), arr(m["y"]),
            m["plot_style"],
            markevery=int(m["markevery"]),
            label=m["label"],
            zorder=3
        )
        handles.append((h, m))

    for h, m in handles:
        add_method_ci(
            ax,
            arr(m["x"]), arr(m["y"]),
            color=h.get_color(),
            half_width=float(m["ci_half_width"]),
            alpha=float(m["ci_alpha"]),
            zorder=1
        )

    ax.set_xscale(bset["xscale"])
    ax.set_xlabel(bset["xlabel"], fontsize=14)
    ax.set_ylabel(bset["ylabel"], fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(results_dir, bset["filename_png"])
    save_figure(fig, out_path, dpi=300, pad_inches=0.18)

    # --------------------------------------------------------
    # 3) LSTM best input window size k
    # --------------------------------------------------------
    kfig = figs["lstm_best_k"]

    k_values = arr(kfig["k_values"])
    test_loss = arr(kfig["test_loss"])
    lower = arr(kfig["lower"])
    upper = arr(kfig["upper"])
    best_k = float(kfig["best_k"])
    best_loss = float(kfig["best_loss"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(k_values, test_loss, marker="o", linewidth=2, markersize=7, label="Test loss")
    ax.fill_between(k_values, lower, upper, alpha=0.2)
    ax.scatter(best_k, best_loss, s=90, zorder=5, label=fr'Elbow point at $k={int(best_k)}$')
    ax.axvline(best_k, linestyle="--", linewidth=1)
    ax.axhline(best_loss, linestyle="--", linewidth=1)

    ax.set_xticks(k_values)
    ax.set_xlabel(kfig["settings"]["xlabel"])
    ax.set_ylabel(kfig["settings"]["ylabel"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(results_dir, kfig["settings"]["filename_png"])
    save_figure(fig, out_path, dpi=300, pad_inches=0.16)

    # --------------------------------------------------------
    # 4) LSTM training loss + unseen test interference trace
    # --------------------------------------------------------
    lstm = figs["lstm_training_and_test"]
    lset = lstm["settings"]

    tr = lstm["training_loss"]
    epochs = arr(tr["epochs"])
    loss_mean = arr(tr["loss_mean"])
    loss_lo_vis = arr(tr["loss_lo_vis"])
    loss_hi_vis = arr(tr["loss_hi_vis"])

    itf = lstm["interference_trace"]
    t = arr(itf["t"])
    target = arr(itf["target"])
    pred_plot = arr(itf["pred_plot"])
    pred_lo_vis = arr(itf["pred_lo_vis"])
    pred_hi_vis = arr(itf["pred_hi_vis"])

    fig, axs = plt.subplots(2, 1, figsize=(6.6, 8.0))
    fig.subplots_adjust(hspace=0.35)

    axs[0].plot(epochs, loss_mean, lw=1.6, zorder=3)
    axs[0].fill_between(epochs, loss_lo_vis, loss_hi_vis, alpha=0.42, zorder=1)
    axs[0].set_yscale("log")
    axs[0].set_xlabel(lset["top_xlabel"])
    axs[0].set_ylabel(lset["top_ylabel"])
    axs[0].grid(True, which="major", linestyle="-", alpha=0.45)
    axs[0].grid(True, which="minor", linestyle="-", alpha=0.15)

    axs[1].plot(t, target, label="target", lw=1.6, zorder=3)
    axs[1].plot(t, pred_plot, label="predicted", lw=1.6, zorder=4)
    axs[1].fill_between(t, pred_lo_vis, pred_hi_vis, alpha=0.36, zorder=2)
    axs[1].set_xlabel(lset["bottom_xlabel"])
    axs[1].set_ylabel(lset["bottom_ylabel"])
    axs[1].set_ylim(lset["bottom_ylim"][0], lset["bottom_ylim"][1])
    axs[1].legend(loc="upper right")
    axs[1].grid(True, which="major", linestyle="-", alpha=0.45)
    axs[1].grid(True, which="minor", linestyle="-", alpha=0.15)

    fig.tight_layout()
    out_path = os.path.join(results_dir, lset["filename_png"])
    save_figure(fig, out_path, dpi=300, pad_inches=0.18)

    print(f"[OK] Saved all_4_figures_bundle outputs to: {results_dir}")


# ============================================================
# Bundle 2: sensitivity_plots_bundle.json
# ============================================================
def plot_sensitivity_bundle(json_path):
    bundle = load_json(json_path)
    figs = bundle["figures"]
    results_dir = ensure_results_dir_from_json(json_path)

    # --------------------------------------------------------
    # 1) sensitivity_analysis_with_ci
    # --------------------------------------------------------
    sens1 = figs["sensitivity_error_probability"]
    sset = sens1["settings"]

    alpha_panel = sens1["alpha_panel"]
    eta_panel = sens1["eta_panel"]
    plot_order = ["async", "sync", "async_nourllc", "ctde", "sync_nourllc"]

    # Use a larger size than the stored one to prevent truncation
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=False)
    fig.subplots_adjust(hspace=0.35, bottom=0.20)

    handles = []
    for key in plot_order:
        c = alpha_panel["curves"][key]
        h, = axs[0].plot(
            arr(alpha_panel["x"]),
            arr(c["y"]),
            marker=c["marker"],
            label=c["label"],
            zorder=3
        )
        axs[0].fill_between(
            arr(alpha_panel["x"]),
            arr(c["lower"]),
            arr(c["upper"]),
            color=h.get_color(),
            alpha=0.30,
            edgecolor=h.get_color(),
            linewidth=0.8,
            zorder=1
        )
        handles.append(h)

    axs[0].set_xscale(sset["xscale"])
    axs[0].set_yscale(sset["yscale"])
    axs[0].set_xlabel(sset["xlabel_alpha"], fontsize=14)
    axs[0].set_ylabel(sset["ylabel"], fontsize=14)
    axs[0].grid(True, which="both", linestyle="--", alpha=0.5)

    for key in plot_order:
        c = eta_panel["curves"][key]
        h, = axs[1].plot(
            arr(eta_panel["x"]),
            arr(c["y"]),
            marker=c["marker"],
            zorder=3
        )
        axs[1].fill_between(
            arr(eta_panel["x"]),
            arr(c["lower"]),
            arr(c["upper"]),
            color=h.get_color(),
            alpha=0.30,
            edgecolor=h.get_color(),
            linewidth=0.8,
            zorder=1
        )

    axs[1].set_xscale(sset["xscale"])
    axs[1].set_yscale(sset["yscale"])
    axs[1].set_xlabel(sset["xlabel_eta"], fontsize=14)
    axs[1].set_ylabel(sset["ylabel"], fontsize=14)
    axs[1].grid(True, which="both", linestyle="--", alpha=0.5)

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=12,
        frameon=True
    )

    out_png = os.path.join(results_dir, sset.get("filename_png", "sensitivity_analysis_with_ci.png"))
    save_figure(fig, out_png, dpi=300, pad_inches=0.22)

    if "filename_pdf" in sset:
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=False)
        fig.subplots_adjust(hspace=0.35, bottom=0.20)

        handles = []
        for key in plot_order:
            c = alpha_panel["curves"][key]
            h, = axs[0].plot(
                arr(alpha_panel["x"]),
                arr(c["y"]),
                marker=c["marker"],
                label=c["label"],
                zorder=3
            )
            axs[0].fill_between(
                arr(alpha_panel["x"]),
                arr(c["lower"]),
                arr(c["upper"]),
                color=h.get_color(),
                alpha=0.30,
                edgecolor=h.get_color(),
                linewidth=0.8,
                zorder=1
            )
            handles.append(h)

        axs[0].set_xscale(sset["xscale"])
        axs[0].set_yscale(sset["yscale"])
        axs[0].set_xlabel(sset["xlabel_alpha"], fontsize=14)
        axs[0].set_ylabel(sset["ylabel"], fontsize=14)
        axs[0].grid(True, which="both", linestyle="--", alpha=0.5)

        for key in plot_order:
            c = eta_panel["curves"][key]
            h, = axs[1].plot(
                arr(eta_panel["x"]),
                arr(c["y"]),
                marker=c["marker"],
                zorder=3
            )
            axs[1].fill_between(
                arr(eta_panel["x"]),
                arr(c["lower"]),
                arr(c["upper"]),
                color=h.get_color(),
                alpha=0.30,
                edgecolor=h.get_color(),
                linewidth=0.8,
                zorder=1
            )

        axs[1].set_xscale(sset["xscale"])
        axs[1].set_yscale(sset["yscale"])
        axs[1].set_xlabel(sset["xlabel_eta"], fontsize=14)
        axs[1].set_ylabel(sset["ylabel"], fontsize=14)
        axs[1].grid(True, which="both", linestyle="--", alpha=0.5)

        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.03),
            fontsize=12,
            frameon=True
        )

        out_pdf = os.path.join(results_dir, sset["filename_pdf"])
        save_figure(fig, out_pdf, dpi=300, pad_inches=0.22)

    # --------------------------------------------------------
    # 2) critic / BLER vs devices and velocity
    # --------------------------------------------------------
    sens2 = figs["sensitivity_bler_devices_velocity"]
    cfg = sens2["settings"]

    old_rc = plt.rcParams.copy()
    plt.rcParams.update(cfg["font_params"])

    # Stored figure was too small; use larger safe size
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), dpi=max(int(cfg["dpi"]), 300))
    fig.subplots_adjust(wspace=0.35, hspace=0.42, bottom=0.12, left=0.12, right=0.97, top=0.96)

    shade_alpha = float(cfg["shade_alpha"])
    BLER_MIN = float(cfg["BLER_MIN"])
    BLER_MAX = float(cfg["BLER_MAX"])

    d = sens2["critic_vs_J"]
    ax = axes[0, 0]
    ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
    ax.fill_between(arr(d["x"]), arr(d["lower"]), arr(d["upper"]), alpha=shade_alpha)
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.set_xticks(arr(d["x"]))

    d = sens2["critic_vs_V"]
    ax = axes[0, 1]
    ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
    ax.fill_between(arr(d["x"]), arr(d["lower"]), arr(d["upper"]), alpha=shade_alpha)
    ax.set_xlabel(d["xlabel"])
    if d["ylabel"]:
        ax.set_ylabel(d["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.set_xticks(arr(d["x"]))

    d = sens2["bler_vs_J"]
    ax = axes[1, 0]
    ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
    ax.fill_between(arr(d["x"]), arr(d["lower"]), arr(d["upper"]), alpha=shade_alpha)
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.set_xticks(arr(d["x"]))
    ax.set_ylim(BLER_MIN, BLER_MAX)
    formatter_c = ScalarFormatter(useMathText=True)
    formatter_c.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter_c)

    d = sens2["bler_vs_V"]
    ax = axes[1, 1]
    ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
    ax.fill_between(arr(d["x"]), arr(d["lower"]), arr(d["upper"]), alpha=shade_alpha)
    ax.set_xlabel(d["xlabel"])
    if d["ylabel"]:
        ax.set_ylabel(d["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.set_xticks(arr(d["x"]))
    ax.set_ylim(BLER_MIN, BLER_MAX)
    formatter_d = ScalarFormatter(useMathText=True)
    formatter_d.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter_d)

    out_path = os.path.join(results_dir, "sensitivity_bler_devices_velocity.png")
    save_figure(fig, out_path, dpi=max(int(cfg["dpi"]), 300), pad_inches=0.18)

    plt.rcParams.update(old_rc)

    print(f"[OK] Saved sensitivity_plots_bundle outputs to: {results_dir}")


# ============================================================
# Main
# ============================================================
def main():
    if not os.path.isfile(ALL4_JSON_PATH):
        raise FileNotFoundError(f"Could not find: {ALL4_JSON_PATH}")

    if not os.path.isfile(SENS_JSON_PATH):
        raise FileNotFoundError(f"Could not find: {SENS_JSON_PATH}")

    plot_all4_bundle(ALL4_JSON_PATH)
    plot_sensitivity_bundle(SENS_JSON_PATH)

    print("\nDone. All plots were saved into a 'results' folder next to the JSON file(s).")


if __name__ == "__main__":
    main()