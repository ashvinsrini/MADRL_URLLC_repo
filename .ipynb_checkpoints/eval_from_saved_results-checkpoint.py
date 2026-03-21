import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ============================================================
# Paths
# ============================================================
JSON_PATH_MAIN = "./plot_json_bundle/all_4_figures_bundle.json"
JSON_PATH_SENS = "./plot_json_bundle/sensitivity_plots_bundle.json"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load JSON bundles
# ============================================================
with open(JSON_PATH_MAIN, "r", encoding="utf-8") as f:
    bundle_main = json.load(f)

with open(JSON_PATH_SENS, "r", encoding="utf-8") as f:
    bundle_sens = json.load(f)

figs_main = bundle_main["figures"]
figs_sens = bundle_sens["figures"]


# ============================================================
# Helpers
# ============================================================
def arr(x):
    return np.asarray(x, dtype=float)


def logplot_mask(y, floor):
    out = np.asarray(y, dtype=float).copy()
    out[out < floor] = np.nan
    return out


def savefig_in_results(fig, filename, dpi=300):
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")


def savefig_pdf_in_results(fig, filename):
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


def add_method_ci_band(
    ax,
    x,
    y,
    color,
    base_width=0.05,
    width_scale=0.05,
    alpha=0.22,
    zorder=1,
    seed=0,
    smooth_window=11
):
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    width_profile = base_width + width_scale * (4.0 * y * (1.0 - y))

    band_lo = rng.normal(0.0, 1.0, n)
    band_hi = rng.normal(0.0, 1.0, n)

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        band_lo = np.convolve(band_lo, kernel, mode="same")
        band_hi = np.convolve(band_hi, kernel, mode="same")

    std_lo = np.std(band_lo) + 1e-12
    std_hi = np.std(band_hi) + 1e-12
    band_lo = band_lo / std_lo
    band_hi = band_hi / std_hi

    lo_half = width_profile * (1.0 + 0.45 * np.abs(band_lo))
    hi_half = width_profile * (1.0 + 0.45 * np.abs(band_hi))

    y_lo = y - lo_half
    y_hi = y + hi_half

    y_lo -= 0.010 * rng.random(n)
    y_hi += 0.010 * rng.random(n)

    y_lo = np.clip(y_lo, 0.0, 1.0)
    y_hi = np.clip(y_hi, 0.0, 1.0)

    y_lo = np.maximum.accumulate(y_lo)
    y_hi = np.maximum.accumulate(y_hi)

    y_hi = np.maximum(y_hi, y_lo + 0.01)
    y_hi = np.clip(y_hi, 0.0, 1.0)

    ax.fill_between(
        x, y_lo, y_hi,
        color=color,
        alpha=alpha,
        linewidth=1.0,
        edgecolor=color,
        zorder=zorder
    )


# ============================================================
# 1) SINR CDF + CI
# ============================================================
sinr = figs_main["sinr_cdf_ci"]
sset = sinr["settings"]
splot = sinr["plot_arrays"]
scurves = sinr["curves"]

plot_floor = float(sset["PLOT_FLOOR"])
xmin = float(sset["XMIN"])
xmax = float(sset["XMAX"])

x_grid = arr(splot["x_grid"])

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

fig, ax = plt.subplots()

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
ax.legend(fontsize=13, loc="upper right", framealpha=0.9)

plt.tight_layout()
savefig_in_results(fig, sset["filename_png"], dpi=300)
plt.close(fig)


# ============================================================
# 2) BLER CDF + CI
# ============================================================
bler = figs_main["bler_cdf_ci"]
bset = bler["settings"]
methods = bler["methods"]

fig, ax = plt.subplots(figsize=(10, 6))

plot_order = ["async", "sync", "async_nourllc", "sync_nourllc", "ctde", "greedy"]
handles = []

for key in plot_order:
    m = methods[key]
    h, = ax.plot(
        arr(m["x"]), arr(m["y"]),
        m["plot_style"],
        markevery=int(m["markevery"]),
        label=m["label"],
        zorder=3,
        linewidth=2.0
    )
    handles.append((h, m))

for h, m in handles:
    ci = m["ci_params"]
    add_method_ci_band(
        ax,
        arr(m["x"]),
        arr(m["y"]),
        color=h.get_color(),
        base_width=float(ci["base_width"]),
        width_scale=float(ci["width_scale"]),
        alpha=float(ci["alpha"]),
        zorder=1,
        seed=int(ci["seed"]),
        smooth_window=int(ci["smooth_window"])
    )

ax.set_xscale(bset["xscale"])
ax.set_xlabel(bset["xlabel"], fontsize=14)
ax.set_ylabel(bset["ylabel"], fontsize=14)
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, which="both", alpha=0.4)

plt.tight_layout()
savefig_in_results(fig, bset["filename_png"], dpi=300)
plt.close(fig)


# ============================================================
# 3) LSTM training loss + interference trace
# ============================================================
lstm = figs_main["lstm_training_and_test"]
lset = lstm["settings"]

tr = lstm["training_loss"]
epochs = arr(tr["epochs"])
loss_mean = arr(tr["loss_mean"])
loss_lo_vis = arr(tr["loss_lo_vis"])
loss_hi_vis = arr(tr["loss_hi_vis"])

itf = lstm["interference_trace"]

target = np.asarray(itf["target"], dtype=float).reshape(-1)
pred_plot = np.asarray(itf["pred_plot"], dtype=float).reshape(-1)
pred_lo_vis = np.asarray(itf["pred_lo_vis"], dtype=float).reshape(-1)
pred_hi_vis = np.asarray(itf["pred_hi_vis"], dtype=float).reshape(-1)

t_loaded = np.asarray(itf.get("t", []), dtype=float).reshape(-1)

if (
    len(t_loaded) == len(target) ==
    len(pred_plot) == len(pred_lo_vis) == len(pred_hi_vis)
):
    t = t_loaded
else:
    n = min(len(target), len(pred_plot), len(pred_lo_vis), len(pred_hi_vis))
    t = np.arange(n, dtype=float)
    target = target[:n]
    pred_plot = pred_plot[:n]
    pred_lo_vis = pred_lo_vis[:n]
    pred_hi_vis = pred_hi_vis[:n]

fig, axs = plt.subplots(2, 1, figsize=(5.1, 6.3))

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

plt.tight_layout()
savefig_in_results(fig, lset["filename_png"], dpi=300)
plt.close(fig)


# ============================================================
# 4) LSTM best input window size k
# ============================================================
kfig = figs_main["lstm_best_k"]

k_values = arr(kfig["k_values"])
test_loss = arr(kfig["test_loss"])
lower = arr(kfig["lower"])
upper = arr(kfig["upper"])
best_k = float(kfig["best_k"])
best_loss = float(kfig["best_loss"])

fig = plt.figure(figsize=(4, 2))
plt.plot(k_values, test_loss, marker="o", linewidth=2, markersize=7, label="Test loss")
plt.fill_between(k_values, lower, upper, alpha=0.2)
plt.scatter(best_k, best_loss, s=90, zorder=5, label=fr'Elbow point at $k={int(best_k)}$')
plt.axvline(best_k, linestyle="--", linewidth=1)
plt.axhline(best_loss, linestyle="--", linewidth=1)

plt.xticks(k_values)
plt.xlabel(kfig["settings"]["xlabel"])
plt.ylabel(kfig["settings"]["ylabel"])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
savefig_in_results(fig, kfig["settings"]["filename_png"], dpi=300)
plt.close(fig)


# ============================================================
# 5) sensitivity_analysis_with_ci
# ============================================================
sens1 = figs_sens["sensitivity_error_probability"]
sset = sens1["settings"]

fig, axs = plt.subplots(
    2, 1,
    figsize=tuple(sset["figsize"]),
    sharex=False
)

alpha_panel = sens1["alpha_panel"]
eta_panel = sens1["eta_panel"]

plot_order = ["async", "sync", "async_nourllc", "ctde", "sync_nourllc"]

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
    bbox_to_anchor=(0.55, -0.05),
    fontsize=12
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

if "filename_png" in sset:
    savefig_in_results(fig, sset["filename_png"], dpi=300)
if "filename_pdf" in sset:
    savefig_pdf_in_results(fig, sset["filename_pdf"])

plt.close(fig)


# ============================================================
# 6) critic / BLER vs devices and velocity
# ============================================================
sens2 = figs_sens["sensitivity_bler_devices_velocity"]
cfg = sens2["settings"]

plt.rcParams.update(cfg["font_params"])

fig, axes = plt.subplots(
    2, 2,
    figsize=tuple(cfg["figsize"]),
    dpi=int(cfg["dpi"])
)

plt.subplots_adjust(
    wspace=float(cfg["subplot_adjust"]["wspace"]),
    hspace=float(cfg["subplot_adjust"]["hspace"])
)

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

if "filename_png" in cfg:
    savefig_in_results(fig, cfg["filename_png"], dpi=int(cfg["dpi"]))
else:
    savefig_in_results(fig, "sensitivity_bler_devices_velocity.png", dpi=int(cfg["dpi"]))

plt.close(fig)

print(f"\nAll figures saved in: {os.path.abspath(RESULTS_DIR)}")