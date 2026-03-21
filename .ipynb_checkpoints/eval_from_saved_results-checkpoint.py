import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ============================================================
# Paths
# ============================================================
JSON_DIR = "./plot_json_bundle"
RESULTS_DIR = "./results"

JSON_MAIN = os.path.join(JSON_DIR, "combined_bler_sinr_cdf_ci_data.json")
JSON_LSTM = os.path.join(JSON_DIR, "lstm_plots_ci_bundle.json")
JSON_SENS = os.path.join(JSON_DIR, "sensitivity_plots_bundle.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def arr(x):
    return np.asarray(x, dtype=float)


def first_existing(dct, keys):
    for k in keys:
        if k in dct:
            return dct[k]
    raise KeyError(f"None of the keys {keys} found in dictionary.")


# ============================================================
# 1) Load combined BLER/SINR JSON and plot
# ============================================================
with open(JSON_MAIN, "r", encoding="utf-8") as f:
    data = json.load(f)

# ------------------------------------------------------------
# Figure 1: Error probability CDFs
# ------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

curve_order_1 = [
    "async",
    "sync",
    "async_nourllc",
    "sync_nourllc",
    "ctde",
    "greedy",
]

markers = {
    "async": "*-",
    "sync": "o-",
    "async_nourllc": "D-",
    "sync_nourllc": "x-",
    "ctde": "v-",
    "greedy": "s-",
}

markevery_map = {
    "async": 1000,
    "sync": 1000,
    "async_nourllc": 1000,
    "sync_nourllc": 1000,
    "ctde": 1000,
    "greedy": 10000,
}

alpha_map = {
    "async": 0.20,
    "sync": 0.20,
    "async_nourllc": 0.20,
    "sync_nourllc": 0.20,
    "ctde": 0.20,
    "greedy": 0.18,
}

for key in curve_order_1:
    c = data["error_probability_cdf"]["curves"][key]

    x = arr(c["x"])
    y = arr(c["y"])
    y_lo = arr(first_existing(c, ["ci_lower", "lower"]))
    y_hi = arr(first_existing(c, ["ci_upper", "upper"]))

    line, = ax1.plot(
        x,
        y,
        markers[key],
        markevery=markevery_map[key],
        linewidth=2.0,
        label=c["label"],
        zorder=3,
    )

    ax1.fill_between(
        x,
        y_lo,
        y_hi,
        color=line.get_color(),
        alpha=alpha_map[key],
        linewidth=1.0,
        edgecolor=line.get_color(),
        zorder=1,
    )

ax1.set_xscale("log")
ax1.set_xlabel("Error Probability", fontsize=14)
ax1.set_ylabel("CDF", fontsize=14)
ax1.legend(loc="lower right", fontsize=12)
ax1.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_cdf1_ci_from_json.png"), dpi=300, bbox_inches="tight")
plt.close(fig1)


# ------------------------------------------------------------
# Figure 2: SINR CDFs
# ------------------------------------------------------------
fig2, ax2 = plt.subplots()

curve_order_2 = [
    "F_gamma",
    "F_gamma_bar",
    "F_gamma_over_gamma_bar_given_gamma_bar",
]

for key in curve_order_2:
    c = data["sinr_cdf"]["curves"][key]

    line_x = arr(first_existing(c, ["line_x", "x"]))
    line_y_for_plot = arr(first_existing(c, ["line_y_for_log_plot", "line_y", "y"]))
    x_grid = arr(first_existing(c, ["x_grid"]))
    ci_lower_clip = arr(first_existing(c, ["ci_lower_clipped_for_visible_plot", "ci_lower_plot", "ci_lower", "lower"]))
    ci_upper_plot = arr(first_existing(c, ["ci_upper_for_log_plot", "ci_upper_plot", "ci_upper", "upper"]))
    visible_mask = np.asarray(first_existing(c, ["visible_mask"])).astype(bool)

    label_txt = c.get("label", c.get("label_math", key))
    if isinstance(label_txt, str) and not label_txt.startswith("$"):
        plot_label = f"${label_txt}$"
    else:
        plot_label = label_txt

    line, = ax2.plot(
        line_x,
        line_y_for_plot,
        lw=2.0,
        label=plot_label
    )

    ax2.fill_between(
        x_grid,
        ci_lower_clip,
        ci_upper_plot,
        where=visible_mask,
        color=line.get_color(),
        alpha=0.16,
    )

plot_limits = data["sinr_cdf"]["plot_limits"]
xmin, xmax = plot_limits["xlim"]
ymin_plot, ymax_plot = plot_limits["ylim"]

ax2.set_xlabel("SINR(dB)", fontsize=14)
ax2.set_ylabel("CDF(log scale)", fontsize=14)
ax2.set_yscale("log")
ax2.set_xlim(float(xmin), float(xmax))
ax2.set_ylim(float(ymin_plot), float(ymax_plot))
ax2.grid(True, which="major", linestyle="-", alpha=0.45)
ax2.grid(True, which="minor", linestyle="-", alpha=0.18)
ax2.legend(fontsize=13, loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "SINR_CDF_ci_from_json.png"), dpi=300, bbox_inches="tight")
plt.close(fig2)


# ============================================================
# 2) Load LSTM JSON and plot
# ============================================================
with open(JSON_LSTM, "r", encoding="utf-8") as f:
    bundle = json.load(f)

# ------------------------------------------------------------
# Figure 3: training loss + interference trace
# ------------------------------------------------------------
fig3, axs = plt.subplots(2, 1, figsize=(5.1, 6.3))

train = bundle["training_and_interference_plot"]["training_loss"]
intrf = bundle["training_and_interference_plot"]["interference_trace"]

epochs = arr(train["x"])
loss_mean = arr(first_existing(train, ["mean", "y"]))
loss_lo_vis = arr(first_existing(train, ["ci_lower_plot", "ci_lower", "lower"]))
loss_hi_vis = arr(first_existing(train, ["ci_upper_plot", "ci_upper", "upper"]))

axs[0].plot(epochs, loss_mean, lw=1.6, zorder=3)
axs[0].fill_between(
    epochs,
    loss_lo_vis,
    loss_hi_vis,
    alpha=0.42,
    zorder=1
)
axs[0].set_yscale('log')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss(log scale)')
axs[0].grid(True, which='major', linestyle='-', alpha=0.45)
axs[0].grid(True, which='minor', linestyle='-', alpha=0.15)

t = arr(intrf["x"])
target = arr(first_existing(intrf, ["target"]))
pred_plot = arr(first_existing(intrf, ["predicted_mean", "mean", "y"]))
pred_lo_vis = arr(first_existing(intrf, ["ci_lower_plot", "ci_lower", "lower"]))
pred_hi_vis = arr(first_existing(intrf, ["ci_upper_plot", "ci_upper", "upper"]))

axs[1].plot(t, target, label='target', lw=1.6, zorder=3)
axs[1].plot(t, pred_plot, label='predicted', lw=1.6, zorder=4)
axs[1].fill_between(
    t,
    pred_lo_vis,
    pred_hi_vis,
    alpha=0.36,
    zorder=2
)

axs[1].set_xlabel('Time Steps')
axs[1].set_ylabel('Interference power (dBm)')
axs[1].set_ylim(-68, -17)
axs[1].legend(loc='upper right')
axs[1].grid(True, which='major', linestyle='-', alpha=0.45)
axs[1].grid(True, which='minor', linestyle='-', alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "LSTM_perf_intpower_1_ci_from_json.png"), dpi=300, bbox_inches='tight')
plt.close(fig3)


# ------------------------------------------------------------
# Figure 4: sensitivity plot
# ------------------------------------------------------------
fig4 = plt.figure(figsize=(4, 2))
sens = bundle["lstm_sensitivity_plot"]

k_values = arr(sens["x"])
test_loss = arr(first_existing(sens, ["mean", "y"]))
lower = arr(first_existing(sens, ["ci_lower", "lower"]))
upper = arr(first_existing(sens, ["ci_upper", "upper"]))

best_k = sens["best_k"]
best_loss = sens["best_loss"]
vline_x = sens["vline_x"]
hline_y = sens["hline_y"]

plt.plot(k_values, test_loss, marker='o', linewidth=2, markersize=7, label='Test loss')
plt.fill_between(k_values, lower, upper, alpha=0.2)
plt.scatter(best_k, best_loss, s=90, zorder=5, label=fr'Elbow point at $k=10$')
plt.axvline(vline_x, linestyle='--', linewidth=1)
plt.axhline(hline_y, linestyle='--', linewidth=1)
plt.xticks(k_values)
plt.xlabel(r'LSTM Input window size $k$')
plt.ylabel('Test loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lstm_sensitivity_plot_from_json.png"), dpi=300, bbox_inches='tight')
plt.close(fig4)


# ============================================================
# 3) Load sensitivity_plots_bundle.json and plot
# ============================================================
with open(JSON_SENS, "r", encoding="utf-8") as f:
    bundle = json.load(f)

figs = bundle["figures"]

# ------------------------------------------------------------
# Figure 5: sensitivity_error_probability
# ------------------------------------------------------------
sens1 = figs["sensitivity_error_probability"]
sset = sens1["settings"]

fig5, axs = plt.subplots(
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
        arr(first_existing(c, ["lower", "ci_lower"])),
        arr(first_existing(c, ["upper", "ci_upper"])),
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
        arr(first_existing(c, ["lower", "ci_lower"])),
        arr(first_existing(c, ["upper", "ci_upper"])),
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

fig5.legend(
    handles,
    [h.get_label() for h in handles],
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.55, -0.05),
    fontsize=12
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_error_probability_from_json.png"), dpi=300, bbox_inches="tight")
plt.close(fig5)


# ------------------------------------------------------------
# Figure 6: critic / BLER vs devices and velocity
# ------------------------------------------------------------
sens2 = figs["sensitivity_bler_devices_velocity"]
cfg = sens2["settings"]

plt.rcParams.update(cfg["font_params"])

fig6, axes = plt.subplots(
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

# (a) Critic loss vs J
d = sens2["critic_vs_J"]
ax = axes[0, 0]
ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
ax.fill_between(
    arr(d["x"]),
    arr(first_existing(d, ["lower", "ci_lower"])),
    arr(first_existing(d, ["upper", "ci_upper"])),
    alpha=shade_alpha
)
ax.set_xlabel(d["xlabel"])
ax.set_ylabel(d["ylabel"])
ax.grid(True, alpha=0.25)
ax.set_xticks(arr(d["x"]))

# (b) Critic loss vs velocity
d = sens2["critic_vs_V"]
ax = axes[0, 1]
ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
ax.fill_between(
    arr(d["x"]),
    arr(first_existing(d, ["lower", "ci_lower"])),
    arr(first_existing(d, ["upper", "ci_upper"])),
    alpha=shade_alpha
)
ax.set_xlabel(d["xlabel"])
if d["ylabel"]:
    ax.set_ylabel(d["ylabel"])
ax.grid(True, alpha=0.25)
ax.set_xticks(arr(d["x"]))

# (c) Average BLER vs J
d = sens2["bler_vs_J"]
ax = axes[1, 0]
ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
ax.fill_between(
    arr(d["x"]),
    arr(first_existing(d, ["lower", "ci_lower"])),
    arr(first_existing(d, ["upper", "ci_upper"])),
    alpha=shade_alpha
)
ax.set_xlabel(d["xlabel"])
ax.set_ylabel(d["ylabel"])
ax.grid(True, alpha=0.25)
ax.set_xticks(arr(d["x"]))
ax.set_ylim(BLER_MIN, BLER_MAX)

formatter_c = ScalarFormatter(useMathText=True)
formatter_c.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter_c)

# (d) Average BLER vs velocity
d = sens2["bler_vs_V"]
ax = axes[1, 1]
ax.plot(arr(d["x"]), arr(d["mean"]), marker=d["marker"])
ax.fill_between(
    arr(d["x"]),
    arr(first_existing(d, ["lower", "ci_lower"])),
    arr(first_existing(d, ["upper", "ci_upper"])),
    alpha=shade_alpha
)
ax.set_xlabel(d["xlabel"])
if d["ylabel"]:
    ax.set_ylabel(d["ylabel"])
ax.grid(True, alpha=0.25)
ax.set_xticks(arr(d["x"]))
ax.set_ylim(BLER_MIN, BLER_MAX)

formatter_d = ScalarFormatter(useMathText=True)
formatter_d.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(formatter_d)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_bler_devices_velocity_from_json.png"), dpi=300, bbox_inches="tight")
plt.close(fig6)

print(f"Saved all figures to: {RESULTS_DIR}")