import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

sys.stdout.reconfigure(encoding='utf-8')

# INCARCA DATASETUL
CSV_FILE = "dataset_01_06.csv"

df = pd.read_csv(CSV_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# PARAMETRI MEDIERE SI KALMAN


KALMAN_PARAMS = {
    "ir_raw": {"process_variance": 0.1, "measurement_variance": 30},
    "tmax":   {"process_variance": 1e-5, "measurement_variance": 1},
    "tmean":  {"process_variance": 1e-5, "measurement_variance": 1},
}



# FILTRU KALMAN 1D

class KalmanFilter1D:
    def __init__(self, process_variance=1e-3, measurement_variance=50):
        self.process_variance     = process_variance
        self.measurement_variance = measurement_variance
        self.estimate             = None
        self.estimate_error       = 1.0

    def filter(self, measurements):
        if self.estimate is None:
            self.estimate = float(measurements.iloc[0])
        filtered = []
        for z in measurements:
            pred_error          = self.estimate_error + self.process_variance
            K                   = pred_error / (pred_error + self.measurement_variance)
            self.estimate       = self.estimate + K * (z - self.estimate)
            self.estimate_error = (1 - K) * pred_error
            filtered.append(self.estimate)
        return pd.Series(filtered, index=measurements.index)

def apply_kalman(df, features, kalman_params):
    df_out = df.copy()
    for feat in features:
        params     = kalman_params[feat]
        col_result = pd.Series(index=df.index, dtype=float)
        for scenario in df["scenario"].unique():
            mask = df["scenario"] == scenario
            sub  = df[mask].sort_values("timestamp")
            kf   = KalmanFilter1D(**params)
            col_result[sub.index] = kf.filter(sub[feat])
        df_out[f"{feat}_kalman"] = col_result
    return df_out

FEATURES = ["ir_raw", "tmax", "tmean"]
df = apply_kalman(df, FEATURES, KALMAN_PARAMS)
print("Filtru Kalman aplicat.")

FEATURES_K     = [f"{f}_kalman" for f in FEATURES]
FEATURE_LABELS = {
    "ir_raw": "IR Raw",
    "tmax":   "Tmax [°C]",
    "tmean":  "Tmean [°C]",
}
FEATURE_LABELS_K = {f"{f}_kalman": f"{FEATURE_LABELS[f]} (Kalman)" for f in FEATURES}

# SCENARII SI CULORI
scenario_order = [
    "Camera umbroasa (baseline)",
    "Lumina solara directa",
    #"Radiator electric pornit",
    "Corp uman aproape senzor",
    "Lumanare 10cm",
    "Lumanare 20cm",
    "Lumanare 30cm",
    "Hartie aprinsa",
    "Flacara aragaz",
    "Flacara alcool izopropilic",
]
scenario_order = [s for s in scenario_order if s in df["scenario"].unique()]

COLORS = {
    "Camera umbroasa (baseline)":  "#00AA00",
    "Lumina solara directa":       "#1f750a",
   # "Radiator electric pornit":    "#ea00ff",
    "Corp uman aproape senzor":    "#8800ff",
    "Lumanare 10cm":               "#ff0000",
    "Lumanare 20cm":               "#F88400",
    "Lumanare 30cm":               "#ccaa00",
    "Hartie aprinsa":              "#590e0e",
    "Flacara aragaz":              "#2400f1",
    "Flacara alcool izopropilic":  "#00aaaa",
}
LABEL_COLORS = {0: "#1565c0", 1: "#c62828"}

# TEMA ALBA
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f8f8",
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   "#333",
    "xtick.color":       "#333",
    "ytick.color":       "#333",
    "text.color":        "#333",
    "grid.color":        "#ccc",
    "grid.alpha":        0.5,
    "figure.dpi":        100,
})

# FIGURA 1 — Boxplot per scenariu (date originale)
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle("Distributia senzorilor per scenariu",
              fontsize=14, fontweight="bold", y=1.01)

for ax, feat in zip(axes1, FEATURES):
    data_by_scenario = [df[df["scenario"] == s][feat].values for s in scenario_order]
    bp = ax.boxplot(
        data_by_scenario, patch_artist=True,
        medianprops=dict(color="#333", linewidth=2),
        whiskerprops=dict(color="#666"),
        capprops=dict(color="#666"),
        flierprops=dict(marker="o", markersize=2, alpha=0.4, color="#666"),
    )
    for patch, scenario in zip(bp["boxes"], scenario_order):
        patch.set_facecolor(COLORS.get(scenario, "#888"))
        patch.set_alpha(0.75)
    ax.set_title(FEATURE_LABELS[feat], fontsize=11, fontweight="bold")
    ax.set_xticks(range(1, len(scenario_order) + 1))
    ax.set_xticklabels([s.replace(" ", "\n") for s in scenario_order], fontsize=7)
    ax.grid(axis="y", alpha=0.4)

legend_patches = [Patch(color=COLORS.get(s, "#888"), label=s) for s in scenario_order]
fig1.legend(handles=legend_patches, loc="lower center", ncol=4,
            fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.08))
fig1.tight_layout()
fig1.savefig("fig1_distributie_per_scenariu.png", dpi=150, bbox_inches="tight")
print("Salvat: fig1_distributie_per_scenariu.png")

# FIGURA 2 — Evolutie in timp cu mediere mobila
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig2.suptitle(f"Evolutie in timp per scenariu ",
              fontsize=14, fontweight="bold")

for ax, feat in zip(axes2, FEATURES):
    for scenario in scenario_order:
        sub = df[df["scenario"] == scenario].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("timestamp")
        sub["t_rel"] = (sub["timestamp"] - sub["timestamp"].iloc[0]).dt.total_seconds()
        color = COLORS.get(scenario, "#888")
        ax.plot(sub["t_rel"], sub[feat], color=color, alpha=0.15, linewidth=0.8)
        ax.plot(sub["t_rel"], sub[feat], label=scenario,
                color=color, alpha=0.9, linewidth=2.0)
    ax.set_ylabel(FEATURE_LABELS[feat], fontsize=10)
    ax.grid(alpha=0.4)

axes2[-1].set_xlabel("Timp relativ [secunde]", fontsize=10)
axes2[0].legend(fontsize=7, loc="upper right", framealpha=0.8)
fig2.tight_layout()
fig2.savefig("fig2_evolutie_timp.png", dpi=150, bbox_inches="tight")
print("Salvat: fig2_evolutie_timp.png")

# FIGURA 3 — Separabilitate foc vs non-foc (date originale)

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("Separabilitate: Foc (1) vs Normal (0)", fontsize=14, fontweight="bold")

for ax, feat in zip(axes3, FEATURES):
    for label, color in LABEL_COLORS.items():
        subset = df[df["label"] == label][feat]
        if feat == "tmax":
            ax.hist(subset, bins=np.linspace(20, 210, 80), alpha=0.65, color=color,
                    label="FOC" if label == 1 else "NORMAL",
                    edgecolor="white", linewidth=0.3, density=True)
        else:
            ax.hist(subset, bins=40, alpha=0.65, color=color,
                    label="FOC" if label == 1 else "NORMAL",
                    edgecolor="none", density=True)
    ax.set_title(FEATURE_LABELS[feat], fontsize=11, fontweight="bold")
    ax.set_xlabel("Valoare", fontsize=9)
    ax.set_ylabel("Densitate", fontsize=9)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9)
    if feat == "tmax":
        ax.set_xlim(20, 210)

fig3.tight_layout()
fig3.savefig("fig3_separabilitate_foc_normal.png", dpi=150, bbox_inches="tight")
print("Salvat: fig3_separabilitate_foc_normal.png")

# FIGURA 4 — Corelatii intre senzori (date originale)
fig4, axes4 = plt.subplots(3, 3, figsize=(12, 10))
fig4.suptitle("Corelatii intre senzori", fontsize=14, fontweight="bold")

for i, feat_y in enumerate(FEATURES):
    for j, feat_x in enumerate(FEATURES):
        ax = axes4[i][j]
        if i == j:
            for label, color in LABEL_COLORS.items():
                ax.hist(df[df["label"] == label][feat_x], bins=30,
                        alpha=0.6, color=color, density=True, edgecolor="none")
            ax.set_title(FEATURE_LABELS[feat_x], fontsize=9, fontweight="bold")
        else:
            for label, color in LABEL_COLORS.items():
                sub = df[df["label"] == label]
                ax.scatter(sub[feat_x], sub[feat_y], c=color, alpha=0.25, s=4,
                           label="FOC" if label == 1 else "NORMAL")
            r = df[[feat_x, feat_y]].corr().iloc[0, 1]
            ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes,
                    fontsize=8, color="#e65100", fontweight="bold")
        if i == 2: ax.set_xlabel(FEATURE_LABELS[feat_x], fontsize=8)
        if j == 0: ax.set_ylabel(FEATURE_LABELS[feat_y], fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

legend_patches2 = [
    Patch(color="#c62828", label="FOC (label=1)"),
    Patch(color="#1565c0", label="NORMAL (label=0)"),
]
fig4.legend(handles=legend_patches2, loc="lower center", ncol=2,
            fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
fig4.tight_layout()
fig4.savefig("fig4_corelatii_senzori.png", dpi=150, bbox_inches="tight")
print("Salvat: fig4_corelatii_senzori.png")

# FIGURA 5 — Comparatie brut vs Kalman per scenariu
fig5, axes5 = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig5.suptitle("Comparatie: date brute vs filtru Kalman",
              fontsize=14, fontweight="bold")

for ax, feat in zip(axes5, FEATURES):
    feat_k = f"{feat}_kalman"
    for scenario in scenario_order:
        sub = df[df["scenario"] == scenario].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("timestamp")
        sub["t_rel"] = (sub["timestamp"] - sub["timestamp"].iloc[0]).dt.total_seconds()
        color = COLORS.get(scenario, "#888")
        ax.plot(sub["t_rel"], sub[feat],   color=color, alpha=0.2, linewidth=0.8)
        ax.plot(sub["t_rel"], sub[feat_k], color=color, alpha=0.9,
                linewidth=2.0, label=scenario)
    ax.set_ylabel(FEATURE_LABELS[feat], fontsize=10)
    ax.grid(alpha=0.4)

axes5[-1].set_xlabel("Timp relativ [secunde]", fontsize=10)
axes5[0].legend(fontsize=7, loc="upper right", framealpha=0.8)
fig5.tight_layout()
fig5.savefig("fig5_kalman_vs_brut.png", dpi=150, bbox_inches="tight")
print("Salvat: fig5_kalman_vs_brut.png")

# FIGURA 6 — Separabilitate pe date Kalman
fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5))
fig6.suptitle("Separabilitate Foc vs Normal (date Kalman)",
              fontsize=14, fontweight="bold")

for ax, feat in zip(axes6, FEATURES):
    feat_k = f"{feat}_kalman"
    for label, color in LABEL_COLORS.items():
        subset = df[df["label"] == label][feat_k]
        if feat == "tmax":
            ax.hist(subset, bins=np.linspace(20, 210, 80), alpha=0.65, color=color,
                    label="FOC" if label == 1 else "NORMAL",
                    edgecolor="white", linewidth=0.3, density=True)
        else:
            ax.hist(subset, bins=40, alpha=0.65, color=color,
                    label="FOC" if label == 1 else "NORMAL",
                    edgecolor="none", density=True)
    ax.set_title(FEATURE_LABELS[feat], fontsize=11, fontweight="bold")
    ax.set_xlabel("Valoare", fontsize=9)
    ax.set_ylabel("Densitate", fontsize=9)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9)
    if feat == "tmax":
        ax.set_xlim(20, 210)

fig6.tight_layout()
fig6.savefig("fig6_separabilitate_kalman.png", dpi=150, bbox_inches="tight")
print("Salvat: fig6_separabilitate_kalman.png")

# SALVEAZA CSV CU COLOANE KALMAN
kalman_csv = CSV_FILE.replace(".csv", "_kalman.csv") if ".csv" in CSV_FILE else CSV_FILE + "_kalman.csv"
df.to_csv(kalman_csv, index=False)
print(f"\nDataset cu Kalman salvat: {kalman_csv}")
print("Foloseste acest CSV in anfis_train.py cu coloanele:")
print("  ir_raw_kalman, tmax_kalman, tmean_kalman")

# STATISTICI
print("\nStatistici rapide per label (date originale):")
print(df.groupby("label")[FEATURES].agg(["mean", "std"]).round(2))
print("\nStatistici rapide per label (date Kalman):")
print(df.groupby("label")[FEATURES_K].agg(["mean", "std"]).round(2))

plt.show()