import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. INCARCARE DATE
# ============================================================
CSV_FILE = "dataset_01_06_kalman.csv"

df = pd.read_csv(CSV_FILE)

print(f"Dataset incarcat: {len(df)} samples")
print(f"  Foc    (label=1): {sum(df['label'] == 1)}")
print(f"  Normal (label=0): {sum(df['label'] == 0)}")
print(f"\nScenarii incluse:")
print(df.groupby(["scenario", "label"]).size())

# ============================================================
# 2. PREPROCESARE
# ============================================================
FEATURES = ["ir_raw_kalman", "tmax_kalman", "tmean_kalman"]

X = df[FEATURES].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

MINS = np.array([0.0, 20.0, 20.0], dtype=np.float32)
MAXS = np.array([4095.0, 300.0, 65.0], dtype=np.float32)

print(f"\nNormalizare:")
for i, feat in enumerate(FEATURES):
    print(f"  {feat}: [{MINS[i]:.1f}, {MAXS[i]:.1f}]")

def scale_manual(X, mins=MINS, maxs=MAXS):
    return (X - mins) / (maxs - mins)

X_scaled = scale_manual(X)
X_scaled = np.clip(X_scaled, 0, 1)

norm_params = {"mins": MINS, "maxs": MAXS, "features": FEATURES}
with open("anfis_scaler.pkl", "wb") as f:
    pickle.dump(norm_params, f)
print("Parametrii normalizare salvati: anfis_scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# 3. IMPLEMENTARE ANFIS — Algoritm Hibrid GD + LSE
#
#  Ordinea corecta per epoca:
#    1. Forward pass (cu MF-urile curente)
#    2. GD pe eroarea INAINTE de LSE (semnal mare)
#    3. Forward pass din nou (cu MF-urile actualizate)
#    4. LSE gaseste p optim pentru noile MF-uri
# ============================================================
class ANFIS:
    def __init__(self, n_inputs=3, n_mf=3, epochs=30,
                 lr=0.01):
        self.n_inputs    = n_inputs
        self.n_mf        = n_mf
        self.n_rules     = n_mf ** n_inputs  # 27
        self.epochs      = epochs
        self.lr          = lr
        self.loss_history = []

        np.random.seed(42)

        # Initializare uniforma: centre la distante egale
        # GD va deplasa centrele spre pozitiile optime
        self.c = np.array([
            [0.25, 0.50, 0.75],   # IR
            [0.25, 0.50, 0.75],   # Tmax
            [0.25, 0.50, 0.75],   # Tmean
        ], dtype=np.float32)

        # Sigma: 80% din distanta dintre centre
        self.sigma = np.array([
            [0.10, 0.10, 0.10],   # IR
            [0.10, 0.10, 0.10],   # Tmax
            [0.10, 0.10, 0.10],   # Tmean
        ], dtype=np.float32)

        self.p = np.zeros(self.n_rules, dtype=np.float64)

        self.rule_index = np.array(
            np.meshgrid(*[range(n_mf)] * n_inputs)
        ).T.reshape(-1, n_inputs)

    def _gaussian_mf(self, x, sigma, c):
        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2 + 1e-9))

    def _layer1(self, X):
        n = X.shape[0]
        mu = np.zeros((n, self.n_inputs, self.n_mf))
        for i in range(self.n_inputs):
            for k in range(self.n_mf):
                mu[:, i, k] = self._gaussian_mf(
                    X[:, i], self.sigma[i, k], self.c[i, k])
        return mu

    def _layer2(self, mu):
        n = mu.shape[0]
        w = np.ones((n, self.n_rules))
        for r, rule in enumerate(self.rule_index):
            for i, k in enumerate(rule):
                w[:, r] *= mu[:, i, k]
        return w

    def _layer3(self, w):
        return w / (w.sum(axis=1, keepdims=True) + 1e-9)

    def _layer4_5(self, w_bar):
        return w_bar @ self.p

    def _forward(self, X):
        mu    = self._layer1(X)
        w     = self._layer2(mu)
        w_bar = self._layer3(w)
        y_hat = self._layer4_5(w_bar)
        return mu, w, w_bar, y_hat

    def predict_proba(self, X):
        _, _, _, y_hat = self._forward(X)
        return np.clip(y_hat, 0, 1)

    def predict(self, X, threshold=0.55):
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X, y):
        print(f"\nAntrenare ANFIS: GD+LSE | {self.n_rules} reguli | "
              f"{self.epochs} epoci | lr={self.lr}")
        print("-" * 60)

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        prev_loss = float('inf')

        # Prima iteratie: LSE initial pentru a avea p nenul
        mu, w, w_bar, _ = self._forward(X)
        self.p = np.linalg.lstsq(w_bar, y, rcond=None)[0]

        for epoch in range(self.epochs):

            # 1. FORWARD PASS (cu MF-urile curente)
            mu, w, w_bar, y_hat = self._forward(X)
            loss = np.mean((y - y_hat) ** 2)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < 1e-12 and epoch > 10:
                print(f"Convergenta la epoca {epoch + 1} | Loss: {loss:.8f}")
                break
            prev_loss = loss

            # 2. BACKWARD PASS — GD pe eroarea INAINTE de LSE
            #    Eroarea e mare => gradientii sunt semnificativi
            error = y - y_hat

            for r, rule in enumerate(self.rule_index):
                d_wbar = -2.0 * error * self.p[r]

                for i, k in enumerate(rule):
                    w_sum = w.sum(axis=1) + 1e-9
                    d_w   = d_wbar * (w_sum - w[:, r]) / (w_sum ** 2)
                    other = w[:, r] / (mu[:, i, k] + 1e-9)
                    d_mu  = d_w * other

                    diff  = X[:, i] - self.c[i, k]
                    d_c   = d_mu * mu[:, i, k] * diff / (self.sigma[i, k] ** 2 + 1e-9)
                    d_sig = d_mu * mu[:, i, k] * (diff ** 2) / (self.sigma[i, k] ** 3 + 1e-9)

                    self.c[i, k]     -= self.lr * d_c.mean()
                    self.sigma[i, k] -= self.lr * d_sig.mean()
                    self.sigma[i, k]  = max(self.sigma[i, k], 0.01)

            # 3. FORWARD PASS DIN NOU (cu MF-urile actualizate)
            mu, w, w_bar, _ = self._forward(X)

            # 4. LSE — gaseste p optim pentru noile MF-uri
            self.p = np.linalg.lstsq(w_bar, y, rcond=None)[0]

            if (epoch + 1) % 20 == 0:
                y_hat_new = self._layer4_5(w_bar)
                loss_new = np.mean((y - y_hat_new) ** 2)
                acc = np.mean(self.predict(X) == y.astype(int))
                print(f"Epoca {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {loss_new:.8f} | Acc train: {acc:.3f} | "
                      f"c_range: [{self.c.min():.3f}, {self.c.max():.3f}]")

        print("-" * 60)
        print(f"Antrenare completa! Loss final: {self.loss_history[-1]:.8f}")
        print(f"p_i:   min={self.p.min():.4f}, max={self.p.max():.4f}")
        print(f"c:     \n{np.round(self.c, 3)}")
        print(f"sigma: \n{np.round(self.sigma, 3)}")

# ============================================================
# 4. ANTRENARE
# ============================================================
model = ANFIS(n_inputs=3, n_mf=3, epochs=30, lr=0.01)
model.fit(X_train, y_train)

# ============================================================
# 5. EVALUARE
# ============================================================
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\nREZULTATE PE TEST SET:")
print("=" * 50)
print(classification_report(y_test.astype(int), y_pred,
      target_names=["Normal", "Foc"]))

# ============================================================
# 6. GRAFICE
# ============================================================
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#333",
    "xtick.color":      "#333",
    "ytick.color":      "#333",
    "text.color":       "#333",
    "grid.color":       "#ccc",
    "grid.alpha":       0.5,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("ANFIS (3 intrari, 3 MF, GD+LSE) - Rezultate antrenare",
             fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.plot(model.loss_history, color="#e65100", linewidth=2)
ax.set_title("Curba de invatare (MSE Loss)", fontweight="bold")
ax.set_xlabel("Epoca"); ax.set_ylabel("MSE")
ax.grid(alpha=0.4)

ax = axes[0, 1]
cm = confusion_matrix(y_test.astype(int), y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Foc"])
disp.plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title("Confusion Matrix", fontweight="bold")

ax = axes[1, 0]
ax.hist(y_proba[y_test == 0], bins=30, alpha=0.7,
        color="#1565c0", label="Normal", density=True)
ax.hist(y_proba[y_test == 1], bins=30, alpha=0.7,
        color="#c62828", label="Foc", density=True)
ax.axvline(0.5, color="#333", linestyle="--", linewidth=1.5,
           label="Threshold 0.5")
ax.set_title("Distributie scoruri ANFIS", fontweight="bold")
ax.set_xlabel("Score ANFIS [0..1]"); ax.set_ylabel("Densitate")
ax.legend(); ax.grid(alpha=0.4)

ax = axes[1, 1]
correct   = y_pred == y_test.astype(int)
incorrect = ~correct
ax.scatter(X_test[correct, 0]   * (MAXS[0]-MINS[0]) + MINS[0],
           X_test[correct, 1]   * (MAXS[1]-MINS[1]) + MINS[1],
           c=["#c62828" if l==1 else "#1565c0" for l in y_test[correct]],
           alpha=0.4, s=10, label="Corect")
ax.scatter(X_test[incorrect, 0] * (MAXS[0]-MINS[0]) + MINS[0],
           X_test[incorrect, 1] * (MAXS[1]-MINS[1]) + MINS[1],
           c="black", alpha=0.9, s=40, marker="x", label="Gresit")
ax.set_title("Predictii test set (x = gresit)", fontweight="bold")
ax.set_xlabel("IR Raw"); ax.set_ylabel("Tmax [°C]")
ax.legend(fontsize=8); ax.grid(alpha=0.4)

fig.tight_layout()
fig.savefig("anfis_rezultate.png", dpi=150, bbox_inches="tight")
print("\nGrafic salvat: anfis_rezultate.png")

# Membership Functions dupa antrenare
feature_names = ["IR (norm.)", "Tmax (norm.)", "Tmean (norm.)"]
mf_labels     = ["LOW", "MEDIUM", "HIGH"]
colors_mf     = ["#1565c0", "#e65100", "#2e7d32"]
x_real_labels = {
    "IR (norm.)":    (MINS[0], MAXS[0], "IR Raw (ADC)"),
    "Tmax (norm.)":  (MINS[1], MAXS[1], "Tmax [°C]"),
    "Tmean (norm.)": (MINS[2], MAXS[2], "Tmean [°C]"),
}

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
fig2.suptitle("Membership Functions dupa antrenare (GD+LSE) — 3 intrari, 3 MF",
              fontsize=12, fontweight="bold")

for idx, (ax, fname) in enumerate(zip(axes2, feature_names)):
    x_norm = np.linspace(0, 1, 300)
    mn, mx, xlabel = x_real_labels[fname]
    x_real = x_norm * (mx - mn) + mn
    for k in range(model.n_mf):
        mf_vals = model._gaussian_mf(x_norm, model.sigma[idx,k], model.c[idx,k])
        c_real  = model.c[idx,k] * (mx - mn) + mn
        ax.plot(x_real, mf_vals, color=colors_mf[k], linewidth=2,
                label=f"{mf_labels[k]}: c={c_real:.0f}, s={model.sigma[idx,k]:.3f}")
    ax.set_title(f"MF — {fname}", fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("Grad apartenenta μ")
    ax.legend(fontsize=7); ax.grid(alpha=0.4)
    ax.set_ylim(0, 1.1)

fig2.tight_layout()
fig2.savefig("anfis_mf.png", dpi=150, bbox_inches="tight")
print("Grafic salvat: anfis_mf.png")

# ============================================================
# 7. SALVEAZA MODELUL
# ============================================================
with open("anfis_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel salvat: anfis_model.pkl")

# ============================================================
# 8. EXPORT AUTOMAT anfis_params.h PENTRU ESP32
# ============================================================
def export_anfis_params_h(model, mins, maxs, filename="anfis_params.h"):
    def fmt_2d(arr):
        rows = []
        for row in arr:
            vals = ", ".join(f"{v:.6f}f" for v in row)
            rows.append(f"  {{{vals}}}")
        return ",\n".join(rows)

    def fmt_rules(arr):
        rows = []
        for row in arr:
            vals = ", ".join(str(v) for v in row)
            rows.append(f"  {{{vals}}}")
        return ",\n".join(rows)

    p_rows = []
    for i in range(0, len(model.p), 3):
        chunk = model.p[i:i+3]
        row = ",   ".join(f"{v:+.8e}f" for v in chunk)
        p_rows.append(f"    {row}")
    p_str = ",\n".join(p_rows)

    h = f"""// ================================================================
// anfis_params.h
// Parametrii ANFIS exportati automat din anfis_train.py
// Arhitectura: {model.n_inputs} intrari | {model.n_mf} MF gaussiene | {model.n_rules} reguli
// Algoritm: GD + LSE (hibrid, GD aplicat inaintea LSE)
// Intrari: IR_raw, Tmax_kalman, Tmean_kalman
// p_i: min={model.p.min():.4f}, max={model.p.max():.4f}
// ================================================================

#pragma once

#define ANFIS_N_INPUTS  {model.n_inputs}
#define ANFIS_N_MF      {model.n_mf}
#define ANFIS_N_RULES   {model.n_rules}
#define ANFIS_WSUM_GUARD 1e-6f

// -- Parametrii de normalizare --------------------------------
const float ANFIS_MINS[ANFIS_N_INPUTS] = {{{mins[0]:.4f}f, {mins[1]:.4f}f, {mins[2]:.4f}f}};
const float ANFIS_MAXS[ANFIS_N_INPUTS] = {{{maxs[0]:.4f}f, {maxs[1]:.4f}f, {maxs[2]:.4f}f}};

// -- Centre gaussiene [n_inputs][n_mf] -----------------------
// IR:    LOW={model.c[0,0]:.3f}, MED={model.c[0,1]:.3f}, HIGH={model.c[0,2]:.3f}
// Tmax:  LOW={model.c[1,0]:.3f}, MED={model.c[1,1]:.3f}, HIGH={model.c[1,2]:.3f}
// Tmean: LOW={model.c[2,0]:.3f}, MED={model.c[2,1]:.3f}, HIGH={model.c[2,2]:.3f}
const float ANFIS_C[ANFIS_N_INPUTS][ANFIS_N_MF] = {{
{fmt_2d(model.c)}
}};

// -- Latimi gaussiene [n_inputs][n_mf] -----------------------
const float ANFIS_SIGMA[ANFIS_N_INPUTS][ANFIS_N_MF] = {{
{fmt_2d(model.sigma)}
}};

// -- Parametrii consecinta [n_rules] -------------------------
const float ANFIS_P[ANFIS_N_RULES] = {{
{p_str}
}};

// -- Indecsi reguli [n_rules][n_inputs] ----------------------
const int ANFIS_RULES[ANFIS_N_RULES][ANFIS_N_INPUTS] = {{
{fmt_rules(model.rule_index)}
}};
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(h)
    print(f"\nExportat: {filename}")
    print(f"  n_inputs={model.n_inputs}, n_mf={model.n_mf}, n_rules={model.n_rules}")
    print(f"  MINS={mins}")
    print(f"  MAXS={maxs}")
    print(f"  p_i: min={model.p.min():.4f}, max={model.p.max():.4f}")
    print(f"  Copiaza {filename} in folderul proiectului Arduino")

export_anfis_params_h(model, MINS, MAXS, filename="anfis_params.h")

plt.show()