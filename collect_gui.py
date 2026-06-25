import serial
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from datetime import datetime
import time

sys.stdout.reconfigure(encoding='utf-8')

SERIAL_PORT = "COM5"
BAUD_RATE = 460800

# ============================================
# SCENARII
# ============================================
scenarios = [
    # NON-FOC (label=0)
    ("Camera umbroasa (baseline)",  180, 0),
    ("Lumina solara directa",       180, 0),
    #("Uscator de par",              60, 0),
    #("Radiator electric pornit",    60, 0),
    ("Corp uman aproape senzor",    180, 0),
    # FOC (label=1)
    ("Lumanare 10cm",               180, 1),
    ("Lumanare 20cm",               180, 1),
    ("Lumanare 30cm",               180, 1),
    ("Hartie aprinsa",              30, 1),
    ("Flacara aragaz",              180, 1),
    ("Flacara alcool izopropilic",  40, 1),
]

# ============================================
# STARE GLOBALA
# ============================================
state = {
    "all_data":        [],
    "scenario_idx":    0,
    "collecting":      False,
    "waiting":         True,
    "count":           0,
    "start_time":      None,
    "done":            False,
}

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ============================================
# PARSARE
# Format ESP32 fara UV:
# parts[0] = ts
# parts[1] = ir_raw
# parts[2] = ir_avg
# parts[3] = tmin
# parts[4] = tmax
# parts[5] = tmean
# parts[6:] = frame (768 valori)
# Total: 6 + 768 = 774
# ============================================
def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != 774:
        return None
    try:
        frame_vals = np.array([float(x) for x in parts[6:]], dtype=np.float32)
        return {
            "ts":     int(parts[0]),
            "ir_raw": int(parts[1]),
            "ir_avg": int(parts[2]),
            "tmin":   float(parts[3]),
            "tmax":   float(parts[4]),
            "tmean":  float(parts[5]),
            "frame":  frame_vals.reshape((24, 32)),
        }
    except (ValueError, IndexError):
        return None

# ============================================
# FIGURA
# ============================================
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("#1e1e1e")

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[6, 1, 1],
    hspace=0.45, wspace=0.35
)

ax_cam  = fig.add_subplot(gs[0, 0])
ax_info = fig.add_subplot(gs[0, 1])
ax_prog = fig.add_subplot(gs[1, :])
ax_btn  = fig.add_subplot(gs[2, :])

for ax in [ax_cam, ax_info, ax_prog, ax_btn]:
    ax.set_facecolor("#2d2d2d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

# --- Camera termica ---
dummy = np.zeros((24, 32), dtype=np.float32)
im = ax_cam.imshow(dummy, cmap="inferno", interpolation="nearest", vmin=20, vmax=40)
cbar = plt.colorbar(im, ax=ax_cam)
cbar.set_label("°C", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
ax_cam.set_title("Camera termica MLX90640", color="white", fontsize=10)
ax_cam.tick_params(colors="white")

# --- Info text ---
ax_info.axis("off")
info_text = ax_info.text(
    0.05, 0.95, "Asteptare date...",
    transform=ax_info.transAxes,
    va="top", ha="left",
    color="white", fontsize=9,
    fontfamily="monospace",
    wrap=True
)
ax_info.set_title("Senzori", color="white", fontsize=10)

# --- Progress bar ---
ax_prog.set_xlim(0, 1)
ax_prog.set_ylim(0, 1)
ax_prog.axis("off")
prog_bg   = ax_prog.barh(0.5, 1.0, height=0.6, color="#444", align="center")
prog_bar  = ax_prog.barh(0.5, 0.0, height=0.6, color="#e05c00", align="center")
prog_text = ax_prog.text(
    0.5, 0.5, "Scenariu 0/0 | Asteapta...",
    ha="center", va="center",
    color="white", fontsize=9, fontweight="bold",
    transform=ax_prog.transAxes
)

# --- Buton START ---
btn_ax = plt.axes([0.35, 0.04, 0.30, 0.07])
btn_ax.set_facecolor("#2d2d2d")
button = Button(btn_ax, "► START / NEXT", color="#e05c00", hovercolor="#ff7f2a")
button.label.set_color("white")
button.label.set_fontweight("bold")

# ============================================
# LOGICA BUTON
# ============================================
def on_button_click(event):
    if state["done"]:
        return

    idx = state["scenario_idx"]
    if idx >= len(scenarios):
        finish_collection()
        return

    name, duration, label = scenarios[idx]
    state["collecting"]       = True
    state["waiting"]          = False
    state["start_time"]       = time.time()
    state["count"]            = 0
    state["current_name"]     = name
    state["current_duration"] = duration
    state["current_label"]    = label

    print(f"\n[START] Scenariu {idx+1}/{len(scenarios)}: {name} ({duration}s)")

button.on_clicked(on_button_click)

# ============================================
# SALVARE & FINAL
# ============================================
def finish_collection():
    state["done"] = True
    ser.close()

    df = pd.DataFrame(state["all_data"])
    filename = f"fire_detection_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

    print("\n" + "=" * 70)
    print("COLECTARE COMPLETA!")
    print(f"Total sample-uri: {len(state['all_data'])}")
    print(f"  Foc    (label=1): {sum(1 for d in state['all_data'] if d['label'] == 1)}")
    print(f"  Normal (label=0): {sum(1 for d in state['all_data'] if d['label'] == 0)}")
    print(f"Dataset salvat: {filename}")

    if not df.empty:
        print("\nSTATISTICI:")
        print(df.groupby('label')[['ir_raw', 'tmax']].agg(['mean', 'std', 'min', 'max']))

    prog_text.set_text("COLECTARE COMPLETA!")
    button.label.set_text("DONE")
    fig.canvas.draw_idle()

# ============================================
# ANIMATIE
# ============================================
def update(_):
    if state["done"]:
        return [im]

    parsed = None
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("READY") or line.startswith("ERROR"):
            continue
        p = parse_line(line)
        if p:
            parsed = p
            break

    if parsed:
        frame = parsed["frame"]
        im.set_data(frame)
        im.set_clim(vmin=np.min(frame), vmax=np.max(frame))

        info_str = (
            f"IR raw : {parsed['ir_raw']}\n"
            f"IR avg : {parsed['ir_avg']}\n"
            f"Tmin   : {parsed['tmin']:.2f} C\n"
            f"Tmax   : {parsed['tmax']:.2f} C\n"
            f"Tmean  : {parsed['tmean']:.2f} C"
        )
        info_text.set_text(info_str)

        if state["collecting"]:
            record = {
                "ir_raw":    parsed["ir_raw"],
                "ir_avg":    parsed["ir_avg"],
                "tmin":      parsed["tmin"],
                "tmax":      parsed["tmax"],
                "tmean":     parsed["tmean"],
                "label":     state["current_label"],
                "scenario":  state["current_name"],
                "timestamp": datetime.now().isoformat(),
            }
            state["all_data"].append(record)
            state["count"] += 1

    if state["collecting"]:
        elapsed   = time.time() - state["start_time"]
        duration  = state["current_duration"]
        progress  = min(elapsed / duration, 1.0)
        remaining = max(duration - elapsed, 0)

        prog_bar[0].set_width(progress)
        idx = state["scenario_idx"] + 1
        prog_text.set_text(
            f"Scenariu {idx}/{len(scenarios)}: {state['current_name']} | "
            f"{state['count']} samples | Timp ramas: {remaining:.0f}s"
        )

        if elapsed >= duration:
            print(f"[DONE] {state['current_name']}: {state['count']} sample-uri colectate")
            state["collecting"]   = False
            state["waiting"]      = True
            state["scenario_idx"] += 1

            if state["scenario_idx"] >= len(scenarios):
                finish_collection()
            else:
                next_name = scenarios[state["scenario_idx"]][0]
                prog_text.set_text(
                    f"Scenariu {state['scenario_idx']+1}/{len(scenarios)}: "
                    f"{next_name} | Apasa START pentru a continua"
                )
                button.label.set_text("► NEXT SCENARIU")

    elif not state["done"] and state["scenario_idx"] == 0:
        prog_text.set_text(
            f"Scenariu 1/{len(scenarios)}: {scenarios[0][0]} | Apasa START"
        )

    return [im]

ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
plt.show()