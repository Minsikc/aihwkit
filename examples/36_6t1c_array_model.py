# type: ignore
# pylint: disable-all
# -*- coding: utf-8 -*-

"""aihwkit example 36: 6T-1C capacitor-synapse array model (load law + half-select).

Builds a 5x5 analog tile that behaves like a measured 6T-1C array and walks through the three
array-level effects this fork adds to the pulsed update (CPU, sparse pulse types):

1. Per-cell gain map
   Every cell has its own step per coincidence (``dwmin_up`` / ``dwmin_down`` hidden parameters).

2. Load law (``load_law`` of ``LinearStepDevice``)
   In every bit-line slot the step of a coincident cell is multiplied by

       f = 1 / (1 + s * (a_row * (n_row - 1)^p_row + a_col * (n_col - 1) + a_other * n_other))
       s = (mean gain / gain of the cell)^beta

   n_row / n_col = coincidences sharing the cell's row / column in that slot (itself included),
   n_other = coincidences on neither line. Driving more cells at once makes every step smaller,
   rows load more than columns, and weak cells suffer more. All coefficients 0 (or
   ``load_law=False``) gives a tile without any load dependence.

3. Line-state half-select (``hs_mode`` 2 / 3, ``hs_rate``, ``hs_attr_up``, ``hs_attr_down``)
   A cell that sees only its row line or only its column line is half-selected. A single line does
   nothing, but a CHANGE of the last line that pulsed on the cell acts like a weak pulse of the
   ordered pair:  w <- A + (w - A) * (1 - hs_rate)
     NORMAL opcodes (``hs_mode=2``): lines N1 (row, potentiation), N2 (column, potentiation),
       N3 (row, depression), N4 (column, depression). N1<->N2 pulls toward the upper rail
       (``hs_attr_up``), N3<->N4 toward the lower rail (``hs_attr_down``), the decay pairs
       N1<->N3 / N2<->N4 toward 0, N1<->N4 / N2<->N3 do nothing.
     DNO opcodes (``hs_mode=3``): a row line is driven in every slot (row bit 0 -> the complementary
       line), so the state is N1 or N3 by the row alone and every change decays the row toward 0.

The array is driven like the hardware: sign-pure rank-1 commands (rows = tile output side ``d``,
columns = tile input side ``x``) whose pulse probabilities are the vector entries; a signed outer
product is split into four such commands. ``tile.tile.set_hs_polarity(+1 / -1)`` tells the
half-select model whether a command is a potentiation or a depression.

Numbers below are a fit of one 5x5 array (pulse width 3 us, 10 slots per command, 1 weight unit =
455 ADC LSB). They illustrate the model; use your own measurements for another array.
"""
# pylint: disable=invalid-name

import numpy as np
import torch

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.parameters.enums import PulseType
from aihwkit.simulator.tiles import AnalogTile

N = 5
LSB_PER_W = 455.0  # ADC LSB per weight unit

# step of a cell driven alone, LSB per coincidence (rows x columns)
GAIN_POT = np.array(
    [
        [12.26, 15.55, 14.45, 17.86, 19.57],
        [11.44, 12.99, 13.64, 14.95, 17.96],
        [11.32, 11.68, 11.34, 14.26, 16.92],
        [10.54, 11.50, 11.41, 12.95, 15.83],
        [8.41, 10.41, 10.91, 12.40, 14.51],
    ]
)
GAIN_DEP = np.array(
    [
        [13.08, 15.68, 15.19, 17.84, 19.70],
        [11.02, 14.42, 13.50, 15.18, 19.66],
        [10.92, 12.33, 12.33, 13.47, 17.24],
        [8.90, 10.05, 11.71, 13.69, 15.56],
        [8.41, 9.39, 10.50, 12.35, 14.04],
    ]
)
LOAD = dict(load_a_row=0.130, load_p_row=1.31, load_a_col=0.046, load_a_other=0.050, load_beta=1.36)
HS_RATE = {"normal": 0.0036, "dno": 0.0042}  # fraction of the distance to the attractor per transition
HS_ATTR_UP, HS_ATTR_DOWN = 520.0, -498.0  # NORMAL pair attractors (the rails), LSB


def make_tile(load=True, per_cell_gain=True, half_select=None, bl=10):
    """5x5 tile of the array model.

    load:          apply the load law
    per_cell_gain: use the measured gain maps (False: one uniform gain)
    half_select:   None | "normal" | "dno"
    bl:            slots per command (pulse probability = |vector entry|)
    """
    up, down = GAIN_POT / LSB_PER_W, GAIN_DEP / LSB_PER_W
    if not per_cell_gain:
        up, down = np.full((N, N), up.mean()), np.full((N, N), down.mean())
    dno = half_select == "dno"
    hs = {}
    if half_select is not None:
        hs = dict(
            hs_mode=3 if dno else 2,
            hs_rate=HS_RATE[half_select],
            hs_attr_up=HS_ATTR_UP / LSB_PER_W,
            hs_attr_down=HS_ATTR_DOWN / LSB_PER_W,
        )
    device = LinearStepDevice(
        dw_min=float((up + down).mean() / 2),
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max=1.5,
        w_min=-1.5,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        gamma_up=0.0,  # no state dependence of the step in this example
        gamma_down=0.0,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
        write_noise_std=0.0,
        load_law=load,
        load_dno=load and dno,  # DNO also loads the lines of the undriven rows
        **(LOAD if load else {}),
        **hs,
    )
    rpu_config = SingleRPUConfig(device=device)
    # the load law and the half-select model need a sparse pulse type
    rpu_config.update.pulse_type = (
        PulseType.HALFSELECTED_STOCHASTIC if half_select else PulseType.STOCHASTIC_COMPRESSED
    )
    rpu_config.update.desired_bl = bl
    rpu_config.update.update_bl_management = False
    rpu_config.update.update_management = False

    tile = AnalogTile(N, N, rpu_config)
    hidden = tile.get_hidden_parameters()
    hidden["dwmin_up"] = torch.tensor(up, dtype=torch.float32)
    hidden["dwmin_down"] = torch.tensor(down, dtype=torch.float32)
    tile.set_hidden_parameters(hidden)
    tile.set_learning_rate(device.dw_min * bl)  # -> pulse probability = |x|, |d|
    tile.set_weights(torch.zeros(N, N))
    if half_select is not None:
        tile.tile.enable_hs_tracking()
    return tile


def lsb(tile):
    """Weights in ADC LSB."""
    return tile.get_weights()[0].numpy().astype(float) * LSB_PER_W


def set_lsb(tile, value):
    tile.set_weights(torch.full((N, N), value / LSB_PER_W))


def command(tile, row_p, col_p, potentiate=True):
    """One sign-pure rank-1 command: row / column pulse probabilities in [0, 1]."""
    rows = torch.as_tensor(np.asarray(row_p), dtype=torch.float32)[None, :]
    cols = torch.as_tensor(np.asarray(col_p), dtype=torch.float32)[None, :]
    tile.tile.set_hs_polarity(1 if potentiate else -1)
    if not (rows.any() or cols.any()):
        tile.tile.apply_hs_idle_slot()  # nothing fires: only the DNO row drive reacts
        return
    tile.update(cols, -rows if potentiate else rows)  # aihwkit applies W -= lr * d x^T


def signed_update(tile, u, v):
    """Signed outer product u v^T as four sign-pure commands (potentiation for equal signs)."""
    u, v = np.clip(u, -1, 1), np.clip(v, -1, 1)
    up, un, vp, vn = np.maximum(u, 0), np.maximum(-u, 0), np.maximum(v, 0), np.maximum(-v, 0)
    for rows, cols, pot in ((up, vp, True), (un, vn, True), (up, vn, False), (un, vp, False)):
        if rows.max() > 0 and cols.max() > 0:
            command(tile, rows, cols, pot)


def one_hot(*idx):
    out = np.zeros(N)
    out[list(idx)] = 1.0
    return out


# ---------------------------------------------------------------------------------------
print("1) Load law: step of cell (2, 2) per coincidence when more cells are driven with it")
print("   (probability-1 pulses, 10 slots; 'no load' = load_law off)")
patterns = (
    ("cell alone", one_hot(2), one_hot(2)),
    ("2 cells in its row", one_hot(2), one_hot(2, 3)),
    ("whole row", one_hot(2), np.ones(N)),
    ("whole column", np.ones(N), one_hot(2)),
    ("whole array", np.ones(N), np.ones(N)),
)
for name, rows, cols in patterns:
    steps = []
    for load in (True, False):
        tile = make_tile(load=load)
        command(tile, rows, cols)
        steps.append(lsb(tile)[2, 2] / 10)
    print(f"   {name:20s} {steps[0]:6.2f} LSB   (no load {steps[1]:5.2f})   f = {steps[0] / steps[1]:.2f}")

tile = make_tile()
command(tile, np.ones(N), np.ones(N))
f_map = lsb(tile) / 10 / GAIN_POT
print(
    f"   whole array, per cell: f from {f_map.min():.2f} (weakest cell, gain {GAIN_POT.min():.1f}) "
    f"to {f_map.max():.2f} (strongest, gain {GAIN_POT.max():.1f})"
)

# ---------------------------------------------------------------------------------------
print("\n2) Signed stochastic rank-1 updates: how well does the array follow u v^T ?")
rng = np.random.default_rng(0)
vectors = [(rng.uniform(-0.7, 0.7, N), rng.uniform(-0.7, 0.7, N)) for _ in range(200)]
torch.manual_seed(0)
variants = (
    ("uniform gain, no load", dict(load=False, per_cell_gain=False)),
    ("per-cell gain", dict(load=False)),
    ("per-cell gain + load law", dict(load=True)),
    ("... + half-select (NORMAL)", dict(load=True, half_select="normal")),
)
for name, kwargs in variants:
    target, got = [], []
    for u, v in vectors:
        tile = make_tile(**kwargs)
        signed_update(tile, u, v)
        target.append(np.outer(u, v).ravel())
        got.append(lsb(tile).ravel())
    target, got = np.concatenate(target), np.concatenate(got)
    gain = (target @ got) / (target @ target) / 10
    print(f"   {name:28s} correlation {np.corrcoef(target, got)[0, 1]:.3f}   mean gain {gain:5.2f} LSB/coincidence")

# ---------------------------------------------------------------------------------------
print("\n3) Half-select, NORMAL opcodes: alternate two lines on cell (0, 0) without ever selecting it")
print("   (deterministic single-slot commands; start level in LSB -> level after 200 / 1000 transitions)")
row_only, col_only, none = (one_hot(0), np.zeros(N)), (np.zeros(N), one_hot(0)), None
sequences = (
    ("N1 / N2  (row, column of a potentiation)", ((row_only, True), (col_only, True)), HS_ATTR_UP),
    ("N3 / N4  (row, column of a depression)", ((row_only, False), (col_only, False)), HS_ATTR_DOWN),
    ("N1 / N3  (row line, pot. then dep.)", ((row_only, True), (row_only, False)), 0.0),
    ("N2 / N4  (column line, pot. then dep.)", ((col_only, True), (col_only, False)), 0.0),
    ("N1 / N4  (inert pair)", ((row_only, True), (col_only, False)), None),
    ("N1 / N1  (same line)", ((row_only, True), (row_only, True)), None),
)
for name, pair, attractor in sequences:
    out = []
    for start in (300.0, -300.0):
        levels = []
        for n_transitions in (200, 1000):
            tile = make_tile(half_select="normal", bl=1)
            set_lsb(tile, start)
            for k in range(n_transitions + 1):
                (rows, cols), pot = pair[k % 2]
                command(tile, rows, cols, pot)
            levels.append(lsb(tile)[0, 0])
        out.append(f"{start:+5.0f} -> {levels[0]:+5.0f} / {levels[1]:+5.0f}")
    target = "no effect" if attractor is None else f"attractor {attractor:+.0f}"
    print(f"   {name:42s} {out[0]}   {out[1]}   ({target})")

# ---------------------------------------------------------------------------------------
print("\n4) Half-select, DNO opcodes: a row whose bit toggles decays as a whole toward 0")
tile = make_tile(half_select="dno", bl=1)
set_lsb(tile, 300.0)
for k in range(200):
    command(tile, one_hot(0) * (k % 2 == 0), np.zeros(N))  # row 0: bit 1, 0, 1, 0, ...; no column pulse
w = lsb(tile)
print(f"   row 0 (bit toggles): {w[0].round(0)}   closed form {300 * (1 - HS_RATE['dno']) ** 199:.0f}")
print(f"   row 1 (bit always 0, line never changes): {w[1].round(0)}")

# ---------------------------------------------------------------------------------------
print("\n5) What the half-select does to an accumulated value during further stochastic commands")
print("   (cell values start at +200 LSB; the same 50 random signed updates with |u|, |v| <= 0.5)")
rng = np.random.default_rng(1)
vectors = [(rng.uniform(-0.5, 0.5, N), rng.uniform(-0.5, 0.5, N)) for _ in range(50)]
for name, half_select in (("no half-select", None), ("NORMAL", "normal"), ("DNO", "dno")):
    torch.manual_seed(1)
    tile = make_tile(half_select=half_select)
    set_lsb(tile, 200.0)
    for u, v in vectors:
        signed_update(tile, u, v)
    print(f"   {name:15s} mean level {lsb(tile).mean():+7.1f} LSB")
