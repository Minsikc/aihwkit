# 6T-1C array model: load law and line-state half-select

This fork adds two array-level effects of a 6T-1C capacitor-synapse crossbar to the pulsed update
of aihwkit. Both were fitted to measurements of 5x5 arrays and both act per bit-line slot, so they
need a sparse pulse type (`STOCHASTIC_COMPRESSED`, `HALFSELECTED_STOCHASTIC`, ...). CPU only.

Runnable walk-through: [`examples/36_6t1c_array_model.py`](examples/36_6t1c_array_model.py).

## 1. Load law (`load_law`, `LinearStepDevice`)

The step of a cell gets smaller when more cells are written in the same slot. For a coincident
cell `(i, j)`:

```
dw   = f_ij * g_ij * (1 - gamma * w / w_bound)        # g_ij: step of the cell driven alone
f_ij = 1 / (1 + s_ij * L)
L    = load_a_row * (n_row - 1)^load_p_row + load_a_col * (n_col - 1) + load_a_other * n_other
s_ij = (mean(g) / g_ij)^load_beta
```

| symbol | meaning |
|---|---|
| `n_row`, `n_col` | coincidences of this slot in the cell's row / column (the cell itself included) |
| `n_other` | coincidences of this slot on neither of the two lines |
| `s_ij` | weak-cell factor: cells with a small gain lose more under load (`load_beta = 0` disables it) |
| `load_dno` | DNO opcodes: the undriven rows are driven with the complementary line, so their cells on the active columns count as column / other load |

`f` multiplies both the scale and the slope of `LinearStepDevice`, so the state dependence
(`gamma_up`, `gamma_down`) stays independent of the load. Line counts ignore the pulse sign: drive
sign-pure commands (all coincidences of a command potentiate, or all depress), as the hardware does.

Setting `load_a_row = load_a_col = load_a_other = 0` (or `load_law=False`) removes the load
dependence. `load_p_row` is an exponent: use 1 for a linear row term, not 0.

Per-cell gains are ordinary hidden parameters (`dwmin_up`, `dwmin_down`).

Fit of one array (pulse width 3 us, 10 slots, steps of 8-20 LSB per coincidence):
`a_row 0.130, p_row 1.31, a_col 0.046, a_other 0.050, beta 1.36`. An average cell keeps 88 % of its
step with one more cell in its row, 55 % with the whole row, 84 % with the whole column and 36 %
when all 25 cells are written at once.

## 2. Line-state half-select (`hs_mode` 2 / 3)

A half-selected cell sees a pulse on its row line or on its column line, but not both. One line
alone does (almost) nothing. What moves the cell is a **change** of the last line that pulsed on it:
two different lines in sequence act like a weak pulse of that ordered pair. Measured on the array,
every such transition moves the cell a fixed fraction of the way to an attractor that depends on
the pair:

```
w <- A + (w - A) * (1 - hs_rate)
```

Lines: N1 = row line of a potentiation, N2 = column line of a potentiation, N3 = row line of a
depression, N4 = column line of a depression.

| pair | `hs_mode = 2` (NORMAL opcodes) | `hs_mode = 3` (DNO opcodes) |
|---|---|---|
| N1 <-> N2 | A = `hs_attr_up` (upper rail) | does not occur |
| N3 <-> N4 | A = `hs_attr_down` (lower rail) | does not occur |
| N1 <-> N3 | A = 0 (`hs_reset_pairs`, on by default) | A = 0 |
| N2 <-> N4 | A = 0 (`hs_reset_pairs`, on by default) | does not occur |
| N1 <-> N4, N2 <-> N3, same line | no effect | no effect |

State of a cell in a slot:

* **NORMAL**: the row line if the row fires, else the column line if the column fires, else the
  state is kept. A selected cell (both fire) takes the row line.
* **DNO**: the row line is driven in every slot. A row whose bit is 0 gets the complementary line
  (potentiation command: N3, depression command: N1) and the column pulse sits inside that
  envelope. The state is therefore N1 or N3 by the row alone, and every change of the row bit
  decays all cells of the row toward 0.

The effect is applied at the transition itself, on half-selected and selected cells alike, before
the coincidences of the slot are written. The state persists across updates;
`tile.tile.reset_hs_states()` clears it (e.g. after an array reset).

Usage:

```python
device = LinearStepDevice(..., hs_mode=2, hs_rate=0.0036, hs_attr_up=1.14, hs_attr_down=-1.09)
rpu_config.update.pulse_type = PulseType.HALFSELECTED_STOCHASTIC
tile = AnalogTile(5, 5, rpu_config)
tile.tile.enable_hs_tracking()

tile.tile.set_hs_polarity(+1)      # next update is a potentiation command (-1: depression)
tile.update(x, d)
tile.tile.apply_hs_idle_slot()     # a slot in which no line fires (matters for hs_mode 3 only)
```

`set_hs_polarity(0)` infers the polarity from the pulse signs of the slot and falls back to the
last known polarity when only one line type fires; with slot-wise replays or rows of zero
probability that is ambiguous, so setting it explicitly is recommended.

Measured rates: 0.0036 per transition (NORMAL), 0.0042 (DNO); attractors of the potentiating /
depressing pair = the saturation levels of the cell.

### Older half-select parameters

`hs_mode` 0 / 1 with `hs_decay`, `hs_pair_step_up`, `hs_pair_step_down`, `hs_reset_decay` are kept
for backward compatibility. They differ from the measurements: `hs_decay` acts only at a
coincidence that follows a state flip (a pure half-select sequence does nothing), and the pair
step is a constant increment (the small-weight limit of the attractor form, without an attractor).
Use `hs_mode` 2 / 3 for new work.

## Limits

* CPU, sparse pulse types, `LinearStepDevice` (load law); the half-select model is in the pulsed
  base device.
* Sign-pure commands are assumed (four-quadrant scheme for signed outer products).
* The coefficients come from 5x5 arrays. The load law describes line loading on that array and
  should not be extrapolated to much larger arrays without new measurements.
* Not modelled: read disturb and read noise (apply them on the host when reading the weights), the
  one-sided decay of positive states under repeated N4 pulses.
