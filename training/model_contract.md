# LazyCatBot `.nnue` model contract (v1, HalfKP)

Status: pilot contract, 2026-11-26. Owner: Agent C (training/export).
The normative format definition is `docs/NNUE_PILOT.md` section 3.3; this
document adds the exact byte interpretation, quantization rules, output units
and loader guidance needed to keep the Python reference and the C++ engine in
agreement. Any change to the layout or units requires the coordinator's
approval and a version bump.

Reference implementations:

- writer + round-trip checker: `training/export_nnue.py`
- NumPy reader + forward: `training/reference_inference.py`
- trainer: `training/train_halfkp.py`

## 1. Conventions

- Little-endian, IEEE-754 `float32`, two's-complement `int16`/`int32`.
- Scores are **side-to-move centipawns**: positive means good for the side to
  move. The value network is trained on side-to-move win probability
  (`targets.f32`), so a position is always fed from the side-to-move
  perspective.
- `wdl = sigmoid(cp / 400)`, `logit = cp / 400`.
- Matrix layout is always **input-major / row-major**: the outer index is the
  input (feature or hidden unit), the inner index is the output. A C++ loader
  can therefore do `out[c] += h[j] * w[j * out_width + c]`.

## 2. Feature function (contract 3.1)

```
perspective = WHITE or BLACK
ownKing    = kingSq[perspective]
bucket     = (perspective == WHITE) ? ownKing : (ownKing ^ 56)
for every piece (color, type, sq):
    if color == perspective and type == KING: continue
    osq = (perspective == WHITE) ? sq : (sq ^ 56)
    rel = (color == perspective) ? 0 : 1
    idx = (bucket * 12 + rel * 6 + type) * 64 + osq
inputs = 64 * 12 * 64 = 49152
```

- `type` uses the engine enum: `PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4,
  KING=5`.
- The own king contributes only through `bucket` (64 king buckets).
- At most 32 features are active (`features.u16` rows are zero-padded, the true
  length is `counts.u8`). Typical middlegame positions use 28-31.
- Training computes features for the side to move. C++ may maintain both
  accumulators, but for this single-perspective v1 model it must select the
  accumulator for `pos.side` at evaluation time. Do not sum or average the two
  perspectives.
- `sq` is `a1=0, b1=1, ..., h8=63`. `sq ^ 56` flips ranks.

## 3. Header (64 bytes)

`struct` format `"<4s7IfIQ16s"`:

| Offset | Size | Type | Field | Value |
|---:|---:|---|---|---|
| 0 | 4 | bytes | magic | `LCNN` (`4C 43 4E 4E`) |
| 4 | 4 | u32 | version | `1` |
| 8 | 4 | u32 | arch | `1` = HalfKP |
| 12 | 4 | u32 | inputs | `49152` |
| 16 | 4 | u32 | hidden | FT width; pilot deployable values 128/256/384/512 |
| 20 | 4 | u32 | buckets | `64` king buckets |
| 24 | 4 | u32 | quant | `1` (int16 FT, float32 dense) |
| 28 | 4 | u32 | reserved | `0` |
| 32 | 4 | f32 | ft_scale | positive global FT dequantization scale |
| 36 | 4 | u32 | policy_size | `0` or `4352` |
| 40 | 8 | u64 | checksum | FNV-1a 64 over the payload (not the header) |
| 48 | 16 | bytes | reserved | all zero |

FNV-1a 64 as used by the exporter:

```text
h = 0xcbf29ce484222325            # 14695981039346656037
for byte in payload:
    h ^= byte
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
```

Do not include the header in the checksum. Validate the checksum before using
the payload; a corrupt file must be rejected (fall back to classical eval), not
clamped or partially loaded.

`hidden` is serialized as u32 and the exporter accepts any positive multiple of
32 (1024/1536 teacher sizes included). The pilot deployment ceiling is 512;
see section 8.

## 4. Payload

Written immediately after the header, in this exact order:

| # | Item | Type | Count | Layout |
|---:|---|---|---:|---|
| 1 | `ft` | int16 | `inputs * hidden` | input-major: `ft[i * hidden + j]` |
| 2 | `ft_bias` | f32 | `hidden` | per hidden unit |
| 3 | `fc1_w` | f32 | `hidden * 32` | input-major: `fc1_w[j * 32 + c]` |
| 4 | `fc1_b` | f32 | `32` | |
| 5 | `fc2_w` | f32 | `32 * 32` | input-major |
| 6 | `fc2_b` | f32 | `32` | |
| 7 | `out_w` | f32 | `32` | **already multiplied by `cp_scale`** |
| 8 | `out_b` | f32 | `1` | **already multiplied by `cp_scale`** |
| 9 | `policy_w` | f32 | `hidden * policy_size` | input-major: `policy_w[j * policy_size + c]` |
| 10 | `policy_b` | f32 | `policy_size` | only if `policy_size > 0` |

Payload size:

```
inputs*hidden*2 + hidden*4 + hidden*32*4 + 32*4
+ 32*32*4 + 32*4 + 32*4 + 4
+ policy_size * (hidden*4 + 4)
```

For `hidden=128, inputs=49152, policy_size=0` this is 12,604,292 bytes; with
`policy_size=4352` it is 14,849,924 bytes (file 14,849,988 with the header).
`export_nnue.py --check` prints and verifies both numbers.

### Output units (important)

The exporter multiplies `out_w` and `out_b` by `cp_scale = 400`:

```
out_cp = sum_j relu(fc2)_j * out_w[j] + out_b[0]     # centipawns, STM
wdl    = sigmoid(out_cp / 400)
```

So a C++ forward pass that returns `out_cp` directly implements "return cp".
There is no separate cp-scale field in the v1 header; `400` is fixed by the
target definition (`winProbFromCp` in the extractor is `sigmoid(cp/400)`) and
recorded in `model_card.json`:

```json
"output": {"units": "centipawns_side_to_move", "cp_scale": 400.0,
           "wdl": "sigmoid(cp / cp_scale)"}
```

## 5. Quantization

Only the FT matrix is quantized. Bias and all dense/policy layers stay
`float32`.

Export:

```text
max_abs = max(|ft|)                       # over the whole matrix
ft_scale = max_abs / 32767                # unless the caller supplies one
q = clip(round_half_away(ft / ft_scale), -32767, 32767)   # int16
```

The exporter uses `numpy.rint` (round-half-to-even) and clips symmetrically to
`[-32767, 32767]`, leaving `-32768` unused. A C++ implementation reading the
file does not round anything; rounding differences of at most one LSB only
matter when regenerating or re-quantizing weights in C++, which is not part of
v1.

Inference:

```text
int32 acc[j] = sum over active features i of ft[feat_i * hidden + j]
float h[j]  = acc[j] * ft_scale + ft_bias[j]
h[j]        = max(h[j], 0)
```

Worst case `acc` magnitude with 32 features is `32 * 32767 ≈ 1.05e6`, safely
inside int32. Do not accumulate in int16: 32 rows can overflow the int16 range.
On AVX2, widen int16 rows to int32 (`_mm256_cvtepi16_epi32`) before adding, or
use `_mm256_add_epi16` only if the accumulation is widened before any possible
overflow.

`ft_scale` applies once per hidden unit; it is *not* applied to `ft_bias` or the
dense weights. Because `ft_scale > 0`, an implementation may fold it into the
first dense layer and policy layer (`relu(scale*a + b) = scale*relu(a +
b/scale)`) and apply it once at the output, but that is an optional
optimization; the reference does the straightforward multiply.

## 6. Forward pass

```cpp
// idx[0..n) are the active HalfKP indices; n <= 32
int32_t acc[H] = {0};
for (i = 0; i < n; ++i)
    for (j = 0; j < H; ++j)
        acc[j] += ft[idx[i] * H + j];

float h[H];
for (j = 0; j < H; ++j)
    h[j] = std::max(0.0f, float(acc[j]) * ft_scale + ft_bias[j]);

float d1[32];
for (c = 0; c < 32; ++c) {
    float s = fc1_b[c];
    for (j = 0; j < H; ++j) s += h[j] * fc1_w[j * 32 + c];
    d1[c] = std::max(0.0f, s);
}

float d2[32];
for (c = 0; c < 32; ++c) {
    float s = fc2_b[c];
    for (j = 0; j < 32; ++j) s += d1[j] * fc2_w[j * 32 + c];
    d2[c] = std::max(0.0f, s);
}

float cp = out_b[0];
for (j = 0; j < 32; ++j) cp += d2[j] * out_w[j];   // already centipawns

// int engine score: round-to-nearest, then clamp to the engine's eval range
int score = (int)std::lround(cp);
```

Policy (independent head, not passed through the 32-wide value trunk):

```cpp
float logits[policy_size];
for (c = 0; c < policy_size; ++c) {
    float s = policy_b[c];
    for (j = 0; j < H; ++j) s += h[j] * policy_w[j * policy_size + c];
    logits[c] = s;
}
```

If `policy_size == 0` the file ends after `out_b` and no policy arrays exist.

## 7. Policy index encoding (contract 3.2)

- normal move: `from * 64 + to` (0..4095)
- promotion: `4096 + (promo - 2) * 64 + from`, `promo: 2=N, 3=B, 4=R, 5=Q`
  (destination follows from the from-square and side)
- castling is normalized to the king move: `e1g1`, `e1c1`, `e8g8`, `e8c8`
- `0xFFFF` means "no target" in `moves.u16` during training; it never appears
  inside a model file.

The policy head is trained only when `moves.u16` exists and `--policy` is not
`off`. It does not affect the exported value path; C++ may ignore it entirely.

## 8. Deployability (teacher vs student)

- `hidden <= 512` is the pilot's CPU inference budget. `model_card.json` sets
  `"deployable": true` and the file may be wired into the engine after
  validation.
- `hidden > 512` (e.g. the 1024/1536 teacher run on the rentable A6000) is
  exported for distillation, labeling and analysis only. The card then sets
  `"deployable": false`, `role: "teacher"` and a `deployability.reason`.
  Do not load such a file in a release engine; the v1 loader/accumulator is
  specified and benchmarked for the deployable sizes.
- The student export (<=512) is the deployable artifact, even when it is
  distilled from a teacher.

`model_card.json` also records the config, train/val group counts, metrics,
calibration bins, per-phase metrics, material-sanity FEN results, the
quantization error summary, the FNV checksum, SHA-256 and file size. A
`model_card.json` without `checks.export_vs_torch.ok == true` is not a validated
export.

## 9. Validation and reference commands

```bash
# structure, checksum, payload size, quantization round-trip
python3 training/export_nnue.py --check training/work/model/model.nnue --json

# header only
python3 training/reference_inference.py --model training/work/model/model.nnue --header

# cp/WDL from FENs using NumPy only
python3 training/reference_inference.py --model training/work/model/model.nnue \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

python3 -m pytest training/tests_train -q
```

Acceptance for the C++ path (`docs/NNUE_PILOT.md` section 5): the engine's cp
must match `reference_inference.py` within 2 cp on 100 FENs. The trainer runs
the same comparison on the export batch and fails the `ok` flag above 2 cp.

## 10. Versioning

Bump `version` (and get coordinator approval) if any of the following changes:
header layout, payload order, matrix orientation, quantization mode, input
feature function, output units or `cp_scale`. Additive metadata belongs in
`model_card.json`, never in the header's reserved bytes.
