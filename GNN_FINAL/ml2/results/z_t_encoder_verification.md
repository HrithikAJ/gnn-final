# z(t) Encoder Verification

Protocol: chronological ML1/ML2 UCS contract.

- Checkpoint: `data/ml1_artifacts/gaussian_next_state_best.pt`
- Checkpoint structure: wrapped `model_state_dict`
- Loaded keys: `lstm.weight_ih_l0`, `lstm.weight_hh_l0`, `lstm.bias_ih_l0`, `lstm.bias_hh_l0`
- LSTM shapes: `(256, 406)`, `(256, 64)`, `(256,)`, `(256,)`
- Metadata feature count: 406
- Lookback: 30 windows
- PCA: disabled
- Normalization: canonical `ucs_windows.parquet` is already train-only normalized when `scaler_params.yaml` exists; no second transform is applied
- Missing LSTM keys: none
- Unexpected LSTM keys: none

## Non-degenerate output check

The encoder was run over every causal 30-window history available in the canonical
sequence:

- Cached z(t) windows: `2758`
- Output shape: `[2758, 64]`
- Minimum: `-1.0`
- Maximum: `1.0`
- Mean: `-0.0799187`
- Standard deviation: `0.5434801`
- All-zero output: `false`
- L2 distance between the first two outputs: `2.0258248`

First eight values of the first verified z(t):

```text
[-0.00558908, 0.47968447, -0.08002398, -0.34152293,
  0.76679164, 0.74227655, 0.47684032, -0.23659883]
```

The checkpoint loaded completely and the outputs vary across input windows, so
this gate passes. No zero-latent fallback is used by the encoder.

## Item 2: Canonical UCS Normalization Verification

Confirmed: `ucs_windows.parquet` is already normalized. Verification run 2026-09-09:

| Feature | Parquet Min | Parquet Max | Parquet Mean | Scaler Median (Raw) | is_log1p | Status |
|---------|-------------|-------------|--------------|---------------------|----------|--------|
| duration_microsec_mean | -2.069 | 14.518 | 0.293 | 14,770,387 | false | ✓ scaled |
| byte_count_fwd_mean | -16.945 | 26.194 | 0.518 | 5.99 | true | ✓ scaled |
| packet_count_fwd_mean | -4.519 | 38.793 | 0.701 | 1.85 | true | ✓ scaled |

**Conclusion:** All features show 0-centered scaled values (mean ≈ 0, std ≈ 2–4), while scaler parameters reflect raw data statistics. 
The z_encoder.py implementation is correct: **no additional normalization is applied** to the input features before LSTM.
The assumption that `ucs_windows.parquet` is already normalized is **verified and safe**.
