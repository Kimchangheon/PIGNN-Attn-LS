# Datasets

This bundle does not ship parquet files. Two paths to get the LVN data ready
for training:

## Path A — you already have the HVN-schema parquet

Drop `LVN_converted_n36000_v2.parquet` (or any `LVN_converted_*.parquet`)
into this directory. The path in `configs/lvn_pignn_attn_ls.yaml`
(`./datasets/LVN_converted_n36000_v2.parquet`) will pick it up.

The dataset class auto-detects `LVN_converted` in the filename and switches
to lazy-load mode (parquet stays on disk, rows decoded on demand) so 36k
samples don't blow up RAM.

## Path B — you only have the raw LVN snapshot

If you have `LVN_snapshot_envelope_..._branchrows_directSI.parquet`, run the
converter once to produce an HVN-schema parquet:

```bash
python scripts/convert_lvn_to_hvn_schema.py \
    --in  ./datasets/LVN_snapshot_envelope_manual_flat_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet \
    --out ./datasets/LVN_converted_n36000_v2.parquet
```

The converter:

1. Decodes binary columns.
2. Rebuilds `Lines_connected` from active branches, summing parallel
   branches that share `(min, max)` endpoints.
3. Reorders `Y_Lines` / `Y_C_Lines` into canonical upper-triangular pair
   positions.
4. Per-unit normalises voltages by per-bus `vn_kv` so buses at any voltage
   level come out near 1.0 p.u.
5. Per-units series + shunt admittances against the from-bus base.
6. Remaps bus type `{1->0 PQ, 2->2 PV, 3->1 slack}` to the HVN convention
   the model expects.
7. Adds the `vn_log` per-bus voltage-class scalar feature consumed by
   `GNSMsg_EdgeSelfAttn_VnFeat`.

Output is a parquet file readable by `data_loading.dataset.ChanghunDataset`
with no further changes.
