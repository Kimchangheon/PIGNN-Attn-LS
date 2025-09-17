
---

# Train, Test & Benchmark

## Requirements

* Python 3.9+ (GPU recommended)
* PyTorch (CUDA build if using GPU)
* Install deps:

```bash
pip install -r requirements.txt
```

## Data

* Input format: **Parquet** power-flow datasets.
* Pass **one or more** paths to `--PARQUET` (they’ll be concatenated at load time).

---

## Training

```bash
python train_valid_test.py \
  --d=4 --d_hi=16 --K=40 --EPOCHS=100 \
  --lr_scheduler=CosineAnnealingLR \
  --PARQUET \
    ./data/HVN_15000_NR_plain_4_to_32_buses.parquet \
    ./data/HVN_15010_NR_plain_4_to_32_buses.parquet \
    ./data/HVN_15020_NR_plain_4_to_32_buses.parquet \
  --vlimit \
  --use_armijo \
  --model GNSMsg_EdgeSelfAttn
```

**Common args**

* `--d`: base embedding dim • `--d_hi`: hidden dim • `--K`: message-passing steps
* `--EPOCHS`: epochs • `--lr_scheduler`: LR schedule (e.g., `CosineAnnealingLR`)
* `--PARQUET`: one or more Parquet files
* `--model`: architecture (here `GNSMsg_EdgeSelfAttn`)
* `--vlimit`: enforce voltage limits • `--use_armijo`: Armijo backtracking

---

## Testing

```bash
python test.py \
  --d=4 --d_hi=16 --K=40 --EPOCHS=100 --BATCH=2 \
  --lr_scheduler=CosineAnnealingLR \
  --PARQUET \
    /proj/ciptmp/HVN_per_bus_4_to_511_one_each_max_att_2000.parquet \
    /proj/ciptmp/HVN_per_bus_512_to_1024_one_each.parquet \
  --use_armijo \
  --model GNSMsg_EdgeSelfAttn \
  --ckpt ./results/ckpt/GNSMsg_EdgeSelfAttn_9al_b64_HVN_15000_NR_and_HVN_15010_NR_and_HVN_15020_NR_K40_d4_dhi16_ep100_TrainRatio0.95_armijo_100_best_model.ckpt \
  --BLOCK_DIAG \
  --vlimit
```

* `--ckpt`: path to trained checkpoint
* `--BATCH`: eval batch size
* `--BLOCK_DIAG`: pack batched graphs as block-diagonal (speed/memory win for large N)

---

## Benchmarking

### Single-PF (per-case latency)

```bash
python bench_inference_batched.py \
  --PARQUET \
    /proj/ciptmp/HVN_per_bus_4_to_511_one_each_max_att_2000.parquet \
    /proj/ciptmp/HVN_per_bus_512_to_1024_one_each.parquet \
  --device cuda \
  --startN 4 --endN 1024 \
  --batch_size 1 \
  --batch_mode replicate \
  --batch_style blockdiag \
  --nr_workers 1 \
  --gns_workers 1 \
  --models GNSMsg_EdgeSelfAttn \
  --warmup 5 --repeat 20 --step 20
```

**Notes:** `--batch_size 1` for true single-case timing; `--warmup` ignored in stats; `--step` controls the N stride.

### Multi-PF (throughput)

```bash
python bench_inference_fair.py \
  --PARQUET \
    /proj/ciptmp/HVN_per_bus_4_to_511_one_each_max_att_2000.parquet \
    /proj/ciptmp/HVN_per_bus_512_to_1024_one_each.parquet \
  --device cuda \
  --startN 4 --endN 1024 \
  --cases_per_N 4096 \
  --batch_style blockdiag \
  --models GNSMsg_EdgeSelfAttn \
  --nr_workers 16 \
  --max_nodes_per_forward 14000 \
  --adaptive_oom_backoff \
  --warmup 2 --repeat 5 --step 20
```

**Notes:** `--cases_per_N` controls samples per size; `--max_nodes_per_forward` caps graph size per forward; `--adaptive_oom_backoff` auto-retries smaller batches on OOM.

---