# Generate SimBench CGMES Snapshot Parquets Locally

This guide explains how to generate the SimBench snapshot-envelope parquet dataset from many zipped CGMES snapshots on a single local machine, without Slurm or a cluster.

The workflow is:

1. Put the original SimBench CGMES export zip files under `ScenarioSynthesis_PPC/CGMES`.
2. Extract them into one snapshot directory.
3. Build a load P/Q min-max bounds cache from all snapshots.
4. Run a small smoke test.
5. Run the full parquet generation.

The current recommended SimBench configuration is `100 MVA`, `ppcY`, `coupled` P/Q sampling, SI Newton-Raphson labels, and `manual_flat` or `dc_compile` start mode.

## 1. Expected Project Layout

Assume the repository is here:

```text
~/PIGNN-Attn-LS/
```

The data-generation code lives here:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/
```

Put the two original SimBench CGMES export archives here:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/CGMES/
```

For the current SimBench dataset, the files are:

```text
ExportSim_260603_1024.zip
ExportSim_260603_1434.zip
```

Create these directories if they do not exist:

```powershell
cd ~/PIGNN-Attn-LS/ScenarioSynthesis_PPC
mkdir CGMES
mkdir out
mkdir CGMES/SimBenchSnapshots
```

## 2. Python Environment

Use the same type of environment used for the other pandapower datasets:

```text
Python 3.12
pandapower 3.4.0
pandas
numpy
pyarrow
scipy
```

Example with conda:

```powershell
conda create -n Python3.12_PandaPower3.4.0 python=3.12 -y
conda activate Python3.12_PandaPower3.4.0
pip install pandapower==3.4.0 pandas numpy pyarrow scipy tqdm
```

If your machine already has the project environment, just activate it.

## 3. Extract the CGMES Snapshot Archives

Extract both export archives into the same snapshot root:

```text
~/PIGNN-Attn-LS/ScenarioSynthesis_PPC/CGMES/SimBenchSnapshots/
```

After extraction, the directory should contain subdirectories like:

```text
CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip
CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_2.zip
...
CGMES/SimBenchSnapshots/ExportSim_260603_1434/data/CIM_GridAssist_*.zip
```

The scripts search recursively for files matching:

```text
CIM_GridAssist_*.zip
```

So the exact subdirectory depth is okay as long as the inner snapshot zip files are under `CGMES/SimBenchSnapshots`.

On Windows, you can use 7-Zip or PowerShell extraction. For large archives, 7-Zip is usually more reliable:

```powershell
7z x CGMES/ExportSim_260603_1024.zip -oCGMES/SimBenchSnapshots
7z x CGMES/ExportSim_260603_1434.zip -oCGMES/SimBenchSnapshots
```

## 4. Choose the Base Snapshot

Use the first snapshot zip as the base network:

```text
CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip
```

The base snapshot defines the grid topology, bus/branch structure, and load table. The many snapshots are then used to compute each load's observed minimum and maximum `P` and `Q`.

## 5. Build and Inspect the Load Bounds Cache

From `ScenarioSynthesis_PPC`, run:

```powershell
cd ~/PIGNN-Attn-LS/ScenarioSynthesis_PPC

python explore_cgmes_snapshot_envelope.py `
  --snapshot_root ./CGMES/SimBenchSnapshots `
  --base_cgmes_path ./CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip `
  --case_name SimBench_snapshot_envelope_manual_flat_100MVA `
  --cgmes_version 2.4.15 `
  --cgmes_ignore_errors `
  --base_sn_mva 100 `
  --bounds_cache_path ./out/SimBench_snapshot_load_bounds_cache.npz `
  --fast_xml_bounds `
  --progress_every 1000 `
  --top_k 20
```

This creates:

```text
./out/SimBench_snapshot_load_bounds_cache.npz
```

Expected high-level result for the current SimBench files:

```text
n_snapshots = 39,268
n_loads     = 58
base net    = 94 buses, 131 branches
```

The cache is important. Without it, full generation has to rebuild the load bounds, which is unnecessary and slow.

## 6. Smoke Test Before Full Generation

Run a tiny dataset first. This checks that the CGMES import, load-bound alignment, Ybus construction, and Newton-Raphson label generation all work.

```powershell
python main_datagen_lvn_snapshot_envelope.py `
  --base_cgmes_path ./CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip `
  --snapshot_root ./CGMES/SimBenchSnapshots `
  --case_name SimBench_snapshot_envelope_manual_flat_100MVA_smoke `
  --cgmes_version 2.4.15 `
  --no_cgmes_model_a_cleanup `
  --cgmes_ignore_errors `
  --base_sn_mva 100 `
  --bounds_cache_path ./out/SimBench_snapshot_load_bounds_cache.npz `
  --sample_mode coupled `
  --ybus_mode ppcY `
  --runs 10 `
  --workers 1 `
  --rows_per_task 1 `
  --save_steps 1 `
  --no_save_y_matrix `
  --K 40 `
  --start_mode manual_flat `
  --diagnose_nr `
  --convergence_mode misinf `
  --save_path ./out `
  --overwrite
```

Expected smoke output file:

```text
./out/SimBench_snapshot_envelope_manual_flat_100MVA_smoke_coupled_ppcY_manual_flat_siNR_10_NR_branchrows_directSI.parquet
```

The run should print Newton-Raphson convergence messages such as:

```text
94- Bus.....|converged successfully (misinf)|
```

## 7. Generate the Full 36k Manual-Flat Dataset

For a normal local workstation, start with `workers` equal to the number of physical CPU cores or slightly less. For example, use `--workers 8` on an 8- or 12-core machine.

```powershell
python main_datagen_lvn_snapshot_envelope.py `
  --base_cgmes_path ./CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip `
  --snapshot_root ./CGMES/SimBenchSnapshots `
  --case_name SimBench_snapshot_envelope_manual_flat_100MVA `
  --cgmes_version 2.4.15 `
  --no_cgmes_model_a_cleanup `
  --cgmes_ignore_errors `
  --base_sn_mva 100 `
  --bounds_cache_path ./out/SimBench_snapshot_load_bounds_cache.npz `
  --sample_mode coupled `
  --ybus_mode ppcY `
  --runs 36000 `
  --workers 8 `
  --rows_per_task 2 `
  --save_steps 100 `
  --no_save_y_matrix `
  --K 40 `
  --start_mode manual_flat `
  --diagnose_nr `
  --convergence_mode misinf `
  --save_path ./out `
  --overwrite
```

Expected output:

```text
./out/SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet
```

For the current SimBench data, this parquet is about `355 MB`.

## 8. Generate the Full 36k DC-Init Dataset

Use this if you want the Newton-Raphson initial voltage to come from a DC compile instead of a flat voltage start.

```powershell
python main_datagen_lvn_snapshot_envelope.py `
  --base_cgmes_path ./CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip `
  --snapshot_root ./CGMES/SimBenchSnapshots `
  --case_name SimBench_snapshot_envelope_dc_compile_100MVA `
  --cgmes_version 2.4.15 `
  --no_cgmes_model_a_cleanup `
  --cgmes_ignore_errors `
  --base_sn_mva 100 `
  --bounds_cache_path ./out/SimBench_snapshot_load_bounds_cache.npz `
  --sample_mode coupled `
  --ybus_mode ppcY `
  --runs 36000 `
  --workers 8 `
  --rows_per_task 2 `
  --save_steps 100 `
  --no_save_y_matrix `
  --K 40 `
  --start_mode dc_compile `
  --diagnose_nr `
  --convergence_mode misinf `
  --save_path ./out `
  --overwrite
```

Expected output:

```text
./out/SimBench_snapshot_envelope_dc_compile_100MVA_coupled_ppcY_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
```

## 9. Important Configuration Choices

`--base_sn_mva 100`

This overrides the pandapower CGMES import default `net.sn_mva`. For SimBench surrogate training, `100 MVA` is recommended so the per-unit normalization is consistent with MATPOWER-style PPC datasets.

`--no_cgmes_model_a_cleanup`

Use this for SimBench. The Model A cleanup was introduced for the LVN/Heo1 grid and should not be applied to SimBench.

`--sample_mode coupled`

Samples load `P` and `Q` together from the observed snapshot envelope. This preserves a more realistic relation between active and reactive load than fully independent sampling.

`--ybus_mode ppcY`

Uses the ppc/Ybus construction path used by the PIGNN training pipeline.

`--no_save_y_matrix`

Recommended for normal dataset generation. The parquet stores compact branch-row quantities instead of a full dense/sparse Y matrix per row.

`--start_mode manual_flat`

Initializes the Newton-Raphson voltage from a flat start. This is simple and consistent.

`--start_mode dc_compile`

Runs a DC compile to initialize voltage angles before Newton-Raphson. This can be useful when you want a stronger initial guess.

## 10. Quick Validation of the Final Parquet

Run this from `ScenarioSynthesis_PPC`:

```powershell
python -c "import pyarrow.parquet as pq; p='./out/SimBench_snapshot_envelope_manual_flat_100MVA_coupled_ppcY_manual_flat_siNR_36000_NR_branchrows_directSI.parquet'; pf=pq.ParquetFile(p); print('rows', pf.metadata.num_rows); t=pq.read_table(p, columns=['S_base','U_base','bus_number','branch_number']); d=t.to_pydict(); print('S_base', sorted(set(d['S_base']))); print('U_base', sorted(set(d['U_base']))); print('bus', sorted(set(d['bus_number']))); print('branch', sorted(set(d['branch_number'])))"
```

Expected values:

```text
rows    36000
S_base  [100000000.0]
U_base  [110000.0]
bus     [94]
branch  [131]
```

For the DC-init file, change the parquet filename in the validation command.

## 11. Troubleshooting

If the script says no snapshots were found, check that the inner files named `CIM_GridAssist_*.zip` are somewhere under:

```text
./CGMES/SimBenchSnapshots/
```

If CGMES loading fails, first confirm the base snapshot exists:

```text
./CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip
```

If the machine becomes unresponsive, lower these values:

```text
--workers
--save_steps
```

A safe local fallback is:

```text
--workers 2
--rows_per_task 1
--save_steps 20
```

If the full run is interrupted, rerun with `--overwrite` only if you want to replace the partial parquet. Otherwise, choose a new `--case_name` or `--save_path`.
