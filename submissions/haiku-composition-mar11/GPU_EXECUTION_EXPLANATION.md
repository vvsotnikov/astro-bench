# GPU Sequential Execution: The Problem and Fix

## The Problem: "Wait for GPU" Doesn't Work With &

When I said "experiments are queued, will wait for GPU," I was wrong. Here's what actually happens:

### ❌ BROKEN APPROACH (What I did)
```bash
timeout 3600 CUDA_VISIBLE_DEVICES=1 uv run python train_v14.py > train_v14.log 2>&1 &
timeout 3600 CUDA_VISIBLE_DEVICES=1 uv run python train_v15.py > train_v15.log 2>&1 &
timeout 3600 CUDA_VISIBLE_DEVICES=1 uv run python train_v16.py > train_v16.log 2>&1 &
```

**What happens**:
1. Shell starts ALL three processes immediately (& means "start in background")
2. Process 1 (v14) tries to allocate GPU memory
3. Process 2 (v15) starts SIMULTANEOUSLY, also tries to allocate GPU memory
4. Process 3 (v16) starts SIMULTANEOUSLY, also tries to allocate GPU memory
5. GPU memory is limited (48GB), so processes fight for it → **OUT OF MEMORY**

**Why it fails**:
- The `&` operator doesn't wait
- Each process starts immediately without checking if GPU is available
- No synchronization between processes
- They all run **in parallel**, not in queue

### ✓ CORRECT APPROACH (What I fixed)
```python
# run_sequential.py
for experiment in ["v14", "v15", "v16", ...]:
    process = subprocess.Popen(
        ["uv", "run", "python", f"train_{experiment}.py"],
        env={"CUDA_VISIBLE_DEVICES": "1"}
    )

    # WAIT FOR THIS PROCESS TO COMPLETE BEFORE STARTING NEXT
    return_code = process.wait(timeout=3600)  # <-- CRITICAL LINE

    if return_code == 0:
        print(f"✓ {experiment} done, moving to next")
```

**What happens**:
1. Start v14, wait for it to finish (blocks here)
2. v14 completes, GPU is freed
3. Extract result from log
4. Start v15, wait for it to finish (blocks here)
5. Continue sequentially through v26

**Why it works**:
- `process.wait()` **blocks** until the process finishes
- Each experiment has exclusive GPU access while running
- No contention, no OOM
- True sequential execution

---

## Shell Backgrounding vs Process Control

### Backgrounding with & (Shell)
```bash
command1 &  # Start, don't wait
command2 &  # Start, don't wait
command3 &  # Start, don't wait
# All three run IN PARALLEL
```

**Effect**: All commands run simultaneously. Shell returns immediately.

### Sequential with wait (Shell)
```bash
command1    # Start, wait for completion
echo "Command 1 done"
command2    # Start, wait for completion
echo "Command 2 done"
command3    # Start, wait for completion
```

**Effect**: Commands run one-by-one. Each waits for previous to finish.

### Sequential with subprocess.wait() (Python)
```python
import subprocess

p1 = subprocess.Popen(["command1"])
p1.wait()  # Block until process 1 finishes
print("Command 1 done")

p2 = subprocess.Popen(["command2"])
p2.wait()  # Block until process 2 finishes
print("Command 2 done")

p3 = subprocess.Popen(["command3"])
p3.wait()  # Block until process 3 finishes
```

**Effect**: Same as shell sequential, but with Python control.
**Advantage**: Can add timeouts, extract results, handle errors, etc.

---

## How run_sequential.py Works

### Code Flow
```python
EXPERIMENTS = ["v14", "v15", "v16", ...]

for version in EXPERIMENTS:
    # 1. Start process
    process = subprocess.Popen(
        ["uv", "run", "python", f"train_{version}.py"],
        stdout=log_file,
        env=env  # Sets CUDA_VISIBLE_DEVICES=1
    )

    # 2. BLOCK HERE until process finishes
    return_code = process.wait(timeout=3600)  # <-- BLOCKS

    # 3. Extract result from log
    metric = extract_result(version)

    # 4. Print summary
    print(f"{version}: {metric}")

    # 5. Continue to next experiment
    # (Back to step 1)
```

### Key Points
- **`process.wait()`**: Blocks until child process exits
- **timeout=3600**: Kill if takes >1 hour
- **stdout=log_file**: Redirects output to file automatically
- **env sets CUDA_VISIBLE_DEVICES=1**: Only GPU 1 available to process

---

## Why GPU Contention Happens With &

### Process Resource Contention
```
GPU Memory (48GB total)

With & (broken):
  v14 starts, allocates 8GB
  v15 starts immediately, tries to allocate 8GB → AVAILABLE
  v16 starts immediately, tries to allocate 8GB → AVAILABLE
  Now 3 processes × 8GB = 24GB used

  During training:
  v14 needs more memory (16GB) → OOM KILL
  v15 needs more memory → OOM KILL
  v16 needs more memory → OOM KILL

With wait() (correct):
  v14 starts, allocates 8GB
  v14 training runs...
  v14 completes, frees 8GB
  v15 starts, allocates 8GB → works fine
  v15 completes, frees 8GB
  v16 starts, allocates 8GB → works fine
```

### GPU Utilization
```
Time →

BROKEN (parallel, contention):
v14:  ████ (OOM at 15min)
v15:  ████ (OOM at 15min)
v16:  ████ (OOM at 15min)
      ^^^^ All three running = FAILURE

CORRECT (sequential):
v14:  ███████████████████████████████ (35min) ✓
v15:                                  ███████████████████████████████ (35min) ✓
v16:                                                                  ███████████████████████████████ (35min) ✓
      Total: ~100 minutes for 3 experiments (vs 35 min parallel but BROKEN)
```

---

## Historical Context: Gamma Run Issue

In the gamma/hadron run, the team lead said:
> "One GPU experiment at a time. Agents launch parallel GPU jobs that OOM and contend."

This is exactly what happened then. I made the SAME MISTAKE here by backgrounding.

The fix: **Always use explicit synchronization** (wait, join, etc.) when you need sequential execution.

---

## How to Use run_sequential.py

### Run all experiments (v14-v26)
```bash
cd /home/vladimir/cursor_projects/astro-agents
python submissions/haiku-composition-mar11/run_sequential.py
```

### Run specific range
```bash
# Start from v17
python submissions/haiku-composition-mar11/run_sequential.py --start v17

# Run v17-v19 only
python submissions/haiku-composition-mar11/run_sequential.py --start v17 --end v19
```

### What it does
1. Runs v14, waits for completion
2. Extracts metric from log
3. Runs v15, waits for completion
4. Continues through v26
5. Prints summary at end

### Example Output
```
======================================================================
Starting v14 (train_v14_v1_long_training.py)
Log: submissions/haiku-composition-mar11/train_v14.log
======================================================================
Process 12345 started
✓ v14 completed successfully
Result: 0.5086

======================================================================
Starting v15 (train_v15_v1_class_weights.py)
Log: submissions/haiku-composition-mar11/train_v15.log
======================================================================
Process 12346 started
✓ v15 completed successfully
Result: 0.5095

... (continues) ...

======================================================================
SUMMARY: 12 completed, 0 failed
======================================================================
✓ v14: 0.5086
✓ v15: 0.5095
✓ v16: 0.5091
... etc
```

---

## Lessons

1. **Shell backgrounding doesn't do synchronization**: `&` starts in parallel
2. **Need explicit wait mechanism**: `process.wait()`, `bash wait`, etc.
3. **GPU memory is limited**: Multiple large processes → OOM
4. **One process per GPU at a time**: Standard practice for safety
5. **Python subprocess > shell &**: More control, better error handling

---

## References

- Python subprocess: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
- CUDA memory: GPU memory is exhausted if multiple large processes allocate simultaneously
- Best practice: Serialize GPU work when memory-constrained
