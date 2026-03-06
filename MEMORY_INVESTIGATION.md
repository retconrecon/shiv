# Memory Accumulation Investigation: SAMURAI Tracker on Long Videos

## Executive Summary

The SAMURAI tracker stores **every frame's output tensors indefinitely** in
`output_dict["non_cond_frame_outputs"]` during `propagate_in_video()`, with no eviction or
pruning mechanism triggered during normal tracking. Each frame accumulates ~1.5 MB of tensor
data (with 2 tracked objects at 1024×1024 input). Over 17,000 frames this totals **~25.5 GB**
— exceeding typical GPU VRAM (without CPU offloading) or straining system RAM (with offloading).
Meanwhile, the transformer attention only ever reads **7 memory frames + 16 object pointers**
at any given timestep, meaning >99.8% of stored frames are dead weight. The existing
`_purge_memory()` function is only called during Cross-Object Interaction mask collisions, not
during normal propagation. There is no automatic memory bank size limit.

---

## Memory Budget Table

### Per-Frame Cost (Hiera-L, image_size=1024, B=2 objects)

| Component | Shape | dtype | Device (offload=True) | Size/frame |
|-----------|-------|-------|-----------------------|------------|
| `maskmem_features` | (2, 64, 64, 64) | bfloat16 | CPU | **1.00 MB** |
| `pred_masks` | (2, 1, 256, 256) | float32 | CPU | **0.50 MB** |
| `obj_ptr` | (2, 256) | float32 | **GPU** | 2.0 KB |
| `object_score_logits` | (2, 1) | float32 | **GPU** | 8 B |
| `best_iou_score` | (2,) | float32 | **GPU** | 8 B |
| `kf_score` | (2,) | float32 | **GPU** | 8 B |
| `maskmem_pos_enc` | shared ref | — | cached once | 0 B |
| Per-obj view dicts (×2) | views only | — | — | ~1.2 KB |
| Python dict overhead | — | — | CPU | ~0.2 KB |
| **Total per frame** | | | | **~1.50 MB** |

### Projection to Long Videos

| Video Length | Frames | Total Stored | CPU (offload=True) | GPU (offload=True) | GPU (offload=False) |
|-------------|--------|-------------|--------------------|--------------------|---------------------|
| Short | 4,000 | 6.0 GB | 6.0 GB | ~8 MB | 6.0 GB |
| Medium | 8,000 | 12.0 GB | 12.0 GB | ~16 MB | 12.0 GB |
| **Problem case** | **17,000** | **25.5 GB** | **25.5 GB** | **~34 MB** | **25.5 GB** |

**Note:** Model weights add ~2-4 GB GPU. With `offload_state_to_cpu=False` (the default in
`sam2_video_predictor.init_state()`), all 25.5 GB lands on GPU. With
`offload_state_to_cpu=True` (the default in `MultiObjectTracker.track()`), GPU is spared but
CPU RAM fills instead.

---

## Accumulation Points

### Primary: Output Dict (the big one)

| File | Line | Variable | Growth Rate | What's Stored |
|------|------|----------|-------------|---------------|
| `sam2/sam2/sam2_video_predictor.py` | 736 | `output_dict["non_cond_frame_outputs"][frame_idx]` | +1.5 MB/frame | maskmem_features + pred_masks + obj_ptr + scores |
| `sam2/sam2/sam2_video_predictor.py` | 778 | `output_dict_per_obj[obj_idx]["non_cond_frame_outputs"][frame_idx]` | views only | Tensor slices sharing storage with above |
| `sam2/sam2/sam2_video_predictor.py` | 714 | `output_dict["cond_frame_outputs"][frame_idx]` | +1.5 MB/frame | Same structure (for conditioning frames) |

**Root cause:** Line 736 stores every new frame's output. No line ever removes it during
`propagate_in_video()`.

### Secondary: Metadata Dicts (small but unbounded)

| File | Line | Variable | Growth Rate |
|------|------|----------|-------------|
| `sam2/sam2/sam2_video_predictor.py` | 742 | `inference_state["frames_already_tracked"]` | ~100 B/frame |
| `sam2/sam2/sam2_video_predictor.py` | 106-109 | `consolidated_frame_inds` (sets) | ~28 B/frame |

### Bounded (not a problem)

| File | Line | Variable | Why It's Bounded |
|------|------|----------|-----------------|
| `sam2/sam2/sam2_video_predictor.py` | 896 | `cached_features` | Replaced each frame (dict reassignment, 1 entry max) |
| `sam2/sam2/sam2_video_predictor.py` | 89 | `constants["maskmem_pos_enc"]` | Cached once, shared across all frames |
| `sam2/sam2/modeling/sam2_base.py` | 203-204 | `kf_mean`, `kf_covariance` | Single state (8,) and (8,8) numpy arrays, overwritten |

### Conditional: Diagnostics (off by default)

| File | Line | Variable | Risk |
|------|------|----------|------|
| `sam2/sam2/multi_object_tracker.py` | 639 | `self.diagnostics.append(diag_entry)` | If `capture_diagnostics=True`: stores full BGR frames + metadata per frame. **Enormous** if enabled. |
| `sam2/sam2/modeling/sam2_base.py` | 208 | `self.history` | Currently disabled (`if False:` at line 482). Would accumulate tensors per frame if enabled. |

---

## Access Pattern Analysis

### What the Transformer Actually Reads

At frame N, `_prepare_memory_conditioned_features()` (sam2_base.py:620) selects:

| Memory Type | Source | Max Count | Selection Method |
|-------------|--------|-----------|-----------------|
| Conditioning frames | `cond_frame_outputs` | `max_cond_frames_in_attn` (default: unlimited) | Temporally closest |
| Non-conditioning memory | `non_cond_frame_outputs` | `num_maskmem - 1` = **6** | SAMURAI: score-filtered backward scan; Standard: stride-based |
| Object pointers | Both dicts | `max_obj_ptrs_in_encoder` = **16** | From selected memory frames |

**Total frames read per inference step: ~7 memory + ~16 pointers = ~23 max**

### SAMURAI Score-Based Selection (sam2_base.py:663-690)

In SAMURAI mode, the tracker iterates **backwards through ALL stored frames** to find
high-quality memory:

```python
for i in range(frame_idx - 1, 1, -1):
    frame_out = output_dict["non_cond_frame_outputs"].get(i)
    if frame_out["best_iou_score"] > threshold and ...:
        valid_indices.insert(0, i)
    if len(valid_indices) >= max_obj_ptrs_in_encoder - 1:
        break
```

This means:
- All frames must remain in the dict for score lookup (even though only scores are read)
- At most 15 frames' maskmem_features are actually loaded per step
- The remaining ~16,985 frames' maskmem_features sit unused in memory

### Waste Ratio

| Video Length | Frames Stored | Frames Read/Step | Waste |
|-------------|--------------|-----------------|-------|
| 4,000 | 4,000 | ~23 | 99.4% |
| 8,000 | 8,000 | ~23 | 99.7% |
| 17,000 | 17,000 | ~23 | **99.86%** |

---

## Existing Mechanisms

### 1. `_purge_memory()` — Manual frame removal
- **File:** `sam2/sam2/multi_object_tracker.py:79-96`
- **What:** Pops a frame from `output_dict["non_cond_frame_outputs"]` and all per-object dicts
- **When called:** Only during Cross-Object Interaction (CoI) mask collisions (~line 406)
- **Not called:** During normal `propagate_in_video()` — so in single-object or non-colliding
  tracking, memory is never pruned

### 2. `_clear_non_cond_mem_around_input()` — Window-based cleanup
- **File:** `sam2/sam2/sam2_video_predictor.py:1163-1180`
- **What:** Pops frames within `±(num_maskmem × memory_temporal_stride_for_eval)` of a
  conditioning frame
- **When called:** Only if `clear_non_cond_mem_around_input=True` AND
  (`clear_non_cond_mem_for_multi_obj=True` OR single object)
- **Default:** Both flags are `False` → **never triggered**

### 3. `_reset_tracking_results()` — Full reset
- **File:** `sam2/sam2/sam2_video_predictor.py:864-881`
- **What:** Clears all output dicts completely
- **When called:** Only on explicit `reset_state()` call — not during normal tracking

### 4. CPU Offloading (`offload_state_to_cpu`)
- **File:** `sam2/sam2/sam2_video_predictor.py:79-82`
- **Default:** `False` in `init_state()`, but `True` in `MultiObjectTracker.track()` (line 216)
- **Effect:** Moves maskmem_features (bf16) and pred_masks (f32) to CPU. Keeps obj_ptr on GPU.
- **Limitation:** Prevents GPU OOM but shifts the problem to CPU RAM (still 25.5 GB for 17K frames)

### 5. Config Parameters

From `configs/samurai/sam2.1_hiera_l.yaml`:

| Parameter | Value | Effect on Memory |
|-----------|-------|-----------------|
| `num_maskmem` | 7 | Only 7 frames used in attention (but all stored) |
| `max_obj_ptrs_in_encoder` | 16 (default) | Only 16 pointers used (but all stored) |
| `memory_temporal_stride_for_eval` | 1 (default) | No frame skipping in stride-based selection |
| `memory_encoder.out_dim` | 64 | mem_dim = 64 (determines maskmem_features channel dim) |
| `image_size` | 1024 | Feature maps at 64×64, masks at 256×256 |
| `memory_bank_iou_threshold` | 0.5 | Score filter for SAMURAI memory selection |
| `memory_bank_obj_score_threshold` | 0.0 | No filtering on object score |
| `memory_bank_kf_score_threshold` | 0.0 | No filtering on Kalman score |

**No parameter exists to limit the total number of stored frames.**

---

## SAMURAI-Specific Memory Additions

SAMURAI adds minimal memory beyond base SAM2:

| Addition | Storage | Impact |
|----------|---------|--------|
| `best_iou_score` per frame | (B,) float32 scalar | Negligible (~8 B/frame) |
| `kf_score` per frame | (B,) float32 scalar | Negligible (~8 B/frame) |
| Kalman filter state | (8,) + (8,8) numpy arrays | Fixed ~320 B total, not per-frame |
| Score-based backward scan | Iterates all frames | No extra storage, but requires all frames to remain in dict |

The SAMURAI memory selection logic (sam2_base.py:663-690) **reads** from all stored frames
but does not **add** additional tensor storage. However, it creates a dependency: frames
cannot be evicted without also evicting their scores, which are embedded in the same dict
entries as the large maskmem_features tensors.

---

## Recommendations

### 1. Sliding Window Eviction (Simplest, ~24 GB savings)

Add automatic eviction in `propagate_in_video()` after line 736. Keep only the most recent
`W` frames (e.g., W=500) plus all conditioning frames. Evict old frames by popping them
from both `output_dict` and `output_dict_per_obj`.

```python
# After line 736: output_dict[storage_key][frame_idx] = current_out
MAX_NON_COND_FRAMES = 500
non_cond = output_dict["non_cond_frame_outputs"]
if len(non_cond) > MAX_NON_COND_FRAMES:
    oldest_key = min(non_cond.keys())  # or use an OrderedDict
    _purge_memory(inference_state, oldest_key)
```

- **Savings:** ~24.75 GB for 17K frames (keep 500 × 1.5 MB = 0.75 GB instead of 25.5 GB)
- **Risk:** SAMURAI backward scan may not find high-quality old frames. Mitigate by also
  keeping top-K scored frames.
- **Complexity:** Low

### 2. Separate Score Index from Heavy Tensors (Moderate, enables smarter eviction)

Store `best_iou_score`, `kf_score`, and `object_score_logits` in a lightweight parallel dict
(e.g., `score_index[frame_idx]`). Then evict maskmem_features and pred_masks from old frames
while keeping scores for the SAMURAI backward scan.

```python
# Per frame: store scores separately (~24 bytes) vs heavy tensors (~1.5 MB)
score_index[frame_idx] = {
    "best_iou_score": best_iou_score,
    "object_score_logits": object_score_logits,
    "kf_score": kf_score,
}
```

- **Savings:** Same as #1 but without losing score information
- **Risk:** Frames selected by score scan but already evicted need graceful fallback
- **Complexity:** Moderate (requires changes in sam2_base.py memory selection too)

### 3. Quality-Based Eviction / Top-K Memory Bank (Best balance)

Keep a fixed-size memory bank of the K highest-scoring frames (by IoU × obj_score), plus
the most recent N frames. This aligns storage with what SAMURAI actually selects.

- **Bank size:** K=50 best + N=50 recent = 100 frames → 150 MB
- **Savings:** ~25.35 GB for 17K frames
- **Risk:** Low — SAMURAI already filters by score, so low-scoring frames would never be
  selected anyway
- **Complexity:** Moderate

### 4. Enable CPU Offloading (Quick fix, partial)

If not already enabled, pass `offload_state_to_cpu=True` to `init_state()`. This is already
the default in `MultiObjectTracker.track()` but NOT in direct `init_state()` calls.

- **Savings:** Moves ~25.5 GB from GPU to CPU (GPU drops to ~34 MB for stored outputs)
- **Risk:** Shifts OOM from GPU to CPU; 25.5 GB CPU RAM still required
- **Complexity:** Trivial (one parameter change)

### 5. Periodic Full Reset with Re-conditioning (Most thorough, most invasive)

Every N frames (e.g., N=2000), reset the memory bank entirely and re-condition from the
most recent high-quality mask. This caps memory at O(N) regardless of video length.

- **Savings:** Caps at N × 1.5 MB (e.g., 3 GB for N=2000)
- **Risk:** Loss of long-range context; may cause tracking drift at reset boundaries
- **Complexity:** High (need careful re-initialization logic)

### Priority Order

1. **#4** (enable offloading) — immediate relief if not already on, zero code changes
2. **#1** (sliding window) — simple, dramatic savings, small code change
3. **#3** (quality-based top-K) — best long-term solution, aligns with SAMURAI's design
4. **#2** (separate scores) — enabler for #3
5. **#5** (periodic reset) — last resort if memory budget is very tight
