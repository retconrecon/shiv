# Review Notes — Memory Pruning Patch

Tracking decisions made for each review file's critiques.

---

## review_lattner.md (C. Lattner — compiler engineer, adversarial review)

### Invariant Verdicts (informational — no action needed)

| Invariant | Verdict | Action |
|-----------|---------|--------|
| Semantic equivalence (short videos) | PASS | None |
| Conditioning frame preservation | PASS | None |
| Per-object dict synchronization | PASS | None |

### BUG-1: `_purge_memory()` does not update `score_index`

**Decision: ACCEPTED — implemented**

CoI-purged frames left stale entries in `score_index`, causing the backward scan to evaluate phantom candidates (functionally safe due to `i in non_cond_outputs` guard, but wasteful and a consistency violation).

**Fix:** Added `inference_state["score_index"].pop(frame_idx, None)` to `_purge_memory()` in `multi_object_tracker.py:96`, guarded by `if "score_index" in inference_state` for backwards compatibility with callers that don't use the pruning system.

### BUG-2: Pruning fires on conditioning-frame-only frames

**Decision: SKIPPED — no impact**

Reviewer noted this is a no-op (the function returns immediately when `len(non_cond) <= target`). The wasted CPU of a single dict length check every 100 frames is negligible. Not worth adding a guard.

### BUG-3: Score index populated with None values

**Decision: SKIPPED — already handled**

Reviewer confirmed this is functionally correct: `None` scores cause the frame to be skipped in the backward scan and ranked at -1.0 during landmark selection (making them eviction candidates). This is the desired behavior.

### BUG-5: Backward scan performance regression (O(N) on long videos)

**Decision: ACCEPTED — implemented as P1 (see below)**

---

### P0: Fix `_purge_memory` to clean up `score_index`

**Decision: ACCEPTED — implemented**

Same as BUG-1 above. File: `sam2/sam2/multi_object_tracker.py`, function `_purge_memory()`.

### P1: Bound the backward scan range

**Decision: ACCEPTED — implemented**

Replaced `range(frame_idx - 1, 1, -1)` with `range(frame_idx - 1, scan_limit, -1)` where:
```python
max_scan = getattr(self, 'max_recent_frames', 500) + getattr(self, 'max_landmark_frames', 50) * 4
scan_limit = max(1, frame_idx - max_scan)
```

This caps the scan at ~700 iterations instead of up to 60,000. Used `getattr` with defaults since `max_recent_frames`/`max_landmark_frames` are defined on `SAM2VideoPredictor` (subclass), not `SAM2Base`. The `* 4` multiplier on landmarks gives a generous buffer to find scattered landmark frames.

File: `sam2/sam2/modeling/sam2_base.py`, SAMURAI backward scan block.

### P2: Add score_index population for consolidated frames

**Decision: ACCEPTED — implemented**

Added `self._update_score_index(inference_state, frame_idx, current_out)` calls in both consolidated-frame branches of `propagate_in_video()`:
- After `cond_frame_outputs` lookup (line ~730)
- After `non_cond_frame_outputs` lookup (line ~738)

This ensures every frame that passes through the propagation loop gets indexed, not just newly-inferred frames. Prevents score gaps if consolidated non-cond frames are later evicted by pruning.

File: `sam2/sam2/sam2_video_predictor.py`, `propagate_in_video()`.

### P3: Add `_clear_non_cond_mem_around_input` cleanup for score_index

**Decision: ACCEPTED — implemented**

Added `score_index.pop(t, None)` alongside the existing `non_cond_frame_outputs.pop(t, None)` in the cleanup loop. Used `inference_state.get("score_index", {})` for safety.

Currently disabled by default (`clear_non_cond_mem_around_input=False`), but this prevents a future landmine if the feature is enabled.

File: `sam2/sam2/sam2_video_predictor.py`, `_clear_non_cond_mem_around_input()`.

### P4: Document the `score_index` aliasing invariant

**Decision: ACCEPTED — implemented**

Added comment to `init_state()`:
```python
# INVARIANT: score_index and output_dict["score_index"] are the SAME dict object.
# Use .clear() for reset, never reassign.
```

File: `sam2/sam2/sam2_video_predictor.py`, `init_state()`.

### P5: Consider pruning `score_index` itself

**Decision: SKIPPED — not needed at target scale**

At 60K frames, `score_index` is ~5 MB (negligible vs the ~25 GB saved). The reviewer marked this low priority. Would only matter at 1M+ frames, which is beyond our current target. Adding pruning would require a separate data structure or scan-range coordination, adding complexity for minimal benefit.

### Concern: Fragile `frame_idx - 1` implicit invariant

**Decision: ACKNOWLEDGED — no code change**

The reviewer noted that `frame_idx - 1` is unconditionally appended to `valid_indices` without checking if it was evicted. This is safe because `frame_idx - 1` is always within the recent window (`recent_window = max_recent_frames - memory_pruning_interval = 400`, and `frame_idx - 1` is 1 frame behind current). The P1 scan bound also ensures this frame is always reachable. No explicit guard added because the arithmetic guarantees safety and adding a redundant check would obscure the invariant rather than clarify it.

---

### Summary of changes from review_lattner.md

| File | Changes |
|------|---------|
| `sam2/sam2/multi_object_tracker.py` | P0: `_purge_memory` cleans `score_index` |
| `sam2/sam2/modeling/sam2_base.py` | P1: Bounded backward scan range |
| `sam2/sam2/sam2_video_predictor.py` | P2: Score index for consolidated frames |
| `sam2/sam2/sam2_video_predictor.py` | P3: `_clear_non_cond_mem_around_input` cleans score_index |
| `sam2/sam2/sam2_video_predictor.py` | P4: Aliasing invariant comment |

---

## review_sveng.md (S. V. Engineering — IEC 62304 Class C FMEA review)

### Failure Mode Analysis (FM items)

#### FM-1: Per-Object Dict Divergence on Partial Eviction Failure

**Decision: ADDRESSED via R2 (assertion)**

The concern is that an exception mid-loop during eviction could leave `output_dict` and `output_dict_per_obj` out of sync. This is theoretically possible but practically unlikely — the eviction loop only calls `dict.pop()`, which doesn't raise on missing keys. The R2 post-pruning assertion catches any divergence immediately rather than letting it propagate silently.

#### FM-2: Score Index Desynchronization / "Best Available" Degradation

**Decision: ACKNOWLEDGED — inherent tradeoff, ADDRESSED via R4 (metric)**

This is the fundamental tradeoff of memory pruning: evicted high-quality frames are unavailable even though their scores remain in `score_index`. The backward scan correctly falls back to the best *available* frame. The R4 `n_evicted_hits` counter quantifies this degradation so it can be monitored. The alternative (keeping all frames) is the 25 GB OOM that motivated this patch.

#### FM-3: Object Pointer Collection Reads Evicted Frames

**Decision: ADDRESSED via R1 (logging), but NOT a real risk for our pruning**

The object pointer loop walks `frame_idx - 1` through `frame_idx - 15`. With `max_recent_frames=500`, the recent window after pruning is 400 frames — so the last 16 frames are never evicted by `_prune_memory_bank()`. They could be evicted by CoI's `_purge_memory()`, but that's pre-existing behavior unrelated to this patch. The R1 counter provides observability as a safety net.

#### FM-4: CUDA Memory Not Actually Freed (Accumulator Drift)

**Decision: ACKNOWLEDGED — not actionable via code change**

The reviewer correctly identifies that PyTorch's caching allocator may not release freed blocks back to CUDA. This is a runtime behavior of PyTorch, not something our code controls. Calling `torch.cuda.empty_cache()` periodically would force release but adds CUDA synchronization latency. The pruning correctly removes all Python references (both packed dict and per-object views are popped), which is all we can do. Memory monitoring at the deployment level is the right approach, not code instrumentation.

Additionally: with `offload_state_to_cpu=True` (the default in `MultiObjectTracker.track()`), heavy tensors are on CPU where Python's reference counting handles deallocation promptly. The CUDA fragmentation concern only applies when offloading is disabled.

#### FM-5: Landmark Promotion Race

**Decision: SKIPPED — not applicable in current architecture**

The reviewer acknowledges this requires concurrent generator stepping, which doesn't happen in the single-threaded architecture. Python's GIL also prevents true concurrent dict mutation. If the architecture ever moves to multi-threaded inference, this would need revisiting, but that would require a much broader concurrency review beyond just pruning.

#### FM-6: Pruning Interval on Frame Index vs Frames Processed

**Decision: ACKNOWLEDGED — reverse propagation is safe (no-op) but provides no savings**

The reviewer correctly identifies that in reverse propagation, `recent_cutoff = frame_idx - recent_window` points to a lower frame index, making all existing frames "recent" and preventing eviction. This is actually safe behavior — nothing gets incorrectly evicted. However, it means pruning provides no memory savings during reverse propagation.

This is acceptable because: (1) reverse propagation is uncommon in production, (2) it's typically used for short segments (backward from a conditioning frame), not 60K-frame runs, and (3) the forward pass already pruned the memory bank, so reverse propagation starts from a bounded state.

#### FM-7: `score_index` Grows Unboundedly

**Decision: SKIPPED — same as lattner P5**

Already addressed in the lattner review section. At 60K frames, `score_index` is ~5 MB — negligible. The `_purge_memory` fix (lattner P0) now cleans CoI-purged entries too.

#### FM-8: `_update_score_index` Only Called for Non-Consolidated Frames

**Decision: ALREADY FIXED — same as lattner P2**

Already addressed in the lattner review: we added `_update_score_index` calls in both consolidated-frame branches of `propagate_in_video()`.

---

### State Mutations (M items)

| # | Mutation | Decision |
|---|----------|----------|
| M1 | `score_index = {}` at init | **Safe** — standard Python dict creation, GC handles re-init |
| M2 | Aliased `output_dict["score_index"]` | **Addressed** — lattner P4 added invariant comment |
| M3 | `_update_score_index` stores None for missing scores | **Safe** — backward scan handles None correctly |
| M4 | `output_dict.get("score_index", {})` fallback | **Safe** — correct graceful degradation for non-patched codepaths |
| M5 | `non_cond.pop(f, None)` in pruning | **Addressed** — R2 assertion verifies both dicts are consistent |
| M6 | Per-object `pop(f, None)` in pruning | **Addressed** — R2 assertion + R3 ordering fix |
| M7 | `score_index.clear()` in reset | **Safe** — preserves alias via `.clear()` not reassignment |
| M8 | Pruning trigger on `frame_idx % interval` | **Acknowledged** — FM-6 analysis above |
| M9 | Landmark sort by aggregate IoU | **Acknowledged** — see R7 below |
| M10 | `valid_indices` only if `i in non_cond_outputs` | **Addressed** — R4 counter quantifies eviction misses |

---

### Interface Boundary Analysis

#### Interface 1: `_prune_memory_bank()` vs `_purge_memory()`

**Decision: ACKNOWLEDGED — two eviction paths now both clean score_index (lattner P0)**

The reviewer's concern about neither function cleaning `score_index` was addressed by lattner P0. The concern about neither cleaning `consolidated_frame_inds` or `frames_already_tracked` is acknowledged but not actionable — these are metadata sets that correctly reflect "this frame was processed at some point," even if its output was later evicted. Callers that check these sets and then try to read output data already handle `None` returns.

#### Interface 2: `_prune_memory_bank()` vs `_add_output_per_object()` ordering

**Decision: ACCEPTED — implemented as R3**

See R3 below. Pruning now runs after `_add_output_per_object()`.

#### Interface 3: `score_index` aliasing via `output_dict`

**Decision: ADDRESSED — lattner P4 (invariant comment)**

The reviewer notes that dict reconstruction could break the alias. This is already documented by the P4 invariant comment. The `init_state()` method is the only place that constructs `output_dict`, and it correctly includes the alias. The `_reset_tracking_results()` method uses `.clear()` which preserves it.

#### Interface 4: Object Pointer Collection unguarded

**Decision: ADDRESSED via R1 (logging)**

See FM-3 analysis above. Not a real risk for our pruning parameters but logged for observability.

---

### Recommendations

#### R1 (P0): Guard Object Pointer Collection Against Eviction

**Decision: ACCEPTED — implemented as diagnostic counter**

Added `n_ptrs_skipped` counter in the object pointer collection loop in `sam2_base.py:783-793`. When a frame at `t >= 0` returns `None` from `non_cond_frame_outputs.get()`, the counter increments. This provides observability without changing behavior.

Note: with `max_recent_frames=500`, the last 16 frames are always within the 400-frame recent window. Our pruning never triggers this counter. It serves as a safety net for CoI purges or future config changes.

File: `sam2/sam2/modeling/sam2_base.py`, object pointer collection loop.

#### R2 (P0): Add Post-Pruning Consistency Assertion

**Decision: ACCEPTED — implemented**

Added `if __debug__:` assertion block at the end of `_prune_memory_bank()` that verifies `output_dict["non_cond_frame_outputs"]` and every per-object dict have identical key sets. This assertion is stripped in optimized mode (`python -O`) so there's zero production overhead.

Tested: intentionally breaking a per-object dict correctly triggers `AssertionError: Dict divergence after pruning...`.

File: `sam2/sam2/sam2_video_predictor.py`, `_prune_memory_bank()`.

#### R3 (P0): Move Pruning After `_add_output_per_object()`

**Decision: ACCEPTED — implemented**

Reordered `propagate_in_video()` so pruning runs after per-object slices are created:

1. Store in `output_dict` + update score_index + set `need_pruning = True`
2. `_add_output_per_object()` (for ALL frame types)
3. If `need_pruning`: check interval/overflow trigger, call `_prune_memory_bank()`

This ensures both `output_dict` and `output_dict_per_obj` are fully populated when pruning inspects them, eliminating the theoretical race where the current frame exists in one dict but not the other.

File: `sam2/sam2/sam2_video_predictor.py`, `propagate_in_video()`.

#### R4 (P1): Log Backward Scan Eviction Misses

**Decision: ACCEPTED — implemented as counter**

Added `n_evicted_hits` counter in the SAMURAI backward scan. When a frame passes all score thresholds but fails the `i in non_cond_outputs` check (i.e., its tensors were evicted), the counter increments. This is the single most important metric for quantifying the tracking quality cost of pruning.

The counter is computed but not currently logged per-frame (to avoid log spam). It can be inspected via debugger or added to a diagnostic callback if needed.

File: `sam2/sam2/modeling/sam2_base.py`, SAMURAI backward scan.

#### R5 (P1): Verify Actual Memory Reclamation

**Decision: SKIPPED — deployment monitoring concern, not code change**

Adding `torch.cuda.memory_allocated()` calls in the pruning path introduces CUDA synchronization points that could slow inference. Memory monitoring belongs at the deployment level (e.g., periodic RSS/VRAM sampling), not embedded in the hot path. With `offload_state_to_cpu=True`, heavy tensors are on CPU where refcount-based deallocation is immediate.

#### R6 (P1): Guard `_purge_memory()` Against Score Index Inconsistency

**Decision: SUPERSEDED by lattner P0**

The sveng review recommended keeping `score_index` entries alive after CoI purge (so the backward scan can "evaluate the frame's quality history"). The lattner review recommended the opposite: cleaning `score_index` on CoI purge to prevent phantom candidates.

**We followed lattner's approach** (clean score_index in `_purge_memory`). Rationale: a CoI-purged frame's tensors are gone, so its scores serve no purpose — the backward scan would evaluate the scores, find the frame passes thresholds, check `i in non_cond_outputs`, get False, and skip it. Cleaning the entry avoids this wasted evaluation. There is no use case where a CoI-purged frame's historical score is needed after purge.

#### R7 (P2): Consider Per-Object Landmark Selection

**Decision: SKIPPED — not applicable in current architecture**

In the `MultiObjectTracker`, each object gets its own predictor with its own `inference_state` and `output_dict`. Each predictor tracks B=1, so `best_iou_score` is already per-object. The aggregate-score concern only applies if using `SAM2VideoPredictor` directly with B>1 in SAMURAI mode, which is not the production usage pattern.

If direct multi-object SAMURAI use becomes a requirement, this should be revisited.

#### R8 (P2): Add `score_index` to Debug Dump

**Decision: SKIPPED — low priority**

The `score_index` dict is directly accessible via `inference_state["score_index"]` for any post-hoc analysis. Adding it to the diagnostics capture system in `multi_object_tracker.py` would require modifying code outside the pruning patch scope. Not worth the cross-cutting change for a debug convenience.

---

### Observability Assessment (Section 4)

#### 4.1 Must-Have (P0)

| Observable | Decision |
|-----------|----------|
| Frames skipped by backward scan due to eviction | **ACCEPTED** — R4 `n_evicted_hits` counter |
| Object pointer count per inference step | **ACCEPTED** — R1 `n_ptrs_skipped` counter |
| Actual memory freed per pruning cycle | **SKIPPED** — deployment monitoring, not code instrumentation |
| Per-object eviction consistency | **ACCEPTED** — R2 `__debug__` assertion |

#### 4.2 Should-Have (P1)

| Observable | Decision |
|-----------|----------|
| Landmark set stability | **SKIPPED** — landmarks are recomputed each cycle by design; churn is expected and harmless |
| Score index size | **SKIPPED** — negligible at target scale (~5 MB at 60K) |
| Pruning in reverse propagation | **SKIPPED** — reverse is safe (no-op); see FM-6 |

#### 4.3 Nice-to-Have (P2)

| Observable | Decision |
|-----------|----------|
| Quality delta: selected vs ideal | **SKIPPED** — would require tracking "ideal" set which doubles score bookkeeping |
| Caching allocator fragmentation | **SKIPPED** — PyTorch internals, not actionable from our code |

---

### Summary of changes from review_sveng.md

| File | Changes |
|------|---------|
| `sam2/sam2/modeling/sam2_base.py` | R1: `n_ptrs_skipped` counter in object pointer loop |
| `sam2/sam2/modeling/sam2_base.py` | R4: `n_evicted_hits` counter in backward scan |
| `sam2/sam2/sam2_video_predictor.py` | R2: Post-pruning `__debug__` consistency assertion |
| `sam2/sam2/sam2_video_predictor.py` | R3: Moved pruning after `_add_output_per_object()` |

---

## review_sre_chaos.md (SRE/Chaos Engineering — operational durability review)

### Scenario Analysis (informational — confirms the patch works)

| Scenario | Verdict | Notes |
|----------|---------|-------|
| S1: 60K-frame marathon | **Survives** | Memory capped at ~1.65 GB across 4 predictors. Primary risk is CPU from backward scan. |
| S2: Overnight 50-video batch | **Clean** | `finally` block drops all references. No cross-video accumulation. |
| S4: Adversarial pruning cycle | **Well-behaved** | Same-size alloc/free patterns, mmap-backed. No fragmentation. |
| S5: Multi-GPU | **Safe** | All tensors have explicit device targets. No default-device landmines. |
| S6: Ctrl+C during pruning | **Safe** | Partially-pruned state is valid. `finally` block cleans up in <500ms. |
| S8: `frames_already_tracked` | **Bounded** | ~18 MB/predictor at 60K. Cleared between videos. |

### Time Bombs

#### TB-1: Diagnostics list OOM (`capture_diagnostics=True` on long videos)

**Decision: ACCEPTED — implemented**

The diagnostics list stores full-resolution composites (~3.15 MB/frame at 1024x1024). At 20K frames this hits ~63 GB, causing OOM on 64 GB machines. The pruning patch solves tensor memory but the diagnostics list is a completely separate, unguarded accumulation path.

**Fix:** Changed `self.diagnostics` from `list` to `collections.deque(maxlen=10000)` in `multi_object_tracker.py`. This automatically drops the oldest entry when the cap is exceeded — O(1) with no performance overhead. The deque is compatible with `save_diagnostics()` which iterates over it.

10,000 entries at 3.15 MB = ~31.5 GB worst case, which is large but stays within 64 GB machines. For tighter environments, the maxlen can be reduced.

File: `sam2/sam2/multi_object_tracker.py`, `track()` initialization and imports.

#### TB-2: Landmark selection degeneracy on static scenes

**Decision: ACCEPTED — implemented**

On static scenes where all frames have IoU ~1.0, the global top-K landmark selection was biased toward the oldest frames (due to CPython's insertion-order dict iteration + stable sort). This meant landmarks clustered at frames 0-49, providing no temporal diversity when the scene changes.

**Fix:** Replaced global top-K with temporal bucketing in `_prune_memory_bank()`. Old frames are divided into `max_landmark_frames` temporal buckets, and the best-scoring frame from each bucket becomes a landmark. If there are fewer buckets than `max_landmark_frames`, remaining slots are filled by global top-scoring frames.

Tested: on a 40K-frame static scene (all IoU=0.95), landmarks now span from frame 1 to frame 39551 (diversity ratio 1.00) instead of clustering at frames 0-49.

File: `sam2/sam2/sam2_video_predictor.py`, `_prune_memory_bank()`.

#### TB-3: Backward scan CPU stall (O(N) worst case)

**Decision: ALREADY ADDRESSED — lattner P1**

The backward scan range was already bounded in the lattner review to `max(1, frame_idx - 700)`. This caps the scan at ~700 iterations instead of O(60K). The reviewer's suggested `max_scan_depth` parameter is functionally equivalent to what we already have, just derived from existing config parameters rather than a new one.

#### TB-4: Re-init peak VRAM spike

**Decision: SKIPPED — outside pruning patch scope**

Re-initialization creates a second predictor temporarily (~2x VRAM). This is a pre-existing behavior in the health-check system unrelated to memory pruning. The reviewer's suggestion to serialize re-inits is valid but would require changes to the health-check orchestration logic, which is outside the scope of this patch.

#### TB-5: `score_index` on ultra-long videos

**Decision: SKIPPED — same as lattner P5, sveng FM-7**

Already addressed in both previous reviews. ~18 MB at 60K, ~180 MB at 600K. Acceptable at our target scale.

---

### Scenario 3: Degenerate Video (Static + Crossings)

**Decision: ADDRESSED via TB-2 (temporal bucketing)**

The reviewer identified that on 40K motionless frames followed by crossings, landmarks would all be from the static period with no visual diversity. The temporal bucketing fix ensures landmarks are spread across the full temporal range, so when crossings begin, the memory bank contains frames from different time periods.

Note: this improves temporal diversity but doesn't guarantee *visual* diversity — if 40K frames are truly identical, no landmark selection strategy can provide different visual content. The improvement helps when there are subtle visual changes over time that the global top-K would miss.

---

### Scenario 7: Diagnostics Shadow Accumulation

**Decision: ADDRESSED via TB-1 (deque cap)**

See TB-1 above. The reviewer's observation that "the pruning system has zero knowledge of this list" is correct — the diagnostics list is a completely separate accumulation path. The deque cap is the minimal fix. The reviewer's alternative suggestion (incremental disk writes) would be more thorough but requires redesigning the diagnostics pipeline, which is outside scope.

---

### Guardrails

| # | Guardrail | Decision |
|---|-----------|----------|
| 1 | Diagnostics auto-cap | **ACCEPTED** — deque(maxlen=10000) |
| 2 | score_index pruning | **SKIPPED** — negligible at target scale |
| 3 | Backward scan depth limit | **ALREADY DONE** — lattner P1 |
| 4 | RSS watchdog | **SKIPPED** — ops infrastructure, not code patch |

### Kill Switches

| # | Kill Switch | Decision |
|---|-------------|----------|
| K-1 | Sentinel file check | **SKIPPED** — useful but outside pruning scope |
| K-2-3 | SIGTERM/SIGINT | **Already safe** — Scenario 6 analysis confirms |
| K-4 | SIGUSR1 diagnostics dump | **SKIPPED** — useful but outside pruning scope |
| K-5 | SIGKILL | **Always safe** — OS handles cleanup |

### Production Config Recommendations

**Decision: ACKNOWLEDGED — no code change needed**

The reviewer's recommended config is already the default:
```yaml
capture_diagnostics: false          # default
offload_state_to_cpu: true          # default in MultiObjectTracker.track()
max_recent_frames: 500              # default
max_landmark_frames: 50             # default
memory_pruning_interval: 100        # default
```

The suggestion to increase `max_landmark_frames` to 100 for crossing-heavy videos is noted but left as a user tuning decision rather than a default change.

### Fragmentation Analysis

**Decision: ACKNOWLEDGED — confirms patch is well-behaved**

The reviewer's fragmentation analysis confirms that:
- mmap-backed tensor allocations (>128 KB) return cleanly to the OS
- CUDA caching allocator handles same-size blocks efficiently
- Python heap fragmentation from small dicts is ~200-500 MB per video cycle — acceptable
- Overall fragmentation ratio is 1.0-1.4x, which is excellent

No code changes needed — this validates the implementation approach.

---

### Summary of changes from review_sre_chaos.md

| File | Changes |
|------|---------|
| `sam2/sam2/multi_object_tracker.py` | TB-1: `deque(maxlen=10000)` for diagnostics + `from collections import deque` |
| `sam2/sam2/sam2_video_predictor.py` | TB-2: Temporal bucketing for landmark selection in `_prune_memory_bank()` |
