"""
Multi-Object Tracker with Cross-Object Interaction (CoI) for SAMURAI.

Manages N independent SAMURAI predictors in lockstep. After each frame,
performs pairwise mask-IoU collision detection. When two masks overlap
above `iou_threshold`, the lower-scoring tracker's memory for that frame
is purged so that it can recover on subsequent frames instead of drifting
onto the other object.

Reference: SAM2MOT (arXiv 2504.04519) Cross-Object Interaction module.

Usage:
    from sam2.multi_object_tracker import MultiObjectTracker

    mot = MultiObjectTracker(
        config_file="configs/samurai/sam2.1_hiera_t.yaml",
        ckpt_path="path/to/checkpoint.pt",
    )
    for frame_idx, masks, purge_events in mot.track(
        video_dir="path/to/frames",
        init_boxes={0: [x1,y1,x2,y2], 1: [x1,y1,x2,y2]},
    ):
        # masks: dict  {obj_id: np.ndarray}  float32 logits, full resolution
        # purge_events: list of dicts describing any CoI purges this frame
        ...

    # If capture_diagnostics=True was passed to track():
    mot.save_diagnostics("diag.npz")
"""

import gc
import logging
import os
import os.path as osp
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple

if TYPE_CHECKING:
    from sam2.identity_verifier import IdentityVerifier

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.diagnostics import save_diagnostics, load_diagnostics


# ---------------------------------------------------------------------------
# Track State Machine
# ---------------------------------------------------------------------------

class TrackTier(str, Enum):
    RELIABLE = "reliable"      # logits > 8.0
    PENDING = "pending"        # 6.0 < logits <= 8.0
    SUSPICIOUS = "suspicious"  # 2.0 < logits <= 6.0
    LOST = "lost"              # logits <= 2.0


# Ordered from best to worst for comparison
_TIER_ORDER = [TrackTier.RELIABLE, TrackTier.PENDING, TrackTier.SUSPICIOUS, TrackTier.LOST]
_TIER_RANK = {t: i for i, t in enumerate(_TIER_ORDER)}


@dataclass
class TrackState:
    tier: TrackTier = TrackTier.RELIABLE
    frames_in_tier: int = 0
    lost_counter: int = 0          # consecutive frames in Lost (for death)
    recovery_counter: int = 0      # consecutive frames above current tier (for hysteresis)
    last_logits: float = 10.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of identical shape."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) centroid of a boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def mask_bbox(mask: np.ndarray) -> List[int]:
    """Return [x, y, w, h] bounding box of a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _purge_memory(inference_state: dict, frame_idx: int) -> int:
    """
    Remove frame `frame_idx` from the tracker's memory bank.

    Pops from both the packed output_dict and every per-object slice in
    output_dict_per_obj. Returns the number of entries actually removed.
    """
    removed = 0
    output_dict = inference_state["output_dict"]

    if output_dict["non_cond_frame_outputs"].pop(frame_idx, None) is not None:
        removed += 1

    for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
        if obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None) is not None:
            removed += 1

    return removed


def _get_score(current_out: dict) -> float:
    """
    Extract a scalar confidence score from a tracker's current output.

    Handles both tensor scores (normal path) and literal int scores
    (from _use_mask_as_output fallback path where best_iou_score=1).
    """
    iou_score = current_out.get("best_iou_score")
    obj_score = current_out.get("object_score_logits")

    score = iou_score if iou_score is not None else obj_score
    if score is None:
        return 0.0
    if isinstance(score, (int, float)):
        return float(score)
    return float(score.item())


def _get_scores_dict(current_out: dict) -> Dict[str, float]:
    """Extract all three score components as a dict."""
    def _to_float(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return float(v.item())

    return {
        "iou": _to_float(current_out.get("best_iou_score")),
        "obj": _to_float(current_out.get("object_score_logits")),
        "kf": _to_float(current_out.get("kf_score")),
    }


def _get_obj_logits(current_out: dict) -> float:
    """
    Extract ``object_score_logits`` as a Python float.

    Handles tensors (normal path), literal ints (``_use_mask_as_output``
    fallback where score=1), and None (missing key).  Returns 10.0 as
    default — conditioning frames emit +10.0, which maps to Reliable.
    """
    v = current_out.get("object_score_logits")
    if v is None:
        return 10.0
    if isinstance(v, (int, float)):
        return float(v)
    return float(v.item())


def _compute_tier(
    logits: float,
    thresholds: Dict[str, float],
) -> TrackTier:
    """Classify *logits* into a tier using *thresholds*."""
    if logits > thresholds["reliable"]:
        return TrackTier.RELIABLE
    if logits > thresholds["pending"]:
        return TrackTier.PENDING
    if logits > thresholds["suspicious"]:
        return TrackTier.SUSPICIOUS
    return TrackTier.LOST


def _update_track_state(
    ts: TrackState,
    logits: float,
    thresholds: Dict[str, float],
    recovery_frames: int = 3,
) -> Optional[Tuple[TrackTier, TrackTier]]:
    """
    Update *ts* in-place and return ``(from_tier, to_tier)`` on transition.

    - **Degradation** (toward Lost): immediate.
    - **Recovery** (toward Reliable): requires *recovery_frames* consecutive
      frames classified above the current tier.
    """
    ts.last_logits = logits
    new_tier = _compute_tier(logits, thresholds)

    old_rank = _TIER_RANK[ts.tier]
    new_rank = _TIER_RANK[new_tier]

    if new_rank > old_rank:
        # Degradation — immediate
        from_tier = ts.tier
        ts.tier = new_tier
        ts.frames_in_tier = 1
        ts.recovery_counter = 0
        if new_tier == TrackTier.LOST:
            ts.lost_counter += 1
        else:
            ts.lost_counter = 0
        return (from_tier, new_tier)

    if new_rank < old_rank:
        # Candidate for recovery — require hysteresis
        ts.recovery_counter += 1
        ts.frames_in_tier += 1
        ts.lost_counter = 0
        if ts.recovery_counter >= recovery_frames:
            from_tier = ts.tier
            ts.tier = new_tier
            ts.frames_in_tier = 1
            ts.recovery_counter = 0
            return (from_tier, new_tier)
        return None

    # Same tier
    ts.frames_in_tier += 1
    ts.recovery_counter = 0
    if ts.tier == TrackTier.LOST:
        ts.lost_counter += 1
    else:
        ts.lost_counter = 0
    return None


def _bake_composite(frame_bgr: np.ndarray, masks: dict, obj_ids: list) -> np.ndarray:
    """Burn colored mask overlays onto a BGR video frame. Returns RGB uint8."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    colors = [
        (255, 50, 50),    # red
        (50, 130, 255),   # blue
        (50, 255, 80),    # green
        (255, 200, 0),    # yellow
        (255, 50, 255),   # magenta
        (0, 255, 255),    # cyan
    ]
    alpha = 0.45
    for i, oid in enumerate(obj_ids):
        mask_logits = masks.get(oid)
        if mask_logits is None:
            continue
        binary = mask_logits > 0.0
        clr = colors[i % len(colors)]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[binary] = clr
        out[binary] = out[binary] * (1 - alpha) + overlay[binary] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_kf_state(predictor) -> Optional[np.ndarray]:
    """
    Get the Kalman Filter state [cx, cy, aspect, height, vx, vy, va, vh]
    from a SAMURAI predictor. Returns None if KF not yet initialized.
    """
    if predictor.kf_mean is None:
        return None
    mean = predictor.kf_mean
    if isinstance(mean, np.ndarray):
        return mean.copy()
    return np.array(mean)


# ---------------------------------------------------------------------------
# MultiObjectTracker
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    """
    Lockstep multi-object tracker with Cross-Object Interaction.

    Each object gets its own SAMURAI predictor instance, sidestepping the
    model-global Kalman Filter state issue (T1.1). Generators are advanced
    one frame at a time; after each step, pairwise mask IoU is checked and
    colliding trackers have their memory purged.

    Parameters
    ----------
    config_file : str
        Hydra config path relative to sam2 package root,
        e.g. "configs/samurai/sam2.1_hiera_t.yaml"
    ckpt_path : str
        Path to the SAM2/SAMURAI checkpoint file.
    device : str
        CUDA device string, e.g. "cuda:0".
    apply_postprocessing : bool
        Whether to apply SAM2 postprocessing (hole filling, etc.).
    """

    def __init__(
        self,
        config_file: str,
        ckpt_path: str,
        device: str = "cuda:0",
        apply_postprocessing: bool = True,
    ):
        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.device = device
        self.apply_postprocessing = apply_postprocessing
        self.diagnostics = []

    def track(
        self,
        video_dir: str,
        init_boxes: Dict[int, list],
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        iou_threshold: float = 0.8,
        capture_diagnostics: bool = False,
        health_check: Optional[Callable] = None,
        health_check_interval: int = 30,
        identity_verifier: Optional["IdentityVerifier"] = None,
        tier_thresholds: Optional[Dict[str, float]] = None,
        tier_recovery_frames: int = 3,
        lost_tolerance: int = 25,
        event_driven_health_check: bool = True,
        event_health_check_cooldown: int = 15,
    ) -> Generator[Tuple[int, Dict[int, np.ndarray], List[dict]], None, None]:
        """
        Run lockstep multi-object tracking with Cross-Object Interaction.

        Parameters
        ----------
        video_dir : str
            Path to directory of video frames (sorted alphabetically).
        init_boxes : dict
            Mapping of {obj_id: [x1, y1, x2, y2]} for first-frame bounding boxes.
        offload_video_to_cpu : bool
            Offload decoded video frames to CPU to save GPU memory.
        offload_state_to_cpu : bool
            Offload inference state tensors to CPU between frames.
        iou_threshold : float
            Pairwise mask IoU above which Cross-Object Interaction triggers.
        capture_diagnostics : bool
            If True, store per-frame diagnostic data in self.diagnostics.
            Access after generator exhaustion, or call self.save_diagnostics().
        health_check : callable, optional
            ``Callable(frame_idx, masks, scores, frame) -> dict[int, list]``.
            Called every *health_check_interval* frames with the current
            frame index, mask dict ``{obj_id: np.ndarray}``, score dict
            ``{obj_id: float}``, and BGR video frame ``np.ndarray``.
            Returns ``{obj_id: [x1, y1, x2, y2]}`` for any trackers that
            should be re-initialized with a new bounding box.
            Return ``{}`` when all trackers are healthy.
        health_check_interval : int
            Run the health check every N frames (default 30).
        identity_verifier : IdentityVerifier, optional
            If provided, monitors crossing events and detects identity swaps
            by comparing visual embeddings before and after crossings.
        tier_thresholds : dict, optional
            Logit thresholds for track tiers: ``{"reliable": 8.0, "pending":
            6.0, "suspicious": 2.0}``.  Defaults provided if None.
        tier_recovery_frames : int
            Consecutive frames above current tier required for promotion.
        lost_tolerance : int
            Consecutive Lost-tier frames before a track is killed.
        event_driven_health_check : bool
            If True, trigger an extra health check when any object enters
            the Pending tier (subject to cooldown).
        event_health_check_cooldown : int
            Minimum frames between event-driven health checks.

        Yields
        ------
        frame_idx : int
            Current frame index.
        masks : dict
            {obj_id: np.ndarray} of float32 mask logits at original video
            resolution. Shape (H, W). Caller decides threshold.
        purge_events : list of dict
            Each entry describes a CoI purge event with keys:
            - frame_idx: int
            - obj_pair: (int, int) — the two colliding object IDs
            - iou: float — their mask IoU
            - scores: (float, float) — confidence scores for the pair
            - purged_obj: int or "both" — which object's memory was purged
              (int = variance-based single victim, "both" = freeze-both fallback)
            - variances: (float, float) — score variance over last 10 frames
            - var_ratio: float — max/min variance ratio (>= 2.0 = discriminative)
            - purged: int — number of memory entries removed
        """
        obj_ids = sorted(init_boxes.keys())
        n_objects = len(obj_ids)
        if n_objects == 0:
            return

        self.reinit_events = []
        self.swap_events = []

        # Pre-load sorted frame file paths (needed for diagnostics and health check)
        _frame_files = None
        if capture_diagnostics or health_check is not None or identity_verifier is not None:
            _exts = (".jpg", ".jpeg", ".png", ".bmp")
            _frame_files = sorted([
                osp.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.lower().endswith(_exts)
            ])

        if capture_diagnostics:
            self.diagnostics = []

        # Build N independent predictors
        predictors = {}
        states = {}
        generators = {}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
            for oid in obj_ids:
                pred = build_sam2_video_predictor(
                    self.config_file,
                    self.ckpt_path,
                    device=self.device,
                    apply_postprocessing=self.apply_postprocessing,
                )
                state = pred.init_state(
                    video_dir,
                    offload_video_to_cpu=offload_video_to_cpu,
                    offload_state_to_cpu=offload_state_to_cpu,
                )
                box = init_boxes[oid]
                pred.add_new_points_or_box(
                    state, box=box, frame_idx=0, obj_id=0,
                )
                predictors[oid] = pred
                states[oid] = state
                generators[oid] = pred.propagate_in_video(state)

            # Score history for variance-based CoI victim selection (SAM2MOT-inspired)
            _VARIANCE_WINDOW = 10
            score_history = {oid: deque(maxlen=_VARIANCE_WINDOW) for oid in obj_ids}

            # Track state machine (4a)
            _tier_thresholds = tier_thresholds or {
                "reliable": 8.0, "pending": 6.0, "suspicious": 2.0,
            }
            track_states: Dict[int, TrackState] = {
                oid: TrackState() for oid in obj_ids
            }
            _last_health_check_frame = -999

            # Step all generators in lockstep
            exhausted = set()
            try:
              while True:
                frame_masks = {}    # {obj_id: np.ndarray} float32 logits
                frame_outputs = {}  # {obj_id: current_out dict} for scoring
                current_frame_idx = None
                yielded_frame_idxs = {}

                for oid in obj_ids:
                    if oid in exhausted:
                        continue
                    try:
                        frame_idx, _obj_ids, video_res_masks = next(generators[oid])
                    except StopIteration:
                        exhausted.add(oid)
                        if identity_verifier is not None:
                            identity_verifier.reset_object(oid)
                        continue
                    except Exception as e:
                        logging.error(
                            "Tracker oid=%d failed: %s: %s. Marking exhausted.",
                            oid, type(e).__name__, e,
                        )
                        exhausted.add(oid)
                        if identity_verifier is not None:
                            identity_verifier.reset_object(oid)
                        try:
                            generators[oid].close()
                        except Exception:
                            pass
                        continue

                    yielded_frame_idxs[oid] = frame_idx
                    current_frame_idx = frame_idx
                    mask_logits = video_res_masks[0, 0].cpu().numpy().astype(np.float32)
                    frame_masks[oid] = mask_logits

                    out_dict = states[oid]["output_dict"]
                    current_out = out_dict["non_cond_frame_outputs"].get(frame_idx)
                    if current_out is None:
                        current_out = out_dict["cond_frame_outputs"].get(frame_idx, {})
                    frame_outputs[oid] = current_out
                    score_history[oid].append(_get_score(current_out))

                if not frame_masks:
                    break

                # Partial exhaustion: some generators done while others active
                if exhausted and frame_masks:
                    logging.warning(
                        "Partial exhaustion: exhausted=%s, active=%s",
                        exhausted, set(frame_masks.keys()),
                    )

                # Lockstep invariant: all generators must yield the same frame_idx
                if len(yielded_frame_idxs) > 1:
                    unique_frames = set(yielded_frame_idxs.values())
                    assert len(unique_frames) == 1, (
                        f"LOCKSTEP VIOLATION: generators yielded different "
                        f"frame_idxs: {yielded_frame_idxs}"
                    )

                # --- Track state update (4b) ---
                state_transitions = []
                pending_oids = []
                dead_oids = []
                for oid in list(frame_masks.keys()):
                    if oid not in track_states:
                        continue
                    logits = _get_obj_logits(frame_outputs[oid])
                    transition = _update_track_state(
                        track_states[oid], logits,
                        _tier_thresholds, tier_recovery_frames,
                    )
                    if transition is not None:
                        state_transitions.append({
                            "obj_id": oid,
                            "from": transition[0].value,
                            "to": transition[1].value,
                            "logits": logits,
                        })
                    ts = track_states[oid]
                    if ts.tier == TrackTier.PENDING:
                        pending_oids.append(oid)
                    if ts.lost_counter >= lost_tolerance:
                        dead_oids.append(oid)

                # --- Track death (4c) ---
                track_death_events = []
                for oid in dead_oids:
                    logging.warning(
                        "TRACK DEATH oid=%d at frame %d "
                        "(lost_counter=%d >= tolerance=%d)",
                        oid, current_frame_idx,
                        track_states[oid].lost_counter, lost_tolerance,
                    )
                    try:
                        generators[oid].close()
                    except Exception:
                        pass
                    exhausted.add(oid)
                    del track_states[oid]
                    score_history[oid] = deque(maxlen=_VARIANCE_WINDOW)
                    if identity_verifier is not None:
                        identity_verifier.reset_object(oid)
                    frame_masks.pop(oid, None)
                    frame_outputs.pop(oid, None)
                    track_death_events.append({
                        "frame_idx": current_frame_idx,
                        "obj_id": oid,
                    })

                # --- Cross-Object Interaction ---
                purge_events = []
                active_ids = list(frame_masks.keys())
                bool_masks = {oid: frame_masks[oid] > 0.0 for oid in active_ids}
                pairwise_ious = {}

                for i in range(len(active_ids)):
                    for j in range(i + 1, len(active_ids)):
                        oid_a = active_ids[i]
                        oid_b = active_ids[j]

                        iou = mask_iou(bool_masks[oid_a], bool_masks[oid_b])
                        pairwise_ious[(oid_a, oid_b)] = iou

                        if iou <= iou_threshold:
                            continue

                        score_a = _get_score(frame_outputs[oid_a])
                        score_b = _get_score(frame_outputs[oid_b])

                        # Variance-based victim selection (SAM2MOT-inspired)
                        # High variance = score dropped abruptly = occluded victim
                        # Low variance = gradual degradation = occluder
                        var_a = float(np.var(score_history[oid_a])) if len(score_history[oid_a]) >= 2 else 0.0
                        var_b = float(np.var(score_history[oid_b])) if len(score_history[oid_b]) >= 2 else 0.0

                        min_var = min(var_a, var_b)
                        max_var = max(var_a, var_b)
                        var_ratio = max_var / min_var if min_var > 1e-8 else float('inf')

                        _VAR_RATIO_THRESH = 2.0

                        if var_ratio >= _VAR_RATIO_THRESH:
                            # Variance is discriminative — purge only the victim
                            victim_oid = oid_a if var_a > var_b else oid_b
                            n_purged = _purge_memory(states[victim_oid], current_frame_idx)
                            purged_label = victim_oid
                        else:
                            # Variances too close — freeze both (fallback)
                            n_purged_a = _purge_memory(states[oid_a], current_frame_idx)
                            n_purged_b = _purge_memory(states[oid_b], current_frame_idx)
                            n_purged = n_purged_a + n_purged_b
                            purged_label = "both"

                        purge_events.append({
                            "frame_idx": current_frame_idx,
                            "obj_pair": (oid_a, oid_b),
                            "iou": iou,
                            "scores": (score_a, score_b),
                            "variances": (var_a, var_b),
                            "var_ratio": var_ratio,
                            "purged_obj": purged_label,
                            "purged": n_purged,
                        })

                # --- Lazy frame load (4d) ---
                _frame_bgr_cache = [None]  # mutable box for closure

                def _load_frame():
                    if _frame_bgr_cache[0] is None and _frame_files is not None:
                        _frame_bgr_cache[0] = cv2.imread(
                            _frame_files[current_frame_idx],
                        )
                    return _frame_bgr_cache[0]

                # Eagerly load for always-on consumers
                if identity_verifier is not None or capture_diagnostics:
                    _load_frame()

                # --- Identity Verification ---
                swap_events = []
                if identity_verifier is not None:
                    # Reset pair states for CoI purge victims (P1: CoI-mid)
                    for pe in purge_events:
                        for oid in pe["obj_pair"]:
                            identity_verifier.reset_object(oid)

                    swap_events = identity_verifier.update(
                        frame_idx=current_frame_idx,
                        pairwise_ious=pairwise_ious,
                        bool_masks=bool_masks,
                        frame_bgr=_load_frame(),
                    )

                    # Resolve conflicting swaps: highest margin wins (S3)
                    actual_swaps = [
                        e for e in swap_events if e["action"] == "swap"
                    ]
                    if len(actual_swaps) > 1:
                        actual_swaps.sort(
                            key=lambda e: e["margin"], reverse=True,
                        )
                        claimed_oids = set()
                        for event in actual_swaps:
                            a, b = event["obj_pair"]
                            if a in claimed_oids or b in claimed_oids:
                                logging.warning(
                                    "Deferring conflicting swap %s at "
                                    "frame %d (oid already claimed)",
                                    event["obj_pair"], current_frame_idx,
                                )
                                event["action"] = "deferred"
                            else:
                                claimed_oids.add(a)
                                claimed_oids.add(b)

                    # Apply non-conflicting swaps
                    for event in swap_events:
                        if event["action"] == "swap":
                            oid_a, oid_b = event["obj_pair"]
                            logging.info(
                                "SWAP DETECTED frame %d: oid %d <-> %d "
                                "(margin=%.4f)",
                                current_frame_idx, oid_a, oid_b,
                                event["margin"],
                            )
                            # Atomic relabel
                            predictors[oid_a], predictors[oid_b] = (
                                predictors[oid_b], predictors[oid_a])
                            states[oid_a], states[oid_b] = (
                                states[oid_b], states[oid_a])
                            generators[oid_a], generators[oid_b] = (
                                generators[oid_b], generators[oid_a])
                            frame_masks[oid_a], frame_masks[oid_b] = (
                                frame_masks[oid_b], frame_masks[oid_a])
                            frame_outputs[oid_a], frame_outputs[oid_b] = (
                                frame_outputs[oid_b], frame_outputs[oid_a])
                            bool_masks[oid_a], bool_masks[oid_b] = (
                                bool_masks[oid_b], bool_masks[oid_a])
                            score_history[oid_a], score_history[oid_b] = (
                                score_history[oid_b], score_history[oid_a])
                            # Swap track states (4g)
                            if oid_a in track_states and oid_b in track_states:
                                track_states[oid_a], track_states[oid_b] = (
                                    track_states[oid_b], track_states[oid_a])
                            # Swap exhausted status (P0: L1.3)
                            if oid_a in exhausted and oid_b not in exhausted:
                                exhausted.discard(oid_a)
                                exhausted.add(oid_b)
                            elif oid_b in exhausted and oid_a not in exhausted:
                                exhausted.discard(oid_b)
                                exhausted.add(oid_a)
                            # Invalidate sibling pair states (P1: S2)
                            identity_verifier.notify_swap(oid_a, oid_b)

                self.swap_events.extend(swap_events)

                # Snapshot KF states before re-init may swap predictors
                pre_reinit_kf = {
                    oid: _get_kf_state(predictors[oid])
                    for oid in active_ids
                }

                # --- Health check & re-initialization (4e) ---
                reinit_events = []
                _interval_trigger = (
                    current_frame_idx > 0
                    and current_frame_idx % health_check_interval == 0
                )
                _event_trigger = (
                    event_driven_health_check
                    and pending_oids
                    and (current_frame_idx - _last_health_check_frame)
                    >= event_health_check_cooldown
                )
                if health_check is not None and (_interval_trigger or _event_trigger):
                    _last_health_check_frame = current_frame_idx
                    try:
                        reinit_boxes = health_check(
                            current_frame_idx,
                            frame_masks,
                            {oid: _get_score(frame_outputs[oid])
                             for oid in active_ids},
                            _load_frame(),
                        )
                    except Exception as e:
                        logging.warning(
                            "Health check raised %s at frame %d: %s. "
                            "Skipping re-init.",
                            type(e).__name__, current_frame_idx, e,
                        )
                        reinit_boxes = {}

                    if reinit_boxes is None:
                        reinit_boxes = {}

                    # Skip re-init near end of video (wasteful checkpoint load)
                    num_frames = next(iter(states.values()))["num_frames"]

                    for oid, box in reinit_boxes.items():
                        if oid not in predictors:
                            logging.warning(
                                "Health check returned unknown oid=%d. "
                                "Skipping.", oid,
                            )
                            continue

                        if current_frame_idx >= num_frames - 2:
                            continue

                        # Gate re-init by tier (4f): Reliable tracks don't need it
                        ts = track_states.get(oid)
                        if ts is not None and ts.tier == TrackTier.RELIABLE:
                            continue

                        try:
                            # BUILD PHASE — old predictor untouched
                            new_pred = build_sam2_video_predictor(
                                self.config_file, self.ckpt_path,
                                device=self.device,
                                apply_postprocessing=self.apply_postprocessing,
                            )
                            new_state = new_pred.init_state(
                                video_dir,
                                offload_video_to_cpu=offload_video_to_cpu,
                                offload_state_to_cpu=offload_state_to_cpu,
                            )
                            new_pred.add_new_points_or_box(
                                new_state, box=box,
                                frame_idx=current_frame_idx, obj_id=0,
                            )
                            new_gen = new_pred.propagate_in_video(
                                new_state,
                                start_frame_idx=current_frame_idx + 1,
                            )

                            # COMMIT PHASE — atomic swap
                            old_gen = generators[oid]
                            predictors[oid] = new_pred
                            states[oid] = new_state
                            generators[oid] = new_gen
                            if oid in exhausted:
                                exhausted.discard(oid)

                            # CLEANUP PHASE — release old resources
                            old_gen.close()
                            del old_gen
                            gc.collect()
                            torch.cuda.empty_cache()

                            # Reset score history and track state after re-init
                            score_history[oid] = deque(maxlen=_VARIANCE_WINDOW)
                            track_states[oid] = TrackState()

                            # Reset verifier pair states (P1: S4)
                            if identity_verifier is not None:
                                identity_verifier.reset_object(oid)

                            reinit_events.append({
                                "frame_idx": current_frame_idx,
                                "obj_id": oid,
                                "box": list(box),
                            })

                        except Exception as e:
                            logging.error(
                                "Re-init failed for oid=%d at frame %d: "
                                "%s: %s. Old tracker continues.",
                                oid, current_frame_idx,
                                type(e).__name__, e,
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue

                self.reinit_events.extend(reinit_events)

                # --- Capture diagnostics ---
                if capture_diagnostics:
                    _fbgr = _load_frame()
                    composite = _bake_composite(
                        _fbgr, frame_masks, obj_ids,
                    ) if _fbgr is not None else None

                    diag_entry = {
                        "frame_idx": current_frame_idx,
                        "composite": composite,
                        "scores": {
                            oid: _get_scores_dict(frame_outputs[oid])
                            for oid in active_ids
                        },
                        "centroids": {
                            oid: mask_centroid(bool_masks[oid])
                            for oid in active_ids
                        },
                        "bboxes": {
                            oid: mask_bbox(bool_masks[oid])
                            for oid in active_ids
                        },
                        "kf_states": {
                            oid: pre_reinit_kf[oid]
                            for oid in active_ids
                        },
                        "pairwise_ious": {
                            f"{a},{b}": v
                            for (a, b), v in pairwise_ious.items()
                        },
                        "purge_events": purge_events,
                        "reinit_events": reinit_events,
                        "swap_events": swap_events,
                        "yielded_frame_idxs": dict(yielded_frame_idxs),
                        "track_states": {
                            oid: {
                                "tier": track_states[oid].tier.value,
                                "frames_in_tier": track_states[oid].frames_in_tier,
                                "lost_counter": track_states[oid].lost_counter,
                                "logits": track_states[oid].last_logits,
                            }
                            for oid in active_ids
                            if oid in track_states
                        },
                        "state_transitions": state_transitions,
                        "track_deaths": track_death_events,
                    }
                    self.diagnostics.append(diag_entry)

                yield current_frame_idx, frame_masks, purge_events

            finally:
                for oid in list(generators.keys()):
                    try:
                        generators[oid].close()
                    except Exception:
                        pass
                predictors.clear()
                states.clear()
                generators.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def save_diagnostics(self, path: str) -> None:
        """Save captured diagnostics to disk. Requires capture_diagnostics=True."""
        if not self.diagnostics:
            raise ValueError("No diagnostics captured. Run track() with capture_diagnostics=True first.")
        save_diagnostics(self.diagnostics, path)
