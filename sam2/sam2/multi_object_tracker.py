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
"""

import gc
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of identical shape."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _purge_memory(inference_state: dict, frame_idx: int) -> int:
    """
    Remove frame `frame_idx` from the tracker's memory bank.

    Pops from both the packed output_dict and every per-object slice in
    output_dict_per_obj. Returns the number of entries actually removed.
    """
    removed = 0
    output_dict = inference_state["output_dict"]

    # Purge from packed (batched) output dict
    if output_dict["non_cond_frame_outputs"].pop(frame_idx, None) is not None:
        removed += 1

    # Purge from per-object output dict slices
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

    # best_iou_score is preferred; fall back to object_score_logits
    score = iou_score if iou_score is not None else obj_score
    if score is None:
        return 0.0
    if isinstance(score, (int, float)):
        return float(score)
    # Tensor — may be on GPU
    return float(score.item())


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

    def track(
        self,
        video_dir: str,
        init_boxes: Dict[int, list],
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        iou_threshold: float = 0.8,
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
            - purged_obj: int — which object's memory was purged
            - purged: int — number of memory entries removed
        """
        obj_ids = sorted(init_boxes.keys())
        n_objects = len(obj_ids)
        if n_objects == 0:
            return

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

            # Step all generators in lockstep
            exhausted = set()
            while True:
                frame_masks = {}    # {obj_id: np.ndarray} float32 logits
                frame_outputs = {}  # {obj_id: current_out dict} for scoring
                current_frame_idx = None

                for oid in obj_ids:
                    if oid in exhausted:
                        continue
                    try:
                        frame_idx, _obj_ids, video_res_masks = next(generators[oid])
                    except StopIteration:
                        exhausted.add(oid)
                        continue

                    current_frame_idx = frame_idx
                    # video_res_masks: (1, 1, H, W) tensor — one object per predictor
                    mask_logits = video_res_masks[0, 0].cpu().numpy().astype(np.float32)
                    frame_masks[oid] = mask_logits

                    # Grab the current output for scoring
                    out_dict = states[oid]["output_dict"]
                    current_out = out_dict["non_cond_frame_outputs"].get(frame_idx)
                    if current_out is None:
                        current_out = out_dict["cond_frame_outputs"].get(frame_idx, {})
                    frame_outputs[oid] = current_out

                if not frame_masks:
                    break  # All generators exhausted

                # --- Cross-Object Interaction ---
                purge_events = []
                active_ids = list(frame_masks.keys())
                for i in range(len(active_ids)):
                    for j in range(i + 1, len(active_ids)):
                        oid_a = active_ids[i]
                        oid_b = active_ids[j]
                        mask_a = frame_masks[oid_a] > 0.0
                        mask_b = frame_masks[oid_b] > 0.0

                        iou = mask_iou(mask_a, mask_b)
                        if iou <= iou_threshold:
                            continue

                        # Collision detected — purge the lower-scoring tracker
                        score_a = _get_score(frame_outputs[oid_a])
                        score_b = _get_score(frame_outputs[oid_b])

                        if score_a <= score_b:
                            victim_oid = oid_a
                        else:
                            victim_oid = oid_b

                        n_purged = _purge_memory(
                            states[victim_oid], current_frame_idx,
                        )

                        purge_events.append({
                            "frame_idx": current_frame_idx,
                            "obj_pair": (oid_a, oid_b),
                            "iou": iou,
                            "scores": (score_a, score_b),
                            "purged_obj": victim_oid,
                            "purged": n_purged,
                        })

                yield current_frame_idx, frame_masks, purge_events

        # Cleanup
        for oid in obj_ids:
            del predictors[oid]
            del states[oid]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
