"""
Crossing-triggered identity verification for multi-object tracking.

Detects identity swaps after mask crossings by comparing visual embeddings
before and after the crossing event. Uses a pluggable backbone (default:
DINOv3 ViT-S/16) to extract embeddings from mask crops.

The key insight: we don't ask "which object is this?" (impossible for
visually identical targets). We ask "did the identity assignment change
across this specific crossing?" — comparing pre-crossing vs post-crossing
crops provides relative context.

Architecture:
    EmbeddingBackbone (ABC)
        └── DINOv3Backbone  (default: facebook/dinov3-vits16-pretrain-lvd1689m)

    IdentityVerifier
        └── Per-pair crossing state machine:
            CLEAR → APPROACHING → OVERLAPPING → SEPARATING → CLEAR
"""

import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Embedding backbone ABC
# ---------------------------------------------------------------------------

class EmbeddingBackbone(ABC):
    """Abstract base class for visual embedding extraction."""

    @abstractmethod
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop (H,W,3) uint8 → 1-D float32 embedding."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""
        ...


# ---------------------------------------------------------------------------
# DINOv3 backbone
# ---------------------------------------------------------------------------

class DINOv3Backbone(EmbeddingBackbone):
    """
    DINOv3 ViT-S/16 backbone for visual embedding extraction.

    Uses the CLS token pooler output as a global descriptor (~384-d).
    Lazy-imports transformers to avoid hard dependency.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: str = "cuda:0",
    ):
        from PIL import Image as _Image  # noqa: F401 — validate availability
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.inference_mode()
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            images=Image.fromarray(rgb), return_tensors="pt",
        ).to(self.device)
        output = self.model(**inputs).pooler_output[0]
        return output.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Crossing state machine
# ---------------------------------------------------------------------------

class CrossingPhase(enum.Enum):
    CLEAR = "clear"
    APPROACHING = "approaching"
    OVERLAPPING = "overlapping"
    SEPARATING = "separating"


@dataclass
class PairCrossingState:
    """Mutable state for one object pair's crossing lifecycle."""
    phase: CrossingPhase = CrossingPhase.CLEAR
    # Pre-crossing reference embeddings {obj_id: embedding}
    ref_embeddings: Dict[int, np.ndarray] = field(default_factory=dict)
    # Frame where we entered APPROACHING (for diagnostics)
    approach_frame: Optional[int] = None
    # Frames spent in OVERLAPPING (for timeout)
    overlap_frames: int = 0


# ---------------------------------------------------------------------------
# IdentityVerifier
# ---------------------------------------------------------------------------

class IdentityVerifier:
    """
    Crossing-triggered identity verification.

    Monitors pairwise IoU between tracked objects to detect crossing events.
    When a pair separates after overlapping, compares pre- and post-crossing
    visual embeddings to detect identity swaps.

    Parameters
    ----------
    backbone : EmbeddingBackbone
        Visual embedding extractor.
    proximity_threshold : float
        IoU level marking entering/exiting the crossing zone.
    overlap_threshold : float
        IoU level marking full overlap (should match CoI iou_threshold).
    swap_margin : float
        How much sim_swapped must exceed sim_correct to declare a swap.
    min_crop_size : int
        Minimum crop width/height in pixels; skip degenerate crops.
    max_overlap_frames : int
        Abandon crossing event if OVERLAPPING persists longer than this.
        Prevents stale ref embeddings when objects rest in contact.
        0 disables the timeout.
    """

    def __init__(
        self,
        backbone: EmbeddingBackbone,
        proximity_threshold: float = 0.1,
        overlap_threshold: float = 0.8,
        swap_margin: float = 0.05,
        min_crop_size: int = 16,
        max_overlap_frames: int = 300,
    ):
        self.backbone = backbone
        self.proximity_threshold = proximity_threshold
        self.overlap_threshold = overlap_threshold
        self.swap_margin = swap_margin
        self.min_crop_size = min_crop_size
        self.max_overlap_frames = max_overlap_frames

        self._pair_states: Dict[Tuple[int, int], PairCrossingState] = {}

    def reset_object(self, obj_id: int) -> None:
        """
        Reset all pair states involving obj_id.

        Call when an object dies (exhaustion), is re-initialized, or has
        its memory purged by CoI. Prevents stale pair states from
        corrupting future swap decisions.
        """
        for pair_key in list(self._pair_states.keys()):
            if obj_id in pair_key:
                state = self._pair_states[pair_key]
                if state.phase != CrossingPhase.CLEAR:
                    logging.debug(
                        "Resetting pair %s (object %d reset, was %s)",
                        pair_key, obj_id, state.phase.value,
                    )
                del self._pair_states[pair_key]

    def notify_swap(self, oid_a: int, oid_b: int) -> None:
        """
        Reset sibling pair states after a swap on (oid_a, oid_b).

        After a swap is applied, any other pair involving A or B that is
        mid-crossing has stale ref_embeddings (they reference pre-swap
        identity assignments). Reset those pairs to prevent corrupt
        three-body swap decisions.
        """
        swapped_pair = (min(oid_a, oid_b), max(oid_a, oid_b))
        for pair_key in list(self._pair_states.keys()):
            if pair_key == swapped_pair:
                continue
            if oid_a in pair_key or oid_b in pair_key:
                state = self._pair_states[pair_key]
                if state.phase != CrossingPhase.CLEAR:
                    logging.info(
                        "Resetting sibling pair %s after swap(%d, %d) "
                        "(was %s)",
                        pair_key, oid_a, oid_b, state.phase.value,
                    )
                del self._pair_states[pair_key]

    def _get_pair_state(self, oid_a: int, oid_b: int) -> PairCrossingState:
        """Get or create state for an ordered pair."""
        key = (min(oid_a, oid_b), max(oid_a, oid_b))
        if key not in self._pair_states:
            self._pair_states[key] = PairCrossingState()
        return self._pair_states[key]

    def _extract_crop(
        self, obj_id: int, bool_masks: Dict[int, np.ndarray],
        frame_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract a masked BGR crop of the object from the frame.

        Pixels outside the object's mask are zeroed to prevent
        contamination from overlapping objects (S5, S10).
        """
        from sam2.multi_object_tracker import mask_bbox

        mask = bool_masks.get(obj_id)
        if mask is None:
            return None

        x, y, w, h = mask_bbox(mask)
        if w < self.min_crop_size or h < self.min_crop_size:
            return None

        # Validate mask coverage — reject degenerate crops
        mask_crop = mask[y:y + h, x:x + w]
        if mask_crop.size == 0:
            return None
        mask_ratio = mask_crop.sum() / mask_crop.size
        if mask_ratio < 0.1:
            return None

        # Zero out pixels outside the object's own mask
        crop = frame_bgr[y:y + h, x:x + w].copy()
        crop[~mask_crop] = 0
        return crop

    def _snapshot_embeddings(
        self, pair: Tuple[int, int], bool_masks: Dict[int, np.ndarray],
        frame_bgr: np.ndarray,
    ) -> bool:
        """
        Compute and store reference embeddings for a pair.

        Returns True if both embeddings were successfully extracted.
        """
        state = self._pair_states[pair]
        state.ref_embeddings = {}

        for oid in pair:
            crop = self._extract_crop(oid, bool_masks, frame_bgr)
            if crop is None:
                logging.debug(
                    "Cannot snapshot embedding for oid %d: bad crop", oid,
                )
                return False
            state.ref_embeddings[oid] = self.backbone.embed(crop)

        return True

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _check_swap(
        self, pair: Tuple[int, int],
        bool_masks: Dict[int, np.ndarray],
        frame_bgr: np.ndarray,
        frame_idx: int,
    ) -> Optional[dict]:
        """
        Compare pre- vs post-crossing embeddings for a pair.

        Returns a swap event dict, or None if embeddings can't be extracted.
        """
        state = self._pair_states[pair]
        oid_a, oid_b = pair

        ref_a = state.ref_embeddings.get(oid_a)
        ref_b = state.ref_embeddings.get(oid_b)
        if ref_a is None or ref_b is None:
            logging.debug(
                "Missing ref embeddings for pair %s, skipping swap check",
                pair,
            )
            return None

        # Extract post-crossing embeddings
        crop_a = self._extract_crop(oid_a, bool_masks, frame_bgr)
        crop_b = self._extract_crop(oid_b, bool_masks, frame_bgr)
        if crop_a is None or crop_b is None:
            logging.debug(
                "Cannot extract post-crossing crops for pair %s", pair,
            )
            return None

        cur_a = self.backbone.embed(crop_a)
        cur_b = self.backbone.embed(crop_b)

        # Compare: correct assignment vs swapped assignment
        sim_correct = (
            self._cosine_similarity(cur_a, ref_a)
            + self._cosine_similarity(cur_b, ref_b)
        )
        sim_swapped = (
            self._cosine_similarity(cur_a, ref_b)
            + self._cosine_similarity(cur_b, ref_a)
        )
        margin = sim_swapped - sim_correct

        action = "swap" if margin > self.swap_margin else "no_swap"

        return {
            "action": action,
            "frame_idx": frame_idx,
            "obj_pair": pair,
            "margin": margin,
            "sim_correct": sim_correct,
            "sim_swapped": sim_swapped,
            "approach_frame": state.approach_frame,
        }

    def update(
        self,
        frame_idx: int,
        pairwise_ious: Dict[Tuple[int, int], float],
        bool_masks: Dict[int, np.ndarray],
        frame_bgr: Optional[np.ndarray],
    ) -> List[dict]:
        """
        Drive the crossing state machine for all pairs.

        Parameters
        ----------
        frame_idx : int
            Current frame index.
        pairwise_ious : dict
            {(oid_a, oid_b): iou} for all active pairs.
        bool_masks : dict
            {obj_id: np.ndarray bool} current frame masks.
        frame_bgr : np.ndarray or None
            Current video frame in BGR. None skips embedding extraction.

        Returns
        -------
        list of dict
            Swap events emitted this frame (may be empty).
        """
        events = []

        # Safety-net: prune pair states for objects no longer active
        active_objs = set()
        for a, b in pairwise_ious.keys():
            active_objs.add(a)
            active_objs.add(b)
        for pair_key in list(self._pair_states.keys()):
            if pair_key[0] not in active_objs or pair_key[1] not in active_objs:
                state = self._pair_states[pair_key]
                if state.phase != CrossingPhase.CLEAR:
                    logging.debug(
                        "Pruning stale pair state %s (object absent)",
                        pair_key,
                    )
                del self._pair_states[pair_key]

        # Hysteresis deadband (S8/S9): enter at proximity_threshold,
        # exit at half that to prevent CLEAR↔APPROACHING flapping.
        exit_threshold = self.proximity_threshold * 0.5

        for (oid_a, oid_b), iou in pairwise_ious.items():
            pair = (min(oid_a, oid_b), max(oid_a, oid_b))
            state = self._get_pair_state(*pair)

            if state.phase == CrossingPhase.CLEAR:
                if iou > self.proximity_threshold:
                    state.phase = CrossingPhase.APPROACHING
                    state.approach_frame = frame_idx
                    # Snapshot pre-crossing embeddings
                    if frame_bgr is not None:
                        ok = self._snapshot_embeddings(
                            pair, bool_masks, frame_bgr,
                        )
                        if not ok:
                            logging.debug(
                                "Failed to snapshot embeddings for %s at "
                                "frame %d, reverting to CLEAR",
                                pair, frame_idx,
                            )
                            state.phase = CrossingPhase.CLEAR
                            state.approach_frame = None
                    else:
                        # No frame available — can't do verification
                        state.phase = CrossingPhase.CLEAR
                        state.approach_frame = None

            elif state.phase == CrossingPhase.APPROACHING:
                if iou > self.overlap_threshold:
                    state.phase = CrossingPhase.OVERLAPPING
                    state.overlap_frames = 0
                elif iou <= exit_threshold:
                    # False alarm — never reached overlap (hysteresis)
                    state.phase = CrossingPhase.CLEAR
                    state.ref_embeddings = {}
                    state.approach_frame = None

            elif state.phase == CrossingPhase.OVERLAPPING:
                state.overlap_frames += 1
                # Timeout: abandon if overlapping too long (S7)
                if (self.max_overlap_frames > 0
                        and state.overlap_frames > self.max_overlap_frames):
                    logging.info(
                        "Overlap timeout for pair %s after %d frames, "
                        "abandoning crossing event",
                        pair, state.overlap_frames,
                    )
                    state.phase = CrossingPhase.CLEAR
                    state.ref_embeddings = {}
                    state.approach_frame = None
                    state.overlap_frames = 0
                elif iou < self.overlap_threshold:
                    state.phase = CrossingPhase.SEPARATING

            elif state.phase == CrossingPhase.SEPARATING:
                if iou > self.overlap_threshold:
                    # Re-entered overlap
                    state.phase = CrossingPhase.OVERLAPPING
                elif iou < exit_threshold:
                    # Fully separated — run swap check (hysteresis)
                    if frame_bgr is not None:
                        event = self._check_swap(
                            pair, bool_masks, frame_bgr, frame_idx,
                        )
                        if event is not None:
                            events.append(event)
                    else:
                        # S12: defensive log
                        logging.warning(
                            "Crossing resolved for pair %s at frame %d "
                            "but frame_bgr is None — swap check skipped",
                            pair, frame_idx,
                        )
                    # Reset state
                    state.phase = CrossingPhase.CLEAR
                    state.ref_embeddings = {}
                    state.approach_frame = None
                    state.overlap_frames = 0

        return events
