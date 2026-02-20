"""
Interactive diagnostic viewer for SAMURAI multi-object tracking.

Uses fastplotlib (>= 0.6, ndwidget branch) for GPU-accelerated visualization.
Requires Python >= 3.10 with fastplotlib installed from:
    pip install "fastplotlib @ git+https://github.com/fastplotlib/fastplotlib.git@ndwidget"
    pip install glfw imgui-bundle opencv-python-headless

Usage:
    from sam2.diagnostics import DiagnosticViewer, load_diagnostics

    # Self-contained mode (baked composites, no video_dir needed):
    diag = load_diagnostics("diag.npz")
    viewer = DiagnosticViewer(diag)
    viewer.show()

    # Or with original video frames:
    viewer = DiagnosticViewer(diag, video_dir="path/to/video/frames")
    viewer.show()

    # Keyboard controls:
    #   Right arrow / d  — next frame
    #   Left arrow / a   — previous frame
    #   Shift+Right      — jump 10 frames forward
    #   Shift+Left       — jump 10 frames backward
    #   p                — jump to next purge event
"""

import json
import os
import os.path as osp
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Diagnostics serialization (no torch dependency — safe to import anywhere)
# ---------------------------------------------------------------------------

def save_diagnostics(diagnostics: list, path: str) -> None:
    """
    Save diagnostic data to a compressed .npz file.

    The diagnostics list contains one dict per frame with numpy arrays
    and nested dicts/lists. Metadata goes to a companion JSON file.
    """
    meta_frames = []
    arrays = {}

    for entry in diagnostics:
        fidx = entry["frame_idx"]
        frame_meta = {
            "frame_idx": fidx,
            "scores": entry["scores"],
            "centroids": entry["centroids"],
            "bboxes": entry["bboxes"],
            "kf_states": {},
            "pairwise_ious": entry["pairwise_ious"],
            "purge_events": entry["purge_events"],
            "has_composite": False,
        }
        for oid, mask in entry["masks"].items():
            arrays[f"mask_{fidx}_{oid}"] = mask
        for oid, kf in entry["kf_states"].items():
            if kf is not None:
                arrays[f"kf_{fidx}_{oid}"] = kf
                frame_meta["kf_states"][str(oid)] = True
            else:
                frame_meta["kf_states"][str(oid)] = False
        # Baked composite (video frame + mask overlays, RGB uint8)
        composite = entry.get("composite")
        if composite is not None:
            arrays[f"composite_{fidx}"] = composite
            frame_meta["has_composite"] = True
        meta_frames.append(frame_meta)

    arrays["__meta__"] = np.array([0])  # placeholder
    np.savez_compressed(path, **arrays)

    meta_path = path.rsplit(".", 1)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"frames": meta_frames}, f)


def load_diagnostics(path: str) -> list:
    """Load diagnostic data from .npz + _meta.json files."""
    meta_path = path.rsplit(".", 1)[0] + "_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    data = np.load(path)
    diagnostics = []

    for frame_meta in meta["frames"]:
        fidx = frame_meta["frame_idx"]

        # JSON stringifies int keys — convert back
        scores = {int(k): v for k, v in frame_meta["scores"].items()}
        centroids = {int(k): v for k, v in frame_meta["centroids"].items()}
        bboxes = {int(k): v for k, v in frame_meta["bboxes"].items()}

        entry = {
            "frame_idx": fidx,
            "masks": {},
            "scores": scores,
            "centroids": centroids,
            "bboxes": bboxes,
            "kf_states": {},
            "pairwise_ious": frame_meta["pairwise_ious"],
            "purge_events": frame_meta["purge_events"],
            "composite": None,
        }
        for key in data.files:
            if key.startswith(f"mask_{fidx}_"):
                oid = int(key.split("_")[-1])
                entry["masks"][oid] = data[key]
        for oid_str, has_kf in frame_meta["kf_states"].items():
            oid = int(oid_str)
            if has_kf:
                entry["kf_states"][oid] = data[f"kf_{fidx}_{oid}"]
            else:
                entry["kf_states"][oid] = None
        # Load baked composite if present
        if frame_meta.get("has_composite", False):
            composite_key = f"composite_{fidx}"
            if composite_key in data:
                entry["composite"] = data[composite_key]

        diagnostics.append(entry)

    return diagnostics


# Object colors: visually distinct, alpha-friendly
OBJ_COLORS = [
    (1.0, 0.2, 0.2, 0.45),   # red
    (0.2, 0.5, 1.0, 0.45),   # blue
    (0.2, 1.0, 0.3, 0.45),   # green
    (1.0, 0.8, 0.0, 0.45),   # yellow
    (1.0, 0.2, 1.0, 0.45),   # magenta
    (0.0, 1.0, 1.0, 0.45),   # cyan
]

KF_COLOR = (1.0, 1.0, 0.0, 0.8)     # yellow for KF predictions
CENTROID_COLOR = (1.0, 1.0, 1.0, 1.0)  # white for actual centroids
PURGE_COLOR = "red"
INDICATOR_COLOR = "yellow"


def _load_video_frames(video_dir: str) -> np.ndarray:
    """Load all frames from a directory of images into an (T, H, W, 3) uint8 array."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(exts)
    ])
    if not frame_files:
        raise ValueError(f"No image files found in {video_dir}")

    frames = []
    for fname in frame_files:
        img = cv2.imread(osp.join(video_dir, fname))
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return np.stack(frames)


def _make_mask_overlay(frame: np.ndarray, masks: dict, obj_ids: list) -> np.ndarray:
    """Composite colored mask overlays onto a video frame. Returns RGB uint8."""
    out = frame.copy().astype(np.float32)
    h, w = frame.shape[:2]

    for i, oid in enumerate(obj_ids):
        mask_logits = masks.get(oid)
        if mask_logits is None:
            continue
        binary = mask_logits > 0.0
        color = OBJ_COLORS[i % len(OBJ_COLORS)]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[binary] = [color[0] * 255, color[1] * 255, color[2] * 255]
        alpha = color[3]
        out[binary] = out[binary] * (1 - alpha) + overlay[binary] * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


class DiagnosticViewer:
    """
    Interactive frame-by-frame viewer for SAMURAI tracking diagnostics.

    Layout:
        [0,0] Video frame + colored mask overlays + centroids + KF predictions
        [0,1] Score timelines (iou_score, kf_score per tracker) + frame indicator
        [1,0] Pairwise IoU timeline + purge event markers + frame indicator
        [1,1] KF residual timeline (distance between KF prediction and mask centroid)

    Parameters
    ----------
    video_dir : str or None
        Path to directory containing video frames. Can be None if diagnostic
        data contains baked composites (captured with capture_diagnostics=True).
    diagnostics : list
        Diagnostic data from MultiObjectTracker (or loaded via load_diagnostics).
    preload_frames : bool
        If True, load all video frames into memory at init.
        If False, load frames on demand (slower scrubbing but less memory).
        Ignored when video_dir is None (composites are used instead).
    """

    def __init__(
        self,
        diagnostics: list,
        video_dir: Optional[str] = None,
        preload_frames: bool = True,
    ):
        self.video_dir = video_dir
        self.diagnostics = diagnostics
        self.n_frames = len(diagnostics)
        self._current_idx = 0

        # Check if baked composites are available
        self._has_composites = diagnostics[0].get("composite") is not None

        if not self._has_composites and video_dir is None:
            raise ValueError(
                "video_dir is required when diagnostic data does not contain "
                "baked composites. Re-run tracking with capture_diagnostics=True "
                "to bake composites into the diagnostic files."
            )

        # Extract object IDs from first frame
        self.obj_ids = sorted(diagnostics[0]["masks"].keys())
        self.n_objects = len(self.obj_ids)

        # Find purge frames for quick navigation
        self.purge_frames = []
        for entry in diagnostics:
            if entry["purge_events"]:
                self.purge_frames.append(entry["frame_idx"])

        # Precompute timelines
        self._build_timelines()

        # Load video frames (only needed when no baked composites)
        self._frames = None
        if not self._has_composites and video_dir is not None:
            if preload_frames:
                self._frames = _load_video_frames(video_dir)
            else:
                self._frame_files = sorted([
                    osp.join(video_dir, f)
                    for f in os.listdir(video_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ])

    def _build_timelines(self):
        """Pre-extract score and IoU arrays for timeline plots."""
        n = self.n_frames

        # Per-object score timelines
        self.iou_scores = {oid: np.full(n, np.nan) for oid in self.obj_ids}
        self.kf_scores = {oid: np.full(n, np.nan) for oid in self.obj_ids}
        self.obj_scores = {oid: np.full(n, np.nan) for oid in self.obj_ids}

        # KF residuals (Euclidean distance between KF predicted center and mask centroid)
        self.kf_residuals = {oid: np.full(n, np.nan) for oid in self.obj_ids}

        # Pairwise IoU timelines
        self.pair_keys = []
        for i in range(len(self.obj_ids)):
            for j in range(i + 1, len(self.obj_ids)):
                self.pair_keys.append((self.obj_ids[i], self.obj_ids[j]))
        self.pairwise_iou_timelines = {k: np.full(n, np.nan) for k in self.pair_keys}

        for idx, entry in enumerate(self.diagnostics):
            scores = entry["scores"]
            centroids = entry["centroids"]
            kf_states = entry["kf_states"]

            for oid in self.obj_ids:
                if oid in scores:
                    s = scores[oid]
                    if s["iou"] is not None:
                        self.iou_scores[oid][idx] = s["iou"]
                    if s["kf"] is not None:
                        self.kf_scores[oid][idx] = s["kf"]
                    if s["obj"] is not None:
                        self.obj_scores[oid][idx] = s["obj"]

                # KF residual
                centroid = centroids.get(oid) if isinstance(centroids, dict) else None
                kf_state = kf_states.get(oid) if isinstance(kf_states, dict) else None
                if centroid is not None and kf_state is not None:
                    kf_cx, kf_cy = kf_state[0], kf_state[1]
                    cx, cy = centroid
                    self.kf_residuals[oid][idx] = np.sqrt(
                        (kf_cx - cx) ** 2 + (kf_cy - cy) ** 2
                    )

            # Pairwise IoU
            pw = entry["pairwise_ious"]
            for pair_key in self.pair_keys:
                str_key = f"{pair_key[0]},{pair_key[1]}"
                if str_key in pw:
                    self.pairwise_iou_timelines[pair_key][idx] = pw[str_key]

    def _get_frame(self, idx: int) -> np.ndarray:
        """Get video frame at index."""
        if self._frames is not None:
            return self._frames[idx]
        img = cv2.imread(self._frame_files[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_composite(self, idx: int) -> np.ndarray:
        """Get video frame with mask overlays composited."""
        entry = self.diagnostics[idx]
        # Use baked composite if available (self-contained mode)
        if self._has_composites and entry.get("composite") is not None:
            return entry["composite"]
        frame = self._get_frame(idx)
        return _make_mask_overlay(frame, entry["masks"], self.obj_ids)

    def show(self):
        """Build and display the interactive diagnostic viewer."""
        import fastplotlib as fpl

        # --- Build figure ---
        fig = fpl.Figure(
            shape=(2, 2),
            size=(1400, 800),
            names=[
                ["video", "scores"],
                ["iou", "kf_residual"],
            ],
        )

        # ===== [0,0] Video + mask overlays =====
        composite = self._get_composite(0)
        img_graphic = fig["video"].add_image(composite, name="frame")

        # Centroid markers (one scatter per object for coloring)
        centroid_scatters = {}
        kf_scatters = {}
        for i, oid in enumerate(self.obj_ids):
            color = OBJ_COLORS[i % len(OBJ_COLORS)]
            bright_color = (min(color[0] + 0.3, 1.0), min(color[1] + 0.3, 1.0),
                           min(color[2] + 0.3, 1.0), 1.0)

            entry0 = self.diagnostics[0]
            c = entry0["centroids"].get(oid)
            pos = np.array([[c[0], c[1], 0]], dtype=np.float32) if c else np.array([[0, 0, 0]], dtype=np.float32)

            centroid_scatters[oid] = fig["video"].add_scatter(
                pos, colors="white", sizes=12, name=f"centroid_{oid}"
            )

            # KF predicted position (hollow square - use different marker)
            kf = entry0["kf_states"].get(oid)
            kf_pos = np.array([[kf[0], kf[1], 0]], dtype=np.float32) if kf is not None else np.array([[0, 0, 0]], dtype=np.float32)
            kf_scatters[oid] = fig["video"].add_scatter(
                kf_pos,
                colors=[bright_color[:3]],
                sizes=18,
                markers="x",
                name=f"kf_{oid}",
            )

        # Frame label
        frame_text = fig["video"].add_text(
            f"Frame: 0 / {self.n_frames - 1}",
            font_size=18,
            face_color="white",
            offset=(10, 10, 0),
            anchor="top-left",
            name="frame_label",
        )

        # ===== [0,1] Score timelines =====
        frames_x = np.arange(self.n_frames, dtype=np.float32)
        for i, oid in enumerate(self.obj_ids):
            color_str = ["red", "dodgerblue", "lime", "yellow", "magenta", "cyan"][i % 6]
            # IoU score line
            y = self.iou_scores[oid].copy().astype(np.float32)
            valid = ~np.isnan(y)
            if valid.any():
                y[~valid] = 0
                fig["scores"].add_line(
                    np.column_stack([frames_x, y, np.zeros_like(y)]),
                    colors=color_str,
                    thickness=1.5,
                    name=f"iou_score_{oid}",
                )

        # Frame indicator line on scores
        score_indicator = fig["scores"].add_line(
            np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            colors=INDICATOR_COLOR,
            thickness=2,
            name="score_indicator",
        )
        fig["scores"].add_text("IoU Scores", font_size=14, face_color="white",
                                offset=(10, 10, 0), anchor="top-left")

        # ===== [1,0] Pairwise IoU timeline + purge markers =====
        for pair_key in self.pair_keys:
            y = self.pairwise_iou_timelines[pair_key].copy().astype(np.float32)
            valid = ~np.isnan(y)
            if valid.any():
                y[~valid] = 0
                fig["iou"].add_line(
                    np.column_stack([frames_x, y, np.zeros_like(y)]),
                    colors="orange",
                    thickness=2,
                    name=f"pair_iou_{pair_key[0]}_{pair_key[1]}",
                )

        # Purge event markers
        if self.purge_frames:
            purge_x = np.array(self.purge_frames, dtype=np.float32)
            purge_y = np.ones_like(purge_x) * 0.5
            fig["iou"].add_scatter(
                np.column_stack([purge_x, purge_y, np.zeros_like(purge_x)]),
                colors=PURGE_COLOR,
                sizes=15,
                markers="^",
                name="purge_markers",
            )

        # Frame indicator on IoU
        iou_indicator = fig["iou"].add_line(
            np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            colors=INDICATOR_COLOR,
            thickness=2,
            name="iou_indicator",
        )
        fig["iou"].add_text("Pairwise IoU", font_size=14, face_color="white",
                             offset=(10, 10, 0), anchor="top-left")

        # ===== [1,1] KF residual timeline =====
        for i, oid in enumerate(self.obj_ids):
            color_str = ["red", "dodgerblue", "lime", "yellow", "magenta", "cyan"][i % 6]
            y = self.kf_residuals[oid].copy().astype(np.float32)
            valid = ~np.isnan(y)
            if valid.any():
                y[~valid] = 0
                fig["kf_residual"].add_line(
                    np.column_stack([frames_x, y, np.zeros_like(y)]),
                    colors=color_str,
                    thickness=1.5,
                    name=f"kf_resid_{oid}",
                )

        kf_indicator = fig["kf_residual"].add_line(
            np.array([[0, 0, 0], [0, 100, 0]], dtype=np.float32),
            colors=INDICATOR_COLOR,
            thickness=2,
            name="kf_indicator",
        )
        fig["kf_residual"].add_text("KF Residual (px)", font_size=14, face_color="white",
                                     offset=(10, 10, 0), anchor="top-left")

        # --- State for keyboard navigation ---
        state = {"idx": 0, "fig": fig}

        def _update_display(new_idx: int):
            new_idx = max(0, min(new_idx, self.n_frames - 1))
            state["idx"] = new_idx

            # Update video frame + masks
            img_graphic.data = self._get_composite(new_idx)

            # Update centroids and KF markers
            entry = self.diagnostics[new_idx]
            for oid in self.obj_ids:
                c = entry["centroids"].get(oid)
                if c is not None:
                    centroid_scatters[oid].data = np.array([[c[0], c[1], 0]], dtype=np.float32)

                kf = entry["kf_states"].get(oid)
                if kf is not None:
                    kf_scatters[oid].data = np.array([[kf[0], kf[1], 0]], dtype=np.float32)

            # Update frame label
            purge_note = ""
            if entry["purge_events"]:
                victims = [str(e["purged_obj"]) for e in entry["purge_events"]]
                purge_note = f"  PURGE: obj {','.join(victims)}"
            frame_text.text = f"Frame: {new_idx} / {self.n_frames - 1}{purge_note}"

            # Update indicator lines on timelines
            fx = float(new_idx)
            score_indicator.data = np.array([[fx, -0.1, 0], [fx, 1.1, 0]], dtype=np.float32)
            iou_indicator.data = np.array([[fx, -0.1, 0], [fx, 1.1, 0]], dtype=np.float32)

            # KF residual indicator - use a tall line
            kf_indicator.data = np.array([[fx, 0, 0], [fx, 200, 0]], dtype=np.float32)

        def _on_key(event):
            if event.type != "key_down":
                return

            idx = state["idx"]
            key = event.key

            if key in ("ArrowRight", "d"):
                if "Shift" in (event.modifiers or []):
                    _update_display(idx + 10)
                else:
                    _update_display(idx + 1)
            elif key in ("ArrowLeft", "a"):
                if "Shift" in (event.modifiers or []):
                    _update_display(idx - 10)
                else:
                    _update_display(idx - 1)
            elif key == "p":
                # Jump to next purge event
                for pf in self.purge_frames:
                    if pf > idx:
                        _update_display(pf)
                        return
                # Wrap around
                if self.purge_frames:
                    _update_display(self.purge_frames[0])

        # Register keyboard handler
        fig.renderer.add_event_handler(_on_key, "key_down")

        self._fig = fig
        self._update_display = _update_display

        fig.show()

        return fig

    def goto(self, frame_idx: int):
        """Jump to a specific frame (call after show())."""
        if hasattr(self, "_update_display"):
            self._update_display(frame_idx)

    def goto_purge(self, n: int = 0):
        """Jump to the nth purge event (call after show())."""
        if n < len(self.purge_frames):
            self.goto(self.purge_frames[n])

    def _build_render_figure(self):
        """
        Build the offscreen fastplotlib Figure for rendering diagnostics.

        Called once, then reused for all render_frame calls. The scene graph
        is updated in-place for each frame — no teardown/rebuild needed.
        """
        import fastplotlib as fpl

        fig = fpl.Figure(
            shape=(2, 2),
            size=(1400, 800),
            names=[["video", "scores"], ["iou", "kf_residual"]],
            canvas="offscreen",
        )

        color_names = ["red", "dodgerblue", "lime", "gold", "magenta", "cyan"]
        frames_x = np.arange(self.n_frames, dtype=np.float32)

        # ===== [0,0] Video + mask overlays =====
        composite = self._get_composite(0)
        self._r_img = fig["video"].add_image(composite, name="frame")

        self._r_centroids = {}
        self._r_kf = {}
        for i, oid in enumerate(self.obj_ids):
            clr = color_names[i % len(color_names)]
            self._r_centroids[oid] = fig["video"].add_scatter(
                np.array([[0, 0, 0]], dtype=np.float32),
                colors="white", sizes=12, name=f"centroid_{oid}",
            )
            self._r_kf[oid] = fig["video"].add_scatter(
                np.array([[0, 0, 0]], dtype=np.float32),
                colors=clr, sizes=18, markers="x", name=f"kf_{oid}",
            )

        self._r_frame_text = fig["video"].add_text(
            "Frame: 0", font_size=16, face_color="white",
            offset=(10, 10, 0), anchor="top-left", name="frame_label",
        )

        # ===== [0,1] IoU score timelines =====
        for i, oid in enumerate(self.obj_ids):
            clr = color_names[i % len(color_names)]
            y = self.iou_scores[oid].copy().astype(np.float32)
            y[np.isnan(y)] = 0
            fig["scores"].add_line(
                np.column_stack([frames_x, y, np.zeros_like(y)]),
                colors=clr, thickness=1.5, name=f"iou_score_{oid}",
            )

        # Purge marker lines on score panel
        for pf in self.purge_frames:
            fig["scores"].add_line(
                np.array([[pf, -0.05, 0], [pf, 1.05, 0]], dtype=np.float32),
                colors="red", thickness=1,
            )

        self._r_score_indicator = fig["scores"].add_line(
            np.array([[0, -0.05, 0], [0, 1.05, 0]], dtype=np.float32),
            colors="yellow", thickness=2, name="score_indicator",
        )

        # ===== [1,0] Pairwise IoU timeline =====
        for pair_key in self.pair_keys:
            y = self.pairwise_iou_timelines[pair_key].copy().astype(np.float32)
            y[np.isnan(y)] = 0
            fig["iou"].add_line(
                np.column_stack([frames_x, y, np.zeros_like(y)]),
                colors="orange", thickness=2,
                name=f"pair_iou_{pair_key[0]}_{pair_key[1]}",
            )

        # Threshold line
        fig["iou"].add_line(
            np.array([[0, 0.8, 0], [float(self.n_frames), 0.8, 0]], dtype=np.float32),
            colors="gray", thickness=1,
        )

        # Purge markers
        if self.purge_frames:
            purge_x = np.array(self.purge_frames, dtype=np.float32)
            fig["iou"].add_scatter(
                np.column_stack([purge_x, np.full_like(purge_x, 0.5), np.zeros_like(purge_x)]),
                colors="red", sizes=12, markers="^", name="purge_markers",
            )
            for pf in self.purge_frames:
                fig["iou"].add_line(
                    np.array([[pf, -0.05, 0], [pf, 1.05, 0]], dtype=np.float32),
                    colors="red", thickness=1,
                )

        self._r_iou_indicator = fig["iou"].add_line(
            np.array([[0, -0.05, 0], [0, 1.05, 0]], dtype=np.float32),
            colors="yellow", thickness=2, name="iou_indicator",
        )

        # ===== [1,1] KF residual timeline =====
        kf_max = 1.0
        for i, oid in enumerate(self.obj_ids):
            clr = color_names[i % len(color_names)]
            y = self.kf_residuals[oid].copy().astype(np.float32)
            valid = ~np.isnan(y)
            if valid.any():
                kf_max = max(kf_max, float(np.nanmax(y)) * 1.1)
            y[np.isnan(y)] = 0
            fig["kf_residual"].add_line(
                np.column_stack([frames_x, y, np.zeros_like(y)]),
                colors=clr, thickness=1.5, name=f"kf_resid_{oid}",
            )

        for pf in self.purge_frames:
            fig["kf_residual"].add_line(
                np.array([[pf, 0, 0], [pf, kf_max, 0]], dtype=np.float32),
                colors="red", thickness=1,
            )

        self._r_kf_indicator = fig["kf_residual"].add_line(
            np.array([[0, 0, 0], [0, kf_max, 0]], dtype=np.float32),
            colors="yellow", thickness=2, name="kf_indicator",
        )
        self._r_kf_max = kf_max

        # Initial render
        fig.show(maintain_aspect=True)
        fig.canvas.force_draw()

        self._r_fig = fig

    def _update_render_frame(self, idx: int):
        """Update the offscreen figure's graphics to show frame `idx`."""
        entry = self.diagnostics[idx]
        fx = float(idx)

        # Video + masks
        self._r_img.data = self._get_composite(idx)

        # Centroids + KF markers
        for oid in self.obj_ids:
            c = entry["centroids"].get(oid)
            if c is not None:
                self._r_centroids[oid].data = np.array([[c[0], c[1], 0]], dtype=np.float32)
            kf = entry["kf_states"].get(oid)
            if kf is not None:
                self._r_kf[oid].data = np.array([[kf[0], kf[1], 0]], dtype=np.float32)

        # Frame label
        purge_note = ""
        if entry["purge_events"]:
            victims = ", ".join([str(e["purged_obj"]) for e in entry["purge_events"]])
            ious = ", ".join([f'{e.get("iou", 0):.3f}' for e in entry["purge_events"]])
            purge_note = f"  PURGE obj {victims} (IoU: {ious})"
        self._r_frame_text.text = f"Frame {idx} / {self.n_frames - 1}{purge_note}"

        # Frame indicators on timelines
        self._r_score_indicator.data = np.array(
            [[fx, -0.05, 0], [fx, 1.05, 0]], dtype=np.float32)
        self._r_iou_indicator.data = np.array(
            [[fx, -0.05, 0], [fx, 1.05, 0]], dtype=np.float32)
        self._r_kf_indicator.data = np.array(
            [[fx, 0, 0], [fx, self._r_kf_max, 0]], dtype=np.float32)

    def render_frame(self, idx: int, out_path: str) -> str:
        """
        Render a single diagnostic frame to a PNG file using fastplotlib.

        Produces a 4-panel GPU-rendered composite:
          Top-left:     Video frame + colored mask overlays + centroids + KF predictions
          Top-right:    Full IoU score timelines (all frames) + frame indicator
          Bottom-left:  Full pairwise IoU timeline + purge markers + frame indicator
          Bottom-right: Full KF residual timeline + frame indicator

        The figure is built once and reused — subsequent calls only update
        the data, making batch rendering fast.

        Parameters
        ----------
        idx : int
            Frame index to render.
        out_path : str
            Output PNG file path.

        Returns
        -------
        str
            The output path (same as out_path).
        """
        if not hasattr(self, "_r_fig"):
            self._build_render_figure()

        self._update_render_frame(idx)
        self._r_fig.canvas.force_draw()
        self._r_fig.export(out_path)

        return out_path

    def render_range(self, start: int, end: int, out_dir: str, step: int = 1) -> list:
        """
        Render a range of diagnostic frames to PNG files.

        The fastplotlib scene is built once and reused for all frames.

        Parameters
        ----------
        start : int
            First frame index.
        end : int
            Last frame index (inclusive).
        out_dir : str
            Output directory for PNG files.
        step : int
            Frame step (default 1 = every frame).

        Returns
        -------
        list of str
            Paths to the rendered PNG files.
        """
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for idx in range(start, min(end + 1, self.n_frames), step):
            path = osp.join(out_dir, f"diag_{idx:05d}.png")
            self.render_frame(idx, path)
            paths.append(path)
        return paths

    def render_purge_events(self, out_dir: str, context: int = 5) -> list:
        """
        Render frames around each purge event.

        For each purge, renders [purge_frame - context, purge_frame + context].
        This is the most useful entry point for diagnosing CoI issues.

        Parameters
        ----------
        out_dir : str
            Output directory for PNG files.
        context : int
            Number of frames before and after each purge to render.

        Returns
        -------
        list of str
            Paths to all rendered PNG files.
        """
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for pf in self.purge_frames:
            start = max(0, pf - context)
            end = min(self.n_frames - 1, pf + context)
            paths.extend(self.render_range(start, end, out_dir))
        return paths
