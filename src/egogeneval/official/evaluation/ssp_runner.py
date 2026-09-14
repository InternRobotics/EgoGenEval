#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ssp_runner.py
=============
The Scene & Spatial Preservation (SSP) object evaluator for generated images.

Motivation
----------
ssp_detection.py focuses on spatial/layout consistency (recall, center
error, topology, depth order). It is weak at catching defects *inside* the
generated objects themselves: deformation, missing/extra parts, texture
corruption, blur, identity drift, hallucinated objects, etc.

This module reuses the heavy models already wired into
SspDetectionRunner (GroundingDINO/SAM3 detection, DINOv3 ROI
features, optional Qwen3-VL) and adds a tiered defect-detection pipeline:

  Tier 0  Detection on GT-target image and generated image (reused).
          Crucially we keep the *unfiltered* pred set so hallucinated / extra
          objects are scored instead of silently dropped.
  Tier 1  Cheap per-object signals (almost free, no extra model):
            - identity_sim   : DINOv3 ROI cosine (graded, not a binary gate)
            - struct_ssim    : SSIM on size-aligned crops  -> deformation
            - edge_iou       : Canny edge overlap          -> missing/extra parts
            - sharpness_ratio: Laplacian-variance ratio     -> blur / melt
            - color_sim      : HSV histogram similarity      -> texture/colour drift
            - shape_sim      : mask IoU(aligned)+Hu-moments  -> warped silhouette
  Tier 2  (optional, --use-vlm) ONE structured VLM call per image over a montage
          of matched-pair crops + hallucination crops -> per-object defect flags.

It then dumps per-object JSON plus a summary to ``ssp_results.json``. The
panel renderer used during development is not bundled here, so ``--no-vis``
is required.

This file does NOT modify ssp_detection.py.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import tempfile
import traceback
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse everything from the existing runner / launcher.
import ssp_detection as B
from ssp_detection import (
    normalize_benchmark_steps,
    group_samples_by_model,
    get_frame_from_scene,
    get_context_frame_for_metric,
    read_image_cv2_local,
    infer_dataset_and_scene_from_path,
    safe_box_xyxy,
    _as_rgb_uint8,
    _box_iou_xyxy,
    normalize_detector_label,
)

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

# DINOv3 (Hugging Face format) for graded identity similarity. Load the local
# checkpoint directly so evaluation also works in offline environments.
DINOV3_HF_PATH = os.environ.get(
    "DINOV3_HF_PATH",
    os.environ.get("DINOV3_MODEL", "checkpoints/dinov3-vitl16"),
)

# Max center distance (as a fraction of the image diagonal) allowed for a VLM
# same-object match. This is the ONLY position constraint on VLM matches: if the
# VLM says "same object" and the center moved by less than this, we accept it
# regardless of box IoU. Generous on purpose so viewpoint-shifted objects still
# match (the strict iou>0 / center<=0.08 gate caused most objects to read as
# "not generated"). Set 1.0 to fully trust the VLM with no position bound.
VLM_MATCH_CENTER_TAU = float(os.environ.get("VLM_MATCH_CENTER_TAU", "0.35"))

# Two-stage matching: the loose global VLM matcher PROPOSES pairs (tolerant to
# viewpoint shift + detector mislabels); the per-pair defect VLM then VERIFIES
# identity. If the defect VLM judges a matched pair `wrong_identity`, we DEMOTE it
# -> the GT object becomes `missing` and the generated box becomes an `extra`.
# This is the precision safety net: it stops clearly-different objects (a window
# matched to a door, a brown chair matched to a different chair, a melted blob
# matched to a chair) from counting as matches, and restores the extra/hallucination
# objects that loose matching had absorbed.
VLM_DEMOTE_WRONG_IDENTITY = os.environ.get("VLM_DEMOTE_WRONG_IDENTITY", "1").lower() not in {"0", "false", "no"}
# Optional appearance floor (DINO identity cos01) below which a match is demoted
# even without the VLM. Default 0.0 = disabled (VLM is the authority).
DEMOTE_IDENTITY_FLOOR = float(os.environ.get("DEMOTE_IDENTITY_FLOOR", "0.0"))
# To OVERRULE a defect-VLM `wrong_identity` flag and KEEP the match, require the
# focused judge_pair to be confident same-object AND DINO identity above a floor.
# Catches "same chair @0.95 but identity_sim 0.75 (actually a dark blob/laptop)".
KEEP_OVERRULE_MIN_CONF = float(os.environ.get("KEEP_OVERRULE_MIN_CONF", "0.9"))
KEEP_OVERRULE_IDENTITY_FLOOR = float(os.environ.get("KEEP_OVERRULE_IDENTITY_FLOOR", "0.78"))

# Stage-3 RECOVERY: the global VLM proposer sometimes fails to propose a true
# correspondence (e.g. door<->generated-door), leaving the GT object `missing`
# while its generated counterpart sits in `extra`. For each still-missing GT we
# take the nearest unmatched generated box within VLM_RECOVER_CENTER_TAU and run a
# focused per-pair VLM same-object check (zoomed REF|GEN montage, more reliable
# than the global board). If confirmed, we PROMOTE missing+extra back into a match.
VLM_RECOVER_MISSING = os.environ.get("VLM_RECOVER_MISSING", "1").lower() not in {"0", "false", "no"}
VLM_RECOVER_CENTER_TAU = float(os.environ.get("VLM_RECOVER_CENTER_TAU", "0.35"))
VLM_RECOVER_MAX_PER_SAMPLE = int(os.environ.get("VLM_RECOVER_MAX_PER_SAMPLE", "12"))


# VLM-correspondence ablation.  This opt-in matcher keeps the detector,
# visibility/matchable filters, prompt protocol, and full-image candidate set
# fixed, but replaces PROPOSE/DEMOTE/RECOVER with a single maximum-weight
# one-to-one assignment.  Edge weights use only DINOv3 ROI cosine similarity;
# the same 0.35 center-distance bound as the formal VLM proposer is retained as
# a geometric sanity gate, not as part of the assignment score.  The 0.78
# cos01 floor is the already-frozen DINO identity floor used by the formal
# pipeline when deciding whether a focused VLM judgement can overrule a
# wrong-identity flag.
DINO_HUNGARIAN_CENTER_TAU = float(os.environ.get("DINO_HUNGARIAN_CENTER_TAU", "0.35"))
DINO_HUNGARIAN_IDENTITY_FLOOR = float(os.environ.get("DINO_HUNGARIAN_IDENTITY_FLOOR", "0.78"))

# Per-sample deep-debug trace (set by main when --dump-sample matches the id).
_DBG = False


def dbg(*a):
    if _DBG:
        print("[DBG]", *a, flush=True)


class DinoFeatureExtractor:
    """Self-contained DINOv3 ROI feature extractor via transformers."""

    def __init__(self, model_path: str, device):
        from transformers import AutoModel, AutoImageProcessor
        self.proc = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval().float()
        self.device = device

    @staticmethod
    def try_build(model_path: str, device) -> Optional["DinoFeatureExtractor"]:
        try:
            ext = DinoFeatureExtractor(model_path, device)
            print(f"[dino] loaded HF DINOv3 from {model_path}")
            return ext
        except Exception as e:
            print(f"[dino] HF DINOv3 unavailable ({type(e).__name__}: {e}); identity_sim will be skipped")
            return None

    def feat(self, crop_bgr: Optional[np.ndarray]):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        from PIL import Image
        rgb = _as_rgb_uint8(crop_bgr)  # pipeline arrays are already RGB
        try:
            inp = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(**inp)
            f = getattr(out, "pooler_output", None)
            if f is None:
                f = out.last_hidden_state[:, 0]
            f = F.normalize(f.float(), dim=-1)
            return f.detach().cpu()
        except Exception:
            return None

    def cos01(self, crop_a: Optional[np.ndarray], crop_b: Optional[np.ndarray]) -> Optional[float]:
        fa, fb = self.feat(crop_a), self.feat(crop_b)
        if fa is None or fb is None:
            return None
        c = float(F.cosine_similarity(fa, fb).item())
        return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0))


# =====================================================================
# Tier-1 cheap signals
# =====================================================================
def _to_gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return crop.astype(np.float32)
    return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _resize_to(crop: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    h = max(8, int(h)); w = max(8, int(w))
    if crop.shape[:2] == (h, w):
        return crop
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)


def ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Single-scale SSIM, numpy/cv2 only (no skimage dependency)."""
    a = a_gray.astype(np.float32)
    b = b_gray.astype(np.float32)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    k = (7, 7)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    m = num / (den + 1e-12)
    return float(np.clip(np.mean(m), -1.0, 1.0))


def edge_iou(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    ea = cv2.Canny(a_gray.astype(np.uint8), 60, 150) > 0
    eb = cv2.Canny(b_gray.astype(np.uint8), 60, 150) > 0
    # dilate a little so thin edges that shifted by a pixel still overlap
    ker = np.ones((3, 3), np.uint8)
    ea_d = cv2.dilate(ea.astype(np.uint8), ker) > 0
    eb_d = cv2.dilate(eb.astype(np.uint8), ker) > 0
    inter = np.logical_and(ea, eb_d).sum() + np.logical_and(eb, ea_d).sum()
    union = ea.sum() + eb.sum()
    if union == 0:
        return 1.0
    return float(np.clip(inter / union, 0.0, 1.0))


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def sharpness_ratio(gt_gray: np.ndarray, pred_gray: np.ndarray) -> float:
    """1.0 == same sharpness; <1 == predicted object is blurrier/melted."""
    sg = sharpness(gt_gray)
    sp = sharpness(pred_gray)
    if sg <= 1e-6 and sp <= 1e-6:
        return 1.0
    r = sp / (sg + 1e-6)
    # symmetric: penalise both over-blur and (rare) over-sharpen
    return float(min(r, 1.0 / max(r, 1e-6)))


def color_sim(gt_bgr: np.ndarray, pred_bgr: np.ndarray) -> float:
    """HSV histogram correlation in [0,1]."""
    def hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h
    c = cv2.compareHist(hist(gt_bgr), hist(pred_bgr), cv2.HISTCMP_CORREL)
    return float(np.clip((c + 1.0) / 2.0, 0.0, 1.0))


def _hu_log(mask: np.ndarray) -> Optional[np.ndarray]:
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < 10:
        return None
    mom = cv2.moments(m, binaryImage=True)
    hu = cv2.HuMoments(mom).flatten()
    sign = np.sign(hu)
    val = np.log10(np.abs(hu) + 1e-30)
    return sign * val


def shape_sim(gt_mask: Optional[np.ndarray], pred_mask: Optional[np.ndarray]) -> Optional[float]:
    if gt_mask is None or pred_mask is None:
        return None
    g = (gt_mask > 0).astype(np.uint8)
    p = (pred_mask > 0).astype(np.uint8)
    if int(g.sum()) < 10 or int(p.sum()) < 10:
        return None
    # Align both masks to their own tight bbox at a common canvas, compare IoU + Hu.
    def crop_to_bbox(m):
        ys, xs = np.where(m > 0)
        return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gc = crop_to_bbox(g)
    pc = crop_to_bbox(p)
    H = 64
    gcr = cv2.resize(gc, (H, H), interpolation=cv2.INTER_NEAREST) > 0
    pcr = cv2.resize(pc, (H, H), interpolation=cv2.INTER_NEAREST) > 0
    inter = np.logical_and(gcr, pcr).sum()
    union = np.logical_or(gcr, pcr).sum()
    iou = inter / union if union > 0 else 0.0
    hu_g = _hu_log(g)
    hu_p = _hu_log(p)
    if hu_g is None or hu_p is None:
        hu_score = iou
    else:
        d = float(np.linalg.norm(hu_g - hu_p))
        hu_score = float(np.exp(-d / 3.0))  # d~0 -> 1, larger -> 0
    return float(np.clip(0.6 * iou + 0.4 * hu_score, 0.0, 1.0))


# =====================================================================
# Per-object Tier-1 signal scoring
# =====================================================================
SSP_SIGNAL_WEIGHTS = OrderedDict([
    ("identity_sim", 0.30),
    ("struct_ssim", 0.20),
    ("edge_iou", 0.15),
    ("sharpness_ratio", 0.12),
    ("color_sim", 0.13),
    ("shape_sim", 0.10),
])


def compute_pair_signals(
    gt_crop_bgr: np.ndarray,
    pred_crop_bgr: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: Optional[np.ndarray],
    identity_sim: Optional[float],
) -> Dict[str, Any]:
    """All Tier-1 signals for one matched (GT,Pred) object pair, plus a composite."""
    h = max(gt_crop_bgr.shape[0], pred_crop_bgr.shape[0])
    w = max(gt_crop_bgr.shape[1], pred_crop_bgr.shape[1])
    h = int(np.clip(h, 16, 256)); w = int(np.clip(w, 16, 256))
    g = _resize_to(gt_crop_bgr, (h, w))
    p = _resize_to(pred_crop_bgr, (h, w))
    gg, pg = _to_gray(g), _to_gray(p)

    sig: Dict[str, Optional[float]] = {
        "identity_sim": None if identity_sim is None else float(np.clip(identity_sim, 0.0, 1.0)),
        "struct_ssim": float(np.clip((ssim(gg, pg) + 1.0) / 2.0, 0.0, 1.0)),
        "edge_iou": edge_iou(gg, pg),
        "sharpness_ratio": sharpness_ratio(gg, pg),
        "color_sim": color_sim(g, p),
        "shape_sim": shape_sim(gt_mask, pred_mask),
    }

    # Weighted composite over available signals.
    num, den = 0.0, 0.0
    for k, wt in SSP_SIGNAL_WEIGHTS.items():
        v = sig.get(k)
        if v is not None:
            num += wt * float(v)
            den += wt
    quality = float(num / den) if den > 0 else None

    # Discrete defect flags (cheap heuristics; thresholds tuned conservatively).
    flags = []
    if sig["identity_sim"] is not None and sig["identity_sim"] < 0.45:
        flags.append("identity_drift")
    if sig["struct_ssim"] < 0.30:
        flags.append("structural_deformation")
    if sig["edge_iou"] < 0.10:
        flags.append("contour_mismatch")
    if sig["sharpness_ratio"] < 0.35:
        flags.append("blur_or_melt")
    if sig["color_sim"] < 0.40:
        flags.append("color_texture_drift")
    if sig["shape_sim"] is not None and sig["shape_sim"] < 0.30:
        flags.append("warped_silhouette")

    return {"signals": sig, "object_quality": quality, "cv_defect_flags": flags}


# =====================================================================
# Tier-2 VLM audit pass: object quality + match verification (optional)
# =====================================================================
VLM_AUDIT_PROMPT = """You are auditing the visual quality of objects in a generated image for a physical-consistency benchmark.

The montage shows numbered object pairs. For each id N:
- LEFT crop = reference (ground-truth) object.
- RIGHT crop = the same object as drawn in the GENERATED image.
Some ids may be HALLUCINATION crops (single crop, no reference) — judge those for intrinsic plausibility only.

For every id, decide whether the generated object is well-formed or defective. Defect types:
- deformed        : melted / warped / wrong geometry
- part_count_wrong: extra or missing legs/arms/parts, duplicated structure
- missing_part    : a clearly expected part is absent
- texture_corrupted: garbled / smeared / nonsensical texture or pattern
- blurry          : object is mushy / out of focus relative to its surroundings
- wrong_identity  : the right crop is clearly a different object than the left

Return JSON ONLY:
{
  "objects": [
    {"id": 0, "well_formed": true, "defects": [], "severity": 0.0, "note": "short reason"},
    {"id": 1, "well_formed": false, "defects": ["deformed","blurry"], "severity": 0.7, "note": "..."}
  ]
}
severity is 0.0 (perfect) .. 1.0 (severe). Be strict but do not invent defects that are not visible."""


def _label_tile(tile: np.ndarray, text: str, color=(40, 220, 255)) -> np.ndarray:
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 18), (0, 0, 0), -1)
    cv2.putText(tile, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return tile


def build_vlm_audit_montage(records: List[Dict[str, Any]], tile: int = 150) -> Tuple[np.ndarray, List[int]]:
    """Lay out matched pairs (gt|pred) and hallucination crops into a grid."""
    rows = []
    ids_in_montage = []
    per_row = 4  # pairs per row
    buf = []
    for rec in records:
        oid = rec["vlm_id"]
        if rec["status"] == "match":
            g = _resize_to(rec["gt_crop"], (tile, tile)).copy()
            p = _resize_to(rec["pred_crop"], (tile, tile)).copy()
            _label_tile(g, f"id{oid} REF", (90, 230, 90))
            _label_tile(p, f"id{oid} GEN", (255, 170, 90))
            cell = np.hstack([g, p])
        else:  # hallucination
            p = _resize_to(rec["pred_crop"], (tile, tile)).copy()
            _label_tile(p, f"id{oid} EXTRA?", (40, 140, 255))
            blank = np.full((tile, tile, 3), 35, np.uint8)
            _label_tile(blank, "no ref", (120, 120, 120))
            cell = np.hstack([blank, p])
        ids_in_montage.append(oid)
        buf.append(cell)
        if len(buf) == per_row:
            rows.append(np.hstack(buf)); buf = []
    if buf:
        while len(buf) < per_row:
            buf.append(np.full_like(buf[0], 20))
        rows.append(np.hstack(buf))
    if not rows:
        return np.full((tile, tile, 3), 20, np.uint8), ids_in_montage
    montage = np.vstack(rows)
    return montage, ids_in_montage


VLM_DEFECT_MAX_TILES = 10  # cap per call: long montages -> long JSON -> parse failures


def _extract_objects_loose(resp: Any) -> List[Dict[str, Any]]:
    """Be liberal in what we accept: dict with 'objects', a bare list, or any
    nested list of per-object dicts (severity/defects/well_formed/id)."""
    if isinstance(resp, dict):
        if isinstance(resp.get("objects"), list):
            return resp["objects"]
        # any value that is a list of dicts with the right shape
        for v in resp.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and (
                "severity" in v[0] or "defects" in v[0] or "well_formed" in v[0] or "id" in v[0]):
                return v
    if isinstance(resp, list):
        return resp
    return []


def _run_vlm_defect_chunk(matcher, audit_chunk: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    from PIL import Image
    montage, _ = build_vlm_audit_montage(audit_chunk)
    tmp = None
    resp = None
    try:
        with tempfile.NamedTemporaryFile(prefix="vlm_defect_", suffix=".jpg", delete=False) as f:
            tmp = f.name
        Image.fromarray(_as_rgb_uint8(montage)).save(tmp, quality=92)
        retries = max(int(getattr(matcher, "max_retries", 2)), 3)
        resp = B.vlm_client.call_with_retries(
            matcher.client, VLM_AUDIT_PROMPT, tmp, retries, getattr(matcher, "retry_sleep", 1.0))
    except Exception as e:
        print(f"[vlm-defect] call failed: {type(e).__name__}: {str(e)[:160]}")
        return {}
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    objs = _extract_objects_loose(resp)
    if not objs:
        keys = list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__
        print(f"[vlm-defect] no parseable 'objects' (resp keys={keys})")
    out: Dict[int, Dict[str, Any]] = {}
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        try:
            oid = int(obj.get("id"))
        except Exception:
            continue
        out[oid] = {
            "well_formed": bool(obj.get("well_formed", True)),
            "defects": list(obj.get("defects", []) or []),
            "severity": float(obj.get("severity", 0.0) or 0.0),
            "note": str(obj.get("note", ""))[:160],
        }
    return out


def run_vlm_audit_pass(runner, records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    matcher = getattr(runner, "vlm_matcher", None)
    if matcher is None or not getattr(matcher, "is_available", lambda: False)():
        return {}
    audit = [r for r in records if r["status"] in ("match", "hallucination") and r.get("pred_crop") is not None]
    if not audit:
        return {}
    for i, r in enumerate(audit):
        r["vlm_id"] = i
    # chunk so each montage stays small enough to parse reliably
    out: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(audit), VLM_DEFECT_MAX_TILES):
        chunk = audit[start:start + VLM_DEFECT_MAX_TILES]
        out.update(_run_vlm_defect_chunk(matcher, chunk))

    # per-tile fallback: any object the chunked pass failed to parse gets its own
    # single-crop call (1 id -> trivial JSON -> robust). This closes the last gap.
    leftover = [r for r in audit if r["vlm_id"] not in out]
    if leftover:
        print(f"[vlm-defect] per-tile fallback for {len(leftover)} unparsed object(s)")
        for r in leftover:
            v = _run_vlm_defect_chunk(matcher, [r])
            if r["vlm_id"] in v:
                out[r["vlm_id"]] = v[r["vlm_id"]]
    # last resort: still-unparsed objects are marked explicitly (not silently OK)
    for r in audit:
        if r["vlm_id"] not in out:
            out[r["vlm_id"]] = {"well_formed": None, "defects": [], "severity": None, "note": "vlm_unparsed"}
    return out


# =====================================================================
# Detection + matching (faithful to the runner, but keeps the full pred set)
# =====================================================================
def _crop(img: np.ndarray, box: List[float], pad: float = 0.06) -> Optional[np.ndarray]:
    sb = safe_box_xyxy(box, img.shape[0], img.shape[1])
    if sb is None:
        return None
    x1, y1, x2, y2 = sb
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(img.shape[1], x2 + mx); y2 = min(img.shape[0], y2 + my)
    c = img[y1:y2, x1:x2]
    return c if c.size else None


def _mask_crop(mask: Optional[np.ndarray], box: List[float]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    sb = safe_box_xyxy(box, mask.shape[0], mask.shape[1])
    if sb is None:
        return None
    x1, y1, x2, y2 = sb
    c = mask[y1:y2, x1:x2]
    return c if c.size else None


def _vlm_match_loose(runner, gt_img, gen_img, g_boxes, g_labels, pm_boxes, pm_labels,
                     center_tau: float = VLM_MATCH_CENTER_TAU):
    """VLM same-object matching, trusting the VLM over the detector's noisy label.

    We reuse the runner's global-match montage + prompt, but do our OWN acceptance:
    keep a match if the VLM says same_object with enough confidence AND the center
    moved less than `center_tau` x diagonal. We deliberately DROP the detector-label
    hard filter (`_labels_compatible_for_object_match`) that judge_image_matches
    applies, because GDINO mislabels the same object across views (window->bathtub,
    sink->toilet) and that filter was killing valid VLM matches before the VLM was
    even consulted. No IoU requirement either."""
    matcher = runner.vlm_matcher
    H, W = gt_img.shape[:2]
    diag = float(max(np.hypot(H, W), 1.0))

    montage = B._make_vlm_global_match_montage(
        gt_img=gt_img, pred_img=gen_img, gt_boxes=g_boxes, pred_boxes=pm_boxes,
        gt_labels=g_labels, pred_labels=pm_labels,
        tile_size=max(640, getattr(matcher, "montage_tile_size", 360) * 2))
    prompt = B._build_vlm_global_match_prompt(g_boxes, pm_boxes, g_labels, pm_labels)

    from PIL import Image
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(prefix="vlm_loose_match_", suffix=".jpg", delete=False) as f:
            tmp = f.name
        Image.fromarray(montage).save(tmp, quality=92)
        resp = B.vlm_client.call_with_retries(
            matcher.client, prompt, tmp, max(int(getattr(matcher, "max_retries", 2)), 2),
            getattr(matcher, "retry_sleep", 1.0))
    except Exception as e:
        print(f"[vlm-match] call failed: {type(e).__name__}: {str(e)[:160]}")
        return [], [], {"vlm_error": f"{type(e).__name__}: {e}"}
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    raw = resp.get("matches", []) if isinstance(resp, dict) else []
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        raw = []
    if _DBG:
        dbg("=== STAGE1 PROPOSE (VLM global match) ===")
        dbg(f"GT boxes ({len(g_boxes)}):", [f"G{i}:{normalize_detector_label(g_labels[i]) if i<len(g_labels) else ''}" for i in range(len(g_boxes))])
        dbg(f"PRED candidates ({len(pm_boxes)}):", [f"P{i}:{normalize_detector_label(pm_labels[i]) if i<len(pm_labels) else ''}" for i in range(len(pm_boxes))])
        dbg("VLM raw matches:", json.dumps(raw, ensure_ascii=False)[:1500])

    min_conf = float(getattr(matcher, "min_match_confidence", 0.8))
    cands = []
    n_far = n_lowconf = n_notsame = 0
    for it in raw:
        if not isinstance(it, dict):
            continue
        gi = B._parse_vlm_index(it.get("gt_id", it.get("gt", it.get("source_id"))), "G", len(g_boxes))
        pi = B._parse_vlm_index(it.get("pred_id", it.get("prediction_id", it.get("generated_id"))), "P", len(pm_boxes))
        if gi is None or pi is None:
            dbg(f"  reject (bad idx): {it.get('gt_id')}->{it.get('pred_id')}")
            continue
        glab = normalize_detector_label(g_labels[gi]) if gi < len(g_labels) else ""
        plab = normalize_detector_label(pm_labels[pi]) if pi < len(pm_labels) else ""
        if not B._bool_from_vlm(it.get("same_object", True), True):
            n_notsame += 1
            dbg(f"  reject (same_object=false): G{gi}({glab}) P{pi}({plab})")
            continue
        conf = B._float_or_none(it.get("same_object_confidence", it.get("confidence", it.get("match_confidence"))))
        conf = float(np.clip(0.5 if conf is None else conf, 0.0, 1.0))
        if conf < min_conf:
            n_lowconf += 1
            dbg(f"  reject (conf {conf:.2f}<{min_conf}): G{gi}({glab}) P{pi}({plab})")
            continue
        gb, pb = g_boxes[gi], pm_boxes[pi]
        c_gt = np.array([(gb[0] + gb[2]) / 2.0, (gb[1] + gb[3]) / 2.0])
        c_pr = np.array([(pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0])
        cerr = float(np.linalg.norm(c_gt - c_pr))
        cnorm = cerr / diag
        if cnorm > center_tau:  # VLM says same object but it moved implausibly far
            n_far += 1
            dbg(f"  reject (center {cnorm:.2f}>{center_tau}): G{gi}({glab}) P{pi}({plab})")
            continue
        # Greedy-assignment rank score. Confidence is the base, but when the VLM
        # proposes one P to several G with equal confidence (it's unsure window-vs-
        # door), the contested box must go to the CLOSER, label-consistent GT.
        # Without this, a tie hands P to whichever was listed first (the bug that
        # gave the door's box to the window). NOT a hard gate -- only a tie-breaker.
        label_eq = 1.0 if (_canon_label(glab) == _canon_label(plab) and glab) else 0.0
        score = conf - 0.4 * cnorm + 0.1 * label_eq
        dbg(f"  ACCEPT G{gi}({glab}) <-> P{pi}({plab}) conf={conf:.2f} center={cnorm:.2f} score={score:.3f}")
        cands.append((score, conf, gi, pi, cerr, cnorm, it))

    # One-to-one assignment over the VLM-proposed pairs. Greedy (legacy) can spuriously
    # leave a GT unmatched when its only candidate is taken by a higher-scored GT; optimal
    # (Hungarian, max total score) resolves such contention and recovers those matches.
    # Toggle with VLM_MATCH_ASSIGN=greedy to reproduce the old behavior.
    edge = {}  # (gi,pi) -> (score, conf, cerr, cnorm, it); keep best-scoring duplicate
    for score, conf, gi, pi, cerr, cnorm, it in cands:
        k = (gi, pi)
        if k not in edge or score > edge[k][0]:
            edge[k] = (score, conf, cerr, cnorm, it)
    matched_pairs, match_details = [], []
    assign_mode = os.environ.get("VLM_MATCH_ASSIGN", "optimal").lower()
    if edge and assign_mode == "optimal":
        from scipy.optimize import linear_sum_assignment
        gis = sorted({gi for gi, _ in edge}); pis = sorted({pi for _, pi in edge})
        gidx = {g: i for i, g in enumerate(gis)}; pidx = {p: j for j, p in enumerate(pis)}
        NEG = -1e6
        M = np.full((len(gis), len(pis)), NEG, dtype=float)
        for (gi, pi), v in edge.items():
            M[gidx[gi], pidx[pi]] = v[0]
        rows, cols = linear_sum_assignment(-M)
        chosen = [(gis[r], pis[c]) for r, c in zip(rows, cols) if M[r, c] > NEG / 2]
        chosen.sort(key=lambda k: -edge[k][0])
    else:  # legacy greedy
        chosen, ug, up = [], set(), set()
        for score, conf, gi, pi, cerr, cnorm, it in sorted(cands, key=lambda x: -x[0]):
            if gi in ug or pi in up:
                dbg(f"  drop G{gi}<->P{pi} (already used) score={score:.3f}"); continue
            ug.add(gi); up.add(pi); chosen.append((gi, pi))
    for gi, pi in chosen:
        score, conf, cerr, cnorm, it = edge[(gi, pi)]
        matched_pairs.append((gi, pi))
        match_details.append({
            "rank": len(match_details) + 1, "gt_idx": gi, "pred_idx": pi,
            "iou": float(_box_iou_xyxy(g_boxes[gi], pm_boxes[pi])), "center_error": cerr, "center_norm": cnorm,
            "match_policy": f"vlm_same_object_loose_center_nolabelgate_{assign_mode}",
            "vlm_same_object_confidence": conf,
            "vlm_reason": str(it.get("reason", ""))[:160],
        })
    stats = {"vlm_raw_matches": int(len(raw)), "vlm_far_rejected": n_far,
             "vlm_lowconf_rejected": n_lowconf, "vlm_notsame": n_notsame,
             "vlm_error": resp.get("vlm_error") if isinstance(resp, dict) else None}
    return matched_pairs, match_details, stats


def _dinov3_hungarian_match(
    gt_img: np.ndarray,
    gen_img: np.ndarray,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    dino: Optional["DinoFeatureExtractor"],
    *,
    center_tau: float = DINO_HUNGARIAN_CENTER_TAU,
    identity_floor: float = DINO_HUNGARIAN_IDENTITY_FLOOR,
):
    """Match detector boxes with DINOv3 appearance and Hungarian assignment.

    The assignment includes zero-weight dummy rows/columns, so a GT or
    generated detection remains unmatched unless its DINOv3 cos01 similarity
    exceeds ``identity_floor``.  Detector labels and IoU never enter either the
    candidate gate or edge score.  A center-distance bound is retained solely
    to prevent implausible cross-image assignments between distant repeated
    objects, matching the formal VLM proposal protocol's geometric bound.
    """
    if dino is None:
        raise RuntimeError("DINOv3-only Hungarian matching requires the frozen DINOv3 extractor")
    if not gt_boxes or not pred_boxes:
        return [], [], {
            "candidate_pairs": 0,
            "accepted_pairs": 0,
            "center_tau": float(center_tau),
            "identity_floor_cos01": float(identity_floor),
        }

    gt_features = [dino.feat(_crop(gt_img, box)) for box in gt_boxes]
    pred_features = [dino.feat(_crop(gen_img, box)) for box in pred_boxes]
    gt_valid = [index for index, feat in enumerate(gt_features) if feat is not None]
    pred_valid = [index for index, feat in enumerate(pred_features) if feat is not None]
    if not gt_valid or not pred_valid:
        return [], [], {
            "candidate_pairs": 0,
            "accepted_pairs": 0,
            "gt_feature_failures": len(gt_boxes) - len(gt_valid),
            "pred_feature_failures": len(pred_boxes) - len(pred_valid),
            "center_tau": float(center_tau),
            "identity_floor_cos01": float(identity_floor),
        }

    gt_matrix = torch.cat([gt_features[index] for index in gt_valid], dim=0).float()
    pred_matrix = torch.cat([pred_features[index] for index in pred_valid], dim=0).float()
    cosine = torch.matmul(gt_matrix, pred_matrix.T).detach().cpu().numpy()
    cos01 = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

    height, width = gt_img.shape[:2]
    diagonal = float(max(np.hypot(height, width), 1.0))
    num_gt, num_pred = len(gt_boxes), len(pred_boxes)
    # Real-real edges occupy the top-left block.  All real-dummy and
    # dummy-dummy edges have weight zero, allowing either side to remain
    # unmatched.  Invalid real-real edges are forbidden with a large negative
    # value.  Positive weights are similarity above the frozen floor.
    size = num_gt + num_pred
    forbidden = -1e6
    weights = np.zeros((size, size), dtype=np.float64)
    weights[:num_gt, :num_pred] = forbidden
    edge_meta: Dict[Tuple[int, int], Dict[str, float]] = {}
    candidate_pairs = 0
    for local_gt, gt_index in enumerate(gt_valid):
        gt_box = gt_boxes[gt_index]
        gt_center = np.array(
            [(gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0]
        )
        for local_pred, pred_index in enumerate(pred_valid):
            pred_box = pred_boxes[pred_index]
            pred_center = np.array(
                [(pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0]
            )
            center_error = float(np.linalg.norm(gt_center - pred_center))
            center_norm = float(center_error / diagonal)
            similarity = float(cos01[local_gt, local_pred])
            if center_norm > center_tau or similarity < identity_floor:
                continue
            candidate_pairs += 1
            # A tiny positive offset makes an exact-floor edge preferable to
            # two dummy assignments without affecting ordering among edges.
            weights[gt_index, pred_index] = similarity - identity_floor + 1e-12
            edge_meta[(gt_index, pred_index)] = {
                "identity_sim": similarity,
                "feature_sim_cosine": float(cosine[local_gt, local_pred]),
                "center_error": center_error,
                "center_norm": center_norm,
            }

    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-weights)
    selected = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < num_gt and column < num_pred and (int(row), int(column)) in edge_meta
    ]
    selected.sort(key=lambda pair: (-edge_meta[pair]["identity_sim"], pair[0], pair[1]))
    match_details = []
    for rank, (gt_index, pred_index) in enumerate(selected, start=1):
        meta = edge_meta[(gt_index, pred_index)]
        match_details.append(
            {
                "rank": rank,
                "gt_idx": gt_index,
                "pred_idx": pred_index,
                "iou": float(_box_iou_xyxy(gt_boxes[gt_index], pred_boxes[pred_index])),
                "center_error": meta["center_error"],
                "center_norm": meta["center_norm"],
                "feature_sim": meta["feature_sim_cosine"],
                "identity_sim": meta["identity_sim"],
                "match_policy": "dinov3_cos01_hungarian_fullcand_center_gate",
            }
        )
    return selected, match_details, {
        "candidate_pairs": int(candidate_pairs),
        "accepted_pairs": int(len(selected)),
        "gt_feature_failures": int(len(gt_boxes) - len(gt_valid)),
        "pred_feature_failures": int(len(pred_boxes) - len(pred_valid)),
        "center_tau": float(center_tau),
        "identity_floor_cos01": float(identity_floor),
        "label_gate": False,
        "iou_gate": False,
        "assignment": "hungarian_max_weight_with_unmatched_dummies",
    }


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def detect_and_match(runner, sample: dict, vlm_match: bool = False,
                     match_policy: str = "relaxed", iou_tau_override: Optional[float] = None,
                     skip_match: bool = False,
                     dino: Optional["DinoFeatureExtractor"] = None) -> Optional[Dict[str, Any]]:
    """
    Reproduce the runner's GT/pred detection + matching, but also retain the
    *unfiltered* pred set (pred_all) so extra/hallucinated objects can be scored
    instead of dropped by the GT-neighborhood filter.

    vlm_match=True uses the runner's VLM same-object matcher (geom gate IoU>=0.15
    OR center<=0.08), which fixes correspondences that brittle IoU>=0.35 matching
    drops under camera motion / detector label disagreement.
    """
    ctx_paths = sample.get("context_paths") or []
    tgt_gt_path = sample.get("target_path")
    tgt_gen_path = sample.get("generated_path")
    if not ctx_paths or not tgt_gt_path or not tgt_gen_path:
        return None

    dataset_name = sample.get("dataset")
    scene_id = sample.get("scene_id")
    if not dataset_name or not scene_id:
        dataset_name, scene_id = infer_dataset_and_scene_from_path(ctx_paths[0])

    ctx_geometry_paths = sample.get("context_geometry_paths") or ctx_paths
    ctx_items = [
        get_context_frame_for_metric(
            dataset_name,
            scene_id,
            p,
            ctx_geometry_paths[i] if i < len(ctx_geometry_paths) else p,
        )
        for i, p in enumerate(ctx_paths)
    ]
    tgt_item = get_frame_from_scene(dataset_name, scene_id, tgt_gt_path)
    H_gt, W_gt = tgt_item.image.shape[:2]

    # generated image
    if os.path.isabs(tgt_gen_path) or os.path.exists(tgt_gen_path):
        gen = read_image_cv2_local(tgt_gen_path)
    else:
        try:
            gen = get_frame_from_scene(dataset_name, scene_id, tgt_gen_path).image.copy()
        except Exception:
            gen = read_image_cv2_local(tgt_gen_path)
    if gen is None:
        return None
    if gen.shape[:2] != (H_gt, W_gt):
        gen = cv2.resize(gen, (W_gt, H_gt))

    sample_prompts, step_prompts = runner._prompts_for_sample(sample)
    sem = runner.sem_tools

    # ---- GT objects: detect on target, visibility-filter, matchable-filter ----
    g_boxes, g_masks, g_labels = sem.get_instances_with_labels(tgt_item.image, sample_prompts)
    g_feats = [sem.extract_roi_feature(tgt_item.image, b) for b in g_boxes]

    # full target-GT set (matchable-filtered, NO visibility restriction): the
    # honest reference for "is this generated object a real scene object?".
    gf_boxes, _, _, gf_labels, _ = runner._filter_instances_for_matching(
        boxes=list(g_boxes), masks=list(g_masks), labels=list(g_labels),
        feats=[None] * len(g_boxes), image_shape=tgt_item.image.shape[:2])

    visible_mask = None
    try:
        visible_mask = runner._compute_target_visibility_mask(
            ctx_items=ctx_items, tgt_item=tgt_item, tgt_shape=(H_gt, W_gt))
        g_boxes, g_masks, g_feats, g_labels, _ = runner._filter_target_instances_by_visibility(
            boxes=g_boxes, masks=g_masks, feats=g_feats, visible_mask=visible_mask, labels=g_labels)
    except Exception as e:
        print(f"[gt-visibility] skipped: {e}")

    g_boxes, g_masks, g_feats, g_labels, _ = runner._filter_instances_for_matching(
        boxes=g_boxes, masks=g_masks, labels=g_labels, feats=g_feats,
        image_shape=tgt_item.image.shape[:2])

    # ---- Pred objects on generated image ----
    p_boxes_raw, p_masks_raw, p_labels_raw = sem.get_instances_with_labels(gen, sample_prompts)

    # pred_all = clean detections over the WHOLE image (no neighborhood filter)
    pa_boxes, pa_masks, _, pa_labels, _ = runner._filter_instances_for_matching(
        boxes=list(p_boxes_raw), masks=list(p_masks_raw), labels=list(p_labels_raw),
        feats=[None] * len(p_boxes_raw), image_shape=gen.shape[:2])

    # raw gen detections (before ANY filtering) — for "was it even detected?" diag
    praw_boxes = [list(map(float, b[:4])) for b in p_boxes_raw]
    praw_labels = [normalize_detector_label(l) for l in p_labels_raw]
    if _DBG:
        def _bx(b): return "[" + ",".join(str(int(x)) for x in b[:4]) + "]"
        dbg("=== STAGE0 DETECTION ===")
        dbg(f"GT (visible+matchable) {len(g_boxes)}:", [f"G{i}:{normalize_detector_label(g_labels[i])}{_bx(g_boxes[i])}" for i in range(len(g_boxes))])
        dbg(f"GEN raw detections {len(praw_boxes)}:", [f"{praw_labels[i]}{_bx(praw_boxes[i])}" for i in range(len(praw_boxes))])
        dbg(f"GEN pred_all (matchable) {len(pa_boxes)}:", [f"{normalize_detector_label(pa_labels[i])}{_bx(pa_boxes[i])}" for i in range(len(pa_boxes))])

    # pred_match = pipeline-faithful set used for matching (visibility + neighborhood)
    pm_boxes, pm_masks, pm_labels = list(p_boxes_raw), list(p_masks_raw), list(p_labels_raw)
    if visible_mask is not None:
        try:
            pm_boxes, pm_masks, pm_labels, _ = runner._filter_candidate_instances_by_visibility(
                boxes=pm_boxes, masks=pm_masks, visible_mask=visible_mask, labels=pm_labels)
        except Exception:
            pass
    try:
        pm_boxes, pm_masks, pm_labels, _ = runner._filter_pred_instances_by_gt_neighborhood(
            pred_boxes=pm_boxes, pred_masks=pm_masks, pred_labels=pm_labels,
            gt_boxes=g_boxes, gt_labels=g_labels, image_shape=gen.shape[:2])
    except Exception:
        pass
    pm_boxes, pm_masks, _, pm_labels, _ = runner._filter_instances_for_matching(
        boxes=pm_boxes, masks=pm_masks, labels=pm_labels,
        feats=[None] * len(pm_boxes), image_shape=gen.shape[:2])

    if skip_match:
        # set-mode only needs detections, not instance matching (skips the global VLM call)
        return {
            "tgt_img": tgt_item.image, "gen_img": gen,
            "gt_boxes": g_boxes, "gt_masks": g_masks, "gt_labels": g_labels, "gt_feats": g_feats,
            "gf_boxes": gf_boxes, "gf_labels": gf_labels,
            "pa_boxes": pa_boxes, "pa_masks": pa_masks, "pa_labels": pa_labels,
            "praw_boxes": praw_boxes, "praw_labels": praw_labels,
            "detector_prompts": sample_prompts, "step_object_prompts": step_prompts,
            "tgt_item": tgt_item, "ctx_items": ctx_items,
            "context_paths": ctx_paths, "context_geometry_paths": ctx_geometry_paths,
            "target_path": tgt_gt_path, "generated_path": tgt_gen_path,
        }

    # ---- match GT <-> pred_match (geometry + label + DINO) ----
    auto_like = runner.proposal_mode in {
        "auto", "cv_auto", "proposal", "prompt_free", "hybrid", "auto_text", "text_auto",
        "gdino", "gdino_box", "grounding_dino", "grounding_dino_box", "gdino_sam3", "grounding_dino_sam3"}
    iou_tau = 0.35 if auto_like else 0.5
    feat_tau = 0.60 if auto_like else 0.65
    eff_mode = runner.eval_mode
    if "dino" in str(eff_mode).lower() and not sem.has_dino():
        eff_mode = "none"

    matched_pairs, match_details = [], []
    match_mode = "geometric_iou_label"
    correspondence_stats: Dict[str, Any] = {}
    can_vlm = (match_policy != "dino_hungarian" and vlm_match
               and getattr(runner, "enable_vlm_match", False)
               and getattr(runner, "vlm_matcher", None) is not None
               and runner.vlm_matcher.is_available())
    # VLM path matches against the FULL detection set (pa_*), NOT the region-filtered
    # pm_* set. The visibility/neighborhood filters drop generated objects that moved
    # away from their GT location, which made displaced-but-real objects read as
    # "missing" (and their detection re-appear as a separate "hallucination"). With
    # VLM same-object + a center bound we no longer need that region restriction.
    dino_hungarian = match_policy == "dino_hungarian"
    if dino_hungarian:
        match_mode = "dinov3_cos01_hungarian_fullcand_center035_floor078"
    if can_vlm or dino_hungarian:
        cand_boxes, cand_masks, cand_labels = pa_boxes, pa_masks, pa_labels
    else:
        cand_boxes, cand_masks, cand_labels = pm_boxes, pm_masks, pm_labels
    if len(g_boxes) and len(cand_boxes):
        if dino_hungarian:
            matched_pairs, match_details, correspondence_stats = _dinov3_hungarian_match(
                tgt_item.image,
                gen,
                g_boxes,
                cand_boxes,
                dino,
            )
        elif can_vlm:
            try:
                matched_pairs, match_details, _ = _vlm_match_loose(
                    runner, tgt_item.image, gen, g_boxes, g_labels, cand_boxes, cand_labels)
                match_mode = "vlm_same_object_loose_center_fullcand"
            except Exception as e:
                print(f"[vlm-match] failed, falling back to geometric: {type(e).__name__}: {e}")
                can_vlm = False
                cand_boxes, cand_masks, cand_labels = pm_boxes, pm_masks, pm_labels
        if not can_vlm and not dino_hungarian:
            if match_policy == "relaxed":
                # synonym-canonicalized labels + lower IoU gate -> rescue motion-shifted
                # objects and detector label disagreement, at a small precision cost.
                gl = [_canon_label(l) for l in g_labels]
                pl = [_canon_label(l) for l in cand_labels]
                tau = iou_tau_override if iou_tau_override is not None else 0.25
                match_mode = "geometric_relaxed_synonym"
            else:
                gl, pl = g_labels, cand_labels
                tau = iou_tau_override if iou_tau_override is not None else iou_tau
                match_mode = "geometric_strict"
            _, _, _, matched_pairs, match_details = runner.sem_evaluator.evaluate_permanence(
                gt_boxes=g_boxes, gt_features=g_feats, pred_boxes=cand_boxes, gen_img_cv2=gen,
                sem_tools=sem, mode=eff_mode, iou_tau=tau, feat_tau=feat_tau,
                gt_masks=g_masks, pred_masks=cand_masks, gt_labels=gl, pred_labels=pl,
                require_label_match=True, mask_iou_tau=None)

    # the set used for matching becomes the "matched-object" set downstream
    used_boxes, used_masks, used_labels = cand_boxes, cand_masks, cand_labels
    return {
        "tgt_img": tgt_item.image, "gen_img": gen,
        "gt_boxes": g_boxes, "gt_masks": g_masks, "gt_labels": g_labels, "gt_feats": g_feats,
        "gf_boxes": gf_boxes, "gf_labels": gf_labels,
        "pm_boxes": used_boxes, "pm_masks": used_masks, "pm_labels": used_labels,
        "pa_boxes": pa_boxes, "pa_masks": pa_masks, "pa_labels": pa_labels,
        "praw_boxes": praw_boxes, "praw_labels": praw_labels,
        "matched_pairs": matched_pairs, "match_details": match_details,
        "match_mode": match_mode,
        "correspondence_stats": correspondence_stats if dino_hungarian else {},
        "detector_prompts": sample_prompts, "step_object_prompts": step_prompts,
        "tgt_item": tgt_item, "ctx_items": ctx_items,
        "context_paths": ctx_paths, "context_geometry_paths": ctx_geometry_paths,
        "target_path": tgt_gt_path, "generated_path": tgt_gen_path,
    }


def diagnose_missing(gbox, glabel, praw_boxes, praw_labels, pm_boxes, pm_labels, hw):
    """Why is a present-looking GT object unmatched? Inspect raw & filtered pred."""
    H, W = hw
    diag = float(np.hypot(H, W)) or 1.0

    def scan(boxes, labels):
        best = None
        for i, b in enumerate(boxes):
            iou = _box_iou_xyxy(gbox, b)
            cg = np.array([(gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2])
            cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            cd = float(np.linalg.norm(cg - cb) / diag)
            lab = labels[i] if i < len(labels) else ""
            lab_ok = B._labels_compatible_for_object_match(glabel, lab) if hasattr(B, "_labels_compatible_for_object_match") else (glabel == lab)
            score = iou - 0.3 * cd
            if best is None or score > best["score"]:
                best = {"score": score, "iou": round(iou, 3), "center_norm": round(cd, 3),
                        "label": lab, "label_compatible": bool(lab_ok)}
        return best

    raw = scan(praw_boxes, praw_labels)
    filt = scan(pm_boxes, pm_labels)
    # classify
    if raw is None:
        reason = "no detections in generated image at all"
    elif raw["iou"] < 0.05 and raw["center_norm"] > 0.25:
        reason = "NOT detected near this location (likely detector miss or object truly absent/displaced)"
    elif raw["iou"] >= 0.05 and (filt is None or filt["iou"] < 0.35):
        if not raw["label_compatible"]:
            reason = f"detected as '{raw['label']}' but label-incompatible with GT '{glabel}' (require_label_match)"
        elif raw["iou"] < 0.35:
            reason = f"detected & label-ok but IoU={raw['iou']} < 0.35 (object shifted under camera motion)"
        else:
            reason = "detected but removed by visibility/neighborhood/quality filter"
    else:
        reason = "detected & geometric-ok in filtered set but lost in assignment (contended; taken by another GT)"
    return {"gt_label": glabel, "best_raw": raw, "best_filtered": filt, "reason": reason}


def _dino_cos(sem, gt_feat, gen_img, pred_box) -> Optional[float]:
    if gt_feat is None:
        return None
    pf = sem.extract_roi_feature(gen_img, pred_box)
    if pf is None:
        return None
    try:
        return float(F.cosine_similarity(gt_feat.float(), pf.float()).item())
    except Exception:
        return None


def _canon_label(lbl: Any) -> str:
    """Map a detector label to its synonym-group representative so that detector
    label noise (chair/stool, sofa/bed, desk/table, monitor/tv, ...) does not
    block an otherwise-valid match. Reuses the runner's synonym groups."""
    n = normalize_detector_label(lbl)
    for group in getattr(B, "_MATCH_LABEL_SYNONYM_GROUPS", []):
        if n in group:
            return sorted(group)[0]
    return n


def _center_in(box_inner: List[float], box_outer: List[float]) -> bool:
    cx = (box_inner[0] + box_inner[2]) * 0.5
    cy = (box_inner[1] + box_inner[3]) * 0.5
    return (box_outer[0] <= cx <= box_outer[2]) and (box_outer[1] <= cy <= box_outer[3])


def analyze_sample(runner, sample: dict, use_vlm: bool = False, dino: Optional["DinoFeatureExtractor"] = None,
                   match_policy: str = "relaxed") -> Optional[Dict[str, Any]]:
    det = detect_and_match(
        runner,
        sample,
        vlm_match=use_vlm,
        match_policy=match_policy,
        dino=dino,
    )
    if det is None:
        return None

    sem = runner.sem_tools
    tgt, gen = det["tgt_img"], det["gen_img"]
    g_boxes, g_masks, g_labels, g_feats = det["gt_boxes"], det["gt_masks"], det["gt_labels"], det["gt_feats"]
    gf_boxes = det.get("gf_boxes", g_boxes)
    gf_labels = det.get("gf_labels", [""] * len(gf_boxes))
    pm_boxes, pm_masks, pm_labels = det["pm_boxes"], det["pm_masks"], det["pm_labels"]
    pa_boxes, pa_masks, pa_labels = det["pa_boxes"], det["pa_masks"], det["pa_labels"]
    matched_pairs = det["matched_pairs"]

    matched_gt = {int(g): int(p) for g, p in matched_pairs}
    H_img, W_img = tgt.shape[:2]
    img_diag = float(max(np.hypot(H_img, W_img), 1.0))
    records: List[Dict[str, Any]] = []

    # ---- matched objects: full Tier-1 defect signals ----
    for gi in range(len(g_boxes)):
        gbox = g_boxes[gi]
        glabel = normalize_detector_label(g_labels[gi]) if gi < len(g_labels) else ""
        if gi in matched_gt:
            pi = matched_gt[gi]
            pbox = pm_boxes[pi]
            gc = _crop(tgt, gbox); pc = _crop(gen, pbox)
            if gc is None or pc is None:
                continue
            # Graded identity similarity from DINOv3 crops (preferred), else None.
            if dino is not None:
                ident01 = dino.cos01(gc, pc)
            else:
                ident = _dino_cos(sem, g_feats[gi] if gi < len(g_feats) else None, gen, pbox)
                ident01 = None if ident is None else (ident + 1.0) / 2.0
            d = compute_pair_signals(
                gc, pc,
                g_masks[gi] if gi < len(g_masks) else None,
                pm_masks[pi] if pi < len(pm_masks) else None,
                ident01)
            c_gt = np.array([(gbox[0] + gbox[2]) / 2.0, (gbox[1] + gbox[3]) / 2.0])
            c_pr = np.array([(pbox[0] + pbox[2]) / 2.0, (pbox[1] + pbox[3]) / 2.0])
            cerr_px = float(np.linalg.norm(c_gt - c_pr))
            records.append({
                "status": "match", "gt_idx": gi, "pred_idx": pi,
                "label": glabel or (normalize_detector_label(pm_labels[pi]) if pi < len(pm_labels) else ""),
                "gt_box": [float(x) for x in gbox[:4]], "pred_box": [float(x) for x in pbox[:4]],
                "gt_crop": gc, "pred_crop": pc,
                "center_err_px": cerr_px, "center_norm": float(cerr_px / img_diag),
                **d,
            })
        else:
            mdiag = diagnose_missing(
                [float(x) for x in gbox[:4]], glabel,
                det["praw_boxes"], det["praw_labels"], pm_boxes, [normalize_detector_label(l) for l in pm_labels],
                tgt.shape[:2])
            records.append({
                "status": "missing", "gt_idx": gi, "pred_idx": None,
                "label": glabel, "gt_box": [float(x) for x in gbox[:4]], "pred_box": None,
                "gt_crop": _crop(tgt, gbox), "pred_crop": None,
                "signals": {}, "object_quality": 0.0, "cv_defect_flags": ["missing_object"],
                "missing_diag": mdiag,
            })

    # ---- hallucination / extra objects ----
    # pa_boxes (full-image detections) and pm_boxes (matching set) come from the
    # SAME raw detections, so a matched generated object also appears in pa_boxes.
    # We must NOT re-count it as extra. Two exclusions:
    #   1) it overlaps an already-MATCHED pred box (same generated object, same
    #      frame -> reliable IoU). This is the key dedup: the matcher is VLM/
    #      cross-view, so a matched object that moved would otherwise fail the
    #      geometric GT test below and be double-counted as extra.
    #   2) it geometrically corresponds to a full-GT box in the target view.
    # Only boxes failing BOTH are true extras / hallucinations.
    matched_pred_boxes = [r["pred_box"] for r in records if r["status"] == "match" and r.get("pred_box")]
    extra_boxes_added: List[List[float]] = []
    for pi in range(len(pa_boxes)):
        pbox = pa_boxes[pi]
        already_matched = any(
            (_box_iou_xyxy(pbox, mb) >= 0.4) or _center_in(pbox, mb) or _center_in(mb, pbox)
            for mb in matched_pred_boxes
        )
        if already_matched:
            continue
        # "corresponds to a real GT object?" — strong overlap counts regardless of
        # label, but the loose center-inside test only counts when labels are
        # compatible. Otherwise a big GT box (e.g. a window spanning the wall)
        # swallows a genuinely-extra object sitting inside it (e.g. an extra door).
        plab_pi = _canon_label(pa_labels[pi]) if pi < len(pa_labels) else ""
        corresponds = False
        for gj, gb in enumerate(gf_boxes):
            if _box_iou_xyxy(pbox, gb) >= 0.25:
                corresponds = True; break
            if (_center_in(pbox, gb) or _center_in(gb, pbox)):
                glab = _canon_label(gf_labels[gj]) if gj < len(gf_labels) else ""
                if plab_pi and glab and plab_pi == glab:
                    corresponds = True; break
        if corresponds:
            continue
        # dedup against extras already recorded (GDINO emits multiple boxes/object)
        if any((_box_iou_xyxy(pbox, eb) >= 0.4) or _center_in(pbox, eb) or _center_in(eb, pbox)
               for eb in extra_boxes_added):
            continue
        extra_boxes_added.append([float(x) for x in pbox[:4]])
        records.append({
            "status": "hallucination", "gt_idx": None, "pred_idx": pi,
            "label": normalize_detector_label(pa_labels[pi]) if pi < len(pa_labels) else "",
            "gt_box": None, "pred_box": [float(x) for x in pbox[:4]],
            "gt_crop": None, "pred_crop": _crop(gen, pbox),
            "signals": {}, "object_quality": None, "cv_defect_flags": ["unmatched_extra_object"],
        })

    # ---- optional Tier-2 VLM defect pass (one call) ----
    vlm_used = False
    if use_vlm:
        verdicts = run_vlm_audit_pass(runner, records)
        if verdicts:
            vlm_used = True
            id2rec = {r.get("vlm_id"): r for r in records if "vlm_id" in r}
            for oid, v in verdicts.items():
                r = id2rec.get(oid)
                if r is not None:
                    r["vlm"] = v

    # ---- STAGE 2 verify: demote a match ONLY if the focused per-pair judge_pair
    # ALSO disagrees. The defect-montage `wrong_identity` flag alone is noisy and can
    # contradict the per-pair call (causing demote->recover thrash). Confirming with
    # the same authority recovery uses makes the two stages consistent. `confirmed_notsame`
    # is handed to recovery so it never re-tests a pair already shown to differ.
    can_pair_vlm = (use_vlm and getattr(runner, "vlm_matcher", None) is not None
                    and runner.vlm_matcher.is_available())
    min_conf = float(getattr(getattr(runner, "vlm_matcher", None), "min_match_confidence", 0.8)) if can_pair_vlm else 0.8
    confirmed_notsame = set()
    demoted = []
    if _DBG:
        dbg("=== VLM AUDIT VERIFY (demote wrong_identity, confirmed by judge_pair) ===")
    for r in records:
        if r["status"] != "match":
            continue
        flagged = bool(VLM_DEMOTE_WRONG_IDENTITY and r.get("vlm") and "wrong_identity" in (r["vlm"].get("defects") or []))
        floor_bad = (DEMOTE_IDENTITY_FLOOR > 0 and (r.get("signals") or {}).get("identity_sim") is not None
                     and r["signals"]["identity_sim"] < DEMOTE_IDENTITY_FLOOR)
        demote = False
        if flagged and can_pair_vlm and r.get("gt_box") and r.get("pred_box"):
            try:
                jp = runner.vlm_matcher.judge_pair(gt_img=tgt, pred_img=gen, gt_box=r["gt_box"],
                                                   pred_box=r["pred_box"], gt_label=r.get("label", ""),
                                                   pred_label=r.get("label", ""))
                same = bool(jp.get("same_object")); conf = float(jp.get("same_object_confidence") or 0.0)
            except Exception as e:
                print(f"[demote-confirm] judge_pair failed: {type(e).__name__}: {str(e)[:120]}")
                same, conf = False, 0.0
            ident = (r.get("signals") or {}).get("identity_sim")
            # To OVERRULE a wrong_identity flag we now require BOTH a confident
            # focused same-object call AND a DINO identity above a floor. A chair
            # whose generated crop looks like a dark blob gets judge_pair=same@0.95
            # but identity_sim ~0.75 (a different/degraded object) -> the flag stands.
            id_ok = (ident is None or ident >= KEEP_OVERRULE_IDENTITY_FLOOR)
            if same and conf >= KEEP_OVERRULE_MIN_CONF and id_ok:
                # overrule the noisy defect-montage flag: keep + strip flag
                r["vlm"]["defects"] = [x for x in (r["vlm"].get("defects") or []) if x != "wrong_identity"]
                r["vlm"]["well_formed"] = (len(r["vlm"]["defects"]) == 0)
                r["vlm"]["note"] = (str(r["vlm"].get("note", "")) + " |identity confirmed")[:160]
                dbg(f"  KEEP G{r['gt_idx']}({r.get('label')}) P{r.get('pred_idx')}: judge_pair same={same} conf={conf:.2f} ident={ident} -> overruled")
            else:
                demote = True
                confirmed_notsame.add((r["gt_idx"], r.get("pred_idx")))
                why = "judge_pair not-same/lowconf" if not (same and conf >= KEEP_OVERRULE_MIN_CONF) else f"identity_sim {ident} < {KEEP_OVERRULE_IDENTITY_FLOOR}"
                dbg(f"  DEMOTE G{r['gt_idx']}({r.get('label')}) P{r.get('pred_idx')}: same={same} conf={conf:.2f} ident={ident} -> {why}")
        elif flagged and not can_pair_vlm:
            demote = True
        if floor_bad:
            demote = True
        if demote:
            demoted.append(r)
    demoted_ids = {id(r) for r in demoted}
    if demoted:
        kept = []
        for r in records:
            if r["status"] == "match" and id(r) in demoted_ids:
                # GT object -> missing (its proposed match was a different object)
                kept.append({
                    "status": "missing", "gt_idx": r["gt_idx"], "pred_idx": None,
                    "label": r["label"], "gt_box": r["gt_box"], "pred_box": None,
                    "gt_crop": r.get("gt_crop"), "pred_crop": None,
                    "signals": {}, "object_quality": 0.0,
                    "cv_defect_flags": ["match_rejected_wrong_identity"],
                    "missing_diag": {"reason": "VLM verify: proposed generated match is a DIFFERENT object (wrong_identity)"},
                })
                # generated box -> extra (it does not faithfully reproduce the GT object)
                kept.append({
                    "status": "hallucination", "gt_idx": None, "pred_idx": r.get("pred_idx"),
                    "label": r["label"], "gt_box": None, "pred_box": r["pred_box"],
                    "gt_crop": None, "pred_crop": r.get("pred_crop"),
                    "signals": r.get("signals", {}), "object_quality": None,
                    "cv_defect_flags": ["extra_from_wrong_identity_match"],
                    "vlm": r.get("vlm"),
                })
            else:
                kept.append(r)
        records = kept
        matched_pairs = [(r["gt_idx"], r["pred_idx"]) for r in records if r["status"] == "match"]

    # ---- STAGE 3 recover: per-pair VLM rescue for still-missing GT objects ----
    # The global proposer can miss a true correspondence (door<->generated-door).
    # For each still-missing GT, take the nearest unmatched generated box and run a
    # focused per-pair VLM same-object check; if confirmed, promote missing+extra
    # back into a match.
    n_recovered = 0
    can_recover = (use_vlm and VLM_RECOVER_MISSING and vlm_used
                   and getattr(runner, "vlm_matcher", None) is not None
                   and runner.vlm_matcher.is_available())
    if _DBG:
        dbg("=== STAGE3 RECOVER (per-pair VLM rescue) ===  can_recover=", can_recover,
            "| #missing=", sum(1 for r in records if r["status"] == "missing"),
            "#extra=", sum(1 for r in records if r["status"] == "hallucination"))
    if can_recover:
        def _ctr(b):
            return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
        used_extra_ids = set()
        for mr in [r for r in records if r["status"] == "missing"]:
            if n_recovered >= VLM_RECOVER_MAX_PER_SAMPLE:
                break
            if mr.get("gt_box") is None or mr.get("gt_crop") is None:
                continue
            cg = _ctr(mr["gt_box"])
            cand, cand_cn = None, None
            for er in [r for r in records if r["status"] == "hallucination"]:
                if id(er) in used_extra_ids or er.get("pred_box") is None or er.get("pred_crop") is None:
                    continue
                # don't re-test a pair Stage-2 already confirmed to be different objects
                if (mr.get("gt_idx"), er.get("pred_idx")) in confirmed_notsame:
                    continue
                cn = float(np.linalg.norm(cg - _ctr(er["pred_box"])) / img_diag)
                if cn <= VLM_RECOVER_CENTER_TAU and (cand is None or cn < cand_cn):
                    cand, cand_cn = er, cn
            if cand is None:
                dbg(f"  missing G{mr['gt_idx']}({mr.get('label')}): no extra within {VLM_RECOVER_CENTER_TAU} -> stays missing")
                continue
            try:
                resp = runner.vlm_matcher.judge_pair(
                    gt_img=tgt, pred_img=gen, gt_box=mr["gt_box"], pred_box=cand["pred_box"],
                    gt_label=mr.get("label", ""), pred_label=cand.get("label", ""))
            except Exception as e:
                print(f"[vlm-recover] judge_pair failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            same = bool(resp.get("same_object"))
            conf = float(resp.get("same_object_confidence") or 0.0)
            dbg(f"  missing G{mr['gt_idx']}({mr.get('label')}) vs nearest extra({cand.get('label')},center={cand_cn:.2f}): judge_pair same={same} conf={conf:.2f} -> {'RECOVER' if (same and conf>=float(getattr(runner.vlm_matcher,'min_match_confidence',0.8))) else 'no'}")
            if not (same and conf >= float(getattr(runner.vlm_matcher, "min_match_confidence", 0.8))):
                continue
            # PROMOTE
            gi = mr["gt_idx"]; pidx = cand.get("pred_idx")
            gmask = g_masks[gi] if (gi is not None and gi < len(g_masks)) else None
            pmask = pa_masks[pidx] if (pidx is not None and pidx < len(pa_masks)) else None
            ident01 = dino.cos01(mr["gt_crop"], cand["pred_crop"]) if dino is not None else None
            d = compute_pair_signals(mr["gt_crop"], cand["pred_crop"], gmask, pmask, ident01)
            cerr = float(np.linalg.norm(cg - _ctr(cand["pred_box"])))
            v = dict(cand.get("vlm") or {})
            if v:  # keep real defects but drop the now-disproven wrong_identity
                v["defects"] = [x for x in (v.get("defects") or []) if x != "wrong_identity"]
                v["well_formed"] = (len(v["defects"]) == 0)
                v["note"] = (str(v.get("note", "")) + " |recovered")[:160]
            new_match = {
                "status": "match", "gt_idx": gi, "pred_idx": pidx,
                "label": mr.get("label") or cand.get("label"),
                "gt_box": mr["gt_box"], "pred_box": cand["pred_box"],
                "gt_crop": mr["gt_crop"], "pred_crop": cand["pred_crop"],
                "center_err_px": cerr, "center_norm": float(cerr / img_diag),
                "recovered": True, "vlm": v or {"well_formed": True, "defects": [], "severity": 0.0, "note": "recovered"},
                **d,
            }
            records = [new_match if r is mr else r for r in records if r is not cand]
            used_extra_ids.add(id(cand))
            n_recovered += 1
        if n_recovered:
            matched_pairs = [(r["gt_idx"], r["pred_idx"]) for r in records if r["status"] == "match"]
            print(f"[vlm-recover] promoted {n_recovered} missing->match for {sample.get('id')}")

    # ---- image-level aggregates ----
    matched = [r for r in records if r["status"] == "match"]
    missing = [r for r in records if r["status"] == "missing"]
    hallu = [r for r in records if r["status"] == "hallucination"]
    qualities = [r["object_quality"] for r in matched if r.get("object_quality") is not None]
    cnorms = [r["center_norm"] for r in matched if r.get("center_norm") is not None]
    cpxs = [r["center_err_px"] for r in matched if r.get("center_err_px") is not None]
    n_gt = len(g_boxes)

    def mean_sig(key):
        vals = [r["signals"].get(key) for r in matched if r.get("signals", {}).get(key) is not None]
        return float(np.mean(vals)) if vals else None

    # object-layout metrics (reuse the runner's evaluator) for the human-correlation study
    try:
        topo = runner.sem_evaluator.compute_topology_consistency(gt_boxes=g_boxes, pred_boxes=pm_boxes, matched_pairs=matched_pairs)
    except Exception:
        topo = {}
    try:
        layout = runner.sem_evaluator.compute_matched_layout_metrics(gt_boxes=g_boxes, pred_boxes=pm_boxes, matched_pairs=matched_pairs)
    except Exception:
        layout = {}

    summary = {
        "id": sample.get("id"),
        "parent_id": sample.get("parent_id"),
        "step": sample.get("step"),
        "num_steps": sample.get("num_steps"),
        "strict_input_local_step": sample.get("strict_input_local_step"),
        "original_benchmark_step": sample.get("original_benchmark_step"),
        "gdino_box_threshold": float(runner.sem_tools.gdino_box_threshold),
        "gdino_text_threshold": float(runner.sem_tools.gdino_text_threshold),
        "grounding_dino_path": getattr(runner.sem_tools, "gdino_path", None),
        "object_prompt_protocol": sample.get("object_prompt_protocol", "legacy_case_prompts_plus_fixed36"),
        "object_prompt_source": sample.get("object_prompt_source", "unspecified"),
        "step_object_prompts": det.get("step_object_prompts", []),
        "detector_prompts": det.get("detector_prompts", []),
        "num_gt_objects": n_gt,
        "num_matched": len(matched),
        "num_missing": len(missing),
        "num_hallucination": len(hallu),
        "num_pred_all": len(pa_boxes),
        "Completeness": float(len(matched) / n_gt) if n_gt else None,
        "ObjectIntegrity": float(np.mean(qualities)) if qualities else None,
        "Center_Error_Norm": float(np.mean(cnorms)) if cnorms else None,  # mean center dist / image diagonal
        "Center_Error_px": float(np.mean(cpxs)) if cpxs else None,
        "Topology": topo.get("Topology"),
        "Topology_MatchedPairCoverage": topo.get("Topology_MatchedPairCoverage"),
        "Matched_Location_Relation": layout.get("Matched_Location_Relation"),
        "Matched_Pair_Area_Ratio_Score": layout.get("Matched_Pair_Area_Ratio_Score"),
        "img_diag_px": img_diag,
        "Hallucination_Rate": float(len(hallu) / max(len(pa_boxes), 1)),
        "match_mode": det.get("match_mode", "geometric_iou_label"),
        "correspondence_stats": det.get("correspondence_stats", {}),
        "mean_identity_sim": mean_sig("identity_sim"),
        "mean_struct_ssim": mean_sig("struct_ssim"),
        "mean_edge_iou": mean_sig("edge_iou"),
        "mean_sharpness_ratio": mean_sig("sharpness_ratio"),
        "mean_color_sim": mean_sig("color_sim"),
        "mean_shape_sim": mean_sig("shape_sim"),
        "vlm_used": vlm_used,
    }
    summary["num_demoted_wrong_identity"] = int(len(demoted))
    summary["num_recovered"] = int(n_recovered)
    summary["objects"] = [
        {"status": r["status"], "label": r.get("label"), "gt_idx": r.get("gt_idx"),
         "pred_idx": r.get("pred_idx"), "center_norm": r.get("center_norm"),
         "object_quality": r.get("object_quality"), "recovered": bool(r.get("recovered")),
         "gt_box": r.get("gt_box"), "pred_box": r.get("pred_box"),
         "cv_defect_flags": r.get("cv_defect_flags"),
         "vlm_well_formed": (r.get("vlm") or {}).get("well_formed"),
         "vlm_defects": (r.get("vlm") or {}).get("defects"),
         "vlm_note": (r.get("vlm") or {}).get("note")}
        for r in records
    ]
    if vlm_used:
        sevs = [r["vlm"]["severity"] for r in records if r.get("vlm") and r["vlm"].get("severity") is not None]
        summary["mean_vlm_severity"] = float(np.mean(sevs)) if sevs else None
        summary["num_vlm_defective"] = int(sum(
            1 for r in records if r.get("vlm") and r["vlm"].get("well_formed") is False))
        summary["num_vlm_defective_matched"] = int(sum(
            1 for r in records if r.get("status") == "match" and r.get("vlm")
            and r["vlm"].get("well_formed") is False))
        summary["num_vlm_defective_hallucination"] = int(sum(
            1 for r in records if r.get("status") == "hallucination" and r.get("vlm")
            and r["vlm"].get("well_formed") is False))
        summary["num_vlm_unparsed"] = int(sum(
            1 for r in records if r.get("vlm") and r["vlm"].get("note") == "vlm_unparsed"))

    return {"summary": summary, "records": records, "tgt_img": tgt, "gen_img": gen,
            "gt_boxes": g_boxes, "matched_pairs": matched_pairs}


# =====================================================================
# SET MODE: class-level count + quality (no which-is-which instance matching)
# =====================================================================
SET_WELLFORMED_TAU = float(os.environ.get("SET_WELLFORMED_TAU", "0.55"))  # object_quality >= this => well-formed


def analyze_sample_set(runner, sample: dict, use_vlm: bool = False,
                       dino: Optional["DinoFeatureExtractor"] = None) -> Optional[Dict[str, Any]]:
    """Class-level evaluation for repeated/ambiguous objects.

    For each detector class we compare GT-count vs generated-count (count consistency)
    and score each generated instance by its BEST resemblance to *any* real instance of
    that same class -- so we never have to decide 'which chair is which'. Unique classes
    (<=1 GT and <=1 gen) fall back to a direct 1-1 comparison.
    """
    det = detect_and_match(runner, sample, skip_match=True)
    if det is None:
        return None
    tgt, gen = det["tgt_img"], det["gen_img"]
    g_boxes, g_masks, g_labels = det["gt_boxes"], det["gt_masks"], det["gt_labels"]
    pa_boxes, pa_masks, pa_labels = det["pa_boxes"], det["pa_masks"], det["pa_labels"]
    H, W = tgt.shape[:2]
    diag = float(max(np.hypot(H, W), 1.0))

    def grp(labels, n):
        d = {}
        for i in range(n):
            d.setdefault(_canon_label(labels[i]) if i < len(labels) else "", []).append(i)
        return d
    gt_by = grp(g_labels, len(g_boxes))
    gen_by = grp(pa_labels, len(pa_boxes))
    classes = sorted(set(gt_by) | set(gen_by))

    def pair_q(gt_i, gen_j):
        gc = _crop(tgt, g_boxes[gt_i]); pc = _crop(gen, pa_boxes[gen_j])
        if gc is None or pc is None:
            return None
        ident = dino.cos01(gc, pc) if dino is not None else None
        d = compute_pair_signals(gc, pc, g_masks[gt_i] if gt_i < len(g_masks) else None,
                                 pa_masks[gen_j] if gen_j < len(pa_masks) else None, ident)
        return d

    class_results = []
    for c in classes:
        if not c:
            continue
        gts = gt_by.get(c, []); gens = gen_by.get(c, [])
        ngt, ngen = len(gts), len(gens)
        count_score = (min(ngt, ngen) / max(ngt, ngen)) if max(ngt, ngen) > 0 else 1.0
        # score each generated instance by its best resemblance to ANY real instance of this class
        inst = []
        for j in gens:
            best = None
            for i in gts:
                d = pair_q(i, j)
                if d is None:
                    continue
                if best is None or (d.get("object_quality") or 0) > (best.get("object_quality") or 0):
                    best = d
            q = best.get("object_quality") if best else None
            inst.append({
                "gen_idx": j, "box": [float(x) for x in pa_boxes[j][:4]],
                "pred_crop": _crop(gen, pa_boxes[j]),
                "object_quality": q,
                "signals": best.get("signals", {}) if best else {},
                "cv_defect_flags": best.get("cv_defect_flags", []) if best else [],
                "well_formed": (q is not None and q >= SET_WELLFORMED_TAU),
                "has_ref": ngt > 0,
            })
        n_well = sum(1 for x in inst if x["well_formed"])
        class_results.append({
            "cls": c, "gt_count": ngt, "gen_count": ngen, "count_score": count_score,
            "extra": max(0, ngen - ngt), "missing": max(0, ngt - ngen),
            "instances": inst,
            "repeated": (ngt >= 2 or ngen >= 2),
            "gt_idxs": gts,
        })

    # optional VLM: judge each generated instance intrinsically (well-formed object of class c?)
    # This can flip an instance's well_formed flag, so counts are computed AFTER it.
    vlm_used = False
    if use_vlm:
        verds = run_vlm_setmode(runner, gen, class_results)
        vlm_used = bool(verds)

    # finalize per-class well-formed/defective counts from the (possibly VLM-updated) instances
    for cr in class_results:
        nwf = sum(1 for x in cr["instances"] if x.get("well_formed"))
        cr["num_well_formed"] = nwf
        cr["num_defective"] = cr["gen_count"] - nwf

    # ---- image-level aggregates ----
    all_inst = [x for cr in class_results for x in cr["instances"]]
    qual = [x["object_quality"] for x in all_inst if x.get("object_quality") is not None]
    tot_gt = sum(cr["gt_count"] for cr in class_results)
    tot_gen = sum(cr["gen_count"] for cr in class_results)
    tot_well = sum(cr["num_well_formed"] for cr in class_results)
    # count-consistency weighted by GT count (how right are the per-class counts)
    cw = [(cr["count_score"], max(cr["gt_count"], cr["gen_count"])) for cr in class_results if max(cr["gt_count"], cr["gen_count"]) > 0]
    count_acc = (sum(s * w for s, w in cw) / sum(w for _, w in cw)) if cw else None

    summary = {
        "id": sample.get("id"), "mode": "set",
        "parent_id": sample.get("parent_id"),
        "step": sample.get("step"),
        "num_steps": sample.get("num_steps"),
        "strict_input_local_step": sample.get("strict_input_local_step"),
        "original_benchmark_step": sample.get("original_benchmark_step"),
        "object_prompt_protocol": sample.get("object_prompt_protocol", "legacy_case_prompts_plus_fixed36"),
        "object_prompt_source": sample.get("object_prompt_source", "unspecified"),
        "step_object_prompts": det.get("step_object_prompts", []),
        "detector_prompts": det.get("detector_prompts", []),
        "num_classes": len([c for c in class_results]),
        "total_gt": tot_gt, "total_gen": tot_gen,
        "Class_Count_Accuracy": count_acc,
        "ObjectIntegrity": float(np.mean(qual)) if qual else None,
        "WellFormed_Rate": float(tot_well / tot_gen) if tot_gen else None,
        "total_extra": sum(cr["extra"] for cr in class_results),
        "total_missing": sum(cr["missing"] for cr in class_results),
        "vlm_used": vlm_used,
        "classes": [{k: v for k, v in cr.items() if k not in ("instances", "gt_idxs")} for cr in class_results],
    }
    return {"summary": summary, "class_results": class_results, "tgt_img": tgt, "gen_img": gen,
            "g_boxes": g_boxes, "g_labels": g_labels, "pa_boxes": pa_boxes, "pa_labels": pa_labels}


def run_vlm_setmode(runner, gen, class_results) -> bool:
    """Ask the VLM, per generated instance, whether it is a well-formed object of its
    class (intrinsic). Updates each instance dict with vlm defect info. Returns True if used."""
    matcher = getattr(runner, "vlm_matcher", None)
    if matcher is None or not getattr(matcher, "is_available", lambda: False)():
        return False
    inst = [(cr["cls"], x) for cr in class_results for x in cr["instances"] if x.get("pred_crop") is not None]
    if not inst:
        return False
    used = False
    for start in range(0, len(inst), VLM_DEFECT_MAX_TILES):
        chunk = inst[start:start + VLM_DEFECT_MAX_TILES]
        recs = [{"status": "hallucination", "vlm_id": k, "pred_crop": x[1]["pred_crop"],
                 "label": x[0], "gt_crop": None} for k, x in enumerate(chunk)]
        verds = _run_vlm_defect_chunk(matcher, recs)
        for k, (cls, x) in enumerate(chunk):
            v = verds.get(k)
            if v:
                used = True
                x["vlm"] = v
                if not v.get("well_formed", True):
                    x["well_formed"] = False
    return used


def _f(v, nd=2):
    if v is None:
        return "NA"
    if isinstance(v, float) and (np.isnan(v)):
        return "NA"
    return f"{v:.{nd}f}"


def _make_json_safe(o):
    if isinstance(o, dict):
        return {k: _make_json_safe(v) for k, v in o.items() if k not in ("gt_crop", "pred_crop")}
    if isinstance(o, (list, tuple)):
        return [_make_json_safe(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser(description="SSP object detection/matching pass (metrics-only).")
    ap.add_argument("--input-jsonl", nargs="+", default=B.INPUT_JSONL,
                    help="one or more generated.jsonl / official_input.jsonl files")
    ap.add_argument("--output-dir", required=True,
                    help="directory for per-model ssp_results.json")
    ap.add_argument("--max-samples-per-model", type=int, default=4)
    ap.add_argument("--use-vlm", action="store_true", help="enable the Tier-2 Qwen3-VL match/quality pass (slow)")
    ap.add_argument("--only-id", default=None, help="only evaluate samples whose id contains this substring")
    ap.add_argument("--debug-missing", action="store_true", help="print why each present-looking GT object is unmatched")
    ap.add_argument("--match-policy", default="relaxed", choices=["strict", "relaxed", "dino_hungarian"],
                    help=("non-VLM matching: 'strict'=IoU0.35+exact label; "
                          "'relaxed'=IoU0.25+synonym labels; 'dino_hungarian'=DINOv3 "
                          "appearance-only Hungarian assignment over full-image candidates"))
    ap.add_argument("--dump-sample", default=None, help="print a full per-step trace for samples whose id contains this substring")
    ap.add_argument("--mode", default="instance", choices=["instance", "set"],
                    help="'instance'=1-to-1 matched view (default); 'set'=class-level count+quality")
    ap.add_argument("--shard-count", type=int, default=1, help="split each model's cases into N shards (parallel jobs)")
    ap.add_argument("--shard-index", type=int, default=0, help="this job processes cases where case_idx %% shard_count == shard_index")
    ap.add_argument("--no-vis", action="store_true", help="skip panel rendering (metrics only); required, see below")
    ap.add_argument(
        "--no-depth-topology",
        action="store_true",
        help="disable DA3 depth front/behind topology scoring; all other SSP fields are unchanged.",
    )
    args = ap.parse_args()
    assert 0 <= args.shard_index < args.shard_count or args.shard_count == 1, "bad shard config"
    if not args.no_vis:
        ap.error("the vendored evaluator does not bundle the panel renderer; pass --no-vis")

    from datetime import datetime
    run_stamp = f"job{os.environ.get('SLURM_JOB_ID', 'local')} {datetime.now():%Y-%m-%d %H:%M:%S}"
    print(f"[init] run stamp = {run_stamp}", flush=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from run_obj_metric_one import build_runner
    if args.use_vlm:
        # load the VLM (local 30B by default) for both same-object matching and defect recheck
        os.environ["BENCHMARK_ENABLE_VLM_MATCH"] = "1"
        os.environ.setdefault("VLM_LOCAL_DEVICE_MAP", "auto")
    else:
        os.environ["BENCHMARK_ENABLE_VLM_MATCH"] = "0"  # don't even load the 30B model
    print("[init] building runner (loading detection / DINO / DA3 models)...")
    runner = build_runner(save_vis=False)
    print("[init] model status:", getattr(runner.sem_tools, "model_status", {}))

    # Self-contained DINOv3 for graded identity similarity.
    dev = getattr(runner.sem_tools, "device", None)
    if dev is None and torch is not None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    dino = DinoFeatureExtractor.try_build(DINOV3_HF_PATH, dev)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    grand_summary = {}

    for jsonl in args.input_jsonl:
        with open(jsonl, "r", encoding="utf-8") as f:
            cases = [json.loads(l) for l in f if l.strip()]
        groups = group_samples_by_model(cases, fallback=Path(jsonl).stem)
        for model_name, model_cases in groups.items():
            model_cases = model_cases[: args.max_samples_per_model]
            if args.shard_count > 1:
                model_cases = [c for ci, c in enumerate(model_cases) if ci % args.shard_count == args.shard_index]
            mdir = out_root / model_name
            mdir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== model={model_name}  shard={args.shard_index}/{args.shard_count}  cases={len(model_cases)} ===")
            per_model = []
            for case in model_cases:
                for sample in normalize_benchmark_steps(case):
                    sid = sample.get("id", "unknown")
                    if args.only_id and args.only_id not in str(sid):
                        continue
                    globals()["_DBG"] = bool(args.dump_sample and args.dump_sample in str(sid))
                    if globals()["_DBG"]:
                        print(f"\n========== DUMP TRACE: {sid} ({model_name}) ==========", flush=True)
                    try:
                        if args.mode == "set":
                            res = analyze_sample_set(runner, sample, use_vlm=args.use_vlm, dino=dino)
                        else:
                            res = analyze_sample(
                                runner,
                                sample,
                                use_vlm=args.use_vlm,
                                dino=dino,
                                match_policy=args.match_policy,
                            )
                    except Exception as e:
                        print(f"  [err] {sid}: {type(e).__name__}: {e}")
                        traceback.print_exc(limit=2)
                        continue
                    if res is None:
                        print(f"  [skip] {sid}: no detections / missing paths")
                        continue
                    per_model.append(res["summary"])
                    s = res["summary"]
                    if args.mode == "set":
                        print(f"  [ok] {sid}: GT={s['total_gt']} Gen={s['total_gen']} "
                              f"CountAcc={_f(s['Class_Count_Accuracy'])} WellFormed={_f(s['WellFormed_Rate'])} "
                              f"extra={s['total_extra']} miss={s['total_missing']}")
                    else:
                        print(f"  [ok] {sid}: GT={s['num_gt_objects']} match={s['num_matched']} "
                              f"miss={s['num_missing']} extra={s['num_hallucination']} "
                              f"Integrity={_f(s['ObjectIntegrity'])}")
                    if args.debug_missing and args.mode == "instance":
                        for r in res["records"]:
                            if r["status"] == "missing" and r.get("missing_diag"):
                                md = r["missing_diag"]
                                print(f"      MISSING G{r['gt_idx']} '{r['label']}': {md.get('reason')}")
                                if md.get('best_raw') is not None:
                                    print(f"          best_raw_detection={md['best_raw']}")
            # model-level mean (keys depend on mode; missing keys are skipped)
            agg = {}
            agg_keys = (("Class_Count_Accuracy", "ObjectIntegrity", "WellFormed_Rate",
                         "total_gt", "total_gen", "total_extra", "total_missing")
                        if args.mode == "set" else
                        ("Completeness", "ObjectIntegrity", "Center_Error_Norm", "Center_Error_px",
                         "Hallucination_Rate", "Topology", "DepthTopology_Matched", "DepthTopology_AllGT",
                         "DepthTopology_ObjectPairCoverage", "DepthTopology_DepthValidPairCoverage",
                         "mean_identity_sim", "mean_struct_ssim", "mean_edge_iou",
                         "mean_sharpness_ratio", "mean_color_sim", "mean_shape_sim"))
            for k in agg_keys:
                vals = [r[k] for r in per_model if r.get(k) is not None]
                agg[k] = float(np.mean(vals)) if vals else None
            agg["num_samples"] = len(per_model)
            grand_summary[model_name] = agg
            jname = ("ssp_results.json" if args.shard_count == 1
                     else f"ssp_results.shard{args.shard_index:02d}of{args.shard_count:02d}.json")
            with open(mdir / jname, "w", encoding="utf-8") as f:
                json.dump(_make_json_safe({"model_summary": agg, "details": per_model}), f, indent=2, ensure_ascii=False)
            print(f"  model summary ({jname}): {json.dumps(agg, ensure_ascii=False)}")

    if args.shard_count == 1:
        with open(out_root / "grand_summary.json", "w", encoding="utf-8") as f:
            json.dump(_make_json_safe(grand_summary), f, indent=2, ensure_ascii=False)
    print(f"\n✅ done. output under: {out_root}")
    print(json.dumps(_make_json_safe(grand_summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
