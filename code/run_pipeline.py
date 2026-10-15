import os
import csv
import math
import cv2
import numpy as np

# Core modules for the final thesis pipeline:
# - stage1.py: motion-guided launch and initial trajectory
# - stage2.py: appearance-guided tiny-object tracking
# - stage3.py: selective least-squares trajectory recovery
import stage1 as s1
import stage2 as s2
import stage3 as s3

# =========================
# Output / settings
# =========================
OUT_DIR = "results"
S1_OUT = os.path.join(OUT_DIR, "stage1")
S2_OUT = os.path.join(OUT_DIR, "stage2")
S3_OUT = os.path.join(OUT_DIR, "stage3")
VIS_OUT = os.path.join(OUT_DIR, "visualization")
FULL_DIR = os.path.join(VIS_OUT, "full_trajectory_frames")
PANEL_DIR = os.path.join(VIS_OUT, "four_panel_frames")

for p in [OUT_DIR, S1_OUT, S2_OUT, S3_OUT, VIS_OUT, FULL_DIR, PANEL_DIR]:
    os.makedirs(p, exist_ok=True)

# =========================
# Handoff rule
# =========================
HANDOFF_WINDOW = 4
HANDOFF_BAD_COUNT = 2
STEP_TAU = 4.0
AREA_TAU = 12.0
TLEN_TAU = 3.0
MHI_CONF_TAU = 0.45
MIN_VALID_BEFORE_HANDOFF = 5

# Compactness Transition Score (CTS) for Stage1 -> Stage2 handoff.
# Goal: describe the actual transition from elongated motion blur to compact target.
# CTS is high when the Stage1 motion component has:
#   smaller area, shorter line-band length, more compact bbox shape, and acceptable Stage1 motion quality.
USE_COMPACTNESS_HANDOFF = True
CTS_AREA_REF = 20.0      # close to previous area<=12 rule, but slightly smoother
CTS_LEN_REF = 6.0        # close to previous t_len<=3 rule, but slightly smoother
CTS_AR_MAX = 3.0         # bbox aspect ratio above 3 is treated as elongated/blur-like
CTS_W_AREA = 0.35
CTS_W_LEN = 0.35
CTS_W_SHAPE = 0.20
CTS_W_Q = 0.10
CTS_TAU = 0.65
CTS_RECENT_COUNT = 2

# Keep the old threshold rule only as a diagnostic/fallback.
# This lets you compare whether CTS handoff is close to the previous experience-based handoff.
USE_OLD_HANDOFF_FALLBACK = True
# =========================
# Output controls
# =========================
RUN_VISUALIZATION = True       # thesis batch mode: no videos by default
SAVE_FULL_FRAMES = True
SAVE_PANEL_FRAMES = True
SAVE_VIDEOS = True
SAVE_STAGE1_COMPONENTS = True

# =========================
# Visualization
# =========================
STAGE2_METHODS = ["lbp", "ltp", "weighted"]
MAIN_MAIN_VIS_METHOD = "ltp"
FPS = 12
ROI_SIZE = 56
ROI_ZOOM = 10
PANEL_W, PANEL_H = 640, 480

C_GT = (0, 255, 0)       # green
C_S1 = (0, 255, 255)     # yellow
C_S2 = (255, 0, 255)     # magenta
C_HYB = (0, 180, 255)    # orange
C_TXT = (255, 255, 255)
C_BLUE = (255, 0, 0)


# =========================================================
# Basic utilities
# =========================================================
def sf(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


def label(img, text, xy, color=C_TXT, scale=0.6, thick=2):
    cv2.putText(img, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def resize_panel(img):
    if img is None:
        img = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def compose4(a, b, c, d):
    return np.vstack([
        np.hstack([resize_panel(a), resize_panel(b)]),
        np.hstack([resize_panel(c), resize_panel(d)])
    ])


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_video(paths, out_path, fps=12):
    if not paths:
        return
    first = cv2.imread(paths[0])
    if first is None:
        return
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        vw.write(im)
    vw.release()


def gt_item(gt, frame_name, idx):
    return gt.get(frame_name, gt.get(idx, None))


def xy_from_stage1(r):
    if r is None:
        return None, None
    return sf(r.get("x")), sf(r.get("y"))


def xy_from_stage2(r):
    if r is None:
        return None, None
    return sf(r.get("pred_x")), sf(r.get("pred_y"))


def xy_from_stage3(r):
    if r is None:
        return None, None
    return sf(r.get("stage3_x")), sf(r.get("stage3_y"))


def normalize_vec(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n


# =========================================================
# ROI / panels
# =========================================================
def crop_zoom(img, cx, cy, size=ROI_SIZE, zoom=ROI_ZOOM):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    r = size // 2
    x1, x2 = max(0, cx - r), min(w, cx + r)
    y1, y2 = max(0, cy - r), min(h, cy + r)
    crop = img[y1:y2, x1:x2].copy()
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    z = cv2.resize(canvas, (size * zoom, size * zoom), interpolation=cv2.INTER_NEAREST)
    for k in range(0, size * zoom, zoom):
        cv2.line(z, (k, 0), (k, size * zoom - 1), (45, 45, 45), 1)
        cv2.line(z, (0, k), (size * zoom - 1, k), (45, 45, 45), 1)
    return z, (x1, y1)


def draw_roi_point(roi, origin, x, y, color, name):
    if x is None or y is None:
        return
    ox, oy = origin
    rx = int(round((x - ox) * ROI_ZOOM))
    ry = int(round((y - oy) * ROI_ZOOM))
    if 0 <= rx < roi.shape[1] and 0 <= ry < roi.shape[0]:
        cv2.circle(roi, (rx, ry), max(3, ROI_ZOOM // 2), color, -1)
        label(roi, name, (rx + 5, max(18, ry - 5)), color, 0.45, 1)


def make_roi_panel(img, pred_xy, gt_rec, title, color_pred):
    px, py = pred_xy
    if px is None or py is None:
        if gt_rec is not None:
            px, py = gt_rec[0], gt_rec[1]
        else:
            px, py = img.shape[1] // 2, img.shape[0] // 2
    roi, org = crop_zoom(img, px, py)
    if gt_rec is not None:
        draw_roi_point(roi, org, gt_rec[0], gt_rec[1], C_GT, "GT")
    draw_roi_point(roi, org, px, py, color_pred, "Pred")
    label(roi, title, (10, 24), C_TXT, 0.52, 1)
    return roi


def make_empty_panel(title, lines=None):
    im = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    label(im, title, (20, 35), (0, 255, 255), 0.85, 2)
    y = 75
    for l in (lines or []):
        label(im, l, (20, y), C_TXT, 0.6, 1)
        y += 28
    return im


def make_diff_panel(gray_list, idx):
    if idx <= 0:
        diff = np.zeros_like(gray_list[0])
    else:
        diff = cv2.absdiff(gray_list[idx], gray_list[idx - 1])
    vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    label(vis, "Frame difference |I_t - I_t-1|", (15, 30), C_TXT, 0.65, 2)
    return vis


def make_candidate_panel(img, comps, s1r, gt_rec, frame_name):
    vis = img.copy()
    for c in comps or []:
        x = int(c.get("bbox_x", 0)); y = int(c.get("bbox_y", 0))
        w = int(c.get("bbox_w", 0)); h = int(c.get("bbox_h", 0))
        if w > 0 and h > 0:
            cv2.rectangle(vis, (x, y), (x + w, y + h), C_BLUE, 1)
    px, py = xy_from_stage1(s1r)
    if px is not None and py is not None:
        cv2.circle(vis, (int(round(px)), int(round(py))), 7, C_S1, -1)
        label(vis, s1r.get("component_type", s1r.get("mode", "")), (int(px) + 8, int(py) - 8), C_S1, 0.52, 1)
    if gt_rec is not None:
        gx, gy, gr = gt_rec
        rr = max(4, int(round(gr)) if gr == gr else 5)
        cv2.circle(vis, (int(round(gx)), int(round(gy))), rr, C_GT, 2)
    label(vis, "Stage1 diff candidates (blue) + selected point", (15, 30), C_TXT, 0.65, 2)
    return vis


def make_status_panel(fname, idx, stage, s1r, s2r, handoff, gt_rec):
    im = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    lines = [
        f"Frame {idx:04d} {fname}",
        f"Stage: {stage}",
        f"Handoff: {handoff.get('handoff_frame', '')}",
        f"Trigger: {handoff.get('trigger_reason', '')}",
    ]
    if s1r is not None:
        lines += [
            f"S1 mode: {s1r.get('mode', '')}",
            f"step: {s1r.get('step', '')}",
            f"area: {s1r.get('area', '')}",
            f"t_len: {s1r.get('t_len', '')}",
            f"Q_mhi: {s1r.get('motion_conf', '')}",
        ]
    if s2r is not None:
        lines += [
            f"S2 method: {s2r.get('_method', MAIN_MAIN_VIS_METHOD)}",
            f"S2 mode: {s2r.get('mode', '')}",
            f"score: {s2r.get('score', '')}",
            f"error: {s2r.get('error_px', '')}",
            f"reliable: {s2r.get('online_reliable', '')}",
            f"reason: {s2r.get('reliable_reason', '')}",
        ]
    if gt_rec is not None:
        lines += [f"GT: ({gt_rec[0]:.1f},{gt_rec[1]:.1f}), r={gt_rec[2]}"]
    label(im, "Status", (20, 35), (0, 255, 255), 0.85, 2)
    y = 75
    for l in lines:
        label(im, l, (20, y), C_TXT, 0.6, 1)
        y += 28
    return im


# =========================================================
# Stage2 heatmap display
# =========================================================
def normalize_scoremap_valid(score_map):
    m = score_map.astype(np.float32)
    valid = m > -1e17
    out = np.zeros_like(m, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = m[valid]
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-6:
        return out
    out[valid] = (vals - mn) / (mx - mn)
    return out


def recompute_stage2_heatmap(gray_list, idx, row, method):
    """
    Recompute a Stage2 score map for visualization using the revised Stage2 API.

    The older visualization code called s2.build_feature() and s2.score_feature(),
    but the current Stage2 version uses normalized multi-cue scoring directly.
    This function mirrors the current Stage2 score design:
        Score = wA*Appearance + wM*Motion + wD*Direction + wP*Position
    and therefore restores the heatmap in stage2_4panel_*.mp4.
    """
    if row is None:
        return None

    # Use the prediction center if available. This is the actual center of the
    # Stage2 search region saved in tracking.csv. Fall back to pred_x/pred_y.
    px = sf(row.get("pred_center_x"))
    py = sf(row.get("pred_center_y"))
    if px is None or py is None:
        px = sf(row.get("pred_x"))
        py = sf(row.get("pred_y"))

    vx = sf(row.get("vx")) or 0.0
    vy = sf(row.get("vy")) or 0.0
    if px is None or py is None:
        return None

    try:
        miss_count = int(float(row.get("miss_count", 0))) if row.get("miss_count", "") != "" else 0
    except Exception:
        miss_count = 0

    gray_cur = cv2.GaussianBlur(gray_list[idx], (3, 3), 0)
    gray_prev = cv2.GaussianBlur(gray_list[max(0, idx - 1)], (3, 3), 0)
    gray_next = cv2.GaussianBlur(gray_list[min(len(gray_list) - 1, idx + 1)], (3, 3), 0)

    if not hasattr(s2, "compute_search_radius"):
        return None

    R = int(s2.compute_search_radius(vx, vy, miss_count))
    R = max(3, min(R, 80))
    side = 2 * R + 1
    score_map = np.full((side, side), -1e18, dtype=np.float32)
    valid_mask = np.zeros((side, side), dtype=np.uint8)

    H, W = gray_cur.shape
    prev_speed = math.hypot(vx, vy)
    cand = []

    for oy, dy_off in enumerate(range(-R, R + 1)):
        for ox, dx_off in enumerate(range(-R, R + 1)):
            cx = px + dx_off
            cy = py + dy_off
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                continue

            dx = cx - px
            dy = cy - py

            # Keep the same candidate gates as Stage2 when those functions exist.
            if hasattr(s2, "speed_ok"):
                ok, *_ = s2.speed_ok(dx, dy, prev_speed)
                if not ok:
                    continue
            if hasattr(s2, "accel_ok"):
                ok, *_ = s2.accel_ok(vx, vy, dx, dy)
                if not ok:
                    continue
            if hasattr(s2, "direction_ok"):
                ok, cand_ang, _ = s2.direction_ok(vx, vy, dx, dy)
                if not ok:
                    continue
            else:
                cand_ang = 0.0
            if hasattr(s2, "vertical_ok"):
                ok, _ = s2.vertical_ok(vy, dy)
                if not ok:
                    continue

            # Revised Stage2 feature computation.
            try:
                s3_raw = s2.temporal_scale_response(gray_prev, gray_cur, gray_next, cx, cy, 3, method)
                s5_raw = s2.temporal_scale_response(gray_prev, gray_cur, gray_next, cx, cy, 5, method)
                s7_raw = s2.temporal_scale_response(gray_prev, gray_cur, gray_next, cx, cy, 7, method)
                fd_raw = s2.compute_frame_diff_raw(gray_prev, gray_cur, gray_next, cx, cy)
            except Exception:
                return None

            pred_dist = math.hypot(cx - px, cy - py)
            x_motion_dev = abs(dx - vx)
            cand.append({
                "ox": ox, "oy": oy, "x": cx, "y": cy,
                "s3_raw": s3_raw, "s5_raw": s5_raw, "s7_raw": s7_raw,
                "fd_raw": fd_raw, "pred_dist": pred_dist,
                "x_motion_dev": x_motion_dev, "cand_ang": cand_ang,
            })

    if not cand:
        return None

    # Normalize each scale response within the visible candidate set, matching Stage2.
    s3n = s2.norm01([c["s3_raw"] for c in cand]) if hasattr(s2, "norm01") else np.zeros(len(cand), np.float32)
    s5n = s2.norm01([c["s5_raw"] for c in cand]) if hasattr(s2, "norm01") else np.zeros(len(cand), np.float32)
    s7n = s2.norm01([c["s7_raw"] for c in cand]) if hasattr(s2, "norm01") else np.zeros(len(cand), np.float32)

    base_w3 = sf(row.get("w3")) or 0.30
    base_w5 = sf(row.get("w5")) or 0.50
    base_w7 = sf(row.get("w7")) or 0.20

    best_score = -1e18
    best_xy = None

    for k, c in enumerate(cand):
        s3_norm = float(s3n[k])
        s5_norm = float(s5n[k])
        s7_norm = float(s7n[k])

        if hasattr(s2, "refine_scale_weights"):
            cw3, cw5, cw7 = s2.refine_scale_weights(base_w3, base_w5, base_w7, s3_norm, s5_norm, s7_norm)
        else:
            cw3, cw5, cw7 = base_w3, base_w5, base_w7

        clip01_fn = getattr(s2, "clip01", clip01)
        appearance = clip01_fn(cw3 * s3_norm + cw5 * s5_norm + cw7 * s7_norm)

        fd_clip = getattr(s2, "FD_CLIP", 0.35)
        motion = clip01_fn(c["fd_raw"] / max(1e-8, fd_clip))

        if hasattr(s2, "normalized_direction_score"):
            direction = s2.normalized_direction_score(c["cand_ang"])
        else:
            direction = 1.0

        if hasattr(s2, "normalized_position_score"):
            pscore = s2.normalized_position_score(c["pred_dist"])
        else:
            pscore = math.exp(-(c["pred_dist"] ** 2) / max(1e-6, 2.0 * 8.0 * 8.0))

        xm_sigma = getattr(s2, "XMOTION_SIGMA", 6.0)
        xscore = math.exp(-(c["x_motion_dev"] ** 2) / max(1e-6, 2.0 * xm_sigma * xm_sigma))
        position = clip01_fn(0.70 * pscore + 0.30 * xscore)

        total = (
            getattr(s2, "W_APPEARANCE", 0.40) * appearance +
            getattr(s2, "W_MOTION", 0.25) * motion +
            getattr(s2, "W_DIRECTION", 0.20) * direction +
            getattr(s2, "W_POSITION", 0.15) * position
        )

        score_map[c["oy"], c["ox"]] = float(total)
        valid_mask[c["oy"], c["ox"]] = 1
        if total > best_score:
            best_score = float(total)
            best_xy = (c["x"], c["y"])

    return {
        "score_map": score_map,
        "valid_mask": valid_mask,
        "center": (px, py),
        "R": R,
        "best_xy": best_xy,
        "best_score": best_score,
    }

def make_stage2_heatmap_panel(gray_list, idx, img, row, gt_rec, method):
    hm = recompute_stage2_heatmap(gray_list, idx, row, method)
    if hm is None:
        return make_empty_panel(f"{method.upper()} heatmap", ["No Stage2 prediction or v6 API mismatch"])

    score_map = hm["score_map"]
    valid_mask = hm["valid_mask"]
    cx, cy = hm["center"]
    R = hm["R"]
    best_xy = hm["best_xy"]

    norm = normalize_scoremap_valid(score_map)
    heat = (norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color[valid_mask == 0] = (20, 20, 20)

    zoom = max(3, min(10, 360 // max(1, heat_color.shape[0])))
    hz = cv2.resize(heat_color, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

    x0 = cx - R
    y0 = cy - R

    def to_h(x, y):
        return int(round((x - x0) * zoom)), int(round((y - y0) * zoom))

    hx, hy = to_h(cx, cy)
    if 0 <= hx < hz.shape[1] and 0 <= hy < hz.shape[0]:
        cv2.circle(hz, (hx, hy), 5, C_TXT, -1)
        label(hz, "C", (hx + 6, max(16, hy - 6)), C_TXT, 0.45, 1)

    if best_xy is not None:
        bx, by = to_h(best_xy[0], best_xy[1])
        if 0 <= bx < hz.shape[1] and 0 <= by < hz.shape[0]:
            cv2.circle(hz, (bx, by), 6, C_S2, -1)
            label(hz, "Best", (bx + 6, max(16, by - 6)), C_S2, 0.45, 1)

    if gt_rec is not None:
        gx, gy, _ = gt_rec
        gx2, gy2 = to_h(gx, gy)
        if 0 <= gx2 < hz.shape[1] and 0 <= gy2 < hz.shape[0]:
            cv2.circle(hz, (gx2, gy2), 6, C_GT, 2)
            label(hz, "GT", (gx2 + 6, gy2 + 16), C_GT, 0.45, 1)

    label(hz, f"{method.upper()} heatmap R={R}", (12, 24), C_TXT, 0.55, 1)
    return hz


# =========================================================
# Full trajectory overlay
# =========================================================
def draw_polyline(vis, pts, color, thick=2):
    if len(pts) < 2:
        return
    arr = np.array([(int(round(x)), int(round(y))) for x, y in pts], dtype=np.int32)
    cv2.polylines(vis, [arr], False, color, thick)


def draw_full_trajectory_overlay(img, frame_idx, frame_name, gt_pts, s1_pts, s2_pts, hybrid_pts,
                                 gt_rec, cur, stage, pred_color, handoff_idx=None):
    full = img.copy()

    draw_polyline(full, gt_pts, C_GT, 2)
    draw_polyline(full, s1_pts, C_S1, 2)
    draw_polyline(full, s2_pts, C_S2, 2)
    draw_polyline(full, hybrid_pts, C_HYB, 3)

    if gt_rec is not None:
        gx, gy, gr = gt_rec
        rr = max(4, int(round(gr)) if gr == gr else 5)
        cv2.circle(full, (int(round(gx)), int(round(gy))), rr, C_GT, 2)
        label(full, "GT", (int(gx) + 8, int(gy) + 16), C_GT, 0.55, 1)

    if cur[0] is not None and cur[1] is not None:
        cv2.circle(full, (int(round(cur[0])), int(round(cur[1]))), 6, pred_color, -1)
        label(full, "Pred", (int(cur[0]) + 8, int(cur[1]) - 8), pred_color, 0.55, 1)

    if handoff_idx is not None and frame_idx == handoff_idx:
        label(full, "HANDOFF", (20, 75), (0, 128, 255), 1.0, 2)

    label(full, f"{frame_name} | {stage}", (20, 35), C_TXT, 0.75, 2)

    y0 = full.shape[0] - 85
    label(full, "GT", (20, y0), C_GT, 0.6, 2)
    label(full, "Stage1", (90, y0), C_S1, 0.6, 2)
    label(full, "Stage2", (210, y0), C_S2, 0.6, 2)
    label(full, "Hybrid", (330, y0), C_HYB, 0.6, 2)
    label(full, "Full trajectory overlay", (20, full.shape[0] - 25), C_TXT, 0.62, 2)
    return full


# =========================================================
# Handoff
# =========================================================
def clip01(v):
    return max(0.0, min(1.0, float(v)))


def compactness_transition_score(r):
    """
    Compute Compactness Transition Score (CTS) for Stage1 -> Stage2 handoff.

    CTS = 0.35*S_area + 0.35*S_len + 0.20*S_shape + 0.10*Q_motion

    Interpretation:
    - S_area increases when the motion component area becomes smaller.
    - S_len increases when the line-band length t_len becomes shorter.
    - S_shape increases when the component bbox becomes compact rather than elongated.
    - Q_motion is the Stage1 motion confidence; it is only a weak safety term.

    This score does not prove the point is correct. It only indicates that the image
    representation has changed from high-speed elongated blur to a compact target,
    which is more suitable for Stage2 local response tracking.
    """
    area = sf(r.get("area"))
    tl = sf(r.get("t_len"))
    q = sf(r.get("motion_conf"))
    w = sf(r.get("bbox_w"))
    h = sf(r.get("bbox_h"))

    if area is None or tl is None:
        return {
            "CTS": 0.0,
            "S_area": 0.0,
            "S_len": 0.0,
            "S_shape": 0.0,
            "Q_motion": 0.0,
            "AR": "",
        }

    # Smaller component area indicates compact target.
    s_area = clip01(1.0 - area / CTS_AREA_REF)

    # Shorter t_len indicates weaker motion blur.
    s_len = clip01(1.0 - tl / CTS_LEN_REF)

    # Shape compactness from bbox aspect ratio if bbox is available.
    # Some Stage1 CSV versions may not include bbox_w/h in trajectory rows;
    # in that case we use neutral score 0.5 instead of blocking the handoff.
    if w is not None and h is not None and w > 0 and h > 0:
        ar = max(w, h) / max(1.0, min(w, h))
        s_shape = clip01(1.0 - abs(ar - 1.0) / max(1e-6, CTS_AR_MAX - 1.0))
    else:
        ar = ""
        s_shape = 0.5

    q_motion = clip01(q if q is not None else 0.5)

    cts = (
        CTS_W_AREA * s_area
        + CTS_W_LEN * s_len
        + CTS_W_SHAPE * s_shape
        + CTS_W_Q * q_motion
    )

    return {
        "CTS": float(cts),
        "S_area": float(s_area),
        "S_len": float(s_len),
        "S_shape": float(s_shape),
        "Q_motion": float(q_motion),
        "AR": ar,
    }


def is_transition_ready_result(r):
    """Stage1 handoff decision based on compactness transition score."""
    mode = r.get("mode", "")
    if mode == "MISS":
        vals = compactness_transition_score(r)
        return False, "MISS", vals

    vals = compactness_transition_score(r)
    cts = vals["CTS"]

    # CTS is the main transition criterion.
    # A very low Stage1 confidence prevents handoff from a clearly unstable point.
    q_ok = vals["Q_motion"] >= 0.30
    ready = (cts >= CTS_TAU) and q_ok

    if ready:
        reason = f"CTS_READY(CTS={cts:.2f})"
    else:
        reason = f"CTS_NOT_READY(CTS={cts:.2f})"
        if not q_ok:
            reason += "+LOW_MOTION_CONF"
    return ready, reason, vals


def is_low_quality_result(r):
    """
    Backward-compatible fallback rule.
    Kept only for comparison with the previous experience-based threshold method.
    """
    mode = r.get("mode", "")
    if mode == "MISS":
        return True, "MISS"

    step = sf(r.get("step"))
    area = sf(r.get("area"))
    tl = sf(r.get("t_len"))
    q = sf(r.get("motion_conf"))

    reasons = []
    if q is not None and q < MHI_CONF_TAU:
        reasons.append("LOW_MHI_CONF")
    if step is not None and step < STEP_TAU:
        reasons.append("LOW_STEP")
    if area is not None and area <= AREA_TAU:
        reasons.append("LOW_AREA")
    if tl is not None and tl <= TLEN_TAU:
        reasons.append("SHORT_TLEN")

    if reasons:
        return True, "+".join(reasons)
    return False, "OK"

def find_last_valid(results, before_idx=None):
    if before_idx is None:
        before_idx = len(results)
    for i in range(before_idx - 1, -1, -1):
        r = results[i]
        if r.get("mode") in ["TRACK_INIT", "TRACK"]:
            x, y = xy_from_stage1(r)
            if x is not None and y is not None:
                return i, r
    return None, None


def estimate_velocity(results, idx, max_back=6):
    if idx is None:
        return 0.0, 0.0
    vals = []
    for j in range(idx, max(-1, idx - max_back), -1):
        r = results[j]
        if r.get("mode") in ["TRACK_INIT", "TRACK"]:
            x, y = xy_from_stage1(r)
            if x is not None and y is not None:
                vals.append((j, x, y))
    if len(vals) < 2:
        return 0.0, 0.0
    j2, x2, y2 = vals[0]
    j1, x1, y1 = vals[1]
    dt = max(1, j2 - j1)
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    sp = math.hypot(vx, vy)
    if sp > 80:
        vx *= 80 / sp
        vy *= 80 / sp
    return vx, vy



def handoff_scale_info(hr):
    """Collect Stage1 scale fields for Stage2 multi-scale weighting."""
    return {
        "bbox_w": sf(hr.get("bbox_w")),
        "bbox_h": sf(hr.get("bbox_h")),
        "est_radius": sf(hr.get("est_radius")),
        "blur_ratio": sf(hr.get("blur_ratio")),
        "scale_class": hr.get("scale_class", ""),
        "motion_conf": sf(hr.get("motion_conf")),
    }

def choose_handoff(frame_names, results):
    q = []
    valid = 0
    fallback_candidate = None

    for i, r in enumerate(results):
        if r.get("mode") in ["TRACK_INIT", "TRACK"]:
            valid += 1

        # New interpretable compactness transition score
        ready, reason, cts_vals = is_transition_ready_result(r)

        # Old rule is kept only as fallback diagnostics
        low, old_reason = is_low_quality_result(r)

        q.append({
            "frame_idx": i,
            "frame_name": frame_names[i],
            "mode": r.get("mode", ""),
            "transition_ready": int(ready),
            "transition_reason": reason,
            "CTS": cts_vals["CTS"],
            "S_area": cts_vals["S_area"],
            "S_len": cts_vals["S_len"],
            "S_shape": cts_vals["S_shape"],
            "Q_motion_norm": cts_vals["Q_motion"],
            "AR": cts_vals["AR"],
            "low_quality_fallback": int(low),
            "fallback_reason": old_reason,
            "step": r.get("step", np.nan),
            "area": r.get("area", np.nan),
            "t_len": r.get("t_len", np.nan),
            "motion_conf": r.get("motion_conf", np.nan),
            "est_scale": r.get("est_scale", np.nan),
            "bbox_w": r.get("bbox_w", np.nan),
            "bbox_h": r.get("bbox_h", np.nan),
            "n": r.get("n", np.nan),
        })

        if i >= HANDOFF_WINDOW:
            recent = q[max(0, i - HANDOFF_WINDOW + 1):i + 1]
            recent_ready = sum(v["transition_ready"] for v in recent)
            recent_old_bad = sum(v["low_quality_fallback"] for v in recent)

            # Primary rule: Motion Shape Transition Index.
            if USE_COMPACTNESS_HANDOFF and valid >= MIN_VALID_BEFORE_HANDOFF and recent_ready >= CTS_RECENT_COUNT:
                hi, hr = find_last_valid(results, i + 1)
                if hr is None:
                    continue
                vx, vy = estimate_velocity(results, hi)
                return {
                    "handoff_idx": hi,
                    "handoff_frame": frame_names[hi],
                    "x": float(hr["x"]),
                    "y": float(hr["y"]),
                    "vx": vx,
                    "vy": vy,
                    "trigger_idx": i,
                    "trigger_frame": frame_names[i],
                    "trigger_reason": f"compactness_ready={recent_ready}/{HANDOFF_WINDOW};CTS_tau={CTS_TAU}",
                    **handoff_scale_info(hr),
                    "quality_rows": q,
                }

            # Conservative fallback: preserve previous behavior, but do not use it
            # before giving K_t a chance. This prevents videos from completely failing.
            if USE_OLD_HANDOFF_FALLBACK and fallback_candidate is None and valid >= MIN_VALID_BEFORE_HANDOFF and recent_old_bad >= HANDOFF_BAD_COUNT:
                hi, hr = find_last_valid(results, i + 1)
                if hr is not None:
                    vx, vy = estimate_velocity(results, hi)
                    fallback_candidate = {
                        "handoff_idx": hi,
                        "handoff_frame": frame_names[hi],
                        "x": float(hr["x"]),
                        "y": float(hr["y"]),
                        "vx": vx,
                        "vy": vy,
                        "trigger_idx": i,
                        "trigger_frame": frame_names[i],
                        "trigger_reason": f"fallback_old_recent_bad={recent_old_bad}/{HANDOFF_WINDOW}",
                        **handoff_scale_info(hr),
                        "quality_rows": q,
                    }

    if fallback_candidate is not None:
        return fallback_candidate

    hi, hr = find_last_valid(results)
    if hr is None:
        raise RuntimeError("No valid Stage1 result for handoff.")
    vx, vy = estimate_velocity(results, hi)
    return {
        "handoff_idx": hi,
        "handoff_frame": frame_names[hi],
        "x": float(hr["x"]),
        "y": float(hr["y"]),
        "vx": vx,
        "vy": vy,
        "trigger_idx": hi,
        "trigger_frame": frame_names[hi],
        "trigger_reason": "fallback_last_valid",
        **handoff_scale_info(hr),
        "quality_rows": q,
    }


# =========================================================
# Stage1 / Stage2 runners
# =========================================================
def run_stage1():
    s1.OUT_DIR = S1_OUT

    names, bgr, gray = s1.load_frames(s1.FRAMES_DIR)
    gt = s1.read_gt_csv(s1.GT_CSV)
    p0_frame, p0, p0r = s1.get_first_gt(gt, names)

    print(f"[S1] P0 frame={p0_frame}, P0=({p0[0]:.1f},{p0[1]:.1f}), r0={p0r}")

    # Newer adaptive Stage1 API (kept for compatibility)
    if hasattr(s1, "make_adaptive_params"):
        params = s1.make_adaptive_params(p0r)
        launch_frame, launch_energy, launch_ths = s1.detect_launch_by_local_energy(gray, p0_frame, p0, params)
        start, end = s1.make_direction_window(launch_frame, len(gray))

        print(f"[S1] Adaptive launch frame = {launch_frame}")
        print(f"[S1] Direction window = {start} ~ {end}")

        ret = s1.estimate_direction_multihypothesis(gray, p0_frame, p0, start, end, params)
        d = ret[0]
        hyps = ret[2] if len(ret) >= 3 else []

        results, components = s1.track_along_direction(
            gray, names, p0_frame, p0, d, launch_frame,
            min(len(gray) - 1, launch_frame + 45), params
        )

        if hasattr(s1, "save_params_csv"):
            s1.save_params_csv(params, os.path.join(S1_OUT, "adaptive_params.csv"))
        if hasattr(s1, "save_launch_csv"):
            s1.save_launch_csv(names, launch_energy, launch_ths, launch_frame, os.path.join(S1_OUT, "launch_energy.csv"))
        if hasattr(s1, "save_direction_hypotheses_csv"):
            s1.save_direction_hypotheses_csv(hyps, os.path.join(S1_OUT, "direction_hypotheses.csv"))

    # Current Stage1 API / compatibility path
    else:
        if not hasattr(s1, "find_effective_window"):
            raise RuntimeError("stage1.py does not provide a supported Stage1 API.")
        start, end, motion, smooth, peak = s1.find_effective_window(gray, p0_frame)
        d, dir_blobs, hyps = s1.estimate_direction_multihypothesis(gray, p0_frame, p0, start, end)
        results, components, launch_frame, energy = s1.track_along_direction(gray, names, p0_frame, p0, d, start, end)

    s1.save_results_csv(names, results, gt, os.path.join(S1_OUT, "stage1_trajectory_results.csv"))
    if SAVE_STAGE1_COMPONENTS:
        s1.save_components_csv(names, components, os.path.join(S1_OUT, "lineband_components.csv"))

    return names, bgr, gray, gt, results, components


def run_stage2(handoff, names, bgr, gray):
    s2.OUT_DIR = S2_OUT
    s2.HANDOFF_FRAME = handoff["handoff_frame"]
    s2.HANDOFF_X = handoff["x"]
    s2.HANDOFF_Y = handoff["y"]
    s2.HANDOFF_VX = handoff["vx"]
    s2.HANDOFF_VY = handoff["vy"]
    s2.HANDOFF_BBOX_W = handoff.get("bbox_w", np.nan)
    s2.HANDOFF_BBOX_H = handoff.get("bbox_h", np.nan)
    s2.HANDOFF_EST_RADIUS = handoff.get("est_radius", np.nan)
    s2.HANDOFF_BLUR_RATIO = handoff.get("blur_ratio", np.nan)
    s2.HANDOFF_SCALE_CLASS = handoff.get("scale_class", "")
    s2.HANDOFF_MOTION_CONF = handoff.get("motion_conf", np.nan)

    ann = s2.read_annotations(s2.ANNOT_CSV)
    summaries = []
    methods = getattr(s2, "METHODS", STAGE2_METHODS)
    for m in methods:
        summaries.append(s2.run_method(m, names, bgr, gray, ann))
    write_csv(os.path.join(S2_OUT, "compare_summary.csv"), summaries)


def load_stage2_by_frame(method):
    rows = read_csv(os.path.join(S2_OUT, method, f"{method}_tracking.csv"))
    return {r["frame"]: r for r in rows if "frame" in r}


def run_stage3_for_methods(methods=None):
    methods = methods or getattr(s2, "METHODS", STAGE2_METHODS)
    outputs = []
    for m in methods:
        tracking_csv = os.path.join(S2_OUT, m, f"{m}_tracking.csv")
        if not os.path.exists(tracking_csv):
            continue
        out_csv = os.path.join(S3_OUT, f"{m}_stage3_recovery.csv")
        s3.run_stage3(tracking_csv, out_csv)
        outputs.append({"method": m, "stage3_csv": out_csv})
    write_csv(os.path.join(S3_OUT, "stage3_outputs.csv"), outputs)
    return outputs


def load_stage3_by_frame(method):
    rows = read_csv(os.path.join(S3_OUT, f"{method}_stage3_recovery.csv"))
    return {r["frame"]: r for r in rows if "frame" in r}




def build_final_hybrid_csv(names, gt, s1res, handoff, method=None):
    """
    Build the thesis-ready final trajectory CSV:
      - frames before/equal handoff: Stage1 output
      - frames after handoff: Stage2+Stage3 recovery output
    This is the main output used for final evaluation.
    """
    method = method or MAIN_MAIN_VIS_METHOD
    s3_by = load_stage3_by_frame(method)
    rows = []
    handoff_idx = handoff["handoff_idx"]

    for i, fname in enumerate(names):
        gt_rec = gt_item(gt, fname, i)
        gt_x = gt_y = gt_r = None
        if gt_rec is not None:
            gt_x, gt_y, gt_r = gt_rec

        if i <= handoff_idx:
            r1 = s1res[i]
            x, y = xy_from_stage1(r1)
            source = "Stage1"
            state = r1.get("mode", "")
            confidence = r1.get("motion_conf", "")
        else:
            r3 = s3_by.get(fname)
            x, y = xy_from_stage3(r3)
            source = "" if r3 is None else r3.get("source", "Stage2+3")
            state = "" if r3 is None else r3.get("stage2_mode", "")
            confidence = "" if r3 is None else r3.get("stage3_conf", "")

        err = ""
        if x is not None and y is not None and gt_x is not None and gt_y is not None:
            err = math.hypot(float(x) - float(gt_x), float(y) - float(gt_y))

        rows.append({
            "frame_idx": i,
            "frame": fname,
            "final_x": "" if x is None else f"{float(x):.3f}",
            "final_y": "" if y is None else f"{float(y):.3f}",
            "gt_x": "" if gt_x is None else f"{float(gt_x):.3f}",
            "gt_y": "" if gt_y is None else f"{float(gt_y):.3f}",
            "gt_r": "" if gt_r is None else f"{float(gt_r):.3f}",
            "error_px": "" if err == "" else f"{err:.3f}",
            "source": source,
            "state": state,
            "confidence": confidence,
            "handoff_frame": handoff.get("handoff_frame", ""),
            "handoff_reason": handoff.get("trigger_reason", ""),
        })

    out_path = os.path.join(OUT_DIR, f"final_hybrid_trajectory_{method}.csv")
    write_csv(out_path, rows)
    return out_path

# =========================================================
# Visualization
# =========================================================
def generate_visuals(names, bgr, gray, gt, s1res, components, handoff):
    s1_by = {names[i]: s1res[i] for i in range(len(names))}
    s2_by_method = {m: load_stage2_by_frame(m) for m in STAGE2_METHODS}
    s3_by_method = {m: load_stage3_by_frame(m) for m in STAGE2_METHODS}
    main_s2 = s2_by_method.get(MAIN_MAIN_VIS_METHOD, {})
    main_s3 = s3_by_method.get(MAIN_MAIN_VIS_METHOD, {})

    handoff_idx = handoff["handoff_idx"]

    gt_pts = []
    s1_pts = []
    s2_main_pts = []
    hybrid_pts = []

    panel_paths = []
    full_paths = []
    s1_paths = []
    s2_paths_by_method = {m: [] for m in STAGE2_METHODS}

    for i, f in enumerate(names):
        im = bgr[i]
        g = gt_item(gt, f, i)

        if g is not None:
            gt_pts.append((g[0], g[1]))

        r1 = s1_by.get(f)
        x1, y1 = xy_from_stage1(r1)

        r2_main = main_s2.get(f)
        if r2_main is not None:
            r2_main["_method"] = MAIN_MAIN_VIS_METHOD
        x2m, y2m = xy_from_stage2(r2_main)

        r3_main = main_s3.get(f)
        x3m, y3m = xy_from_stage3(r3_main)

        # 分段軌跡：Stage1 只到 handoff；Stage2 從 handoff 後開始。
        if x1 is not None and y1 is not None and i <= handoff_idx:
            s1_pts.append((x1, y1))
        if x2m is not None and y2m is not None and i >= handoff_idx:
            s2_main_pts.append((x2m, y2m))

        # Hybrid 最終軌跡：handoff 前 Stage1，handoff 後 Stage2。
        if i <= handoff_idx and x1 is not None and y1 is not None:
            cur = (x1, y1)
            stage = "Stage1 MHI / direction diff"
            pred_color = C_S1
        elif i >= handoff_idx and x3m is not None and y3m is not None:
            cur = (x3m, y3m)
            src = r3_main.get("source", "Stage3") if r3_main is not None else "Stage3"
            stage = f"Stage2+3 {MAIN_MAIN_VIS_METHOD} | {src}"
            pred_color = C_HYB
        elif i >= handoff_idx and x2m is not None and y2m is not None:
            cur = (x2m, y2m)
            stage = f"Stage2 {MAIN_MAIN_VIS_METHOD}"
            pred_color = C_S2
        else:
            cur = (None, None)
            stage = "MISS"
            pred_color = C_HYB

        if cur[0] is not None and cur[1] is not None:
            hybrid_pts.append(cur)

        full = draw_full_trajectory_overlay(
            im, i, f, gt_pts, s1_pts, s2_main_pts, hybrid_pts,
            g, cur, stage, pred_color, handoff_idx=handoff_idx
        )

        fp = os.path.join(FULL_DIR, f"{i:04d}_{f}")
        if SAVE_FULL_FRAMES or SAVE_VIDEOS:
            cv2.imwrite(fp, full)
            full_paths.append(fp)

        if i <= handoff_idx:
            p1 = full
            p2 = make_diff_panel(gray, i)
            p3 = make_candidate_panel(im, components[i] if i < len(components) else [], r1, g, f)
            p4 = make_roi_panel(im, cur, g, "ROI zoom | Stage1", C_S1)
            panel = compose4(p1, p2, p3, p4)

            pp = os.path.join(PANEL_DIR, f"{i:04d}_{f}_4panel.png")
            sp = os.path.join(PANEL_DIR, f"{i:04d}_{f}_stage1.png")
            if SAVE_PANEL_FRAMES or SAVE_VIDEOS:
                cv2.imwrite(pp, panel)
                cv2.imwrite(sp, panel)
                panel_paths.append(pp)
                s1_paths.append(sp)

        else:
            for m in STAGE2_METHODS:
                r2 = s2_by_method.get(m, {}).get(f)
                if r2 is not None:
                    r2["_method"] = m
                x2, y2 = xy_from_stage2(r2)
                pred = (x2, y2)

                # 左上圖仍是完整軌跡圖；目前點使用該 method 的 pred。
                if x2 is not None and y2 is not None:
                    method_cur = (x2, y2)
                    method_stage = f"Stage2 {m}"
                    method_color = C_S2
                else:
                    method_cur = cur
                    method_stage = f"Stage2 {m} MISS"
                    method_color = C_HYB

                p1 = draw_full_trajectory_overlay(
                    im, i, f, gt_pts, s1_pts, s2_main_pts, hybrid_pts,
                    g, method_cur, method_stage, method_color, handoff_idx=handoff_idx
                )
                p2 = make_stage2_heatmap_panel(gray, i, im, r2, g, m)
                p3 = make_roi_panel(im, pred, g, f"ROI zoom | {m.upper()}", C_S2)
                p4 = make_status_panel(f, i, f"Stage2 {m}", r1, r2, handoff, g)

                panel = compose4(p1, p2, p3, p4)
                sp = os.path.join(PANEL_DIR, f"{i:04d}_{f}_stage2_{m}.png")
                if SAVE_PANEL_FRAMES or SAVE_VIDEOS:
                    cv2.imwrite(sp, panel)
                    s2_paths_by_method[m].append(sp)

                if m == MAIN_MAIN_VIS_METHOD:
                    pp = os.path.join(PANEL_DIR, f"{i:04d}_{f}_4panel.png")
                    if SAVE_PANEL_FRAMES or SAVE_VIDEOS:
                        cv2.imwrite(pp, panel)
                        panel_paths.append(pp)

    if SAVE_VIDEOS:
        save_video(panel_paths, os.path.join(VIS_OUT, "integrated_4panel.mp4"), FPS)
        save_video(s1_paths, os.path.join(VIS_OUT, "stage1_4panel.mp4"), FPS)
        for m, paths in s2_paths_by_method.items():
            save_video(paths, os.path.join(VIS_OUT, f"stage2_4panel_{m}.mp4"), FPS)
        save_video(full_paths, os.path.join(VIS_OUT, "full_trajectory.mp4"), FPS)

    # If only videos are needed, remove temporary frame images after the mp4 files are written.
    if SAVE_VIDEOS and not SAVE_FULL_FRAMES:
        for fp in full_paths:
            try:
                os.remove(fp)
            except OSError:
                pass
    if SAVE_VIDEOS and not SAVE_PANEL_FRAMES:
        for paths in [panel_paths, s1_paths, *s2_paths_by_method.values()]:
            for fp in paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass


# =========================================================
# Main
# =========================================================
def main():
    names, bgr, gray, gt, s1res, components = run_stage1()
    handoff = choose_handoff(names, s1res)

    write_csv(os.path.join(OUT_DIR, "handoff_summary.csv"), [{k: v for k, v in handoff.items() if k != "quality_rows"}])
    write_csv(os.path.join(OUT_DIR, "stage1_quality_rows.csv"), handoff["quality_rows"])

    print("[HANDOFF]", handoff["handoff_frame"], handoff["x"], handoff["y"], handoff["trigger_reason"])

    run_stage2(handoff, names, bgr, gray)
    run_stage3_for_methods()
    final_csv = build_final_hybrid_csv(names, gt, s1res, handoff, MAIN_MAIN_VIS_METHOD)
    print("Final hybrid CSV:", final_csv)
    if RUN_VISUALIZATION:
        generate_visuals(names, bgr, gray, gt, s1res, components, handoff)

    print("Done. Output:", OUT_DIR)
    if RUN_VISUALIZATION:
        print("4-panel video:", os.path.join(VIS_OUT, "integrated_4panel.mp4"))
        print("Full trajectory video:", os.path.join(VIS_OUT, "full_trajectory.mp4"))


if __name__ == "__main__":
    main()
