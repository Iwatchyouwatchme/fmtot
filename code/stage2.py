# stage2.py
# Stage 2: normalized multi-cue candidate scoring for tiny golf ball localization.
import os
import csv
import math
import cv2
import numpy as np

FRAMES_DIR = "frames"
ANNOT_CSV = "annotations.csv"
OUT_DIR = "stage2"

# Fallback handoff values; integrated runner overwrites these from Stage1.
HANDOFF_FRAME = None
HANDOFF_X = None
HANDOFF_Y = None
HANDOFF_VX = None
HANDOFF_VY = None
HANDOFF_SCALE_CLASS = "unknown"  # tiny_3x3 / mid_5x5 / blur_7x7 / unknown
HANDOFF_BBOX_W = np.nan
HANDOFF_BBOX_H = np.nan
HANDOFF_EST_RADIUS = np.nan
HANDOFF_BLUR_RATIO = np.nan
HANDOFF_MOTION_CONF = np.nan

METHODS = ["lbp", "ltp", "weighted"]  # internal key "weighted" corresponds to the CSWC variant in the thesis

# Output controls
SAVE_DETAILED_CSV = True
SAVE_EVIDENCE_CSV = True
SAVE_SCALE_CSV = True
SAVE_SCOREMAP_FRAMES = set()
SAVE_CONTEXT_FRAMES = False
SCOREMAP_ZOOM = 10

SEARCH_RADIUS_BASE = 15
SEARCH_RADIUS_MIN = 15
SEARCH_RADIUS_MAX = 40
SEARCH_RADIUS_SPEED_K = 2.0
MISS_EXPAND_K = 2

# Temporal appearance weights: same location response at t-1, t, t+1.
WT_PREV = 0.6
WT_CURR = 1.0
WT_NEXT = 0.6
LTP_THRESH = 5.0

# Normalized cue fusion weights. All cues are 0~1.
# These weights are fixed and thesis-friendly: appearance is the main evidence,
# motion suppresses static bright background, direction/position maintain continuity.
W_APPEARANCE = 0.40
W_MOTION = 0.25
W_DIRECTION = 0.20
W_POSITION = 0.15

# Multi-scale appearance weights.
# Thesis design:
#   3x3 / 5x5 / 7x7 are not interpreted as the exact ball size.
#   They are observation windows.  A visible 3~4 px bright blob often benefits
#   from a 5x5 or 7x7 window because surrounding background contrast is useful.
#   Therefore the scale prior is conservative and never lets one scale dominate
#   completely.
SCALE_WEIGHT_TINY = (0.50, 0.30, 0.20)          # 1x1, 1x2, 2x1, 2x2: 3x3 observation is main
SCALE_WEIGHT_MID = (0.20, 0.50, 0.30)           # about 3x3: 5x5 observation is main
SCALE_WEIGHT_BLUR = (0.20, 0.30, 0.50)          # >=4 px or mild blur: 7x7 observation is main
SCALE_RESPONSE_BLEND = 0.10                     # weak per-frame response refinement only
APPEARANCE_SHAPE_BLEND = 0.10                   # local bright component is auxiliary, not decisive

# Local bright-component analysis around each candidate.
# It is used only for scale estimation and center consistency.  It does not
# replace multi-cue scoring, so the method is not simply "find the brightest blob".
BRIGHT_PATCH_SIZE = 9
BRIGHT_CENTER_MAX_DIST = 3.0
BRIGHT_MAX_REASONABLE_AREA = 36
BRIGHT_MAX_REASONABLE_ASPECT = 2.8
BRIGHT_TOPK_REFINE = 80

# Stage decision thresholds for normalized score.
TRACK_SCORE_TAU = {"lbp": 0.55, "ltp": 0.55, "weighted": 0.55}
HOLD_SCORE_TAU = {"lbp": 0.40, "ltp": 0.40, "weighted": 0.40}
D_TRACK = 8.0
D_HOLD = 14.0
MIN_MOTION_FOR_TRACK = 0.05
USE_MOTION_GATE_FOR_TRACK = True

# Online reliability / offline accuracy.
RELIABLE_MAX_PRED_DIST = 18.0
RELIABLE_MAX_CAND_ANGLE = 95.0
ACC_OK_PX = 5.0
ACC_GOOD_PX = 2.0

# Motion model gates.
MAX_SPEED_BASE = 10.0
MAX_SPEED_MARGIN = 4.0
MAX_ACCEL_BASE = 8.0
MAX_ACCEL_MARGIN = 5.0
LOW_SPEED_DIR_DISABLE = 1.5
MED_SPEED_SOFT_DIR = 4.0
SOFT_MAX_TURN_DEG = 95.0
MAX_TURN_DEG = 65.0
VERTICAL_STRONG_DOWN_TAU = 2.0
VERTICAL_STRONG_UP_TAU = -2.0
VERTICAL_REV_TOL = 2.0
VY_APEX_TAU = 0.8
VY_DOWN_ENTER = 0.8
VY_DOWN_KEEP = -0.35
APEX_MAX_FRAMES = 15
MIN_DOWN_VY = 0.8
G_DOWN = 0.15
VEL_ALPHA = 0.7
VX_HOLD_DECAY = 0.92
VY_UP_HOLD_DECAY = 0.95
VX_MISS_DECAY = 0.94
VY_UP_MISS_DECAY = 0.97
LAST_GOOD_BLEND = 0.20
DOWN_VX_SMOOTH_OLD = 0.9
DOWN_VX_SMOOTH_NEW = 0.1
VX_SIGN_TAU = 0.6
VX_SIGN_MARGIN_STRONG = 10.0
VX_SIGN_MARGIN_STRONG_WEIGHTED = 0.10

FRAME_DIFF_PATCH = 5
FRAME_DIFF_NORM = 255.0 * FRAME_DIFF_PATCH * FRAME_DIFF_PATCH
FD_CLIP = 0.35
POS_SIGMA = D_TRACK / 1.5
DIR_SIGMA = 45.0
XMOTION_SIGMA = 6.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_frames(folder):
    fs = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    fs.sort()
    return fs


def to_gray(img):
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def safe_float(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


def read_annotations(csv_path):
    ann = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            raise ValueError("annotations.csv must contain frame column")
        for row in reader:
            fn = row["frame"].strip()
            x = safe_float(row.get("x", row.get("gt_x", "")))
            y = safe_float(row.get("y", row.get("gt_y", "")))
            r = safe_float(row.get("r", row.get("radius", "")))
            vis_raw = str(row.get("visible", "")).strip()
            visible = (x is not None and y is not None) if vis_raw == "" else (vis_raw == "1")
            ann[fn] = {"x": x, "y": y, "r": r, "visible": visible}
    return ann


def get_gt(ann, frame_name):
    rec = ann.get(frame_name)
    if rec is None:
        return None, None, None, False
    if not rec["visible"] or rec["x"] is None or rec["y"] is None:
        return None, None, rec["r"], False
    return rec["x"], rec["y"], rec["r"], True


def compute_error(px, py, gx, gy):
    return None if px is None or gx is None else math.hypot(px - gx, py - gy)


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def norm01(vals):
    vals = np.asarray(vals, dtype=np.float32)
    out = np.zeros_like(vals, dtype=np.float32)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out
    mn = float(vals[finite].min())
    mx = float(vals[finite].max())
    if mx - mn < 1e-8:
        out[finite] = 0.0
    else:
        out[finite] = (vals[finite] - mn) / (mx - mn)
    return out


def angle_between_deg(v1x, v1y, v2x, v2y):
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    c = (v1x * v2x + v1y * v2y) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def direction_ok(vx, vy, dx, dy):
    prev_speed = math.hypot(vx, vy)
    cand_speed = math.hypot(dx, dy)
    if prev_speed < LOW_SPEED_DIR_DISABLE or cand_speed < LOW_SPEED_DIR_DISABLE:
        return True, 0.0, "DIR_DISABLED_LOW_SPEED"
    ang = angle_between_deg(vx, vy, dx, dy)
    if prev_speed < MED_SPEED_SOFT_DIR or cand_speed < MED_SPEED_SOFT_DIR:
        return ang <= SOFT_MAX_TURN_DEG, ang, "DIR_SOFT"
    return ang <= MAX_TURN_DEG, ang, "DIR_NORMAL"


def accel_ok(vx, vy, dx, dy):
    acc = math.hypot(dx - vx, dy - vy)
    prev_speed = math.hypot(vx, vy)
    lim = MAX_ACCEL_BASE + MAX_ACCEL_MARGIN + 0.2 * prev_speed
    return acc <= lim, acc, lim


def speed_ok(dx, dy, prev_speed):
    sp = math.hypot(dx, dy)
    lim = MAX_SPEED_BASE + MAX_SPEED_MARGIN + 0.4 * prev_speed
    return sp <= lim, sp, lim


def vertical_ok(vy, dy):
    if abs(vy) <= VERTICAL_STRONG_DOWN_TAU:
        return True, "VERT_ALLOW_NEAR_APEX"
    if vy > VERTICAL_STRONG_DOWN_TAU:
        return (dy >= -VERTICAL_REV_TOL), "VERT_OK_DOWN" if dy >= -VERTICAL_REV_TOL else "VERT_BLOCK_DOWN_TO_UP"
    if vy < VERTICAL_STRONG_UP_TAU:
        return (dy <= VERTICAL_REV_TOL), "VERT_OK_UP" if dy <= VERTICAL_REV_TOL else "VERT_BLOCK_UP_TO_DOWN"
    return True, "VERT_OK_DEFAULT"


def compute_search_radius(vx, vy, miss_count):
    speed = math.hypot(vx, vy)
    R = SEARCH_RADIUS_BASE + SEARCH_RADIUS_SPEED_K * speed + MISS_EXPAND_K * miss_count
    return int(round(max(SEARCH_RADIUS_MIN, min(SEARCH_RADIUS_MAX, R))))


def infer_motion_phase_hysteresis(prev_phase, vy):
    if prev_phase == "UP":
        if vy < -VY_APEX_TAU:
            return "UP"
        if vy > VY_DOWN_ENTER:
            return "DOWN"
        return "TRANSITION"
    if prev_phase == "TRANSITION":
        if vy > VY_DOWN_ENTER:
            return "DOWN"
        if vy < -VY_APEX_TAU:
            return "UP"
        return "TRANSITION"
    return "DOWN" if vy >= VY_DOWN_KEEP else "DOWN"


def stabilize_vx(prev_vx, new_vx, best_score, second_score, method_name):
    margin = best_score - second_score if second_score > -1e17 else 1e9
    strong_margin = VX_SIGN_MARGIN_STRONG_WEIGHTED if method_name == "weighted" else VX_SIGN_MARGIN_STRONG
    if abs(prev_vx) > VX_SIGN_TAU and (prev_vx * new_vx < 0):
        if margin < strong_margin and abs(new_vx) < abs(prev_vx) * 1.2:
            return 0.5 * prev_vx
    return new_vx


def build_weighted_kernels():
    # Center-surround filters. Center/inner positive weights emphasize bright compact target;
    # negative outer ring suppresses uniform bright background. k3/k5/k7 all sum to zero.
    k3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
    k5 = np.array([[-1,-1,-1,-1,-1],[-1,0,0,0,-1],[-1,0,16,0,-1],[-1,0,0,0,-1],[-1,-1,-1,-1,-1]], dtype=np.float32)
    k7 = np.array([[-1,-1,-1,-1,-1,-1,-1],[-1,0,0,0,0,0,-1],[-1,0,1,1,1,0,-1],[-1,0,1,16,1,0,-1],[-1,0,1,1,1,0,-1],[-1,0,0,0,0,0,-1],[-1,-1,-1,-1,-1,-1,-1]], dtype=np.float32)
    return {3: k3, 5: k5, 7: k7}


WEIGHTED_KERNELS = build_weighted_kernels()
KABS = {k: float(np.sum(np.abs(v))) for k, v in WEIGHTED_KERNELS.items()}


def get_patch(gray, x, y, size):
    r = size // 2
    h, w = gray.shape
    xi = int(round(x)); yi = int(round(y))
    if xi - r < 0 or xi + r >= w or yi - r < 0 or yi + r >= h:
        return None
    return gray[yi-r:yi+r+1, xi-r:xi+r+1].astype(np.float32)


def neighbors(size):
    c = size // 2
    return [(yy, xx) for yy in range(size) for xx in range(size) if not (yy == c and xx == c)]


def value_lbp(gray, x, y, size):
    p = get_patch(gray, x, y, size)
    if p is None:
        return 0.0
    c = size // 2
    return float(sum(1 for yy, xx in neighbors(size) if p[c, c] > p[yy, xx]))


def value_ltp(gray, x, y, size):
    p = get_patch(gray, x, y, size)
    if p is None:
        return 0.0
    c = size // 2
    score = 0.0
    for yy, xx in neighbors(size):
        d = p[c, c] - p[yy, xx]
        if d > LTP_THRESH:
            score += 1.0
        elif d < -LTP_THRESH:
            score -= 1.0
    return score


def value_weighted(gray, x, y, size):
    p = get_patch(gray, x, y, size)
    if p is None:
        return 0.0
    return max(0.0, float(np.sum(p * WEIGHTED_KERNELS[size]) / max(1.0, KABS[size])))


def compute_value(gray, x, y, size, method):
    if method == "lbp":
        return value_lbp(gray, x, y, size)
    if method == "ltp":
        return value_ltp(gray, x, y, size)
    if method == "weighted":
        return value_weighted(gray, x, y, size)
    raise ValueError(method)


def temporal_scale_response(gray_prev, gray_cur, gray_next, x, y, size, method):
    return (WT_PREV * compute_value(gray_prev, x, y, size, method) +
            WT_CURR * compute_value(gray_cur, x, y, size, method) +
            WT_NEXT * compute_value(gray_next, x, y, size, method))


def temporal_motion_compensated_response(gray_prev, gray_cur, gray_next, x, y, vx, vy, size, method):
    """Temporal response along the expected motion path.

    The old same-location t-1/t/t+1 response can accidentally strengthen a
    static bright background point.  Here the previous and next responses are
    sampled at x-v and x+v. Around the apex, velocity is small, so the formula
    naturally becomes close to same-location checking without forcing a sharp
    physical turn.
    """
    speed = math.hypot(vx, vy)
    if speed < 1.0:
        return temporal_scale_response(gray_prev, gray_cur, gray_next, x, y, size, method)
    prev_val = compute_value(gray_prev, x - vx, y - vy, size, method)
    curr_val = compute_value(gray_cur, x, y, size, method)
    next_val = compute_value(gray_next, x + vx, y + vy, size, method)
    return 0.35 * prev_val + 1.00 * curr_val + 0.35 * next_val


def analyze_local_bright_component(gray, x, y, patch_size=BRIGHT_PATCH_SIZE):
    """Estimate local bright region near a single candidate point.

    The returned bbox is only a scale/consistency cue.  Large or elongated
    components are treated as low-trust because they may include background
    highlights.
    """
    p = get_patch(gray, x, y, patch_size)
    if p is None:
        return {
            "local_area": 0, "local_w": 0, "local_h": 0, "local_aspect": np.nan,
            "center_dist": np.nan, "shape_score": 0.5, "scale_trust": 0.0,
            "local_scale_mode": "unknown"
        }

    p8 = np.clip(p, 0, 255).astype(np.uint8)
    mean = float(np.mean(p8)); std = float(np.std(p8))
    thr = max(mean + 0.75 * std, float(np.percentile(p8, 82)))
    _, bw = cv2.threshold(p8, thr, 255, cv2.THRESH_BINARY)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    c0 = patch_size // 2
    if num <= 1:
        return {
            "local_area": 0, "local_w": 0, "local_h": 0, "local_aspect": np.nan,
            "center_dist": np.nan, "shape_score": 0.5, "scale_trust": 0.0,
            "local_scale_mode": "none"
        }

    best_lab = None
    best_cost = 1e18
    for lab in range(1, num):
        x0, y0, w, h, area = stats[lab]
        cx, cy = centroids[lab]
        d = math.hypot(cx - c0, cy - c0)
        contains_center = (labels[c0, c0] == lab)
        if d > BRIGHT_CENTER_MAX_DIST and not contains_center:
            continue
        # Prefer a component close to the candidate point, not the largest region.
        cost = d + 0.03 * area + (0.0 if contains_center else 1.0)
        if cost < best_cost:
            best_cost = cost
            best_lab = lab

    if best_lab is None:
        return {
            "local_area": 0, "local_w": 0, "local_h": 0, "local_aspect": np.nan,
            "center_dist": np.nan, "shape_score": 0.5, "scale_trust": 0.0,
            "local_scale_mode": "none"
        }

    x0, y0, w, h, area = stats[best_lab]
    cx, cy = centroids[best_lab]
    center_dist = math.hypot(cx - c0, cy - c0)
    aspect = max(w, h) / max(1.0, min(w, h))
    mx = max(w, h)

    area_score = clip01(1.0 - max(0.0, area - BRIGHT_MAX_REASONABLE_AREA) / max(1.0, BRIGHT_MAX_REASONABLE_AREA))
    aspect_score = clip01(1.0 - max(0.0, aspect - 1.0) / max(1e-6, BRIGHT_MAX_REASONABLE_ASPECT - 1.0))
    center_score = clip01(1.0 - center_dist / max(1.0, BRIGHT_CENTER_MAX_DIST))
    shape_score = clip01(0.35 * area_score + 0.35 * aspect_score + 0.30 * center_score)

    # Conservative scale classification.  A sudden 6x6 region is allowed to
    # suggest 7x7 only if its shape remains compact and centered.
    if shape_score < 0.35:
        scale_mode = "untrusted"
        scale_trust = 0.0
    else:
        scale_trust = shape_score
        if mx <= 2:
            scale_mode = "tiny_3x3"
        elif mx < 4:
            scale_mode = "mid_5x5"
        elif mx <= 6:
            scale_mode = "blur_7x7"
        else:
            scale_mode = "large_untrusted"
            scale_trust *= 0.35

    return {
        "local_area": int(area), "local_w": int(w), "local_h": int(h),
        "local_aspect": float(aspect), "center_dist": float(center_dist),
        "shape_score": float(shape_score), "scale_trust": float(scale_trust),
        "local_scale_mode": scale_mode
    }


def scale_weights_from_local_component(base_w3, base_w5, base_w7, info):
    mode = info.get("local_scale_mode", "unknown")
    trust = clip01(info.get("scale_trust", 0.0))
    if mode == "tiny_3x3":
        target = np.asarray(SCALE_WEIGHT_TINY, dtype=np.float32)
    elif mode == "mid_5x5":
        target = np.asarray(SCALE_WEIGHT_MID, dtype=np.float32)
    elif mode == "blur_7x7":
        target = np.asarray(SCALE_WEIGHT_BLUR, dtype=np.float32)
    else:
        return base_w3, base_w5, base_w7
    base = np.asarray([base_w3, base_w5, base_w7], dtype=np.float32)
    # local bright scale is auxiliary; even high trust only changes 35% of weights.
    alpha = 0.35 * trust
    out = (1.0 - alpha) * base + alpha * target
    out = out / max(1e-8, float(np.sum(out)))
    return float(out[0]), float(out[1]), float(out[2])


def compute_frame_diff_raw(gray_prev, gray_cur, gray_next, x, y, patch_size=FRAME_DIFF_PATCH):
    p1 = get_patch(cv2.absdiff(gray_cur, gray_prev), x, y, patch_size)
    p2 = get_patch(cv2.absdiff(gray_next, gray_cur), x, y, patch_size)
    if p1 is None or p2 is None:
        return 0.0
    return float((np.sum(p1) + np.sum(p2)) / max(1.0, FRAME_DIFF_NORM))


def _scale_tuple(name):
    if name == "tiny_3x3":
        return (*SCALE_WEIGHT_TINY, name)
    if name == "blur_7x7":
        return (*SCALE_WEIGHT_BLUR, name)
    return (*SCALE_WEIGHT_MID, "mid_5x5")


def scale_weights_from_handoff():
    """
    Decide the base multi-scale weights from Stage1 handoff information.

    The handoff scale is only the initial prior.  Stage2 will still refine the
    scale using a small local bright-component cue around each candidate.  The
    bbox rule is intentionally conservative:
      max(w,h) <= 2 : 3x3-main
      2 < max(w,h) < 4 : 5x5-main
      max(w,h) >= 4 or mild blur : 7x7-main
    """
    cls = str(HANDOFF_SCALE_CLASS or "").lower()
    bw = safe_float(HANDOFF_BBOX_W)
    bh = safe_float(HANDOFF_BBOX_H)
    br = safe_float(HANDOFF_BLUR_RATIO)
    er = safe_float(HANDOFF_EST_RADIUS)

    # Strong blur evidence has priority, but its weight is still only 0.50 on 7x7.
    if cls in ["blur_7x7", "blur"] or (br is not None and br >= 2.5):
        return _scale_tuple("blur_7x7")

    if bw is not None and bh is not None and bw > 0 and bh > 0:
        mx = max(bw, bh)
        mn = max(1.0, min(bw, bh))
        aspect = mx / mn
        if aspect >= 3.0 or mx >= 4.0:
            return (*SCALE_WEIGHT_BLUR, "blur_7x7_bbox")
        if mx <= 2.0 or (er is not None and er <= 1.5):
            return (*SCALE_WEIGHT_TINY, "tiny_3x3_bbox")
        return (*SCALE_WEIGHT_MID, "mid_5x5_bbox")

    if cls in ["tiny_3x3", "tiny"]:
        return _scale_tuple("tiny_3x3")
    if cls in ["mid_5x5", "mid", "compact_5x5"]:
        return _scale_tuple("mid_5x5")
    if er is not None:
        if er <= 1.5:
            return _scale_tuple("tiny_3x3")
        if er >= 2.0:
            return _scale_tuple("blur_7x7")
    return _scale_tuple("mid_5x5")

def refine_scale_weights(base_w3, base_w5, base_w7, s3n, s5n, s7n):
    """Weakly adapt scale weights using normalized per-frame responses.

    This avoids the previous problem where the largest raw response directly
    decided the scale.  The response only contributes a small refinement, while
    Stage1 bbox remains the main prior.
    """
    vals = np.asarray([max(0.0, s3n), max(0.0, s5n), max(0.0, s7n)], dtype=np.float32)
    total = float(np.sum(vals))
    if total < 1e-8:
        return base_w3, base_w5, base_w7
    prop = vals / total
    base = np.asarray([base_w3, base_w5, base_w7], dtype=np.float32)
    out = (1.0 - SCALE_RESPONSE_BLEND) * base + SCALE_RESPONSE_BLEND * prop
    out = out / max(1e-8, float(np.sum(out)))
    return float(out[0]), float(out[1]), float(out[2])


def normalized_position_score(pred_dist):
    return math.exp(-(pred_dist ** 2) / max(1e-6, 2.0 * POS_SIGMA * POS_SIGMA))


def normalized_direction_score(cand_ang):
    if cand_ang is None:
        return 1.0
    return math.exp(-(cand_ang ** 2) / max(1e-6, 2.0 * DIR_SIGMA * DIR_SIGMA))


def decide_stage2_state(best_score, best_pred_dist, motion_norm, method_name):
    if best_score is None or best_pred_dist is None:
        return "MID_MISS", "no_candidate", False, False, False, False, False
    score_ok_high = best_score >= TRACK_SCORE_TAU[method_name]
    score_ok_low = best_score >= HOLD_SCORE_TAU[method_name]
    dist_ok_track = best_pred_dist <= D_TRACK
    dist_ok_hold = best_pred_dist <= D_HOLD
    motion_ok = (motion_norm is None) or (motion_norm >= MIN_MOTION_FOR_TRACK)
    if USE_MOTION_GATE_FOR_TRACK and not motion_ok and score_ok_high:
        # Strong appearance but little temporal change: safe to hold, not reliable track.
        if score_ok_low and dist_ok_hold:
            return "MID_HOLD", "low_motion_evidence_hold", score_ok_high, score_ok_low, dist_ok_track, dist_ok_hold, motion_ok
        return "MID_MISS", "low_motion_evidence_reject", score_ok_high, score_ok_low, dist_ok_track, dist_ok_hold, motion_ok
    if score_ok_high and dist_ok_track:
        return "MID_OK", "normalized_score_and_prediction_gate", score_ok_high, score_ok_low, dist_ok_track, dist_ok_hold, motion_ok
    if score_ok_low and dist_ok_hold:
        return "MID_HOLD", "hold_candidate_plausible", score_ok_high, score_ok_low, dist_ok_track, dist_ok_hold, motion_ok
    return "MID_MISS", "candidate_rejected", score_ok_high, score_ok_low, dist_ok_track, dist_ok_hold, motion_ok


def judge_online_reliability(mode, pred_dist, cand_ang, motion_norm):
    if mode != "MID_OK":
        return 0, "mode_not_ok"
    if pred_dist is not None and pred_dist > RELIABLE_MAX_PRED_DIST:
        return 0, "far_from_prediction"
    if cand_ang is not None and cand_ang > RELIABLE_MAX_CAND_ANGLE:
        return 0, "large_turn_angle"
    if USE_MOTION_GATE_FOR_TRACK and motion_norm is not None and motion_norm < MIN_MOTION_FOR_TRACK:
        return 0, "low_motion_evidence"
    return 1, "reliable"


def judge_offline_accuracy(err):
    if err is None:
        return ""
    if err <= ACC_GOOD_PX:
        return "GOOD_2PX"
    if err <= ACC_OK_PX:
        return "OK_5PX"
    return "BAD_GT_ERROR"


def apply_motion_prediction_update(vx, vy, last_good_vx, last_good_vy, motion_phase, miss_kind):
    if motion_phase == "UP":
        vx *= VX_MISS_DECAY if miss_kind == "MISS" else VX_HOLD_DECAY
        vy *= VY_UP_MISS_DECAY if miss_kind == "MISS" else VY_UP_HOLD_DECAY
    elif motion_phase == "TRANSITION":
        vx *= VX_HOLD_DECAY
    else:
        vx *= VX_HOLD_DECAY
        vx = DOWN_VX_SMOOTH_OLD * vx + DOWN_VX_SMOOTH_NEW * last_good_vx
        vy = max(vy + G_DOWN, MIN_DOWN_VY)
    vx = (1.0 - LAST_GOOD_BLEND) * vx + LAST_GOOD_BLEND * last_good_vx
    if motion_phase == "DOWN":
        vy = max((1.0 - LAST_GOOD_BLEND) * vy + LAST_GOOD_BLEND * max(last_good_vy, MIN_DOWN_VY), MIN_DOWN_VY)
    else:
        vy = (1.0 - LAST_GOOD_BLEND) * vy + LAST_GOOD_BLEND * last_good_vy
    return vx, vy


def run_method(method_name, frame_names, frames_bgr, frames_gray, ann):
    method_dir = os.path.join(OUT_DIR, method_name)
    ensure_dir(method_dir)
    if HANDOFF_FRAME not in frame_names:
        raise ValueError(f"HANDOFF_FRAME not found: {HANDOFF_FRAME}")

    start_idx = frame_names.index(HANDOFF_FRAME)
    x, y = float(HANDOFF_X), float(HANDOFF_Y)
    vx, vy = float(HANDOFF_VX), float(HANDOFF_VY)
    last_good_vx, last_good_vy = vx, vy
    motion_phase = infer_motion_phase_hysteresis("UP", vy)
    apex_count = 0
    miss_count = 0
    w3, w5, w7, scale_mode = scale_weights_from_handoff()

    track_rows, evidence_rows, scale_rows = [], [], []

    for i in range(start_idx, len(frame_names)):
        fname = frame_names[i]
        gray_cur = cv2.GaussianBlur(frames_gray[i], (3, 3), 0)
        gray_prev = cv2.GaussianBlur(frames_gray[max(0, i-1)], (3, 3), 0)
        gray_next = cv2.GaussianBlur(frames_gray[min(len(frames_gray)-1, i+1)], (3, 3), 0)
        gt_x, gt_y, gt_r, gt_visible = get_gt(ann, fname)

        if i == start_idx:
            continue

        prev_speed = math.hypot(vx, vy)
        pred_cx, pred_cy = x + vx, y + vy
        R = compute_search_radius(vx, vy, miss_count)
        H, W = gray_cur.shape

        cand = []
        for dy_off in range(-R, R + 1):
            for dx_off in range(-R, R + 1):
                cx, cy = pred_cx + dx_off, pred_cy + dy_off
                if cx < 0 or cx >= W or cy < 0 or cy >= H:
                    continue
                dx, dy = cx - x, cy - y
                sp_ok, cand_speed, speed_lim = speed_ok(dx, dy, prev_speed)
                if not sp_ok:
                    continue
                ac_ok, cand_acc, acc_lim = accel_ok(vx, vy, dx, dy)
                if not ac_ok:
                    continue
                dir_ok_flag, cand_ang, dir_mode = direction_ok(vx, vy, dx, dy)
                if not dir_ok_flag:
                    continue
                vert_ok_flag, vert_reason = vertical_ok(vy, dy)
                if not vert_ok_flag:
                    continue

                s3 = temporal_motion_compensated_response(gray_prev, gray_cur, gray_next, cx, cy, vx, vy, 3, method_name)
                s5 = temporal_motion_compensated_response(gray_prev, gray_cur, gray_next, cx, cy, vx, vy, 5, method_name)
                s7 = temporal_motion_compensated_response(gray_prev, gray_cur, gray_next, cx, cy, vx, vy, 7, method_name)
                fd = compute_frame_diff_raw(gray_prev, gray_cur, gray_next, cx, cy)
                pred_dist = math.hypot(cx - pred_cx, cy - pred_cy)
                x_motion_dev = abs(dx - vx)
                cand.append({
                    "x": cx, "y": cy, "dx": dx, "dy": dy,
                    "s3_raw": s3, "s5_raw": s5, "s7_raw": s7, "fd_raw": fd,
                    "pred_dist": pred_dist, "x_motion_dev": x_motion_dev,
                    "cand_speed": cand_speed, "cand_acc": cand_acc, "cand_ang": cand_ang,
                    "speed_lim": speed_lim, "acc_lim": acc_lim, "dir_mode": dir_mode, "vert_reason": vert_reason,
                })

        best = None
        second_score = -1e18
        if cand:
            s3n = norm01([c["s3_raw"] for c in cand])
            s5n = norm01([c["s5_raw"] for c in cand])
            s7n = norm01([c["s7_raw"] for c in cand])
            fdn = np.asarray([clip01(c["fd_raw"] / FD_CLIP) for c in cand], dtype=np.float32)
            # First pass: compute a preliminary score without local component analysis.
            # Local bright-component analysis is applied only to top candidates for speed and robustness.
            for k, c in enumerate(cand):
                c["s3_norm"] = float(s3n[k]); c["s5_norm"] = float(s5n[k]); c["s7_norm"] = float(s7n[k])
                cw3, cw5, cw7 = refine_scale_weights(w3, w5, w7, c["s3_norm"], c["s5_norm"], c["s7_norm"])
                c["cand_w3"] = cw3; c["cand_w5"] = cw5; c["cand_w7"] = cw7
                c["shape_score"] = 0.5
                c["local_area"] = 0; c["local_w"] = 0; c["local_h"] = 0
                c["local_aspect"] = np.nan; c["local_center_dist"] = np.nan
                c["local_scale_mode"] = "not_refined"
                app_core = clip01(cw3*c["s3_norm"] + cw5*c["s5_norm"] + cw7*c["s7_norm"])
                c["appearance_core"] = app_core
                c["appearance_norm"] = app_core
                c["motion_norm"] = float(fdn[k])
                c["direction_norm"] = normalized_direction_score(c["cand_ang"])
                pscore = normalized_position_score(c["pred_dist"])
                xscore = math.exp(-(c["x_motion_dev"] ** 2) / max(1e-6, 2.0 * XMOTION_SIGMA * XMOTION_SIGMA))
                c["position_norm"] = clip01(0.70 * pscore + 0.30 * xscore)
                c["score"] = (W_APPEARANCE * c["appearance_norm"] + W_MOTION * c["motion_norm"] +
                              W_DIRECTION * c["direction_norm"] + W_POSITION * c["position_norm"])

            cand.sort(key=lambda z: z["score"], reverse=True)
            refine_n = min(BRIGHT_TOPK_REFINE, len(cand))
            for c in cand[:refine_n]:
                info = analyze_local_bright_component(gray_cur, c["x"], c["y"])
                c.update({
                    "local_area": info["local_area"],
                    "local_w": info["local_w"],
                    "local_h": info["local_h"],
                    "local_aspect": info["local_aspect"],
                    "local_center_dist": info["center_dist"],
                    "shape_score": info["shape_score"],
                    "local_scale_mode": info["local_scale_mode"],
                })
                lw3, lw5, lw7 = scale_weights_from_local_component(c["cand_w3"], c["cand_w5"], c["cand_w7"], info)
                c["cand_w3"] = lw3; c["cand_w5"] = lw5; c["cand_w7"] = lw7
                app_core = clip01(lw3*c["s3_norm"] + lw5*c["s5_norm"] + lw7*c["s7_norm"])
                c["appearance_core"] = app_core
                c["appearance_norm"] = clip01((1.0 - APPEARANCE_SHAPE_BLEND) * app_core + APPEARANCE_SHAPE_BLEND * c["shape_score"])
                c["score"] = (W_APPEARANCE * c["appearance_norm"] + W_MOTION * c["motion_norm"] +
                              W_DIRECTION * c["direction_norm"] + W_POSITION * c["position_norm"])
            cand.sort(key=lambda z: z["score"], reverse=True)
            best = cand[0]
            second_score = cand[1]["score"] if len(cand) > 1 else -1e18

        if best is None:
            mode, state_reason = "MID_MISS", "no_valid_candidate"
            score = None
            score_ok_high = score_ok_low = dist_ok_track = dist_ok_hold = motion_ok = False
            miss_count += 1
            motion_phase = infer_motion_phase_hysteresis(motion_phase, vy)
            if motion_phase == "TRANSITION":
                apex_count += 1
                if apex_count > APEX_MAX_FRAMES:
                    motion_phase = "DOWN"
                    vy = max(last_good_vy + G_DOWN, MIN_DOWN_VY)
            vx, vy = apply_motion_prediction_update(vx, vy, last_good_vx, last_good_vy, motion_phase, "MISS")
            pred_x, pred_y = x + vx, y + vy
            x, y = pred_x, pred_y
            best_pred_dist = None
            cand_ang = None
        else:
            best_pred_dist = best["pred_dist"]
            cand_ang = best["cand_ang"]
            margin = best["score"] - second_score if second_score > -1e17 else 1e9
            mode, state_reason, score_ok_high, score_ok_low, dist_ok_track, dist_ok_hold, motion_ok = decide_stage2_state(
                best["score"], best_pred_dist, best["motion_norm"], method_name)
            score = best["score"]

            if mode == "MID_OK":
                pred_x, pred_y = best["x"], best["y"]
                new_vx, new_vy = pred_x - x, pred_y - y
                new_vx = stabilize_vx(vx, new_vx, best["score"], second_score, method_name)
                if motion_phase == "DOWN":
                    vx = DOWN_VX_SMOOTH_OLD * vx + DOWN_VX_SMOOTH_NEW * new_vx
                else:
                    vx = VEL_ALPHA * vx + (1.0 - VEL_ALPHA) * new_vx
                vy = VEL_ALPHA * vy + (1.0 - VEL_ALPHA) * new_vy
                x, y = pred_x, pred_y
                miss_count = 0
                last_good_vx, last_good_vy = vx, vy
            elif mode == "MID_HOLD":
                # HOLD means the candidate is plausible but not reliable.
                # Do NOT update the internal tracked state with this candidate;
                # otherwise a static bright background point can contaminate velocity
                # and all later predictions.  Keep candidate_x/y in CSV for analysis,
                # but output a model-predicted point based on the last reliable state.
                miss_count += 1
                motion_phase = infer_motion_phase_hysteresis(motion_phase, vy)
                if motion_phase == "TRANSITION":
                    apex_count += 1
                    if apex_count > APEX_MAX_FRAMES:
                        motion_phase = "DOWN"
                        vy = max(last_good_vy + G_DOWN, MIN_DOWN_VY)
                vx, vy = apply_motion_prediction_update(vx, vy, last_good_vx, last_good_vy, motion_phase, "HOLD")
                pred_x, pred_y = x + vx, y + vy
                x, y = pred_x, pred_y
            else:
                miss_count += 1
                motion_phase = infer_motion_phase_hysteresis(motion_phase, vy)
                vx, vy = apply_motion_prediction_update(vx, vy, last_good_vx, last_good_vy, motion_phase, "MISS")
                pred_x, pred_y = x + vx, y + vy
                x, y = pred_x, pred_y

        motion_phase = infer_motion_phase_hysteresis(motion_phase, vy)
        if motion_phase != "TRANSITION":
            apex_count = 0
        err = compute_error(pred_x, pred_y, gt_x, gt_y)
        online_reliable, reliable_reason = judge_online_reliability(
            mode, best_pred_dist, cand_ang, None if best is None else best["motion_norm"])
        offline_acc = judge_offline_accuracy(err)

        def bf(key, default=""):
            return default if best is None else best.get(key, default)

        track_rows.append({
            "frame": fname,
            "gt_x": "" if gt_x is None else f"{gt_x:.3f}", "gt_y": "" if gt_y is None else f"{gt_y:.3f}", "gt_r": "" if gt_r is None else f"{gt_r:.3f}",
            "pred_x": f"{pred_x:.3f}", "pred_y": f"{pred_y:.3f}",
            "candidate_x": "" if best is None else f"{best['x']:.3f}", "candidate_y": "" if best is None else f"{best['y']:.3f}",
            "pred_center_x": f"{pred_cx:.3f}", "pred_center_y": f"{pred_cy:.3f}",
            "best_pred_dist": "" if best_pred_dist is None else f"{best_pred_dist:.3f}",
            "error_px": "" if err is None else f"{err:.3f}",
            "vx": f"{vx:.3f}", "vy": f"{vy:.3f}", "speed": f"{math.hypot(vx, vy):.3f}",
            "score": "" if score is None else f"{score:.6f}",
            "appearance_score": "" if best is None else f"{best['appearance_norm']:.6f}",
            "appearance_core": "" if best is None else f"{best.get('appearance_core', best['appearance_norm']):.6f}",
            "shape_score": "" if best is None else f"{best.get('shape_score', 0.5):.6f}",
            "motion_score": "" if best is None else f"{best['motion_norm']:.6f}",
            "direction_score": "" if best is None else f"{best['direction_norm']:.6f}",
            "position_score": "" if best is None else f"{best['position_norm']:.6f}",
            "w_app": f"{W_APPEARANCE:.3f}", "w_motion": f"{W_MOTION:.3f}", "w_dir": f"{W_DIRECTION:.3f}", "w_pos": f"{W_POSITION:.3f}",
            "scale_mode": scale_mode, "w3": f"{w3:.3f}", "w5": f"{w5:.3f}", "w7": f"{w7:.3f}",
            "cand_w3": "" if best is None else f"{best.get('cand_w3', w3):.3f}",
            "cand_w5": "" if best is None else f"{best.get('cand_w5', w5):.3f}",
            "cand_w7": "" if best is None else f"{best.get('cand_w7', w7):.3f}",
            "local_area": "" if best is None else best.get("local_area", ""),
            "local_w": "" if best is None else best.get("local_w", ""),
            "local_h": "" if best is None else best.get("local_h", ""),
            "local_aspect": "" if best is None or not np.isfinite(best.get("local_aspect", np.nan)) else f"{best.get('local_aspect'):.3f}",
            "local_center_dist": "" if best is None or not np.isfinite(best.get("local_center_dist", np.nan)) else f"{best.get('local_center_dist'):.3f}",
            "local_scale_mode": "" if best is None else best.get("local_scale_mode", ""),
            "handoff_bbox_w": "" if safe_float(HANDOFF_BBOX_W) is None else f"{safe_float(HANDOFF_BBOX_W):.3f}",
            "handoff_bbox_h": "" if safe_float(HANDOFF_BBOX_H) is None else f"{safe_float(HANDOFF_BBOX_H):.3f}",
            "handoff_est_radius": "" if safe_float(HANDOFF_EST_RADIUS) is None else f"{safe_float(HANDOFF_EST_RADIUS):.3f}",
            "handoff_blur_ratio": "" if safe_float(HANDOFF_BLUR_RATIO) is None else f"{safe_float(HANDOFF_BLUR_RATIO):.3f}",
            "mode": mode, "state_reason": state_reason,
            "score_ok_high": int(score_ok_high), "score_ok_low": int(score_ok_low), "dist_ok_track": int(dist_ok_track), "dist_ok_hold": int(dist_ok_hold), "motion_ok": int(motion_ok),
            "T_high": f"{TRACK_SCORE_TAU[method_name]:.3f}", "T_low": f"{HOLD_SCORE_TAU[method_name]:.3f}", "D_track": f"{D_TRACK:.3f}", "D_hold": f"{D_HOLD:.3f}",
            "search_R": R, "miss_count": miss_count, "motion_phase": motion_phase, "apex_count": apex_count,
            "online_reliable": online_reliable, "reliable_reason": reliable_reason, "offline_accuracy": offline_acc,
        })

        if SAVE_EVIDENCE_CSV:
            evidence_rows.append({
                "frame": fname, "best_score": "" if best is None else f"{best['score']:.6f}",
                "second_score": "" if second_score <= -1e17 else f"{second_score:.6f}",
                "margin": "" if (best is None or second_score <= -1e17) else f"{best['score']-second_score:.6f}",
                "cand_speed": "" if best is None else f"{best['cand_speed']:.4f}", "cand_acc": "" if best is None else f"{best['cand_acc']:.4f}",
                "cand_angle": "" if best is None else f"{best['cand_ang']:.4f}", "dir_mode": "" if best is None else best['dir_mode'],
                "vert_reason": "" if best is None else best['vert_reason'], "state_reason": state_reason,
                "online_reliable": online_reliable, "reliable_reason": reliable_reason, "offline_accuracy": offline_acc,
            })
        if SAVE_SCALE_CSV:
            scale_rows.append({
                "frame": fname, "gt_r": "" if gt_r is None else f"{gt_r:.3f}",
                "best_s3_raw": "" if best is None else f"{best['s3_raw']:.6f}", "best_s5_raw": "" if best is None else f"{best['s5_raw']:.6f}", "best_s7_raw": "" if best is None else f"{best['s7_raw']:.6f}",
                "best_s3_norm": "" if best is None else f"{best['s3_norm']:.6f}", "best_s5_norm": "" if best is None else f"{best['s5_norm']:.6f}", "best_s7_norm": "" if best is None else f"{best['s7_norm']:.6f}",
                "w3": f"{w3:.3f}", "w5": f"{w5:.3f}", "w7": f"{w7:.3f}",
                "cand_w3": "" if best is None else f"{best.get('cand_w3', w3):.3f}",
                "cand_w5": "" if best is None else f"{best.get('cand_w5', w5):.3f}",
                "cand_w7": "" if best is None else f"{best.get('cand_w7', w7):.3f}",
                "local_w": "" if best is None else best.get("local_w", ""),
                "local_h": "" if best is None else best.get("local_h", ""),
                "local_area": "" if best is None else best.get("local_area", ""),
                "local_scale_mode": "" if best is None else best.get("local_scale_mode", ""),
                "shape_score": "" if best is None else f"{best.get('shape_score', 0.5):.6f}",
                "scale_mode": scale_mode,
            })

    def save_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)

    if SAVE_DETAILED_CSV:
        save_csv(os.path.join(method_dir, f"{method_name}_tracking.csv"), track_rows)
    if SAVE_EVIDENCE_CSV:
        save_csv(os.path.join(method_dir, f"{method_name}_evidence.csv"), evidence_rows)
    if SAVE_SCALE_CSV:
        save_csv(os.path.join(method_dir, f"{method_name}_scale_response.csv"), scale_rows)

    errs = []
    succ2 = succ5 = succ10 = 0
    for row in track_rows:
        if row["error_px"] != "":
            e = float(row["error_px"]); errs.append(e)
            succ2 += int(e <= 2); succ5 += int(e <= 5); succ10 += int(e <= 10)
    return {
        "method": method_name,
        "frames_with_error": len(errs),
        "mean_error_px": "" if not errs else float(np.mean(errs)),
        "median_error_px": "" if not errs else float(np.median(errs)),
        "max_error_px": "" if not errs else float(np.max(errs)),
        "success_2px": "" if not errs else 100.0 * succ2 / len(errs),
        "success_5px": "" if not errs else 100.0 * succ5 / len(errs),
        "success_10px": "" if not errs else 100.0 * succ10 / len(errs),
        "scale_mode": scale_mode,
        "w3": w3, "w5": w5, "w7": w7,
    }


def main():
    ensure_dir(OUT_DIR)
    frame_names = list_frames(FRAMES_DIR)
    if not frame_names:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}")
    frames_bgr, frames_gray = [], []
    for fn in frame_names:
        img = cv2.imread(os.path.join(FRAMES_DIR, fn))
        if img is None:
            raise RuntimeError(f"Cannot read {fn}")
        frames_bgr.append(img); frames_gray.append(to_gray(img))
    ann = read_annotations(ANNOT_CSV)
    summaries = [run_method(m, frame_names, frames_bgr, frames_gray, ann) for m in METHODS]
    with open(os.path.join(OUT_DIR, "compare_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader(); writer.writerows(summaries)
    print("Done. Output:", OUT_DIR)


if __name__ == "__main__":
    main()
