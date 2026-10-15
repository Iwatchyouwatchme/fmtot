# stage3.py
# Stage 3: selective trajectory recovery using least-squares curve fitting.
#
# Thesis model:
#   x(t) = a1*t + a0
#   y(t) = b2*t^2 + b1*t + b0
# Only reliable Stage2 (MID_OK) observations are stored. When Stage3 is
# triggered, the fit uses reliable observations from the image-plane apex
# (minimum y) through the latest reliable observation. Generated predictions
# are never inserted back into the reliable buffer.

import os
import csv
import math
import argparse
import numpy as np

T_RELIABLE = 1
UNRELIABLE_TRIGGER_K = 8
N_FIT_MIN = 4
N_PRED_MAX = 6
CONF_DECAY = 0.85
FALLING_ONLY = True
REDETECT_DIST_TAU = 18.0

SAVE_DIAGNOSTIC_COLUMNS = True

# Confidence weights for reliable Stage2 points, all 0~1 if available.
CONF_W_SCORE = 0.35
CONF_W_MOTION = 0.25
CONF_W_POSITION = 0.25
CONF_W_MODE = 0.15


def sf(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def frame_index_from_name(frame, fallback_idx):
    """Return the actual frame index when frame names are numeric (e.g. 0123.png)."""
    try:
        stem = os.path.splitext(os.path.basename(str(frame)))[0]
        return int(stem)
    except Exception:
        return int(fallback_idx)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def is_stage2_reliable(row):
    if "online_reliable" in row and str(row.get("online_reliable", "")).strip() != "":
        return int(float(row["online_reliable"])) >= T_RELIABLE
    return row.get("mode", "") == "MID_OK"


def is_falling(row):
    phase = str(row.get("motion_phase", "")).upper()
    if phase == "DOWN":
        return True
    vy = sf(row.get("vy"))
    return vy is not None and vy > 0.8


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def score_to_conf(row):
    score = sf(row.get("score"))
    motion = sf(row.get("motion_score", row.get("motion_bonus")))
    pos = sf(row.get("position_score"))
    mode_ok = 1.0 if row.get("mode", "") == "MID_OK" else 0.0

    if score is None:
        score_c = 0.6
    elif 0.0 <= score <= 1.0:
        score_c = score
    else:
        score_c = abs(score) / (abs(score) + 5.0)

    motion_c = 0.5 if motion is None else clip01(motion)
    pos_c = 0.5 if pos is None else clip01(pos)
    q = (CONF_W_SCORE * score_c +
         CONF_W_MOTION * motion_c +
         CONF_W_POSITION * pos_c +
         CONF_W_MODE * mode_ok)
    return clip01(q)


def falling_points_from_apex(reliable_points):
    """
    reliable_points: ordered tuples (frame_idx, x, y, confidence).
    The image-plane apex is the reliable observation with minimum y.
    """
    if not reliable_points:
        return []
    apex_pos = min(range(len(reliable_points)), key=lambda i: reliable_points[i][2])
    return reliable_points[apex_pos:]


def fit_recovery_curve(reliable_points):
    """Fit x(t) linearly and y(t) quadratically by least squares."""
    pts = falling_points_from_apex(reliable_points)
    if len(pts) < N_FIT_MIN:
        return None

    t = np.asarray([p[0] for p in pts], dtype=np.float64)
    x = np.asarray([p[1] for p in pts], dtype=np.float64)
    y = np.asarray([p[2] for p in pts], dtype=np.float64)

    # Keep actual frame gaps, but shift the origin for numerical stability.
    t0 = float(t[0])
    tau = t - t0

    # Least-squares polynomial fits: x first-order, y second-order.
    ax = np.polyfit(tau, x, 1)       # [a1, a0]
    by = np.polyfit(tau, y, 2)       # [b2, b1, b0]

    return {
        "t0": t0,
        "ax": ax,
        "by": by,
        "n_used": len(pts),
        "apex_frame": int(pts[0][0]),
        "last_reliable_frame": int(pts[-1][0]),
        "last_conf": float(pts[-1][3]),
    }


def predict_from_curve(reliable_points, target_frame_idx):
    fit = fit_recovery_curve(reliable_points)
    if fit is None:
        return None

    tau = float(target_frame_idx) - fit["t0"]
    ax = fit["ax"]
    by = fit["by"]

    x = float(np.polyval(ax, tau))
    y = float(np.polyval(by, tau))

    # Derivatives are diagnostic only; the recovery position itself comes from the fit.
    vx = float(ax[0])
    vy = float(2.0 * by[0] * tau + by[1])

    dt = max(1, int(target_frame_idx) - fit["last_reliable_frame"])
    conf = clip01(fit["last_conf"] * (CONF_DECAY ** dt))

    return x, y, vx, vy, conf, fit


def run_stage3(tracking_csv, out_csv):
    rows = read_csv_rows(tracking_csv)
    reliable_points = []
    out = []
    unreliable_count = 0
    recovery_pred_count = 0
    last_recovery_xy = None

    for row_idx, r in enumerate(rows):
        frame = r.get("frame", f"{row_idx:04d}.png")
        frame_idx = frame_index_from_name(frame, row_idx)
        sx, sy = sf(r.get("pred_x")), sf(r.get("pred_y"))
        mode = r.get("mode", "")
        phase = r.get("motion_phase", "")
        reliable = is_stage2_reliable(r) and sx is not None and sy is not None

        if reliable:
            q = score_to_conf(r)
            status = "STAGE2_TRACK"
            d = ""
            if last_recovery_xy is not None:
                d = math.hypot(sx - last_recovery_xy[0], sy - last_recovery_xy[1])
                status = ("RECOVERY_TO_STAGE2" if d <= REDETECT_DIST_TAU
                          else "STAGE2_RELIABLE_FAR_FROM_RECOVERY")

            # Only reliable observations enter the fit buffer.
            reliable_points.append((frame_idx, sx, sy, q))
            unreliable_count = 0
            recovery_pred_count = 0
            last_recovery_xy = None

            row = {
                "frame": frame,
                "stage3_x": f"{sx:.3f}",
                "stage3_y": f"{sy:.3f}",
                "source": status,
                "stage3_conf": f"{q:.3f}",
                "recovery_active": 0,
                "stage2_x": r.get("pred_x", ""),
                "stage2_y": r.get("pred_y", ""),
                "stage2_mode": mode,
                "stage2_reliable": 1,
                "motion_phase": phase,
            }
            if SAVE_DIAGNOSTIC_COLUMNS:
                fit_pts = falling_points_from_apex(reliable_points)
                row.update({
                    "unreliable_count": 0,
                    "fit_points": len(fit_pts),
                    "apex_frame": "" if not fit_pts else fit_pts[0][0],
                    "vx_recovery": "",
                    "vy_recovery": "",
                    "distance_to_recovery": "" if d == "" else f"{d:.3f}",
                    "x_coef_a1": "",
                    "x_coef_a0": "",
                    "y_coef_b2": "",
                    "y_coef_b1": "",
                    "y_coef_b0": "",
                })
            out.append(row)
            continue

        unreliable_count += 1
        falling_ok = (not FALLING_ONLY) or is_falling(r)
        fit_pts = falling_points_from_apex(reliable_points)
        enough_history = len(fit_pts) >= N_FIT_MIN
        trigger_ok = unreliable_count >= UNRELIABLE_TRIGGER_K
        pred_ok = recovery_pred_count < N_PRED_MAX

        pred = (predict_from_curve(reliable_points, frame_idx)
                if (trigger_ok and falling_ok and enough_history and pred_ok)
                else None)

        fit = None
        if pred is not None:
            px, py, rvx, rvy, conf, fit = pred
            recovery_pred_count += 1
            last_recovery_xy = (px, py)
            source = "RECOVERY_PRED"
            final_x, final_y = px, py
            recovery_active = 1
        else:
            # Stage2 continues producing/searching while Stage3 eligibility is checked.
            # MID_HOLD/MID_MISS positions are not accepted into the reliable fit buffer.
            final_x, final_y = sx, sy
            rvx = rvy = None
            conf = 0.0
            recovery_active = 0

            if not trigger_ok:
                if str(mode).upper() == "MID_HOLD":
                    source = "STAGE2_HOLD_MODEL_BUFFER"
                elif str(mode).upper() == "MID_MISS":
                    source = "STAGE2_MISS_MODEL_BUFFER"
                else:
                    source = "STAGE2_UNRELIABLE_BUFFER"
            elif not falling_ok:
                source = "NO_RECOVERY_NOT_FALLING"
            elif not enough_history:
                source = "NO_RECOVERY_NOT_ENOUGH_RELIABLE_POINTS"
            elif not pred_ok:
                source = "NO_RECOVERY_MAX_PRED_REACHED"
            else:
                source = "NO_RECOVERY_MODEL"

        row = {
            "frame": frame,
            "stage3_x": "" if final_x is None else f"{final_x:.3f}",
            "stage3_y": "" if final_y is None else f"{final_y:.3f}",
            "source": source,
            "stage3_conf": f"{conf:.3f}",
            "recovery_active": recovery_active,
            "stage2_x": r.get("pred_x", ""),
            "stage2_y": r.get("pred_y", ""),
            "stage2_mode": mode,
            "stage2_reliable": 0,
            "motion_phase": phase,
        }

        if SAVE_DIAGNOSTIC_COLUMNS:
            row.update({
                "unreliable_count": unreliable_count,
                "fit_points": len(fit_pts),
                "apex_frame": "" if not fit_pts else fit_pts[0][0],
                "vx_recovery": "" if rvx is None else f"{rvx:.3f}",
                "vy_recovery": "" if rvy is None else f"{rvy:.3f}",
                "distance_to_recovery": "",
                "x_coef_a1": "" if fit is None else f"{fit['ax'][0]:.8f}",
                "x_coef_a0": "" if fit is None else f"{fit['ax'][1]:.8f}",
                "y_coef_b2": "" if fit is None else f"{fit['by'][0]:.8f}",
                "y_coef_b1": "" if fit is None else f"{fit['by'][1]:.8f}",
                "y_coef_b0": "" if fit is None else f"{fit['by'][2]:.8f}",
            })
        out.append(row)

    write_csv_rows(out_csv, out)
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Stage 3 least-squares trajectory recovery")
    ap.add_argument("--tracking_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    print("Done. Stage3 output:", run_stage3(args.tracking_csv, args.out_csv))


if __name__ == "__main__":
    main()
