# stage1.py
# Stage 1: motion-guided launch direction estimation and initial trajectory tracking.
import os
import csv
import math
import cv2
import numpy as np

# =========================================================
# 使用者設定
# =========================================================
FRAMES_DIR = "frames"
GT_CSV = "annotations.csv"      # 需要第一個 GT 當初始位置 P0
OUT_DIR = "stage1_mhi_prior"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# Step A: 前段 MHI / multi-hypothesis 方向估計參數
# 目的：只估球大致飛行方向 d，不直接決定每一幀球心
# =========================================================
DIFF_THRESH_DIR = 35
MORPH_OPEN_SIZE = 3

MOTION_DIFF_THRESH = 35
MOTION_SMOOTH_WIN = 5
MHI_PRE_FRAMES = 8
MHI_POST_FRAMES = 24
MAX_DIR_WINDOW = 45

# P1 candidates：不假設球往上/垂直，只找離 P0 合理距離的 motion blob
P1_MIN_DIST = 70
P1_MAX_DIST = 330
P1_SEARCH_RADIUS = 380
P1_MAX_CANDIDATES = 80

# trajectory growing for direction estimation
DIR_STEP_MIN = 35
DIR_STEP_MAX = 330
DIR_LINE_BAND = 35
DIR_ANGLE_DIFF_MAX = 45
DIR_T_FORWARD_MIN = 5
DIR_MAX_MISS = 5

# blob filter for direction estimation
DIR_BLOB_AREA_MIN = 2
DIR_BLOB_AREA_MAX = 900
DIR_BLOB_W_MAX = 120
DIR_BLOB_H_MAX = 120
DIR_BLOB_ASPECT_MAX = 10.0

# =========================================================
# Step B: 方向約束式 frame difference 軌跡抽取參數
# =========================================================
DIFF_THRESH_TRACK = 22          # 線帶內可低一點，提高球變化敏感度
LINE_BAND_TRACK = 26            # 方向線附近寬度
LAUNCH_LOCAL_RADIUS = 45        # P0 附近偵測起飛
LAUNCH_DIFF_THRESH = 18
LAUNCH_MIN_PIXELS = 8

T_MIN_FROM_P0_FOR_LAUNCH = 60   # P0 往前至少多少才算球已飛出
T_MAX_FROM_P0_FOR_LAUNCH = 650   # 初期高速球可能在第一個可見frame已離P0很遠

# 線帶內 component 分類
LONG_BLOB_T_LENGTH = 35         # 沿方向長度大於此值，視為長白線/拖影
SMALL_BLOB_AREA_MAX = 150

# 速度/時間連續性
TRACK_STEP_MIN_INIT = 45
TRACK_STEP_MAX_INIT = 650        # 初始點允許較大位移，避免t=400~600被擋掉
TRACK_STEP_MIN = 4
TRACK_STEP_MAX = 260
SPEED_RATIO_MIN = 0.25
SPEED_RATIO_MAX = 2.0
EXPECTED_STEP_DECAY = 0.80
MAX_TRACK_MISS = 8

# =========================================================
# mhi_7: early tracking selection 參數
# 方向已經由 launch-local validation 決定後，前幾幀不能再用一般速度比例限制。
# 目的：避免 21/22 選到舊殘影後，23 真正球候選因 step 太大被擋掉。
# =========================================================
EARLY_TRACK_FRAMES = 6          # launch 後前 6 個成功點視為高速擊球初期
EARLY_STEP_MAX = 560            # 前段允許大 step，例如 t=196 -> 468 -> 610
EARLY_STEP_MIN = 12
EARLY_N_ABS_MAX = 22            # line band 內仍避免左右偏移太大
EARLY_N_JUMP_MAX = 18           # 避免 n 左右大幅跳動
NORMAL_N_ABS_MAX = 26
NORMAL_N_JUMP_MAX = 15
EARLY_FRONT_REWARD = 0.035      # 初始/前段給較前方 t 一點獎勵，避免選舊殘影
EARLY_TLEN_PENALTY = 0.28       # 長 blur 可用，但不要讓巨大長條過度主導

# mhi_8: launch 初期不要只看單一 frame，改用短時間 lookahead 選起始路徑
INIT_LOOKAHEAD_FRAMES = 4          # 從 launch 開始往後看幾個 frame，例如 17~20 或 21~24
INIT_LOOKAHEAD_STEP_MIN = 18       # 起始路徑每一步至少要往前
INIT_LOOKAHEAD_STEP_MAX = 620      # 擊球初期允許大位移
INIT_LOOKAHEAD_N_ABS_MAX = 26
INIT_LOOKAHEAD_N_JUMP_MAX = 24
# 只限制 launch 第一個候選點的橫向偏移。
# 目的：避免第一點被球桿、人體殘影、近端錯誤 motion 拉歪，形成不合理折線。
# 注意：這不是整段軌跡的硬限制；後續幀仍使用 INIT_LOOKAHEAD_N_ABS_MAX / EARLY_N_ABS_MAX。
INIT_FIRST_N_MAX = 12.0            # 建議 12；若部分影片方向線較歪，可放寬到 15
INIT_LOOKAHEAD_MIN_HITS = 2        # 至少要能接到幾個點才採用 lookahead
INTENSITY_LOG_REWARD = 2.2      # intensity 用 log，不直接用原始總和

# 若第一段抓不到，允許用方向線前方最強 component 啟動
ALLOW_FALLBACK_LAUNCH = True

SAVE_DEBUG_IMAGES = False  # thesis batch mode: keep False unless visual debugging is needed
SAVE_CSV = True
SAVE_COMPONENTS_CSV = True
SAVE_AUX_CSV = True


# =========================================================
# Step A-2: launch-local direction estimation 參數
# 目的：方向不再用整段 19~43 的最大 hit，而是只用起飛後前幾幀的局部 motion。
# =========================================================
LOCAL_DIR_FRAMES = 5              # launch 後觀察 5 個 frame，例如 21~25，不再固定只看 21~24
LOCAL_ANGLE_BIN_DEG = 8.0         # 角度分群 bin 大小
LOCAL_CLUSTER_TOL_DEG = 12.0      # cluster 內允許角度標準差
LOCAL_MIN_DIST = 55               # 候選離 P0 太近不要
LOCAL_MAX_DIST = 650              # 允許 21~24 的高速遠距 motion
LOCAL_MAX_AREA = 2200             # 排除太大的身體區域，但保留 blur
LOCAL_MAX_W = 220
LOCAL_MAX_H = 280
LOCAL_MIN_FRAMES_IN_CLUSTER = 2   # 至少跨 2 個 frame，才像連續飛行
LOCAL_REQUIRE_UPWARD = True       # 高爾夫球通常往畫面上方飛；若影片方向不同可改 False
LOCAL_UPWARD_MARGIN = 20          # y 不能比 P0 更下面太多

# =========================================================
# 新版：畫面上方優先的先驗方向限制
# OpenCV 影像座標：x 向右為正，y 向下為正。
# atan2(dy, dx) 的角度定義：
#   0°   = 往右
#   90°  = 往下
#   -90° = 往畫面上方
# 因此「球大致往畫面上方」可設定成 -90° ± 20~25°。
# 若影片是往右上或左上，可手動調整 PRIOR_CENTER_DEG。
# =========================================================
USE_PRIOR_DIRECTION_CONE = True
PRIOR_CENTER_DEG = -90.0          # 理論上往畫面上方
PRIOR_HALF_ANGLE_DEG = 25.0       # 建議 20~25；影片差異大可放寬到 30
PRIOR_SOFT_HALF_ANGLE_DEG = 35.0  # fallback 用；避免完全找不到候選

LOCAL_SAVE_CSV = True

# mhi_6: 方向 cluster 不再用 blob count 當主因。
# 每個角度 cluster 會先用該方向嘗試在 launch 後 5 個 frame 做短程 line-band tracking，
# 能連續產生較多合理點、t 有往前推進、n 不亂跳的方向才會被選中。
LOCAL_VALIDATE_STEP_MIN = 3
LOCAL_VALIDATE_STEP_MAX = 720
LOCAL_VALIDATE_N_JUMP_TAU = 40
LOCAL_VALIDATE_MIN_HITS = 2

# Runtime record from direction validation. Used to avoid invalid launch fallback.
LAST_VALIDATED_LAUNCH_FRAME = None
LAST_VALIDATED_LAUNCH_REASON = ""


# =========================================================
# I/O
# =========================================================
def load_frames(frames_dir):
    names = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])
    if not names:
        raise RuntimeError(f"No frames found in {frames_dir}")

    frames_bgr, frames_gray = [], []
    for n in names:
        img = cv2.imread(os.path.join(frames_dir, n))
        if img is None:
            print(f"[WARN] Cannot read {n}")
            continue
        frames_bgr.append(img)
        frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return names, frames_bgr, frames_gray


def read_gt_csv(csv_path):
    gt = {}
    if not os.path.exists(csv_path):
        raise RuntimeError(f"GT CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise RuntimeError("GT CSV has no header.")

        frame_col = None
        for c in ["frame", "filename", "file"]:
            if c in headers:
                frame_col = c
                break

        x_col = "x" if "x" in headers else ("gt_x" if "gt_x" in headers else None)
        y_col = "y" if "y" in headers else ("gt_y" if "gt_y" in headers else None)
        r_col = "r" if "r" in headers else ("radius" if "radius" in headers else None)

        if frame_col is None or x_col is None or y_col is None:
            raise RuntimeError("GT CSV must contain frame,x,y or frame,gt_x,gt_y.")

        for row in reader:
            try:
                name = row[frame_col].strip()
                x = float(row[x_col])
                y = float(row[y_col])
                r = float(row[r_col]) if r_col is not None and row.get(r_col, "") != "" else np.nan

                if "visible" in row and row["visible"] not in ["", None]:
                    if int(float(row["visible"])) == 0:
                        continue

                item = (x, y, r)
                gt[name] = item
                try:
                    idx = int(os.path.splitext(name)[0])
                    gt[idx] = item
                except Exception:
                    pass
            except Exception:
                continue

    return gt


def get_first_gt(gt, frame_names):
    for i, n in enumerate(frame_names):
        if n in gt:
            x, y, r = gt[n]
            return i, np.array([x, y], dtype=np.float32), r
        if i in gt:
            x, y, r = gt[i]
            return i, np.array([x, y], dtype=np.float32), r
    raise RuntimeError("No valid GT found. Need first GT as P0.")


# =========================================================
# Geometry / utility
# =========================================================
def moving_average(x, win):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < win or win <= 1:
        return x
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same")


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n


def angle_between(v1, v2):
    v1, v2 = np.asarray(v1, dtype=np.float32), np.asarray(v2, dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def project_to_line(pt, origin, d):
    pt = np.asarray(pt, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    d = normalize(d)
    nvec = np.array([-d[1], d[0]], dtype=np.float32)
    v = pt - origin
    t = float(np.dot(v, d))
    n = float(np.dot(v, nvec))
    return t, n


def point_from_tn(origin, d, t, n=0.0):
    d = normalize(d)
    nvec = np.array([-d[1], d[0]], dtype=np.float32)
    return origin + t * d + n * nvec


def fit_direction(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return None
    mean = np.mean(pts, axis=0)
    X = pts - mean
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    d = vt[0].astype(np.float32)
    if np.dot(d, pts[-1] - pts[0]) < 0:
        d = -d
    return normalize(d)


# =========================================================
# Motion energy / effective window
# =========================================================
def diff_mask(frames_gray, idx, thresh):
    h, w = frames_gray[0].shape[:2]
    if idx <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    diff = cv2.absdiff(frames_gray[idx], frames_gray[idx - 1])
    _, bw = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((MORPH_OPEN_SIZE, MORPH_OPEN_SIZE), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw


def compute_motion_energy(frames_gray, thresh):
    vals = [0]
    for i in range(1, len(frames_gray)):
        bw = diff_mask(frames_gray, i, thresh)
        vals.append(int(np.count_nonzero(bw)))
    return np.asarray(vals, dtype=np.float32)


def find_effective_window(frames_gray, p0_frame):
    motion = compute_motion_energy(frames_gray, MOTION_DIFF_THRESH)
    smooth = moving_average(motion, MOTION_SMOOTH_WIN)

    peak = int(np.argmax(smooth))
    start = max(0, peak - MHI_PRE_FRAMES)
    end = min(len(frames_gray) - 1, peak + MHI_POST_FRAMES)

    if end - start + 1 > MAX_DIR_WINDOW:
        end = start + MAX_DIR_WINDOW - 1

    start = min(start, p0_frame)
    end = max(end, min(len(frames_gray) - 1, p0_frame + 8))
    return start, end, motion, smooth, peak


# =========================================================
# Step A: MHI / multi-hypothesis 方向估計
# =========================================================
def get_blobs_from_mask(mask, frame_idx, area_min, area_max, w_max, h_max, aspect_max):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        cx, cy = centroids[lab]

        if area < area_min or area > area_max:
            continue
        if w > w_max or h > h_max:
            continue

        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > aspect_max:
            continue

        blobs.append({
            "frame": frame_idx,
            "cx": float(cx),
            "cy": float(cy),
            "area": float(area),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "aspect": float(aspect)
        })
    return blobs


def build_direction_blobs(frames_gray):
    all_blobs = []
    for i in range(len(frames_gray)):
        mask = diff_mask(frames_gray, i, DIFF_THRESH_DIR)
        blobs = get_blobs_from_mask(
            mask, i,
            DIR_BLOB_AREA_MIN, DIR_BLOB_AREA_MAX,
            DIR_BLOB_W_MAX, DIR_BLOB_H_MAX, DIR_BLOB_ASPECT_MAX
        )
        all_blobs.append(blobs)
    return all_blobs


def collect_p1_candidates(all_blobs, p0_frame, p0, start, end):
    candidates = []
    s = max(p0_frame + 1, start)
    e = min(end, len(all_blobs) - 1)

    for f in range(s, e + 1):
        for b in all_blobs[f]:
            p = np.array([b["cx"], b["cy"]], dtype=np.float32)
            dist0 = float(np.linalg.norm(p - p0))
            if dist0 < P1_MIN_DIST or dist0 > P1_MAX_DIST:
                continue
            if dist0 > P1_SEARCH_RADIUS:
                continue

            item = b.copy()
            item["dist0"] = dist0
            # 只做候選排序，不是最終判斷
            item["pre_score"] = 0.06 * b["area"] + 0.03 * abs(dist0 - 200.0) + 2.0 * (f - s)
            candidates.append(item)

    candidates.sort(key=lambda x: x["pre_score"])
    return candidates[:P1_MAX_CANDIDATES]


def grow_direction_hypothesis(all_blobs, p0_frame, p0, p1, end_frame):
    p1_pt = np.array([p1["cx"], p1["cy"]], dtype=np.float32)
    d = normalize(p1_pt - p0)
    if np.linalg.norm(d) < 1e-6:
        return None

    traj = [(p0_frame, p0, 0.0, 0.0, 0.0, 0.0), (p1["frame"], p1_pt, p1["area"], 0.0, 0.0, 0.0)]
    pprev = p1_pt
    prev_t, _ = project_to_line(p1_pt, p0, d)
    expected_step = float(np.linalg.norm(p1_pt - p0)) * 0.85
    misses = 0

    line_errs = []
    ang_errs = []
    areas = [p1["area"]]

    for f in range(p1["frame"] + 1, min(end_frame + 1, len(all_blobs))):
        if len(traj) >= 3:
            d_fit = fit_direction([x[1] for x in traj])
            if d_fit is not None:
                d = normalize(0.75 * d + 0.25 * d_fit)

        best = None
        best_cost = 1e18

        for b in all_blobs[f]:
            p = np.array([b["cx"], b["cy"]], dtype=np.float32)
            step_vec = p - pprev
            step = float(np.linalg.norm(step_vec))
            if step < DIR_STEP_MIN or step > DIR_STEP_MAX:
                continue

            t, n = project_to_line(p, p0, d)
            if t < prev_t + DIR_T_FORWARD_MIN:
                continue
            if abs(n) > DIR_LINE_BAND:
                continue

            ang = angle_between(step_vec, d)
            if ang > DIR_ANGLE_DIFF_MAX:
                continue

            cost = 12.0 * abs(n) + 4.0 * ang + abs(step - expected_step) + 0.04 * b["area"]
            if cost < best_cost:
                best_cost = cost
                best = (b, p, step, t, n, ang)

        if best is None:
            misses += 1
            if misses >= DIR_MAX_MISS:
                break
            continue

        b, p, step, t, n, ang = best
        traj.append((f, p, b["area"], step, n, ang))
        pprev = p
        prev_t = t
        expected_step = max(DIR_STEP_MIN, 0.75 * expected_step + 0.25 * step)
        misses = 0
        line_errs.append(abs(n))
        ang_errs.append(ang)
        areas.append(b["area"])

    hits = len(traj)
    mean_line = float(np.mean(line_errs)) if line_errs else 999.0
    mean_ang = float(np.mean(ang_errs)) if ang_errs else 999.0
    mean_area = float(np.mean(areas)) if areas else 999.0

    # 方向估計的選擇：重視可形成穩定直線，而不是直接當軌跡輸出
    score = 1000.0 * hits - 20.0 * mean_line - 8.0 * mean_ang - 0.05 * mean_area - 60.0 * misses

    d_final = fit_direction([x[1] for x in traj])
    if d_final is None:
        d_final = d

    return {
        "p1": p1,
        "traj": traj,
        "hits": hits,
        "misses": misses,
        "mean_line": mean_line,
        "mean_angle": mean_ang,
        "mean_area": mean_area,
        "score": score,
        "direction": d_final
    }


def _angle_deg(v):
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def _angle_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _in_prior_direction_cone(angle_deg, half_angle=None):
    """判斷候選角度是否落在先驗方向扇形內。"""
    if not USE_PRIOR_DIRECTION_CONE:
        return True
    if half_angle is None:
        half_angle = PRIOR_HALF_ANGLE_DEG
    return _angle_diff_deg(angle_deg, PRIOR_CENTER_DEG) <= half_angle


def get_prior_direction_vector():
    """回傳目前先驗方向向量。-90° 對應畫面上方 (0,-1)。"""
    a = math.radians(PRIOR_CENTER_DEG)
    return np.array([math.cos(a), math.sin(a)], dtype=np.float32)


def _detect_launch_by_p0_energy_only(frames_gray, p0, start, end):
    """
    只用 P0 附近的 frame difference 找 possible motion-change frame。

    修正版重點：
    - 若 P0 附近沒有達到 local energy 門檻，不再回傳 start/frame1 當作 launch。
    - 回傳 None 代表 P0-local evidence 不足，後續需依靠方向線帶的
      temporal validation 來確認保守起飛幀。
    """
    energy = compute_launch_energy(frames_gray, p0)
    s = max(1, start)
    e = min(end, len(frames_gray) - 1)
    for f in range(s, e + 1):
        if energy[f] >= LAUNCH_MIN_PIXELS:
            return f, energy
    return None, energy


def _collect_launch_local_vectors(all_blobs, p0, launch_frame, end_frame):
    """
    從 launch_frame ~ launch_frame+LOCAL_DIR_FRAMES-1 收集候選向量。
    每個候選都是 P0 -> motion blob center。
    """
    items = []
    s = max(1, launch_frame)
    e = min(end_frame, len(all_blobs) - 1, launch_frame + LOCAL_DIR_FRAMES - 1)

    for f in range(s, e + 1):
        for b in all_blobs[f]:
            p = np.array([b["cx"], b["cy"]], dtype=np.float32)
            v = p - p0
            dist = float(np.linalg.norm(v))
            if dist < LOCAL_MIN_DIST or dist > LOCAL_MAX_DIST:
                continue

            # 排除過大的身體 / 手臂 motion，但保留球的長條 blur
            if b.get("area", 0) > LOCAL_MAX_AREA:
                continue
            if b.get("w", 0) > LOCAL_MAX_W or b.get("h", 0) > LOCAL_MAX_H:
                continue

            # 弱方向先驗：球應該從 P0 往上飛。若影片方向不同，可把 LOCAL_REQUIRE_UPWARD=False。
            if LOCAL_REQUIRE_UPWARD and p[1] > p0[1] + LOCAL_UPWARD_MARGIN:
                continue

            ang = _angle_deg(v)

            # 新版：優先只保留「往畫面上方 ± PRIOR_HALF_ANGLE_DEG」的候選方向。
            # 這裡不是假設完全垂直，而是利用場景先驗先排除明顯不可能的方向。
            if USE_PRIOR_DIRECTION_CONE and (not _in_prior_direction_cone(ang, PRIOR_HALF_ANGLE_DEG)):
                continue

            items.append({
                "frame": f,
                "point": p,
                "vx": float(v[0]),
                "vy": float(v[1]),
                "dist": dist,
                "angle": ang,
                "area": float(b.get("area", 0)),
                "w": float(b.get("w", 0)),
                "h": float(b.get("h", 0)),
                "blob": b,
            })
    return items


def _cluster_vectors_by_angle(items):
    """
    用角度分群。mhi_6 的重點：
    - count 只代表該方向附近 blob 數量，不能直接當球的證據。
    - 這裡只先做角度 grouping 與基本統計。
    - 真正選方向會在 estimate_direction_multihypothesis() 裡，
      用 launch 後 5 個 frame 的短程 line-band tracking 支援度來決定。
    """
    clusters = []

    for it in sorted(items, key=lambda x: x["angle"]):
        placed = False
        for cl in clusters:
            if _angle_diff_deg(it["angle"], cl["mean_angle"]) <= LOCAL_ANGLE_BIN_DEG:
                cl["items"].append(it)
                # 用單位向量平均更新 mean angle，避免 -179/179 問題
                vecs = []
                for z in cl["items"]:
                    a = math.radians(z["angle"])
                    vecs.append([math.cos(a), math.sin(a)])
                mv = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
                cl["mean_angle"] = _angle_deg(mv)
                placed = True
                break
        if not placed:
            clusters.append({"mean_angle": it["angle"], "items": [it]})

    scored = []
    for cl in clusters:
        its = cl["items"]
        frames = sorted(set(z["frame"] for z in its))
        angles = [z["angle"] for z in its]
        mean_angle = cl["mean_angle"]
        angle_std = float(np.mean([_angle_diff_deg(a, mean_angle) for a in angles])) if angles else 999.0
        mean_area = float(np.mean([z["area"] for z in its])) if its else 999.0
        mean_dist = float(np.mean([z["dist"] for z in its])) if its else 0.0

        # intrinsic_score 只是初步分數，不作最終選擇主因。
        # 只重視跨 frame 與角度集中，不把 count 當主要優勢。
        intrinsic_score = (
            900.0 * len(frames)
            - 80.0 * angle_std
            - 0.04 * mean_area
        )
        if len(frames) < LOCAL_MIN_FRAMES_IN_CLUSTER:
            intrinsic_score -= 1000.0
        if angle_std > LOCAL_CLUSTER_TOL_DEG:
            intrinsic_score -= 800.0

        scored.append({
            "items": its,
            "frames": frames,
            "mean_angle": mean_angle,
            "angle_std": angle_std,
            "mean_area": mean_area,
            "mean_dist": mean_dist,
            "intrinsic_score": intrinsic_score,
            "score": intrinsic_score,  # 之後會被 local validation score 覆蓋
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def _direction_from_cluster(cluster, p0):
    """
    用 P0 + cluster 內候選點做 line fitting，得到穩定方向。
    """
    pts = [p0]
    for it in cluster["items"]:
        pts.append(it["point"])
    d = fit_direction(pts)
    if d is None:
        # fallback：平均單位向量
        vecs = []
        for it in cluster["items"]:
            v = normalize(it["point"] - p0)
            if np.linalg.norm(v) > 1e-6:
                vecs.append(v)
        if not vecs:
            return np.array([0.0, -1.0], dtype=np.float32)
        d = normalize(np.mean(np.asarray(vecs, dtype=np.float32), axis=0))
    return normalize(d)



def _validate_cluster_by_local_lineband(cluster, frames_gray, p0, launch_frame, end_frame):
    """
    對某個 launch-local angle cluster 做短程 line-band validation。

    目的不是直接輸出完整軌跡，而是檢查：
    1. 沿著這個方向，在 launch 後幾幀是否真的能找到往前的 motion component。
    2. t 是否能持續前進。
    3. n 是否不要太大、不要左右大幅跳動。

    這樣可以避免只因為 blob count 很多，就選到人或球桿的方向。
    """
    d = _direction_from_cluster(cluster, p0)

    frames_used = []
    chosen_ts = []
    chosen_ns = []
    chosen_pts = []

    prev_t = 0.0
    prev_n = None
    misses = 0

    local_end = min(len(frames_gray) - 1, end_frame, launch_frame + LOCAL_DIR_FRAMES - 1)

    for f in range(launch_frame, local_end + 1):
        comps = extract_lineband_components(frames_gray, f, p0, d)
        best = None
        best_cost = 1e18

        for c in comps:
            t = float(c.get("t", 0.0))
            n = float(c.get("n", 999.0))
            step = t - prev_t

            if step < LOCAL_VALIDATE_STEP_MIN or step > LOCAL_VALIDATE_STEP_MAX:
                continue

            if abs(n) > LINE_BAND_TRACK:
                continue

            if prev_n is not None and abs(n - prev_n) > LOCAL_VALIDATE_N_JUMP_TAU:
                continue

            # validation 階段：優先能往前走、n 不大、component 不過大。
            # 不用 intensity 當主因，避免人/球桿強 motion 影響方向選擇。
            progress_reward = -0.08 * step
            n_cost = 2.0 * abs(n)
            area_cost = 0.004 * float(c.get("area", 0.0))
            tlen_cost = 0.15 * float(c.get("t_len", 0.0))
            intensity_reward = -0.8 * math.log1p(float(c.get("intensity_sum", 0.0)))

            cost = n_cost + area_cost + tlen_cost + intensity_reward + progress_reward

            if cost < best_cost:
                best_cost = cost
                best = c

        if best is None:
            misses += 1
            continue

        frames_used.append(f)
        chosen_ts.append(float(best["t"]))
        chosen_ns.append(float(best["n"]))
        chosen_pts.append(np.array([best["x"], best["y"]], dtype=np.float32))
        prev_t = float(best["t"])
        prev_n = float(best["n"])

    hits = len(frames_used)
    progress = (chosen_ts[-1] - chosen_ts[0]) if hits >= 2 else (chosen_ts[0] if hits == 1 else 0.0)
    mean_abs_n = float(np.mean(np.abs(chosen_ns))) if hits > 0 else 999.0
    step_std = 999.0
    if hits >= 3:
        steps = np.diff(np.asarray(chosen_ts, dtype=np.float32))
        step_std = float(np.std(steps))
    elif hits == 2:
        step_std = 0.0

    # cluster 本身的 temporal consistency 只看「跨幀數」，不是 blob count。
    frame_count = len(set(cluster.get("frames", [])))
    intrinsic = float(cluster.get("intrinsic_score", 0.0))

    score = (
        3500.0 * hits
        + 900.0 * frame_count
        + 2.5 * progress
        - 45.0 * mean_abs_n
        - 7.0 * step_std
        - 450.0 * misses
        + 0.05 * intrinsic
    )

    if hits < LOCAL_VALIDATE_MIN_HITS:
        score -= 5000.0

    return {
        "direction": d,
        "track_hits": hits,
        "track_misses": misses,
        "track_frames": frames_used,
        "track_progress": float(progress),
        "track_mean_abs_n": float(mean_abs_n),
        "track_step_std": float(step_std),
        "track_points": chosen_pts,
        "score": float(score),
    }

def save_local_direction_candidates_csv(clusters, out_path):
    if not clusters:
        return
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "cluster_rank", "cluster_score", "cluster_mean_angle", "cluster_angle_std",
            "cluster_frames", "cluster_count", "cluster_mean_dist", "cluster_mean_area",
            "track_hits", "track_frames", "track_progress", "track_mean_abs_n", "track_step_std",
            "frame", "x", "y", "dist0", "angle", "area", "w", "h"
        ])
        for rank, cl in enumerate(clusters, start=1):
            for it in cl["items"]:
                wr.writerow([
                    rank, cl.get("score", ""), cl.get("mean_angle", ""), cl.get("angle_std", ""),
                    ";".join(map(str, cl.get("frames", []))), len(cl.get("items", [])),
                    cl.get("mean_dist", ""), cl.get("mean_area", ""),
                    cl.get("track_hits", ""), ";".join(map(str, cl.get("track_frames", []))),
                    cl.get("track_progress", ""), cl.get("track_mean_abs_n", ""), cl.get("track_step_std", ""),
                    it["frame"], float(it["point"][0]), float(it["point"][1]),
                    it["dist"], it["angle"], it["area"], it["w"], it["h"]
                ])

def estimate_direction_multihypothesis(frames_gray, p0_frame, p0, start, end, *args, **kwargs):
    """
    mhi_6 版本：launch-local direction estimation + local line-band validation。

    設計：
    1. 先用 P0 local energy 找 launch frame。
    2. 觀察 launch 後 5 個 frame 的 motion blobs。
    3. 所有 blob 都以 P0 為原點轉成角度，做 angle clustering。
    4. 不直接用 blob count 選方向。
    5. 每個方向 cluster 都拿去做 5-frame 短程 line-band tracking 測試，
       能沿該方向產生最多連續合理預測點的方向才被選為 d。
    """
    all_blobs = build_direction_blobs(frames_gray)

    launch_frame, local_energy = _detect_launch_by_p0_energy_only(frames_gray, p0, start, end)

    # 若 P0-local energy 沒有明確觸發，仍需用一個搜尋種子建立方向候選；
    # 但這個 seed 只用於方向估計，不視為 confirmed launch frame。
    launch_seed = launch_frame if launch_frame is not None else max(1, start)
    local_end = min(end, launch_seed + LOCAL_DIR_FRAMES - 1)
    if launch_frame is None:
        print(f"[INFO] P0-local motion change not confirmed. Use seed window = {launch_seed} ~ {local_end} for direction search")
    else:
        print(f"[INFO] Launch-local direction window = {launch_seed} ~ {local_end} ({LOCAL_DIR_FRAMES} frames after motion change)")

    items = _collect_launch_local_vectors(all_blobs, p0, launch_seed, end)
    print(f"[INFO] Launch-local direction candidates in prior cone = {len(items)}")

    # 如果 ±25° 太嚴格導致沒有候選，放寬到 soft cone，仍然比全方向搜尋穩定。
    if USE_PRIOR_DIRECTION_CONE and len(items) == 0:
        old_half = PRIOR_HALF_ANGLE_DEG
        globals()["PRIOR_HALF_ANGLE_DEG"] = PRIOR_SOFT_HALF_ANGLE_DEG
        items = _collect_launch_local_vectors(all_blobs, p0, launch_seed, end)
        globals()["PRIOR_HALF_ANGLE_DEG"] = old_half
        print(f"[WARN] No candidates in strict cone. Retry with soft cone ±{PRIOR_SOFT_HALF_ANGLE_DEG}°, candidates={len(items)}")

    clusters = _cluster_vectors_by_angle(items)

    # 對每個 cluster 做短程 line-band validation。
    for cl in clusters:
        val = _validate_cluster_by_local_lineband(cl, frames_gray, p0, launch_seed, end)
        cl.update(val)
        cl["score"] = val["score"]

    clusters.sort(key=lambda x: x["score"], reverse=True)

    if LOCAL_SAVE_CSV:
        try:
            save_local_direction_candidates_csv(
                clusters,
                os.path.join(OUT_DIR, "launch_local_direction_candidates.csv")
            )
        except Exception as e:
            print(f"[WARN] Cannot save launch local direction candidates: {e}")

    if clusters:
        print("[INFO] Top launch-local direction clusters after line-band validation:")
        for rank, cl in enumerate(clusters[:8], start=1):
            print(
                f"  #{rank}: score={cl['score']:.1f}, frames={cl['frames']}, "
                f"count={len(cl['items'])}, angle={cl['mean_angle']:.1f}, "
                f"std={cl['angle_std']:.1f}, mean_area={cl['mean_area']:.1f}, "
                f"mean_dist={cl['mean_dist']:.1f}, "
                f"track_hits={cl.get('track_hits', 0)}, track_frames={cl.get('track_frames', [])}, "
                f"progress={cl.get('track_progress', 0.0):.1f}, mean|n|={cl.get('track_mean_abs_n', 999.0):.1f}"
            )

        best = clusters[0]
        d = normalize(best["direction"])

        # 若方向驗證已形成穩定短軌跡，將第一個 validated track frame
        # 記錄為保守 launch frame。這可避免 P0-local energy 不明顯時
        # fallback 到 frame1 / start frame。
        global LAST_VALIDATED_LAUNCH_FRAME, LAST_VALIDATED_LAUNCH_REASON
        tf = best.get("track_frames", [])
        if tf and best.get("track_hits", 0) >= LOCAL_VALIDATE_MIN_HITS:
            LAST_VALIDATED_LAUNCH_FRAME = int(min(tf))
            LAST_VALIDATED_LAUNCH_REASON = (
                f"validated_lineband_start;hits={best.get('track_hits', 0)};"
                f"progress={best.get('track_progress', 0.0):.1f};"
                f"mean_abs_n={best.get('track_mean_abs_n', 999.0):.1f}"
            )
        else:
            LAST_VALIDATED_LAUNCH_FRAME = None
            LAST_VALIDATED_LAUNCH_REASON = ""

        # 建立與舊 save_direction_hypotheses_csv 相容的 hyps
        p1_item = min(best["items"], key=lambda z: z["frame"])
        p1_blob = p1_item["blob"].copy()
        p1_blob["dist0"] = p1_item["dist"]
        hyps = [{
            "p1": p1_blob,
            "traj": [(p0_frame, p0, 0.0, 0.0, 0.0, 0.0)] + [
                (it["frame"], it["point"], it["area"], it["dist"], 0.0, 0.0)
                for it in best["items"]
            ],
            "hits": best.get("track_hits", len(best["frames"])),
            "misses": best.get("track_misses", 0),
            "mean_line": best.get("track_mean_abs_n", best["angle_std"]),
            "mean_angle": best["angle_std"],
            "mean_area": best["mean_area"],
            "score": best["score"],
            "direction": d,
        }]

        return d, all_blobs, hyps

    # fallback：如果 local direction 找不到，再回到舊版 multi-hypothesis，避免完全失敗。
    print("[WARN] No launch-local direction cluster. Fallback to original multi-hypothesis.")
    p1_candidates = collect_p1_candidates(all_blobs, p0_frame, p0, start, end)
    print(f"[INFO] Direction P1 candidates = {len(p1_candidates)}")

    hyps = []
    for p1 in p1_candidates:
        h = grow_direction_hypothesis(all_blobs, p0_frame, p0, p1, end)
        if h is not None:
            hyps.append(h)

    if not hyps:
        print("[WARN] No direction hypothesis. Fallback to upward direction.")
        return get_prior_direction_vector(), all_blobs, []

    hyps.sort(key=lambda h: h["score"], reverse=True)
    best = hyps[0]
    return normalize(best["direction"]), all_blobs, hyps


# =========================================================
# Step B: direction-constrained diff trajectory extraction
# =========================================================
def compute_launch_energy(frames_gray, p0):
    h, w = frames_gray[0].shape[:2]
    cx, cy = int(round(p0[0])), int(round(p0[1]))
    x1, x2 = max(0, cx - LAUNCH_LOCAL_RADIUS), min(w, cx + LAUNCH_LOCAL_RADIUS + 1)
    y1, y2 = max(0, cy - LAUNCH_LOCAL_RADIUS), min(h, cy + LAUNCH_LOCAL_RADIUS + 1)

    vals = [0]
    kernel = np.ones((MORPH_OPEN_SIZE, MORPH_OPEN_SIZE), np.uint8)
    for i in range(1, len(frames_gray)):
        diff = cv2.absdiff(frames_gray[i], frames_gray[i - 1])
        roi = diff[y1:y2, x1:x2]
        _, bw = cv2.threshold(roi, LAUNCH_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        vals.append(int(np.count_nonzero(bw)))
    return np.asarray(vals, dtype=np.float32)


def extract_lineband_components(frames_gray, frame_idx, p0, d):
    """
    在 direction band 內找 frame difference component。
    每個 component 會計算：
      t_min, t_max, t_mid, t_weighted, n_mean, area
    """
    if frame_idx <= 0:
        return []

    diff = cv2.absdiff(frames_gray[frame_idx], frames_gray[frame_idx - 1])
    _, bw = cv2.threshold(diff, DIFF_THRESH_TRACK, 255, cv2.THRESH_BINARY)

    # 只保留方向線附近 pixels
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return []

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    d = normalize(d)
    nvec = np.array([-d[1], d[0]], dtype=np.float32)
    v = pts - p0[None, :]
    ts = v @ d
    ns = v @ nvec

    keep = (np.abs(ns) <= LINE_BAND_TRACK) & (ts > 0)
    band_mask = np.zeros_like(bw)
    if np.any(keep):
        kpts = pts[keep].astype(np.int32)
        band_mask[kpts[:, 1], kpts[:, 0]] = 255

    kernel = np.ones((MORPH_OPEN_SIZE, MORPH_OPEN_SIZE), np.uint8)
    band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_OPEN, kernel)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(band_mask, connectivity=8)
    comps = []

    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        if area < 1:
            continue

        yy, xx = np.where(labels == lab)
        cpts = np.stack([xx, yy], axis=1).astype(np.float32)

        v2 = cpts - p0[None, :]
        tvals = v2 @ d
        nvals = v2 @ nvec

        valid = (np.abs(nvals) <= LINE_BAND_TRACK) & (tvals > 0)
        if not np.any(valid):
            continue

        cpts = cpts[valid]
        tvals = tvals[valid]
        nvals = nvals[valid]

        # 用 diff intensity 做 weighted center
        intensities = diff[cpts[:, 1].astype(int), cpts[:, 0].astype(int)].astype(np.float32)
        weight_sum = float(np.sum(intensities) + 1e-6)
        t_weighted = float(np.sum(tvals * intensities) / weight_sum)
        n_weighted = float(np.sum(nvals * intensities) / weight_sum)

        t_min = float(np.min(tvals))
        t_max = float(np.max(tvals))
        t_mid = 0.5 * (t_min + t_max)
        t_len = t_max - t_min

        # 長白線/拖影：用沿方向中點；小點：用強度重心
        if t_len >= LONG_BLOB_T_LENGTH:
            chosen_t = t_mid
            chosen_n = n_weighted
            comp_type = "LONG_LINE_MID"
        else:
            chosen_t = t_weighted
            chosen_n = n_weighted
            comp_type = "SMALL_WEIGHTED"

        chosen_pt = point_from_tn(p0, d, chosen_t, chosen_n)

        comps.append({
            "frame": frame_idx,
            "x": float(chosen_pt[0]),
            "y": float(chosen_pt[1]),
            "t": float(chosen_t),
            "n": float(chosen_n),
            "t_min": t_min,
            "t_max": t_max,
            "t_len": float(t_len),
            "area": float(area),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
            "type": comp_type,
            "intensity_sum": float(np.sum(intensities))
        })

    comps.sort(key=lambda c: c["t"])
    return comps


def detect_launch_frame(frames_gray, p0, d, start, end):
    """
    起飛 frame 條件：
    1. P0 附近有 local difference
    2. 同時在方向線前方一定距離內有 component

    修正版：
    - 不再於失敗時強制回傳 max(1,start)。
    - 若 P0-local evidence 不明顯，但方向估計階段已取得 validated line-band
      track_frames，則使用第一個 validated frame 作為保守 launch frame。
    - 若仍無法確認，回傳 None，由上層標記 Stage1 initialization failed。
    """
    local_energy = compute_launch_energy(frames_gray, p0)
    launch = None

    for f in range(max(1, start), min(end, len(frames_gray)-1) + 1):
        if local_energy[f] < LAUNCH_MIN_PIXELS:
            continue

        comps = extract_lineband_components(frames_gray, f, p0, d)
        forward = [
            c for c in comps
            if T_MIN_FROM_P0_FOR_LAUNCH <= c["t"] <= T_MAX_FROM_P0_FOR_LAUNCH
        ]

        if forward:
            launch = f
            break

    # fallback 不再忽略 local evidence 直接找任意 forward component；
    # 改為使用方向驗證的短軌跡起點，因為它同時滿足連續性、progress 與 line-band consistency。
    if launch is None and ALLOW_FALLBACK_LAUNCH:
        global LAST_VALIDATED_LAUNCH_FRAME, LAST_VALIDATED_LAUNCH_REASON
        if LAST_VALIDATED_LAUNCH_FRAME is not None:
            launch = int(LAST_VALIDATED_LAUNCH_FRAME)
            print(f"[INFO] Use validated line-band launch frame = {launch} ({LAST_VALIDATED_LAUNCH_REASON})")

    return launch, local_energy


def _log_intensity(c):
    """用 log 強度，避免巨大 blur 的 intensity_sum 直接壓過其他條件。"""
    return math.log1p(max(0.0, float(c.get("intensity_sum", 0.0))))


def choose_component_for_frame(comps, prev_t, expected_step, is_initial=False, prev_n=None, track_count=0):
    """
    mhi_7 選點規則：
    - 方向已由 launch-local cluster 決定。
    - launch 後前幾個成功點屬於高速擊球初期，不使用嚴格 SPEED_RATIO。
    - 初期優先選「合理範圍內更前方的 motion component」，避免停在舊位置殘影。
    - LONG_LINE 仍用 component 內部的 t_mid，這在 extract_lineband_components() 已完成。
    - n 只作穩定限制，避免左右大幅跳動，不再過度要求完全貼直線。
    """
    if not comps:
        return None

    candidates = []
    early_phase = is_initial or (track_count < EARLY_TRACK_FRAMES)

    for c in comps:
        t = float(c["t"])
        n = float(c["n"])
        area = float(c.get("area", 0.0))
        t_len = float(c.get("t_len", 0.0))
        inten_log = _log_intensity(c)

        # -------------------------
        # 初始點：不要被 P0 附近或舊殘影吸住。
        # 在 launch frame 內，允許 t 在起飛範圍，並偏好較前方但不是亂跳的 component。
        # -------------------------
        if is_initial:
            if not (T_MIN_FROM_P0_FOR_LAUNCH <= t <= T_MAX_FROM_P0_FOR_LAUNCH):
                continue
            step = t
            if step < TRACK_STEP_MIN_INIT or step > TRACK_STEP_MAX_INIT:
                continue
            if abs(n) > EARLY_N_ABS_MAX:
                continue

            cost = (
                - EARLY_FRONT_REWARD * t          # 偏好前方，避免選舊殘影
                + 0.90 * abs(n)                  # n 輔助，不可太大
                + 0.010 * area                   # 避免身體/大區塊
                + EARLY_TLEN_PENALTY * t_len     # 長條不要過度主導
                - INTENSITY_LOG_REWARD * inten_log
            )

        # -------------------------
        # 後續點：前幾幀不使用 ratio 硬限制，允許大位移。
        # 中後段才回到一般速度預測。
        # -------------------------
        else:
            step = t - float(prev_t)

            if early_phase:
                if step < EARLY_STEP_MIN or step > EARLY_STEP_MAX:
                    continue
                if abs(n) > EARLY_N_ABS_MAX:
                    continue
                if prev_n is not None and abs(n - prev_n) > EARLY_N_JUMP_MAX:
                    continue

                # 若有 expected_step，作軟性參考，不硬擋。
                if expected_step is not None and expected_step > 1:
                    pred_cost = abs(step - expected_step)
                else:
                    pred_cost = 0.0

                cost = (
                    0.28 * pred_cost
                    - 0.030 * step               # 初期鼓勵往前推進
                    + 0.95 * abs(n)
                    + (0.75 * abs(n - prev_n) if prev_n is not None else 0.0)
                    + 0.010 * area
                    + 0.20 * t_len
                    - 1.8 * inten_log
                )

            else:
                if step < TRACK_STEP_MIN or step > TRACK_STEP_MAX:
                    continue
                if abs(n) > NORMAL_N_ABS_MAX:
                    continue
                if prev_n is not None and abs(n - prev_n) > NORMAL_N_JUMP_MAX:
                    continue

                if expected_step is not None and expected_step > 1:
                    ratio = step / expected_step
                    if ratio < SPEED_RATIO_MIN or ratio > SPEED_RATIO_MAX:
                        continue
                    speed_cost = abs(step - expected_step)
                else:
                    speed_cost = 0.0

                cost = (
                    0.85 * speed_cost
                    + 1.15 * abs(n)
                    + (0.85 * abs(n - prev_n) if prev_n is not None else 0.0)
                    + 0.012 * area
                    + 0.12 * t_len
                    - 1.4 * inten_log
                )

        candidates.append((cost, c, step))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    cost, c, step = candidates[0]
    out = c.copy()
    out["step"] = float(step)
    out["select_cost"] = float(cost)
    return out


def _valid_initial_candidate(c):
    """
    launch 第一個候選點的基本條件。

    這裡只針對「起飛第一點」加較嚴格的 n 偏移限制。
    原因：第一個候選點最容易被球桿、人體或殘影干擾；
    若第一點離已估計的飛行方向線太遠，後面即使接回正確方向，
    顯示出的軌跡仍會形成明顯折線。

    注意：
    - 這裡不特別偏好 t 遠或 t 近。
    - 只要求候選點不要偏離方向線太誇張。
    - 後續 lookahead frame 仍使用 INIT_LOOKAHEAD_N_ABS_MAX。
    """
    t = float(c["t"])
    n = float(c["n"])

    # 必須已經離開初始位置一段距離，避免 P0 附近殘影被當成球。
    if not (T_MIN_FROM_P0_FOR_LAUNCH <= t <= T_MAX_FROM_P0_FOR_LAUNCH):
        return False

    # 初始位移需落在合理範圍。
    if t < TRACK_STEP_MIN_INIT or t > max(TRACK_STEP_MAX_INIT, T_MAX_FROM_P0_FOR_LAUNCH):
        return False

    # 新增：起飛第一點必須接近方向線。
    # 這可排除例如 n=-16.5 的錯誤近端 motion，而保留 n≈0~數 px 的合理球候選。
    if abs(n) > INIT_FIRST_N_MAX:
        return False

    return True


def _transition_ok_early(prev_c, c):
    """launch lookahead 期間的連續性條件：t 要往前，n 不可左右大跳。"""
    step = float(c["t"]) - float(prev_c["t"])
    if step < INIT_LOOKAHEAD_STEP_MIN or step > INIT_LOOKAHEAD_STEP_MAX:
        return False
    if abs(float(c["n"])) > INIT_LOOKAHEAD_N_ABS_MAX:
        return False
    if abs(float(c["n"]) - float(prev_c["n"])) > INIT_LOOKAHEAD_N_JUMP_MAX:
        return False
    return True


def _initial_path_score(path):
    """
    對 launch 初期短路徑評分。
    目標不是選單幀 cost 最小，而是選能在後續幀持續往前的路徑。
    """
    if not path:
        return -1e18

    hits = len(path)
    t0 = float(path[0][1]["t"])
    t_last = float(path[-1][1]["t"])
    progress = max(0.0, t_last - t0)
    total_t = sum(float(c["t"]) for _, c in path)
    mean_abs_n = sum(abs(float(c["n"])) for _, c in path) / hits
    mean_area = sum(float(c.get("area", 0.0)) for _, c in path) / hits
    mean_tlen = sum(float(c.get("t_len", 0.0)) for _, c in path) / hits
    inten = sum(_log_intensity(c) for _, c in path)

    # step 平滑只是輔助，擊球初期不應強迫等速度。
    steps = [float(path[i][1]["t"]) - float(path[i-1][1]["t"]) for i in range(1, hits)]
    step_var = float(np.std(steps)) if len(steps) >= 2 else 0.0

    # 起始 t 不能太後方；但也不要固定 t。用整條路徑 progress 主導。
    score = (
        1200.0 * hits
        + 2.2 * progress
        + 0.10 * total_t
        - 35.0 * mean_abs_n
        - 0.030 * mean_area
        - 0.40 * mean_tlen
        - 0.30 * step_var
        + 2.0 * inten
    )
    return score


def select_initial_path_with_lookahead(frames_gray, p0, d, launch_frame, end):
    """
    mhi_8 核心修改：
    TRACK_INIT 不再只看 launch 單一 frame，而是看 launch 後數個 frame 能否接成合理路徑。

    回傳：path, comps_cache
      path = [(frame_idx, chosen_component), ...]
      comps_cache[f] = 該 frame 的 lineband components
    """
    last_f = min(end, len(frames_gray) - 1, launch_frame + INIT_LOOKAHEAD_FRAMES - 1)
    frame_range = list(range(launch_frame, last_f + 1))
    comps_cache = {}
    cand_by_frame = {}

    for f in frame_range:
        comps = extract_lineband_components(frames_gray, f, p0, d)
        comps_cache[f] = comps
        if f == launch_frame:
            cand_by_frame[f] = [c for c in comps if _valid_initial_candidate(c)]
        else:
            cand_by_frame[f] = [
                c for c in comps
                if float(c["t"]) > T_MIN_FROM_P0_FOR_LAUNCH
                and abs(float(c["n"])) <= INIT_LOOKAHEAD_N_ABS_MAX
            ]

    if not cand_by_frame.get(launch_frame):
        return [], comps_cache

    # beam search：避免候選太多爆量。每個 frame 保留較有可能的前幾個。
    def pre_rank(c):
        return (
            -0.015 * float(c["t"])
            + 0.8 * abs(float(c["n"]))
            + 0.004 * float(c.get("area", 0.0))
            + 0.10 * float(c.get("t_len", 0.0))
            - 1.5 * _log_intensity(c)
        )

    for f in frame_range:
        cand_by_frame[f] = sorted(cand_by_frame[f], key=pre_rank)[:24]

    paths = [[(launch_frame, c)] for c in cand_by_frame[launch_frame]]

    for f in frame_range[1:]:
        new_paths = []
        for path in paths:
            prev_c = path[-1][1]
            extended = False
            for c in cand_by_frame.get(f, []):
                if _transition_ok_early(prev_c, c):
                    new_paths.append(path + [(f, c)])
                    extended = True
            # 允許短暫沒有接上，但有接上路徑會優先；避免一個 missing 直接放棄
            if not extended:
                new_paths.append(path)

        # 去除完全相同最後狀態，並保留高分路徑
        uniq = {}
        for path in new_paths:
            key = (path[-1][0], round(float(path[-1][1]["t"]), 1), round(float(path[-1][1]["n"]), 1), len(path))
            sc = _initial_path_score(path)
            if key not in uniq or sc > uniq[key][0]:
                uniq[key] = (sc, path)
        paths = [v[1] for v in sorted(uniq.values(), key=lambda x: x[0], reverse=True)[:80]]

    if not paths:
        return [], comps_cache

    best_path = max(paths, key=_initial_path_score)
    if len(best_path) < INIT_LOOKAHEAD_MIN_HITS:
        return [], comps_cache

    print("[INFO] Initial lookahead path:", " -> ".join(
        f"{frame_names if False else f}:{c['t']:.1f}/{c['n']:.1f}" for f, c in best_path
    ))
    return best_path, comps_cache

def estimate_radius_from_component(c):
    """Equivalent radius from component area: r = sqrt(area / pi)."""
    area = max(1.0, float(c.get("area", 1.0)))
    return float(max(1.0, min(20.0, math.sqrt(area / math.pi))))


def estimate_aspect_from_component(c):
    w = float(c.get("bbox_w", 0.0) or 0.0)
    h = float(c.get("bbox_h", 0.0) or 0.0)
    if w <= 0 or h <= 0:
        return float("nan")
    return float(max(w, h) / max(1.0, min(w, h)))


def estimate_blur_ratio_from_component(c):
    """Blur indicator using length along the flight direction relative to equivalent radius."""
    r = estimate_radius_from_component(c)
    t_len = max(0.0, float(c.get("t_len", 0.0) or 0.0))
    return float(t_len / max(1.0, 2.0 * r))


def estimate_scale_from_component(c):
    """Scale indicator used for backward compatibility. Not a physical radius."""
    area = max(1.0, float(c.get("area", 1.0)))
    t_len = max(0.0, float(c.get("t_len", 0.0)))
    scale_area = math.sqrt(area / math.pi)
    scale_len = 0.5 * t_len if t_len > 0 else scale_area
    return float(max(1.0, min(12.0, 0.6 * scale_area + 0.4 * scale_len)))


def estimate_scale_class_from_component(c):
    """Classify the handoff component into tiny/mid/blur for Stage2 scale weighting."""
    w = float(c.get("bbox_w", 0.0) or 0.0)
    h = float(c.get("bbox_h", 0.0) or 0.0)
    t_len = max(0.0, float(c.get("t_len", 0.0) or 0.0))
    comp_type = str(c.get("type", c.get("component_type", ""))).upper()
    radius = estimate_radius_from_component(c)
    aspect = estimate_aspect_from_component(c)
    blur_ratio = estimate_blur_ratio_from_component(c)

    if comp_type.startswith("LONG") or t_len >= LONG_BLOB_T_LENGTH or (aspect == aspect and aspect >= 3.0) or blur_ratio >= 2.5:
        return "blur_7x7"
    if max(w, h) <= 4.0 or radius <= 2.0:
        return "tiny_3x3"
    return "mid_5x5"


def component_scale_fields(c):
    return {
        "est_scale": estimate_scale_from_component(c),
        "est_radius": estimate_radius_from_component(c),
        "blur_ratio": estimate_blur_ratio_from_component(c),
        "bbox_aspect": estimate_aspect_from_component(c),
        "scale_class": estimate_scale_class_from_component(c),
        "bbox_x": c.get("bbox_x", np.nan),
        "bbox_y": c.get("bbox_y", np.nan),
        "bbox_w": c.get("bbox_w", np.nan),
        "bbox_h": c.get("bbox_h", np.nan),
    }


def compute_motion_confidence(c):
    """
    Stage1 可靠度 Q_mhi，範圍約 0~1。
    不是用來保證正確，而是給 handoff 使用：
    n 越貼近方向線、面積越合理、step 合理、強度越明顯，信心越高。
    """
    n = abs(float(c.get("n", 999.0)))
    area = float(c.get("area", 0.0))
    step = abs(float(c.get("step", 0.0)))
    inten = math.log1p(max(0.0, float(c.get("intensity_sum", 0.0))))
    t_len = float(c.get("t_len", 0.0))

    n_score = max(0.0, 1.0 - n / max(1.0, LINE_BAND_TRACK))
    area_score = 1.0 if 2.0 <= area <= SMALL_BLOB_AREA_MAX else max(0.0, 1.0 - abs(area - SMALL_BLOB_AREA_MAX) / max(1.0, SMALL_BLOB_AREA_MAX))
    step_score = max(0.0, min(1.0, step / max(1.0, TRACK_STEP_MIN_INIT)))
    inten_score = max(0.0, min(1.0, inten / 8.0))
    # blur 長度太誇張會扣一點，但擊球初期允許一定長條
    tlen_score = max(0.0, 1.0 - max(0.0, t_len - 120.0) / 200.0)

    q = 0.30 * n_score + 0.20 * area_score + 0.20 * step_score + 0.15 * inten_score + 0.15 * tlen_score
    return float(max(0.0, min(1.0, q)))


def track_along_direction(frames_gray, frame_names, p0_frame, p0, d, start, end, p0_r=np.nan):
    results = []
    all_components = []
    for i in range(len(frames_gray)):
        results.append({
            "frame": i,
            "x": np.nan,
            "y": np.nan,
            "t": np.nan,
            "n": np.nan,
            "step": np.nan,
            "area": np.nan,
            "t_len": np.nan,
            "component_type": "",
            "mode": "MISS",
            "cost": np.nan,
            "motion_conf": np.nan,
            "est_scale": np.nan,
            "est_radius": np.nan,
            "blur_ratio": np.nan,
            "bbox_aspect": np.nan,
            "scale_class": "",
            "bbox_x": np.nan,
            "bbox_y": np.nan,
            "bbox_w": np.nan,
            "bbox_h": np.nan
        })
        all_components.append([])

    results[p0_frame].update({
        "x": float(p0[0]),
        "y": float(p0[1]),
        "t": 0.0,
        "n": 0.0,
        "step": 0.0,
        "area": 0.0,
        "t_len": 0.0,
        "component_type": "INIT",
        "mode": "INIT_GT",
        "cost": 0.0,
        "motion_conf": 1.0,
        "est_scale": p0_r if np.isfinite(p0_r) else np.nan,
        "est_radius": p0_r if np.isfinite(p0_r) else np.nan,
        "blur_ratio": 0.0,
        "bbox_aspect": 1.0,
        "scale_class": "static",
        "bbox_x": np.nan,
        "bbox_y": np.nan,
        "bbox_w": np.nan,
        "bbox_h": np.nan
    })

    launch_frame, launch_energy = detect_launch_frame(frames_gray, p0, d, start, end)
    if launch_frame is None:
        print("[WARN] Direction-constrained launch frame not confirmed. Stage1 returns INIT only.")
        return results, all_components, launch_frame, launch_energy

    print(f"[INFO] Direction-constrained launch frame = {launch_frame}, P0 local energy = {launch_energy[launch_frame]:.1f}")

    # ---------------------------------------------------------
    # Pre-launch hold state:
    # Before the confirmed launch frame, the ball is assumed to remain at P0.
    # This avoids unnecessary NaN values in the static interval and makes the
    # Stage1 trajectory CSV consistent with the physical interpretation.
    # These PRE_LAUNCH points are not used as tracking evidence or handoff points.
    # ---------------------------------------------------------
    prelaunch_end = min(max(p0_frame + 1, 0), len(results))
    for f_pre in range(p0_frame + 1, max(p0_frame + 1, launch_frame)):
        if 0 <= f_pre < len(results):
            results[f_pre].update({
                "x": float(p0[0]),
                "y": float(p0[1]),
                "t": 0.0,
                "n": 0.0,
                "step": 0.0,
                "area": 0.0,
                "t_len": 0.0,
                "component_type": "STATIC",
                "mode": "PRE_LAUNCH",
                "cost": 0.0,
                "motion_conf": 1.0,
                "est_scale": p0_r if np.isfinite(p0_r) else np.nan,
                "est_radius": p0_r if np.isfinite(p0_r) else np.nan,
                "blur_ratio": 0.0,
                "bbox_aspect": 1.0,
                "scale_class": "static",
                "bbox_x": np.nan,
                "bbox_y": np.nan,
                "bbox_w": np.nan,
                "bbox_h": np.nan
            })

    prev_t = 0.0
    prev_n = None
    expected_step = None
    miss_count = 0
    started = False
    track_count = 0

    # ---------------------------------------------------------
    # mhi_8：launch 初始點用短時間 lookahead 決定。
    # 這可以同時處理：
    # - 有些影片第一幀應選較前方 blur/球點
    # - 有些影片第一幀不能只用 intensity 或單幀 cost
    # ---------------------------------------------------------
    init_path, comps_cache = select_initial_path_with_lookahead(
        frames_gray, p0, d, launch_frame, min(end, len(frames_gray)-1)
    )

    last_seed_frame = launch_frame - 1
    if init_path:
        prev_record = None
        for k, (f, c) in enumerate(init_path):
            all_components[f] = comps_cache.get(f, all_components[f])
            chosen = c.copy()
            if prev_record is None:
                chosen["step"] = float(chosen["t"])
                mode = "TRACK_INIT"
            else:
                chosen["step"] = float(chosen["t"]) - float(prev_record["t"])
                mode = "TRACK"
            chosen["select_cost"] = -_initial_path_score(init_path[:k+1])

            motion_conf = compute_motion_confidence(chosen)
            results[f].update({
                "x": chosen["x"],
                "y": chosen["y"],
                "t": chosen["t"],
                "n": chosen["n"],
                "step": chosen["step"],
                "area": chosen["area"],
                "t_len": chosen["t_len"],
                "component_type": chosen["type"],
                "mode": mode,
                "cost": chosen["select_cost"],
                "motion_conf": motion_conf,
                **component_scale_fields(chosen)
            })

            if prev_record is None:
                expected_step = max(TRACK_STEP_MIN, chosen["step"] * EXPECTED_STEP_DECAY)
            else:
                if expected_step is None:
                    expected_step = max(TRACK_STEP_MIN, chosen["step"])
                else:
                    expected_step = max(TRACK_STEP_MIN, 0.45 * expected_step + 0.55 * chosen["step"])

            prev_t = chosen["t"]
            prev_n = chosen["n"]
            prev_record = chosen
            started = True
            track_count += 1
            last_seed_frame = f

        print(f"[INFO] Initial lookahead selected {len(init_path)} points, continue from frame {last_seed_frame + 1}")

    loop_start = max(launch_frame, last_seed_frame + 1)

    for f in range(loop_start, min(end, len(frames_gray)-1) + 1):
        comps = extract_lineband_components(frames_gray, f, p0, d)
        all_components[f] = comps

        chosen = choose_component_for_frame(
            comps,
            prev_t,
            expected_step,
            is_initial=(not started),
            prev_n=prev_n,
            track_count=track_count
        )

        if chosen is None:
            miss_count += 1
            if started and miss_count >= MAX_TRACK_MISS:
                print(f"[INFO] Stop track at frame {f}, miss_count={miss_count}")
                break
            continue

        motion_conf = compute_motion_confidence(chosen)
        results[f].update({
            "x": chosen["x"],
            "y": chosen["y"],
            "t": chosen["t"],
            "n": chosen["n"],
            "step": chosen["step"],
            "area": chosen["area"],
            "t_len": chosen["t_len"],
            "component_type": chosen["type"],
            "mode": "TRACK_INIT" if not started else "TRACK",
            "cost": chosen["select_cost"],
            "motion_conf": motion_conf,
            **component_scale_fields(chosen)
        })

        if not started:
            expected_step = max(TRACK_STEP_MIN, chosen["step"] * EXPECTED_STEP_DECAY)
            started = True
        else:
            if track_count < EARLY_TRACK_FRAMES:
                expected_step = max(TRACK_STEP_MIN, 0.45 * expected_step + 0.55 * chosen["step"])
            else:
                expected_step = max(TRACK_STEP_MIN, 0.75 * expected_step + 0.25 * chosen["step"])

        prev_t = chosen["t"]
        prev_n = chosen["n"]
        miss_count = 0
        track_count += 1

    return results, all_components, launch_frame, launch_energy


# =========================================================
# Save / draw
# =========================================================
def save_motion_curve_csv(frame_names, motion, smooth, peak, start, end, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_idx", "frame_name", "motion_count", "motion_smooth", "is_peak", "in_window"])
        for i in range(len(motion)):
            wr.writerow([i, frame_names[i], float(motion[i]), float(smooth[i]), 1 if i == peak else 0, 1 if start <= i <= end else 0])


def save_results_csv(frame_names, results, gt, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "frame_idx", "frame_name",
            "pred_x", "pred_y",
            "gt_x", "gt_y", "gt_r", "error_px",
            "t", "n", "step", "area", "t_len", "component_type", "mode", "cost", "motion_conf",
            "est_scale", "est_radius", "blur_ratio", "bbox_aspect", "scale_class",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h"
        ])
        for i, r in enumerate(results):
            name = frame_names[i]
            gt_item = gt.get(name, gt.get(i, None))
            gt_x = gt_y = gt_r = ""
            err = ""
            if gt_item is not None:
                gt_x, gt_y, gt_r = gt_item
                if np.isfinite(r["x"]) and np.isfinite(r["y"]):
                    err = math.hypot(float(r["x"]) - gt_x, float(r["y"]) - gt_y)

            wr.writerow([
                i, name,
                r["x"], r["y"],
                gt_x, gt_y, gt_r, err,
                r["t"], r["n"], r["step"], r["area"], r["t_len"], r["component_type"], r["mode"], r["cost"],
                r.get("motion_conf", np.nan), r.get("est_scale", np.nan), r.get("est_radius", np.nan),
                r.get("blur_ratio", np.nan), r.get("bbox_aspect", np.nan), r.get("scale_class", ""),
                r.get("bbox_x", np.nan), r.get("bbox_y", np.nan), r.get("bbox_w", np.nan), r.get("bbox_h", np.nan)
            ])


def save_components_csv(frame_names, all_components, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "frame_idx", "frame_name", "x", "y", "t", "n",
            "t_min", "t_max", "t_len", "area", "type", "intensity_sum",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h"
        ])
        for i, comps in enumerate(all_components):
            for c in comps:
                wr.writerow([
                    i, frame_names[i], c["x"], c["y"], c["t"], c["n"],
                    c["t_min"], c["t_max"], c["t_len"], c["area"], c["type"], c["intensity_sum"],
                    c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"]
                ])


def save_launch_energy_csv(frame_names, energy, launch_frame, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_idx", "frame_name", "p0_local_energy", "is_launch_frame"])
        for i, e in enumerate(energy):
            wr.writerow([i, frame_names[i], float(e), 1 if (launch_frame is not None and i == launch_frame) else 0])


def save_direction_hypotheses_csv(hyps, out_path):
    if not hyps:
        return
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "score", "hits", "misses", "p1_frame", "p1_x", "p1_y", "p1_dist0", "mean_line", "mean_angle", "direction_angle"])
        for rank, h in enumerate(hyps, start=1):
            p1 = h["p1"]
            ang = math.degrees(math.atan2(h["direction"][1], h["direction"][0]))
            wr.writerow([rank, h["score"], h["hits"], h["misses"], p1["frame"], p1["cx"], p1["cy"], p1["dist0"], h["mean_line"], h["mean_angle"], ang])


def draw_debug(frame_names, frames_bgr, results, gt, p0, d, start, end, launch_frame):
    out_dir = os.path.join(OUT_DIR, "debug_frames")
    os.makedirs(out_dir, exist_ok=True)

    d = normalize(d)
    p1_line = p0 - d * 2000
    p2_line = p0 + d * 2000

    for i, img in enumerate(frames_bgr):
        vis = img.copy()
        h, w = vis.shape[:2]

        if start <= i <= end:
            cv2.putText(vis, "DIR WINDOW", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if i == launch_frame:
            cv2.putText(vis, "LAUNCH", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 2)

        # direction line
        cv2.line(vis, (int(p1_line[0]), int(p1_line[1])), (int(p2_line[0]), int(p2_line[1])), (0, 0, 255), 2)

        # band lines
        nvec = np.array([-d[1], d[0]], dtype=np.float32)
        for offset in [-LINE_BAND_TRACK, LINE_BAND_TRACK]:
            a = p0 - d * 2000 + nvec * offset
            b = p0 + d * 2000 + nvec * offset
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (80, 80, 255), 1)

        r = results[i]
        if np.isfinite(r["x"]) and np.isfinite(r["y"]):
            px, py = int(round(r["x"])), int(round(r["y"]))
            cv2.circle(vis, (px, py), 7, (0, 255, 255), -1)
            cv2.putText(vis, r["mode"], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        gt_item = gt.get(frame_names[i], gt.get(i, None))
        if gt_item is not None:
            gx, gy, gr = gt_item
            rr = int(round(gr)) if np.isfinite(gr) else 6
            cv2.circle(vis, (int(round(gx)), int(round(gy))), max(4, rr), (0, 255, 0), 2)
            cv2.putText(vis, "GT", (int(gx) + 8, int(gy) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.putText(vis, f"Frame {i:04d}", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, frame_names[i]), vis)


# =========================================================
# Main
# =========================================================
def main():
    frame_names, frames_bgr, frames_gray = load_frames(FRAMES_DIR)
    gt = read_gt_csv(GT_CSV)
    print(f"[INFO] Loaded {len(frames_gray)} frames")

    p0_frame, p0, p0_r = get_first_gt(gt, frame_names)
    print(f"[INFO] Initial GT P0: frame={p0_frame}, point=({p0[0]:.1f},{p0[1]:.1f}), r={p0_r}")

    start, end, motion, smooth, peak = find_effective_window(frames_gray, p0_frame)
    print(f"[INFO] motion peak frame = {peak}, value = {smooth[peak]:.1f}")
    print(f"[INFO] direction effective window = {start} ~ {end}")
    print(f"[INFO] Prior direction cone: center={PRIOR_CENTER_DEG} deg, half_angle=±{PRIOR_HALF_ANGLE_DEG} deg")

    # Step A: 只估方向
    d, direction_blobs, hyps = estimate_direction_multihypothesis(frames_gray, p0_frame, p0, start, end)
    angle = math.degrees(math.atan2(d[1], d[0]))
    print(f"[INFO] Estimated direction d=({d[0]:.3f},{d[1]:.3f}), angle={angle:.1f} deg")

    # Step B: 沿方向線帶，用 frame difference 抽軌跡點
    results, all_components, launch_frame, launch_energy = track_along_direction(frames_gray, frame_names, p0_frame, p0, d, start, end, p0_r)

    if SAVE_CSV:
        if SAVE_AUX_CSV:
            save_motion_curve_csv(frame_names, motion, smooth, peak, start, end, os.path.join(OUT_DIR, "motion_energy.csv"))
            save_direction_hypotheses_csv(hyps, os.path.join(OUT_DIR, "direction_hypotheses.csv"))
            save_launch_energy_csv(frame_names, launch_energy, launch_frame, os.path.join(OUT_DIR, "p0_launch_energy.csv"))
        save_results_csv(frame_names, results, gt, os.path.join(OUT_DIR, "trajectory_results.csv"))
        if SAVE_COMPONENTS_CSV:
            save_components_csv(frame_names, all_components, os.path.join(OUT_DIR, "lineband_components.csv"))

    if SAVE_DEBUG_IMAGES:
        draw_debug(frame_names, frames_bgr, results, gt, p0, d, start, end, launch_frame)

    hit_count = sum(1 for r in results if r["mode"] in ["INIT_GT", "TRACK_INIT", "TRACK"])
    print(f"[INFO] trajectory hit count = {hit_count}")
    print(f"[INFO] Done. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
