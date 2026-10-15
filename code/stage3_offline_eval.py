#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage3_offline_eval.py

Canonical offline Stage 3 evaluation used to reproduce the thesis/paper
descending-trajectory experiment (Table V).

IMPORTANT
---------
This version follows the original experiment logic that reproduces the
reported Table V values from results_summary_updated.

Input
-----
Either:
  1) results_summary_updated.zip
or
  2) the extracted results_summary_updated/ folder containing:
       02_all_stage2_lbp_frames.csv
       03_all_stage2_ltp_frames.csv
       04_all_stage2_weighted_frames.csv

Example
-------
python stage3_offline_eval.py ^
    --input "D:\\Thesis_Data\\results_summary_updated.zip" ^
    --output_dir "D:\\Thesis_Data\\stage3_offline_results"

Paper protocol
--------------
Reliable point:
    mode == MID_OK and online_reliable == 1

Robust apex detection:
    1. Find the minimum reliable predicted y.
    2. Treat reliable points within y_min + 2 px as the apex plateau.
    3. Keep the connected plateau around the minimum-y point
       (maximum reliable-frame gap = 3).
    4. Search backward from the end of that plateau and confirm descending
       motion with the following 3 reliable points:
         - at least 2 positive y increments
         - net y increase >= 0.8 px
    5. If no candidate satisfies the confirmation, use the last point
       in the plateau.

Eligibility:
    - at least 4 reliable fitting points from the detected apex onward
    - at least 4 GT frames from the detected apex onward
    - a video is included only if LBP, LTP, and CSWC are all eligible

Curve:
    x(t): first-order least squares
    y(t): second-order least squares
    using all reliable points from apex to the last GT frame.

Evaluation:
    For each appearance source, evaluate from that source's detected apex
    through its last valid GT frame.

Three outputs per appearance source:
    - Stage2
    - Curve by <source> (Keep reliable)
    - Curve by <source> (All curve)

Ground truth is used only for offline evaluation / interval end and is NOT
used to fit the curve or determine Stage 2 reliability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


VALID_VIDEOS = [
    "v0", "v4", "v10", "v12", "v21", "v22", "v23", "v24",
    "v31", "v32", "v33", "v34", "v35", "v37", "v40",
]

SOURCE_FILES = {
    "LBP": "02_all_stage2_lbp_frames.csv",
    "LTP": "03_all_stage2_ltp_frames.csv",
    "CSWC": "04_all_stage2_weighted_frames.csv",
}

# These values are part of the original offline apex-selection protocol.
APEX_DELTA_PX = 2.0
APEX_CONFIRM_N = 3
APEX_MIN_NET_Y_PX = 0.8
APEX_MAX_RELIABLE_GAP = 3

MIN_FIT_POINTS = 4
MIN_GT_FRAMES = 4


def to_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


def frame_number(name: str) -> int:
    digits = "".join(ch for ch in Path(str(name)).stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame number from: {name}")
    return int(digits)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item: dict[str, Any] = dict(row)
            item["frame_no"] = frame_number(
                item.get("frame", item.get("frame_name", ""))
            )
            for key in (
                "gt_x", "gt_y", "pred_x", "pred_y", "error_px",
                "online_reliable", "frame_idx",
            ):
                if key in item:
                    item[key] = to_float(item[key])
            rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["video"]), []).append(row)
    for video_rows in groups.values():
        video_rows.sort(key=lambda r: int(r["frame_no"]))
    return groups


def prepare_input(input_path: Path, work_dir: Path) -> Path:
    """
    Return the folder that directly contains the three consolidated Stage 2 CSVs.
    """
    if input_path.is_dir():
        if (input_path / SOURCE_FILES["LBP"]).is_file():
            return input_path

        nested = input_path / "results_summary_updated"
        if (nested / SOURCE_FILES["LBP"]).is_file():
            return nested

        raise FileNotFoundError(
            "Cannot find consolidated Stage 2 CSVs in the input folder."
        )

    if input_path.suffix.lower() != ".zip":
        raise ValueError(
            "--input must be results_summary_updated.zip or the extracted folder."
        )

    extract_dir = work_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)

    nested = extract_dir / "results_summary_updated"
    if (nested / SOURCE_FILES["LBP"]).is_file():
        return nested

    if (extract_dir / SOURCE_FILES["LBP"]).is_file():
        return extract_dir

    raise FileNotFoundError(
        "The ZIP does not contain the expected consolidated Stage 2 CSVs."
    )


def is_reliable(row: dict[str, Any]) -> bool:
    return (
        str(row.get("mode", "")).strip().upper() == "MID_OK"
        and int(row.get("online_reliable", 0) or 0) == 1
        and finite(row.get("pred_x"))
        and finite(row.get("pred_y"))
    )


def detect_apex(
    rows: list[dict[str, Any]],
    delta: float = APEX_DELTA_PX,
    confirm_n: int = APEX_CONFIRM_N,
    min_net: float = APEX_MIN_NET_Y_PX,
    max_gap: int = APEX_MAX_RELIABLE_GAP,
) -> dict[str, Any] | None:
    """
    Reproduce the robust apex detector used for the paper experiment.

    Image y increases downward.
    """
    gt_rows = [
        row for row in rows
        if finite(row.get("gt_x")) and finite(row.get("gt_y"))
    ]
    if not gt_rows:
        return None

    last_gt = max(int(row["frame_no"]) for row in gt_rows)

    reliable = [
        row for row in rows
        if is_reliable(row)
        and int(row["frame_no"]) <= last_gt
    ]
    if len(reliable) < MIN_FIT_POINTS:
        return None

    y_min = min(float(row["pred_y"]) for row in reliable)

    # Apex plateau = reliable points close to the minimum y.
    candidates = [
        row for row in reliable
        if float(row["pred_y"]) <= y_min + delta
    ]
    candidates.sort(key=lambda row: int(row["frame_no"]))
    if not candidates:
        return None

    min_row = min(reliable, key=lambda row: float(row["pred_y"]))
    candidate_frames = [int(row["frame_no"]) for row in candidates]
    center = candidate_frames.index(int(min_row["frame_no"]))

    # Retain the connected plateau containing the minimum-y point.
    left = center
    while (
        left > 0
        and candidate_frames[left] - candidate_frames[left - 1] <= max_gap
    ):
        left -= 1

    right = center
    while (
        right < len(candidate_frames) - 1
        and candidate_frames[right + 1] - candidate_frames[right] <= max_gap
    ):
        right += 1

    plateau = candidates[left:right + 1]

    # Choose the last plateau point before a confirmed descending trend.
    chosen = None
    reliable_frames = [int(row["frame_no"]) for row in reliable]

    for candidate in sorted(
        plateau,
        key=lambda row: int(row["frame_no"]),
        reverse=True,
    ):
        position = reliable_frames.index(int(candidate["frame_no"]))
        following = reliable[position + 1:position + 1 + confirm_n]

        if len(following) < confirm_n:
            continue

        ys = (
            [float(candidate["pred_y"])]
            + [float(row["pred_y"]) for row in following]
        )
        differences = np.diff(np.asarray(ys, dtype=float))

        if (
            int(np.sum(differences > 0)) >= 2
            and ys[-1] - ys[0] >= min_net
        ):
            chosen = candidate
            break

    if chosen is None:
        chosen = plateau[-1]

    apex_frame = int(chosen["frame_no"])

    fit_points = [
        row for row in reliable
        if int(row["frame_no"]) >= apex_frame
    ]
    eval_rows = [
        row for row in gt_rows
        if int(row["frame_no"]) >= apex_frame
    ]

    return {
        "apex_frame": apex_frame,
        "last_gt": last_gt,
        "fit_n": len(fit_points),
        "gt_n": len(eval_rows),
        "y_min": y_min,
        "plateau_start": int(plateau[0]["frame_no"]),
        "plateau_end": int(plateau[-1]["frame_no"]),
    }


def fit_curve(
    rows: list[dict[str, Any]],
    apex_frame: int,
    fit_end: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    """
    Paper model:
      x(t): degree 1
      y(t): degree 2
    """
    reliable = [
        row for row in rows
        if is_reliable(row)
        and int(row["frame_no"]) >= apex_frame
        and int(row["frame_no"]) <= fit_end
    ]

    if len(reliable) < MIN_FIT_POINTS:
        return None

    t = np.asarray(
        [int(row["frame_no"]) - apex_frame for row in reliable],
        dtype=float,
    )
    xs = np.asarray([float(row["pred_x"]) for row in reliable], dtype=float)
    ys = np.asarray([float(row["pred_y"]) for row in reliable], dtype=float)

    coef_x = np.polyfit(t, xs, 1)
    coef_y = np.polyfit(t, ys, 2)

    return coef_x, coef_y, reliable


def evaluate_curve(
    rows: list[dict[str, Any]],
    apex_frame: int,
    coefficients: tuple[np.ndarray, np.ndarray, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    coef_x, coef_y, _ = coefficients
    output: list[dict[str, Any]] = []

    for row in rows:
        frame = int(row["frame_no"])

        if frame < apex_frame:
            continue
        if not finite(row.get("gt_x")) or not finite(row.get("gt_y")):
            continue

        t = frame - apex_frame
        curve_x = float(np.polyval(coef_x, t))
        curve_y = float(np.polyval(coef_y, t))

        gx = float(row["gt_x"])
        gy = float(row["gt_y"])
        sx = float(row["pred_x"])
        sy = float(row["pred_y"])

        stage2_error = math.hypot(sx - gx, sy - gy)
        curve_error = math.hypot(curve_x - gx, curve_y - gy)

        reliable = is_reliable(row)

        if reliable:
            keep_x, keep_y = sx, sy
            keep_source = "STAGE2_RELIABLE"
        else:
            keep_x, keep_y = curve_x, curve_y
            keep_source = "CURVE_RECOVERY"

        keep_error = math.hypot(keep_x - gx, keep_y - gy)

        output.append({
            "video": row["video"],
            "frame_no": frame,
            "gt_x": gx,
            "gt_y": gy,
            "stage2_x": sx,
            "stage2_y": sy,
            "stage2_mode": row.get("mode", ""),
            "stage2_reliable": int(reliable),
            "stage2_error": stage2_error,
            "curve_x": curve_x,
            "curve_y": curve_y,
            "all_curve_error": curve_error,
            "keep_x": keep_x,
            "keep_y": keep_y,
            "keep_source": keep_source,
            "keep_error": keep_error,
        })

    return output


def metrics(errors: list[float]) -> dict[str, float | int]:
    values = np.asarray(
        [value for value in errors if finite(value)],
        dtype=float,
    )

    if len(values) == 0:
        return {
            "Frames": 0,
            "Mean Error": math.nan,
            "Median Error": math.nan,
            "RMSE": math.nan,
            "Max Error": math.nan,
            "Success@2 px": math.nan,
            "Success@5 px": math.nan,
        }

    return {
        "Frames": int(len(values)),
        "Mean Error": float(np.mean(values)),
        "Median Error": float(np.median(values)),
        "RMSE": float(np.sqrt(np.mean(values * values))),
        "Max Error": float(np.max(values)),
        "Success@2 px": float(np.mean(values <= 2.0) * 100.0),
        "Success@5 px": float(np.mean(values <= 5.0) * 100.0),
    }


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stage3_offline_") as tmp:
        data_dir = prepare_input(input_path, Path(tmp))

        source_groups = {
            source: group_by_video(read_rows(data_dir / filename))
            for source, filename in SOURCE_FILES.items()
        }

        # -------------------------------------------------------------
        # 1. Robust apex / eligibility
        # -------------------------------------------------------------
        apex_map: dict[tuple[str, str], dict[str, Any] | None] = {}
        eligible_sets: dict[str, set[str]] = {}

        for source, groups in source_groups.items():
            eligible: set[str] = set()

            for video in VALID_VIDEOS:
                info = detect_apex(groups.get(video, []))
                apex_map[(source, video)] = info

                if (
                    info is not None
                    and info["fit_n"] >= MIN_FIT_POINTS
                    and info["gt_n"] >= MIN_GT_FRAMES
                ):
                    eligible.add(video)

            eligible_sets[source] = eligible

        common_videos = [
            video for video in VALID_VIDEOS
            if all(
                video in eligible_sets[source]
                for source in SOURCE_FILES
            )
        ]

        eligibility_rows: list[dict[str, Any]] = []

        for video in VALID_VIDEOS:
            row: dict[str, Any] = {
                "Video": video,
                "Included in paper 9-row comparison": (
                    "Yes" if video in common_videos else "No"
                ),
            }

            for source in SOURCE_FILES:
                info = apex_map[(source, video)]
                row[f"{source} ApexFrame"] = (
                    "" if info is None else info["apex_frame"]
                )
                row[f"{source} PlateauStart"] = (
                    "" if info is None else info["plateau_start"]
                )
                row[f"{source} PlateauEnd"] = (
                    "" if info is None else info["plateau_end"]
                )
                row[f"{source} FitReliablePoints"] = (
                    0 if info is None else info["fit_n"]
                )
                row[f"{source} EvalGTFrames"] = (
                    0 if info is None else info["gt_n"]
                )

            reasons = []
            for source in SOURCE_FILES:
                info = apex_map[(source, video)]

                if info is None:
                    reasons.append(f"{source}: apex unavailable")
                elif info["fit_n"] < MIN_FIT_POINTS:
                    reasons.append(
                        f"{source}: only {info['fit_n']} post-apex reliable points"
                    )
                elif info["gt_n"] < MIN_GT_FRAMES:
                    reasons.append(
                        f"{source}: only {info['gt_n']} post-apex GT frames"
                    )

            row["Exclusion reason"] = (
                "" if video in common_videos else "; ".join(reasons)
            )
            eligibility_rows.append(row)

        # -------------------------------------------------------------
        # 2. Paper 9-row comparison
        # -------------------------------------------------------------
        overall_rows: list[dict[str, Any]] = []
        per_video_rows: list[dict[str, Any]] = []
        per_frame_rows: list[dict[str, Any]] = []

        for source, groups in source_groups.items():
            pooled_stage2: list[float] = []
            pooled_keep: list[float] = []
            pooled_all: list[float] = []

            # Per-video data cache for this source.
            evaluated_by_video: dict[str, list[dict[str, Any]]] = {}

            for video in common_videos:
                info = apex_map[(source, video)]
                assert info is not None

                rows = groups[video]

                coefficients = fit_curve(
                    rows=rows,
                    apex_frame=info["apex_frame"],
                    fit_end=info["last_gt"],
                )
                assert coefficients is not None

                evaluated = evaluate_curve(
                    rows=rows,
                    apex_frame=info["apex_frame"],
                    coefficients=coefficients,
                )
                evaluated_by_video[video] = evaluated

                coef_x, coef_y, reliable_fit = coefficients

                for item in evaluated:
                    item_out = dict(item)
                    item_out.update({
                        "method_source": source,
                        "apex_frame": info["apex_frame"],
                        "last_gt_frame": info["last_gt"],
                        "fit_reliable_points": len(reliable_fit),
                        "x_a1": float(coef_x[0]),
                        "x_a0": float(coef_x[1]),
                        "y_b2": float(coef_y[0]),
                        "y_b1": float(coef_y[1]),
                        "y_b0": float(coef_y[2]),
                    })
                    per_frame_rows.append(item_out)

                stage2_errors = [r["stage2_error"] for r in evaluated]
                keep_errors = [r["keep_error"] for r in evaluated]
                all_errors = [r["all_curve_error"] for r in evaluated]

                pooled_stage2.extend(stage2_errors)
                pooled_keep.extend(keep_errors)
                pooled_all.extend(all_errors)

                for label, errs in (
                    (source, stage2_errors),
                    (f"Curve by {source} (Keep reliable)", keep_errors),
                    (f"Curve by {source} (All curve)", all_errors),
                ):
                    m = metrics(errs)
                    per_video_rows.append({
                        "Method": label,
                        "Video": video,
                        "ApexFrame": info["apex_frame"],
                        "EvalFrames": m["Frames"],
                        "FitReliablePoints": len(reliable_fit),
                        "MeanError": m["Mean Error"],
                        "MedianError": m["Median Error"],
                        "RMSE": m["RMSE"],
                        "MaxError": m["Max Error"],
                        "Success@2px": m["Success@2 px"],
                        "Success@5px": m["Success@5 px"],
                    })

            # Preserve the ordering used in the paper.
            for label, errs in (
                (source, pooled_stage2),
                (f"Curve by {source} (Keep reliable)", pooled_keep),
                (f"Curve by {source} (All curve)", pooled_all),
            ):
                m = metrics(errs)
                overall_rows.append({
                    "Method": label,
                    "Videos": len(common_videos),
                    "Frames": m["Frames"],
                    "Mean Error": m["Mean Error"],
                    "Median Error": m["Median Error"],
                    "RMSE": m["RMSE"],
                    "Max Error": m["Max Error"],
                    "Success@2 px": m["Success@2 px"],
                    "Success@5 px": m["Success@5 px"],
                })

        # Reorder rows exactly as the thesis/paper table:
        method_order = [
            "LBP",
            "LTP",
            "CSWC",
            "Curve by LBP (Keep reliable)",
            "Curve by LTP (Keep reliable)",
            "Curve by CSWC (Keep reliable)",
            "Curve by LBP (All curve)",
            "Curve by LTP (All curve)",
            "Curve by CSWC (All curve)",
        ]
        order_map = {name: i for i, name in enumerate(method_order)}
        overall_rows.sort(key=lambda r: order_map[r["Method"]])
        per_video_rows.sort(
            key=lambda r: (
                order_map[r["Method"]],
                VALID_VIDEOS.index(r["Video"]),
            )
        )

        # -------------------------------------------------------------
        # 3. Write outputs
        # -------------------------------------------------------------
        write_csv(
            output_dir / "01_stage3_paper_summary.csv",
            overall_rows,
        )
        write_csv(
            output_dir / "02_stage3_per_video.csv",
            per_video_rows,
        )
        write_csv(
            output_dir / "03_stage3_eligibility.csv",
            eligibility_rows,
        )
        write_csv(
            output_dir / "04_stage3_per_frame.csv",
            per_frame_rows,
        )

        metadata = {
            "input": str(input_path),
            "sources": SOURCE_FILES,
            "valid_videos": VALID_VIDEOS,
            "common_eligible_videos": common_videos,
            "common_eligible_count": len(common_videos),
            "reliability": "mode == MID_OK and online_reliable == 1",
            "apex_detection": {
                "delta_px": APEX_DELTA_PX,
                "confirm_n": APEX_CONFIRM_N,
                "min_net_y_px": APEX_MIN_NET_Y_PX,
                "max_reliable_frame_gap": APEX_MAX_RELIABLE_GAP,
            },
            "eligibility": {
                "min_fit_points": MIN_FIT_POINTS,
                "min_gt_frames": MIN_GT_FRAMES,
                "all_three_sources_required": True,
            },
            "curve": {
                "x_degree": 1,
                "y_degree": 2,
                "fit_points": "all reliable observations from detected apex through last GT frame",
            },
            "evaluation": {
                "start": "source-specific detected apex",
                "end": "last valid GT frame",
                "aggregation": "pooled per-frame errors across the common eligible videos",
            },
        }

        with (output_dir / "stage3_offline_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("=" * 72)
        print("Stage 3 offline evaluation finished")
        print("=" * 72)
        print(f"Input:  {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Common eligible videos ({len(common_videos)}):")
        print(", ".join(common_videos))
        print()
        print("Paper summary:")
        for row in overall_rows:
            print(
                f"{row['Method']:<38s} "
                f"N={row['Frames']:>3d}  "
                f"Mean={row['Mean Error']:.6f}  "
                f"Median={row['Median Error']:.6f}  "
                f"RMSE={row['RMSE']:.6f}  "
                f"S@2={row['Success@2 px']:.6f}%  "
                f"S@5={row['Success@5 px']:.6f}%"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the thesis/paper Stage 3 offline descending-trajectory experiment."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="results_summary_updated.zip or extracted results_summary_updated folder",
    )
    parser.add_argument(
        "--output_dir",
        default="stage3_offline_results",
        help="Output directory (default: ./stage3_offline_results)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        Path(args.input).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
