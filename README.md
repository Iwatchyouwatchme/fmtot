# FMTOT — Knowledge-Guided Tracking of Fast-Moving Tiny Objects under Motion Blur

A Golf-Ball Case Study.

**Author:** Royal Shih, National Central University

**Project page:** https://iwatchyouwatchme.github.io/fmtot/

A three-stage, knowledge-guided pipeline for tracking a fast-moving, tiny golf ball
under severe motion blur:

1. **Stage 1 — motion-guided launch tracking.** Estimates launch direction from a
   Motion History Image and multi-hypothesis search, then grows an initial trajectory.
2. **Stage 2 — appearance-guided tiny-object tracking.** Normalized multi-cue candidate
   scoring with three texture descriptors: LBP, LTP, and CSWC (the `weighted` variant in code).
3. **Stage 3 — selective trajectory recovery.** Least-squares fit of a linear-x /
   parabolic-y motion model over reliable observations.

Stages are connected by a knowledge-guided handoff (Compactness Transition Score).

## Results (summary)

| Stage | Best method | Mean error | Success @ 5 px |
|---|---|---|---|
| Stage 1 vs. baselines | Proposed | 2.19 px | 96.4% (100% completion, 15/15) |
| Stage 2 (888 frames) | LTP | 1.28 px | 97.9% |
| Stage 3 offline (~472 frames) | LTP | 1.11 px | 99.2% |

Stage 1 → Stage 2 handoff: 15/15 valid, mean 0.81 px. Full tables in [`summary/`](summary/).

## Repository layout

```
code/      pipeline source (Python + OpenCV + NumPy)
summary/   result tables (paper_summary/ and raw_summary/)
index.html GitHub Pages project page
```

The Stage-3 motion model: `x(t) = a1*t + a0`, `y(t) = b2*t^2 + b1*t + b0`,
fitted only on reliable (MID_OK) Stage 2 points from the image-plane apex onward.

## Dataset

The full dataset (`Thesis_Data.zip`, ~5.6 GB) holds the raw videos, per-frame images,
ground-truth `annotations.csv`, and per-video results for 21 sequences. Because it exceeds
GitHub's per-file limit, it is attached to [**GitHub Releases**](https://github.com/Iwatchyouwatchme/fmtot/releases/latest)
as three split parts (each < 2 GB). Download all parts, then rejoin:

```bash
# Windows
copy /b Thesis_Data_part_00 + Thesis_Data_part_01 + Thesis_Data_part_02 Thesis_Data.zip
# macOS / Linux
cat Thesis_Data_part_* > Thesis_Data.zip
```

Mirror (single file) on Google Drive:
**https://drive.google.com/file/d/1gHv1lzOXS9tcFB7K17-iWNxIwfYTJEUJ/view?usp=sharing**

The `code/` and `summary/` folders in this repository mirror `02_code/` and `03_summary/`
from that archive. See [`DATA_README.txt`](DATA_README.txt) for the original dataset notes.

## Usage

```bash
# general tracking, run inside a video folder containing frames/ and annotations.csv
python code/run_pipeline.py

# offline Stage 3 evaluation (reads existing Stage 2 outputs, reproduces Table V)
python code/stage3_offline_eval.py
```
