
# training_files — How to run everything

This folder contains:
- **CNN/torch training** scripts for **FER2013 emotion**, **UTKFace gender**, and **UTKFace age**
- **YOLOv8** data-prep + training scripts (separate section below)
- plotting utilities (confusion matrices + training curves)

---

# Dataset Links

FER_2013 - https://www.kaggle.com/datasets/msambare/fer2013

UTKFace - https://susanqq.github.io/UTKFace/

UTKFace (Cropped) - https://www.kaggle.com/datasets/jangedoo/utkface-new

---

## 0) File map (what exists here)

Top-level scripts:
- `train_expr_fer.py` — train emotion model on FER2013 
- `train_gender_utk.py` — train gender model on UTKFace 
- `train_age_utk.py` — train age model on UTKFace 
- `datasets_fer2013.py` — FER dataset loader 
- `datasets_utkface_2.py` — UTKFace dataset loader
- `models_common.py` — shared model/training utilities 
- `confusion_plot.py` — confusion matrix plotting 
- `plot.py` — training curve plotting 
- `plot_mae.py` — MAE-related plotting
- `requirements.txt` — python deps for this folder

YOLO section:
- `YOLO/prepare_fer2013_yolo.py`
- `YOLO/prepare_utkface_yolo_mae.py`
- `YOLO/train_yolo_emotion.py`
- `YOLO/train_yolo_gender.py`
- Example output runs (already in repo):
  - `YOLO/y8n_emotion19/...` (includes `weights/best.pt`, `results.csv`, images, etc.)
  - `YOLO/y8n_agegender_gender20/...` (includes `weights/best.pt`, `results.csv`, images, etc.)

---

## 1) Environment setup

From the **repo root**:

```bash
# (recommended) create a venv
python -m venv .venv
# activate:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

# install deps for these training scripts
pip install -r training_files/requirements.txt
````

If you plan to run YOLO training, you will also need the **Ultralytics** stack 

---

## 2) REQUIRED directory layout (datasets)

### 2.1 FER2013 layout (used by CNN training AND YOLO prep)

The YOLO prep script explicitly expects FER images in this layout:

```text
fer/
  train/
    angry/
      *.jpg
    disgust/
      *.jpg
    fear/
      *.jpg
    happy/
      *.jpg
    neutral/
      *.jpg
    sad/
      *.jpg
    surprise/
      *.jpg
  val/
    angry/
      *.jpg
    ... same 7 class folders ...
```

The emotion labels are the 7-class set:
`["angry","disgust","fear","happy","neutral","sad","surprise"]`

**Where to place `fer/`:**

* Safest: put it so the scripts can find it via a relative path (common pattern is `training_files/fer/...` or repo-root `fer/...`).
* If a script crashes with “path not found”, run it with `--help` and pass the dataset root explicitly, or move the dataset folder to match the script’s expected relative location.

### 2.2 UTKFace layout (used by age/gender CNN scripts + YOLO prep)

The file list shows dedicated UTKFace loaders/trainers: `datasets_utkface_2.py`, `train_age_utk.py`, `train_gender_utk.py`, and `YOLO/prepare_utkface_yolo_mae.py`.
* Put UTKFace images under a single folder (e.g., `utkface/`), then run the scripts with `--help` to see the expected `--data_root` / `--utk_root` argument (or adjust the variable at the top of the script).

A common UTKFace download is a flat directory of images like:

```text
utkface/
  UTKFace/
    <age>_<gender>_<race>_<date>.jpg
    ...
```

---

## 3) How to run each NON-YOLO file

> Run these from **repo root** unless your script uses relative paths that assume `cd training_files` first. If you see missing-file errors, try `cd training_files` and rerun.

### 3.1 `train_expr_fer.py` (CNN emotion training)

Example usage using resent18:

```bash
python train_expr_fer_store.py --fer ./fer2013 --epochs 30 --batch 128 --backbone resnet18 --pretrained --device cuda

```

What it needs:

* FER folder present in the expected location (see FER layout above).

What it produces:

* model checkpoints
* training logs/metrics (used later by `plot.py` and `confusion_plot.py`)

### 3.2 `train_gender_utk.py` (CNN gender training)

Example usage using resent18:

```bash
python train_gender_utk_store.py --epochs 30 --batch 128 --backbone resnet18 --pretrained --device cuda
```

What it needs:

* UTKFace images in the expected location (see UTK section), must be in a directory called `UTKFace`

### 3.3 `train_age_utk.py` (CNN age training)

Example usage using resent18

```bash
python train_age_utk_store.py --epochs 50 --batch 64 --backbone resnet18 --pretrained --freeze-backbone-epochs 1

```

### 3.4 `datasets_utkface_2.py` / `models_common.py`

These are imported by the training scripts; you generally **do not run them directly**.

### 3.5 `plot.py` (training curves)

```bash
python training_files/plot.py --help
python training_files/plot.py
```

Use after training if it expects a log path / CSV path.

### 3.6 `plot_mae.py`

```bash
python training_files/plot_mae.py --help
python training_files/plot_mae.py
```

### 3.7 `confusion_plot.py`

This is the only file that must be changed manually within the code, it must point directly to where the confusion matrix is in the file tree and must also be customized to where you would like the output image to be saved.

```bash
python training_files/confusion_plot.py
```

---

## 4) YOLOv8 section (data prep + training)

YOLO scripts live under `training_files/YOLO/`

### 4.1 Prepare FER2013 for YOLO (`prepare_fer2013_yolo.py`)

This script converts FER2013 into **YOLO detect format**, treating each image as a single “face” box and the **emotion** as the detection class.

Expected input layout is the `fer/train/<class>/*.jpg` and `fer/val/<class>/*.jpg` structure shown above.

Run:

```bash
python training_files/YOLO/prepare_fer2013_yolo.py --help
python training_files/YOLO/prepare_fer2013_yolo.py
```

> If it writes outputs to a default folder, keep that output folder path because `train_yolo_emotion.py` will need the YOLO dataset YAML or root it generates.

### 4.2 Prepare UTKFace for YOLO/MAE (`prepare_utkface_yolo_mae.py`)

Run:

```bash
python training_files/YOLO/prepare_utkface_yolo_mae.py --help
python training_files/YOLO/prepare_utkface_yolo_mae.py
```

### 4.3 Train YOLO emotion (`train_yolo_emotion.py`)

Run:

```bash
python training_files/YOLO/train_yolo_emotion.py --help
python training_files/YOLO/train_yolo_emotion.py
```

Outputs (example run already committed) include:

* `YOLO/y8n_emotion19/weights/best.pt`
* `YOLO/y8n_emotion19/results.csv`
* confusion matrices + sample batch images

### 4.4 Train YOLO gender (`train_yolo_gender.py`)

Run:

```bash
python training_files/YOLO/train_yolo_gender.py --help
python training_files/YOLO/train_yolo_gender.py
```

Outputs (example run already committed) include:

* `YOLO/y8n_agegender_gender20/weights/best.pt`
* `YOLO/y8n_agegender_gender20/results.csv`
* confusion matrices + sample batch images

---

## 5) Using the provided YOLO weights (optional)

Pretrained weights that are already in the repo:

* `training_files/YOLO/y8n_emotion19/weights/best.pt`
* `training_files/YOLO/y8n_agegender_gender20/weights/best.pt`

If you want quick inference with Ultralytics CLI (example pattern):

```bash
yolo predict model=training_files/YOLO/y8n_emotion19/weights/best.pt source=path/to/images
```

---

## 6) Common “it can’t find my data” fixes

1. **Run from the correct working directory**

   * Try:

     ```bash
     cd training_files
     python train_expr_fer.py
     ```
   * If that fails, go back to repo root and run with explicit paths.

2. **Confirm expected paths**

   * Always do:

     ```bash
     python <script>.py --help
     ```
   * If there are no CLI args, open the script and look for a `DATA_ROOT`, `FER_ROOT`, `UTK_ROOT`, etc.

3. **Match class folder names exactly**

   * FER class dirs must match the emotion names used by the YOLO prep script. 

