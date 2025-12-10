import argparse, csv, random, shutil
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
PART_DIR_NAMES = ['part1','part2','part3']
EMIT_NAMES = ['male', 'female']

import re

def parse_age_gender(name: str):
    """
    Robust to any number of underscores or stray characters.
    Extracts the first numeric token as age, then the first 0/1 token after that as gender.
    Works for:
      25_0_2_201701....jpg
      53__0_201701....jpg
      53_1__3_2017...jpg
    """
    stem = name.rsplit('.', 1)[0]
    nums = re.findall(r'\d+', stem)   # grab ALL digit runs
    if len(nums) < 2:
        raise ValueError(f"cannot find age+gender in {name}")
    age = int(nums[0])

    gender = None
    for tok in nums[1:]:
        if tok in ('0', '1'):
            gender = int(tok)
            break
    if gender is None:
        raise ValueError(f"no gender 0/1 token found in {name}")
    return age, gender



def collect_images(utk_root: Path):
    items = []
    for part in PART_DIR_NAMES:
        pdir = utk_root/part
        if not pdir.exists(): continue
        for p in pdir.rglob('*'):
            if p.suffix.lower() in IMG_EXTS:
                items.append((part, p))
    if not items:
        raise FileNotFoundError(f"No images found under {utk_root} in {PART_DIR_NAMES}")
    return items

def ensure_dirs(base: Path):
    (base/'train'/'images').mkdir(parents=True, exist_ok=True)
    (base/'train'/'labels').mkdir(parents=True, exist_ok=True)
    (base/'val'/'images').mkdir(parents=True, exist_ok=True)
    (base/'val'/'labels').mkdir(parents=True, exist_ok=True)

def write_pair(src_path: Path, split: str, out_root: Path, new_name: str, gender: int):
    img_dst = out_root/split/'images'/new_name
    lbl_dst = out_root/split/'labels'/(new_name.rsplit('.',1)[0] + '.txt')
    shutil.copy2(src_path, img_dst)
    with open(lbl_dst, 'w', encoding='utf-8') as f:
        # full-image YOLO label: cls cx cy w h
        f.write(f"{gender} 0.5 0.5 1.0 1.0\n")
    return img_dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--utk', required=True, help='Root containing part1/ part2/ part3 (images only)')
    ap.add_argument('--out', default='datasets/utk_yolo')
    ap.add_argument('--yaml', default='cfg/data_agegender.yaml')
    ap.add_argument('--ages_csv', default='datasets/utk_yolo/ages.csv')
    ap.add_argument('--val_pct', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    utk = Path(args.utk)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_root)

    items = collect_images(utk)
    # parse metadata
    rows = []
    parsed = []
    for part, p in items:
        try:
            age, gender = parse_age_gender(p.name)
            parsed.append((part, p, age, gender))
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    # shuffle and split
    random.seed(args.seed)
    random.shuffle(parsed)
    n = len(parsed)
    n_val = max(1, int(round(args.val_pct * n))) if n > 0 else 0
    val_set = set(range(n_val))

    for idx, (part, p, age, gender) in enumerate(parsed):
        # avoid collisions by prefixing filename with part
        new_name = f"{part}_{p.name}"
        split = 'val' if idx in val_set else 'train'
        img_dst = write_pair(p, split, out_root, new_name, gender)
        rel_path = (img_dst.relative_to(out_root.parent)).as_posix()  # e.g., train/images/part1_xxx.jpg
        rows.append((rel_path, age))

    # write ages.csv
    ages_csv = Path(args.ages_csv)
    ages_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(ages_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rel_path','age'])
        w.writerows(rows)

    # write YOLO yaml
    yaml_text = f"""# gender detection (full-image box) + ages.csv for MAE
    path: {out_root.resolve()}
    train: train/images
    val: val/images
    names: {EMIT_NAMES}
    """
    Path(args.yaml).write_text(yaml_text, encoding='utf-8')

    # quick counts
    n_tr = sum(1 for _ in (out_root/'train'/'images').rglob('*'))
    n_va = sum(1 for _ in (out_root/'val'/'images').rglob('*'))
    print(f"[done] train images: {n_tr} | val images: {n_va}")
    print("Wrote:", args.yaml, "and", ages_csv)

if __name__ == '__main__':
    main()
