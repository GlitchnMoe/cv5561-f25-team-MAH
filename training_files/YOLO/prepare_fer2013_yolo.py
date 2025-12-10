"""
Prepare FER2013 into YOLO-detect format where each image contains a single face
and the emotion is the detection class.

Expected FER folder structure is:
  fer/
    train/
      emotion_label_name/
        *.jpg
    val/
      emotion_label_name/
        *.jpg

"""
import argparse, os, pathlib, shutil
from pathlib import Path

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def write_yolo_split(in_dir: Path, out_dir: Path):
    (out_dir/'images').mkdir(parents=True, exist_ok=True)
    (out_dir/'labels').mkdir(parents=True, exist_ok=True)
    for cls_idx, cls_name in enumerate(EMOTIONS):
        src = in_dir/cls_name
        if not src.exists(): 
            print(f"[warn] missing class folder:", src)
            continue
        for p in src.rglob('*'):
            if p.suffix.lower() not in ['.jpg','.jpeg','.png','.bmp']: continue
            # copy image
            rel = p.name
            dst_img = out_dir/'images'/rel
            shutil.copy2(p, dst_img)
            # write full-image box label
            # YOLO format: class cx cy w h (normalized)
            # full box is centered (0.5,0.5) with size (1.0,1.0)
            dst_lbl = out_dir/'labels'/(p.stem+'.txt')
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fer', type=str, required=True, help='Path to FER root having train/ and val/')
    ap.add_argument('--out', type=str, default='datasets/fer_yolo')
    ap.add_argument('--yaml', type=str, default='cfg/data_emotion.yaml')
    args = ap.parse_args()
    fer = Path(args.fer)
    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)

    write_yolo_split(fer/'train', out/'train')
    write_yolo_split(fer/'test', out/'val')


    # write Ultralytics yaml
    yaml = f"""# emotion detection (full-image box)
    path: {out.resolve()}
    train: train/images
    val: val/images
    names: {EMOTIONS}
    """
    Path(args.yaml).write_text(yaml, encoding='utf-8')
    print("Wrote", args.yaml)

if __name__ == '__main__':
    main()
