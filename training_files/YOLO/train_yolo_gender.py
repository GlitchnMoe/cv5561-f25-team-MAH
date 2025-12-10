import argparse, os, sys, subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to data directory')
    ap.add_argument('--model', default='yolov8n-cls.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--name', default='y8n_agegender_gender')
    args = ap.parse_args()
    cmd = [
        'yolo', 'task=classify', 'mode=train', f'data={args.data}', f'model={args.model}',
        f'imgsz={args.imgsz}', f'epochs={args.epochs}', f'batch={args.batch}',
        f'device={args.device}', f'project=runs_agegender', f'name={args.name}',
        'amp=True'
    ]
    print('Running:', ' '.join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == '__main__':
    main()
