import sys
import os
import hashlib
from pathlib import Path

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import joblib
from PIL import Image
from torchvision import transforms

from backend.config import EFFICIENTNET_WEIGHTS_PATH, CALIBRATOR_PATH, LOCKED_DIAGNOSTIC_THRESHOLD
from backend.model.diagnostic_model import get_model
from backend.preprocessing.nail_detection import NailDetector
from backend.preprocessing.image_quality import assess_image_quality

def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()

def get_image_file_sha256(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return compute_sha256(f.read())

def main():
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Default 5 images
        image_paths = [
            "data/samples/real_index_nail.jpg",
            "data/samples/real_koilonychia_anemia.jpg",
            "data/test_images/01_known_anemia_nail.jpg",
            "data/test_images/02_known_healthy_nail.jpg",
            "data/test_images/03_normal_index_nail.jpg",
        ]

    debug_roi_dir = Path("reports/debug_constant_prediction")
    debug_roi_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Calibrator
    device = torch.device("cpu")
    model = get_model("efficientnet_b0", pretrained=False)
    ckpt = torch.load(EFFICIENTNET_WEIGHTS_PATH, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    calibrator = joblib.load(CALIBRATOR_PATH)
    detector = NailDetector()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("=" * 110)
    print("AI NAIL INFERENCE PIPELINE COMPONENT AUDIT TABLE")
    print("=" * 110)
    print(f"{'Image':<28} | {'Orig Dim':<10} | {'ROI BBox':<22} | {'ROI Mean (RGB)':<18} | {'Raw Logit':<16} | {'Sigmoid':<10} | {'Calib Prob':<10} | {'State':<10}")
    print("-" * 110)

    detailed_records = []

    for idx, path_str in enumerate(image_paths):
        p = Path(path_str)
        if not p.exists():
            print(f"File not found: {p}")
            continue

        img_hash = get_image_file_sha256(str(p))
        img = Image.open(p).convert("RGB")
        w, h = img.size

        # ROI Crop
        roi_img, bbox, roi_meta = detector.detect_and_crop(img)
        roi_bytes = roi_img.tobytes()
        roi_hash = compute_sha256(roi_bytes)

        # Save debug ROI
        saved_roi_path = debug_roi_dir / f"roi_{idx+1}_{p.stem}.png"
        roi_img.save(saved_roi_path)

        roi_np = np.array(roi_img)
        roi_mean_rgb = [round(float(m), 2) for m in np.mean(roi_np, axis=(0, 1))]
        roi_std_rgb = [round(float(s), 2) for s in np.std(roi_np, axis=(0, 1))]

        # Preprocessing Tensor
        tensor = transform(roi_img).unsqueeze(0).to(device)
        tensor_bytes = tensor.numpy().tobytes()
        tensor_hash = compute_sha256(tensor_bytes)
        t_mean = float(tensor.mean())
        t_std = float(tensor.std())
        t_min = float(tensor.min())
        t_max = float(tensor.max())

        # EfficientNet-B0 Raw Output
        with torch.no_grad():
            output = model(tensor)
            raw_logit = output.squeeze().item()
            raw_sigmoid = 1.0 / (1.0 + np.exp(-raw_logit))

        # Calibrator
        cal_input = raw_sigmoid
        cal_output = float(calibrator.calibrate(np.array([cal_input]))[0])
        final_prob = round(max(0.001, min(0.999, cal_output)), 3)
        final_state = "ANEMIA" if final_prob >= LOCKED_DIAGNOSTIC_THRESHOLD else "NO_ANEMIA"

        record = {
            "index": idx + 1,
            "filename": p.name,
            "img_hash": img_hash,
            "dimensions": f"{w}x{h}",
            "bbox": str(bbox),
            "roi_hash": roi_hash,
            "roi_mean": roi_mean_rgb,
            "roi_std": roi_std_rgb,
            "tensor_hash": tensor_hash,
            "tensor_mean": round(t_mean, 6),
            "tensor_std": round(t_std, 6),
            "tensor_min": round(t_min, 6),
            "tensor_max": round(t_max, 6),
            "raw_logit": raw_logit,
            "raw_sigmoid": raw_sigmoid,
            "cal_input": cal_input,
            "cal_output": cal_output,
            "final_prob": final_prob,
            "final_state": final_state,
        }
        detailed_records.append(record)

        print(f"{p.name:<28} | {w}x{h:<7} | {str(bbox):<22} | {str(roi_mean_rgb):<18} | {raw_logit:<16.12f} | {raw_sigmoid:<10.6f} | {final_prob:<10.3f} | {final_state:<10}")

    print("=" * 110)
    print("\nDETAILED PER-IMAGE AUDIT BREAKDOWN:")
    for r in detailed_records:
        print(f"\n--- [Image {r['index']}: {r['filename']}] ---")
        print(f"  Input File SHA-256:       {r['img_hash']}")
        print(f"  Input Dimensions:         {r['dimensions']}")
        print(f"  ROI Bounding Box:         {r['bbox']}")
        print(f"  ROI Image SHA-256:        {r['roi_hash']}")
        print(f"  ROI Mean RGB:             {r['roi_mean']}")
        print(f"  ROI Std RGB:              {r['roi_std']}")
        print(f"  Tensor SHA-256:           {r['tensor_hash']}")
        print(f"  Tensor Stats (min/max):   [{r['tensor_min']}, {r['tensor_max']}] | mean={r['tensor_mean']}, std={r['tensor_std']}")
        print(f"  Raw CNN Logit:            {r['raw_logit']:.12f}")
        print(f"  Raw Sigmoid Probability:  {r['raw_sigmoid']:.12f}")
        print(f"  Calibrator Input:         {r['cal_input']:.12f}")
        print(f"  Calibrator Output:        {r['cal_output']:.12f}")
        print(f"  Final Clamped Prob:       {r['final_prob']}")
        print(f"  Final Decision State:     {r['final_state']}")

if __name__ == "__main__":
    main()
