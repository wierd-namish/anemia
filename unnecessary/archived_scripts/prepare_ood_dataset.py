"""
Generates synthetic and realistic Out-Of-Distribution (OOD) test images in test_ood/
to evaluate model rejection and abstention behavior.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def generate_ood_samples(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Wood background / Desk texture
    arr_wood = np.zeros((300, 300, 3), dtype=np.uint8)
    arr_wood[:, :, 0] = 139
    arr_wood[:, :, 1] = 69
    arr_wood[:, :, 2] = 19
    noise = np.random.randint(-15, 15, (300, 300, 3), dtype=np.int16)
    arr_wood = np.clip(arr_wood.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr_wood).save(output_dir / "ood_wood_desk.jpg")
    
    # 2. Skin only (flat dorsal forearm patch without nail)
    arr_skin = np.zeros((300, 300, 3), dtype=np.uint8)
    arr_skin[:, :, 0] = 185
    arr_skin[:, :, 1] = 135
    arr_skin[:, :, 2] = 110
    noise_skin = np.random.normal(0, 4, (300, 300, 3))
    arr_skin = np.clip(arr_skin + noise_skin, 0, 255).astype(np.uint8)
    Image.fromarray(arr_skin).save(output_dir / "ood_skin_only.jpg")
    
    # 3. Clothing fabric / Blue denim
    arr_fabric = np.zeros((300, 300, 3), dtype=np.uint8)
    arr_fabric[:, :, 0] = 30
    arr_fabric[:, :, 1] = 60
    arr_fabric[:, :, 2] = 160
    # Add weave pattern
    for i in range(300):
        if i % 4 == 0:
            arr_fabric[i, :, :] = np.clip(arr_fabric[i, :, :] + 25, 0, 255)
    Image.fromarray(arr_fabric).save(output_dir / "ood_clothing_fabric.jpg")
    
    # 4. Random object (white ceramic coffee mug on black table)
    arr_obj = np.zeros((300, 300, 3), dtype=np.uint8)
    arr_obj[50:250, 80:220] = 235
    Image.fromarray(arr_obj).save(output_dir / "ood_random_object.jpg")
    
    # 5. Blurred surface (featureless grey)
    arr_blur = np.full((300, 300, 3), 150, dtype=np.uint8)
    Image.fromarray(arr_blur).save(output_dir / "ood_blurred_surface.jpg")
    
    # 6. Overexposed image (white washout)
    arr_over = np.full((300, 300, 3), 245, dtype=np.uint8)
    Image.fromarray(arr_over).save(output_dir / "ood_overexposed_image.jpg")
    
    # 7. Underexposed image (pitch black / near black)
    arr_under = np.full((300, 300, 3), 20, dtype=np.uint8)
    Image.fromarray(arr_under).save(output_dir / "ood_underexposed_image.jpg")
    
    print(f"✓ Generated 7 OOD test images in {output_dir}")

if __name__ == "__main__":
    generate_ood_samples(Path(__file__).resolve().parent.parent / "test_ood")
