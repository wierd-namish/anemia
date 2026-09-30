"""
Live Demo Verification Script.
Tests:
1. Health and Model Info
2. Nail Image 1 (Anemic)
3. Nail Image 2 (Non-Anemic)
4. OOD Distractor 1 (Blurred surface)
5. OOD Distractor 2 (Random object)
"""

import urllib.request
import mimetypes
import uuid
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def check_endpoints():
    print("=" * 60)
    print("1. CHECKING /health & /model-info")
    print("=" * 60)
    
    with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
        health = json.loads(resp.read().decode())
        print("HEALTH RESPONSE:")
        print(json.dumps(health, indent=2))
        
    with urllib.request.urlopen(f"{BASE_URL}/model-info") as resp:
        info = json.loads(resp.read().decode())
        print("\nMODEL INFO RESPONSE:")
        print(json.dumps(info, indent=2))
        
    return health, info

def post_image(img_path):
    path = Path(img_path)
    boundary = uuid.uuid4().hex
    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    
    with open(path, 'rb') as f:
        img_bytes = f.read()
        
    mime_type = mimetypes.guess_type(str(path))[0] or 'image/png'
    
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f'Content-Type: {mime_type}\r\n\r\n'
    ).encode('utf-8') + img_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(f"{BASE_URL}/predict", data=body, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        
    # Print clean version without truncating key data
    display_data = dict(data)
    if 'roi_image_base64' in display_data and display_data['roi_image_base64']:
        display_data['roi_image_base64'] = display_data['roi_image_base64'][:35] + f"... [Length: {len(data['roi_image_base64'])} bytes]"
        
    return data, display_data

def run_all_tests():
    health, info = check_endpoints()
    
    test_cases = [
        ("TEST 1: Positive Fingernail Image", "Fingernails/Anemic-Fin-008 (10).png"),
        ("TEST 2: Negative Fingernail Image", "Fingernails/Non-Anrmic-FN-002 (2).png"),
        ("TEST 3: OOD Blurry Image", "test_ood/ood_blurred_surface.jpg"),
        ("TEST 4: OOD Random Object", "test_ood/ood_random_object.jpg"),
        ("TEST 5: Second Negative Fingernail Image", "Fingernails/Non-Anrmic-FN-002 (3).png"),
    ]
    
    results = {}
    for title, path in test_cases:
        print("\n" + "=" * 60)
        print(f"{title}: {path}")
        print("=" * 60)
        raw, display = post_image(path)
        print(json.dumps(display, indent=2))
        results[title] = raw
        
    return results

if __name__ == "__main__":
    run_all_tests()
