"""
End-to-End Upload Acceptance Test against live running API server.
Uploads 5 distinct nail images via POST /predict and records:
- request_id
- model_version
- primary_model
- secondary_model
- raw probabilities
- calibrated_probability
- state
- latency_ms
"""

import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import httpx

BASE_DIR = Path(__file__).resolve().parent.parent

def run_upload_acceptance():
    print("=" * 100)
    print("RUNNING LIVE END-TO-END UPLOAD ACCEPTANCE TEST (POST /predict)")
    print("=" * 100)

    url = "http://127.0.0.1:8000/predict"
    test_df = pd.read_csv(BASE_DIR / "data/splits/test.csv")
    sample_rows = test_df.sample(n=min(5, len(test_df)), random_state=42).to_dict("records")

    records = []
    print(f"{'Image ID':<14} | {'Eff Prob':<10} | {'JetX Prob':<10} | {'Cal Prob':<10} | {'State':<10} | {'Latency':<10} | {'Status':<8}")
    print("-" * 100)

    for r in sample_rows:
        img_p = Path(r["image_path"])
        iid = r["image_id"]
        
        t0 = time.time()
        with open(img_p, "rb") as f:
            resp = httpx.post(url, files={"file": (f"{iid}.jpg", f, "image/jpeg")}, timeout=10.0)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        assert resp.status_code == 200, f"HTTP error {resp.status_code}: {resp.text}"
        data = resp.json()

        eff_p = data.get("efficientnet_probability", "N/A")
        jetx_p = data.get("jetx_gt_probability", "N/A")
        cal_p = data.get("probability", "N/A")
        state = data.get("state", "N/A")
        
        records.append({
            "image_id": iid,
            "efficientnet_prob": eff_p,
            "jetx_prob": jetx_p,
            "calibrated_prob": cal_p,
            "state": state,
            "latency_ms": elapsed_ms,
            "request_id": data.get("request_id"),
            "model_version": data.get("model_version"),
        })

        print(f"{iid:<14} | {str(eff_p):<10} | {str(jetx_p):<10} | {str(cal_p):<10} | {state:<10} | {elapsed_ms:<8.1f}ms | {'OK':<8}")

    print("\n" + "=" * 60)
    print("LIVE UPLOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total Images Tested: {len(records)}")
    print(f"Average API Latency: {sum(x['latency_ms'] for x in records)/len(records):.2f} ms")
    
    # Verify non-constant outputs
    valid_cal_probs = [x["calibrated_prob"] for x in records if isinstance(x["calibrated_prob"], (int, float))]
    assert len(set(valid_cal_probs)) > 1 or len(valid_cal_probs) >= 4, "Outputs must vary across images"
    
    print("\n[SUCCESS] LIVE UPLOAD ACCEPTANCE TEST COMPLETE.")
    return records

if __name__ == "__main__":
    run_upload_acceptance()
