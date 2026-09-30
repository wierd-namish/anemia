import json
import time
import socket
import urllib.request
import urllib.parse
import mimetypes
import uuid

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def post_multipart(url, file_path, field_name="file"):
    boundary = uuid.uuid4().hex
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    filename = file_path.split("/")[-1].split("\\")[-1]
    
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode())
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def get_json(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def test_runtime():
    base_url = "http://127.0.0.1:8000"
    print("=" * 60)
    print("1. HEALTH CHECK (GET /health)")
    print("=" * 60)
    t0 = time.perf_counter()
    status, data = get_json(f"{base_url}/health")
    dt = (time.perf_counter() - t0) * 1000
    print(f"Status: {status} ({dt:.2f} ms)")
    print(json.dumps(data, indent=2))
    assert status == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

    print("\n" + "=" * 60)
    print("2. MODEL INFO (GET /model-info)")
    print("=" * 60)
    status, data = get_json(f"{base_url}/model-info")
    print(f"Status: {status}")
    print(json.dumps(data, indent=2))
    assert status == 200
    assert data["model_name"] == "EfficientNet-B0"
    assert data["locked_threshold"] == 0.48

    print("\n" + "=" * 60)
    print("3. ACTUAL REAL IMAGE INFERENCE (POST /predict)")
    print("=" * 60)
    nail_img_path = "data/samples/real_koilonychia_anemia.jpg"
    t0 = time.perf_counter()
    status, resp_json = post_multipart(f"{base_url}/predict", nail_img_path)
    dt_total = (time.perf_counter() - t0) * 1000
    print(f"File: {nail_img_path}")
    print(f"Status: {status}")
    print(f"Total Client Latency: {dt_total:.2f} ms")
    print(json.dumps(resp_json, indent=2))
    assert status == 200

    print("\n" + "=" * 60)
    print("4. INVALID IMAGE TESTS (Must be INCONCLUSIVE)")
    print("=" * 60)
    invalid_samples = [
        ("wood_desk", "test_ood/ood_wood_desk.jpg"),
        ("skin_only", "test_ood/ood_skin_only.jpg"),
        ("blurry_image", "test_ood/ood_blurred_surface.jpg"),
        ("overexposed", "test_ood/ood_overexposed_image.jpg"),
    ]
    for label, path in invalid_samples:
        t0 = time.perf_counter()
        status, data = post_multipart(f"{base_url}/predict", path)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[{label}] -> State: {data.get('state')} | Reason: {data.get('description')} ({dt:.1f}ms)")
        assert data.get("state") == "INCONCLUSIVE", f"Expected INCONCLUSIVE for {label}, got {data.get('state')}"

    print("\n" + "=" * 60)
    print("5. DETAILED LATENCY BREAKDOWN (Over 10 Passes on Real Nail)")
    print("=" * 60)
    latencies = []
    for i in range(10):
        t0 = time.perf_counter()
        post_multipart(f"{base_url}/predict", nail_img_path)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    print(f"Average Total Latency: {avg_lat:.2f} ms")
    print(f"Min Latency: {min_lat:.2f} ms | Max Latency: {max_lat:.2f} ms")

    print("\n" + "=" * 60)
    print("6. LOCAL LAN NETWORK INFO")
    print("=" * 60)
    local_ip = get_local_ip()
    print(f"Local LAN IPv4: {local_ip}")
    print(f"Mobile Browser URL: http://{local_ip}:8000")

if __name__ == "__main__":
    test_runtime()
