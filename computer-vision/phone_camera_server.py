"""Phone camera capture server.

Run this on the computer, open the shown HTTPS URL on a phone on the same
network, then use the phone browser camera to stream frames or save a capture
into ``computer-vision/captures``.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from start_stream import _load_face_detector, _make_square_face_shoulder_bbox


SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = SCRIPT_DIR / "captures"
LATEST_FRAME_PATH = CAPTURES_DIR / "phone_latest.jpg"
CERT_PATH = SCRIPT_DIR / ".phone-camera-cert.pem"
KEY_PATH = SCRIPT_DIR / ".phone-camera-key.pem"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8443
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Picasso Phone Camera</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f4f4f5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: 1fr auto;
      background: #101114;
    }
    main {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 10px;
    }
    video {
      width: 100%;
      max-width: 900px;
      max-height: calc(100vh - 145px);
      aspect-ratio: 3 / 4;
      object-fit: cover;
      background: #050506;
      border: 1px solid #30323a;
    }
    footer {
      display: grid;
      gap: 10px;
      padding: 12px 12px max(34px, env(safe-area-inset-bottom));
      border-top: 1px solid #2a2c33;
      background: #17191f;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      min-height: 48px;
      border: 1px solid #3a3d46;
      border-radius: 8px;
      background: #242732;
      color: #f4f4f5;
      font: inherit;
      font-weight: 650;
    }
    #capture {
      grid-column: 1 / -1;
      min-height: 64px;
      font-size: 18px;
    }
    button.primary { background: #2f6fed; border-color: #5d91ff; }
    button.danger { background: #8d2b2b; border-color: #c25757; }
    button:disabled { opacity: 0.5; }
    body.capture-flash::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: rgba(34, 197, 94, 0.42);
      animation: flash-capture 420ms ease-out;
      z-index: 10;
    }
    @keyframes flash-capture {
      from { opacity: 1; }
      to { opacity: 0; }
    }
    .status {
      min-height: 22px;
      color: #c8cad2;
      font-size: 14px;
      line-height: 1.4;
    }
    canvas { display: none; }
  </style>
</head>
<body>
  <main>
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas"></canvas>
  </main>
  <footer>
    <div class="controls">
      <button id="start" class="primary">Start Camera</button>
      <button id="record" disabled>Start Feed</button>
      <button id="capture" class="primary" disabled>Save Capture</button>
    </div>
    <div id="status" class="status">Open this page on your phone, then start the camera.</div>
  </footer>
  <script>
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const statusEl = document.getElementById("status");
    const startBtn = document.getElementById("start");
    const recordBtn = document.getElementById("record");
    const captureBtn = document.getElementById("capture");

    let stream = null;
    let facingMode = "user";
    let feedTimer = null;
    let sentFrames = 0;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function flashCapture() {
      document.body.classList.remove("capture-flash");
      void document.body.offsetWidth;
      document.body.classList.add("capture-flash");
      window.setTimeout(() => document.body.classList.remove("capture-flash"), 450);
    }

    async function startCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1280 },
          height: { ideal: 960 }
        },
        audio: false
      });
      video.srcObject = stream;
      recordBtn.disabled = false;
      captureBtn.disabled = false;
      setStatus("Camera ready.");
    }

    function frameBlob(quality = 0.86) {
      const w = video.videoWidth || 1280;
      const h = video.videoHeight || 960;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, w, h);
      return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
    }

    async function sendFrame(endpoint, quality) {
      const blob = await frameBlob(quality);
      if (!blob) {
        throw new Error("Could not encode frame.");
      }
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob
      });
      if (!res.ok) {
        let message = await res.text();
        try {
          const errorData = JSON.parse(message);
          message = errorData.message || errorData.error || message;
        } catch (_) {}
        throw new Error(message);
      }
      return res.json();
    }

    async function toggleFeed() {
      if (feedTimer) {
        clearInterval(feedTimer);
        feedTimer = null;
        recordBtn.textContent = "Start Feed";
        recordBtn.classList.remove("danger");
        setStatus(`Feed stopped. Sent ${sentFrames} frames.`);
        return;
      }

      sentFrames = 0;
      recordBtn.textContent = "Stop Feed";
      recordBtn.classList.add("danger");
      setStatus("Sending live frames...");
      feedTimer = setInterval(async () => {
        try {
          await sendFrame("/frame", 0.72);
          sentFrames += 1;
          setStatus(`Sending live frames... ${sentFrames}`);
        } catch (err) {
          setStatus(`Feed error: ${err.message}`);
        }
      }, 250);
    }

    startBtn.addEventListener("click", async () => {
      try {
        await startCamera();
      } catch (err) {
        setStatus(`Camera error: ${err.message}`);
      }
    });

    recordBtn.addEventListener("click", toggleFeed);

    captureBtn.addEventListener("click", async () => {
      flashCapture();
      try {
        const data = await sendFrame("/capture", 0.92);
        setStatus(`Saved ${data.path}`);
      } catch (err) {
        setStatus(`Retake: ${err.message}`);
      }
    });
  </script>
</body>
</html>
"""


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def make_capture_name() -> str:
    return datetime.now().strftime("phone_%Y%m%dT%H%M%S.jpg")


def _decode_jpeg(body: bytes):
    data = np.frombuffer(body, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _largest_face(
    detector: cv2.CascadeClassifier,
    image,
    min_size: tuple[int, int],
    min_neighbors: int = 3,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def _validated_capture_jpeg(body: bytes) -> tuple[bytes, dict]:
    frame = _decode_jpeg(body)
    if frame is None:
        raise ValueError("camera frame could not be read")

    detector = _load_face_detector()
    min_frame_side = min(frame.shape[:2])
    min_face_side = max(50, int(min_frame_side * 0.06))
    face = _largest_face(detector, frame, (min_face_side, min_face_side))
    if face is None:
        raise ValueError("no clear human face detected")

    _, _, face_w, face_h = face
    if min(face_w, face_h) < min_face_side:
        raise ValueError("detected face is too small")

    x1, y1, x2, y2 = _make_square_face_shoulder_bbox(face, frame.shape)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("face crop was empty")

    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise ValueError("could not encode face crop")

    return encoded.tobytes(), {
        "crop": [int(x1), int(y1), int(x2), int(y2)],
        "face": [int(v) for v in face],
    }


def ensure_self_signed_cert(cert_path: Path = CERT_PATH, key_path: Path = KEY_PATH) -> None:
    if cert_path.exists() and key_path.exists():
        return

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "365",
        "-nodes",
        "-subj",
        "/CN=picasso-phone-camera",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not create a self-signed HTTPS cert. Install openssl or run with --http."
        ) from exc


class PhoneCameraHandler(BaseHTTPRequestHandler):
    server_version = "PicassoPhoneCamera/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_text(self, status: int, text: str, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(200, INDEX_HTML, "text/html")
            return
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_text(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/frame", "/capture"}:
            self._send_text(404, "Not found")
            return

        content_type = self.headers.get("Content-Type", "")
        if "image/jpeg" not in content_type:
            self._send_text(415, "Expected Content-Type: image/jpeg")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_text(400, "Invalid Content-Length")
            return

        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send_text(413, "Invalid upload size")
            return

        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        body = self.rfile.read(length)

        if path == "/frame":
            LATEST_FRAME_PATH.write_bytes(body)
            self._send_json(
                200,
                {
                    "ok": True,
                    "path": str(LATEST_FRAME_PATH.relative_to(SCRIPT_DIR)),
                    "bytes": len(body),
                },
            )
            return

        try:
            capture_body, capture_meta = _validated_capture_jpeg(body)
        except ValueError as exc:
            self._send_json(
                422,
                {
                    "ok": False,
                    "message": f"{exc}; please retake in better light with your face centered.",
                },
            )
            return

        capture_path = CAPTURES_DIR / make_capture_name()
        capture_path.write_bytes(capture_body)
        self._send_json(
            200,
            {
                "ok": True,
                "path": str(capture_path.relative_to(SCRIPT_DIR)),
                "bytes": len(capture_body),
                **capture_meta,
            },
        )


def serve(host: str, port: int, use_https: bool) -> None:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), PhoneCameraHandler)

    scheme = "http"
    if use_https:
        ensure_self_signed_cert()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=CERT_PATH, keyfile=KEY_PATH)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    ip = local_ip()
    print("Phone camera server running.", flush=True)
    print(f"  Local:   {scheme}://127.0.0.1:{port}/", flush=True)
    print(f"  Phone:   {scheme}://{ip}:{port}/", flush=True)
    print(f"  Saves:   {CAPTURES_DIR}", flush=True)
    if use_https:
        print("  On the phone, accept the self-signed certificate warning once.", flush=True)
    else:
        print("  Warning: phone browsers may block camera access over plain HTTP.", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping phone camera server.")
    finally:
        httpd.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a phone-camera capture UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use plain HTTP instead of HTTPS. Most phone browsers block camera on HTTP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(args.host, args.port, use_https=not args.http)


if __name__ == "__main__":
    main()
