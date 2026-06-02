"""Calibration entrypoint: serve a phone UI, capture 4 corner photos,
extract X/Y/Z via OpenAI vision, write the canvas calibration JSON, exit.

State machine (sequential, advanced by the phone capture button):
    INIT  ->  TL  ->  TR  ->  BR  ->  BL  ->  DONE  ->  (exit)

Each corner capture is saved to ``calibration/captures/<corner>.jpg`` and
its extracted coordinates are stored in
``robot/canvas_calibration.json`` under ``raw_corners_xyz`` once all four
have been captured. The existing ``stroke_plane_offset_xyz_m`` field, if
present, is preserved.

The robot-side state machine (moving the arm through the same corners)
is intentionally NOT part of this script.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from extract_coordinates import extract_xyz_from_image


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CAPTURES_DIR = SCRIPT_DIR / "captures"

# Reuse the existing phone_camera_server cert files so:
#   (a) we don't have to generate a fresh self-signed cert from this script
#       (which fails inside Git Bash on Windows because MSYS rewrites the
#       openssl `-subj "/CN=..."` argument into a Windows path),
#   (b) the phone only has to accept one self-signed cert across both
#       servers.
COMPUTER_VISION_DIR = REPO_ROOT / "computer-vision"
CERT_PATH = COMPUTER_VISION_DIR / ".phone-camera-cert.pem"
KEY_PATH = COMPUTER_VISION_DIR / ".phone-camera-key.pem"

CANVAS_CALIBRATION_PATH = REPO_ROOT / "robot" / "canvas_calibration.json"

CORNERS = ["TL", "TR", "BR", "BL"]

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8444
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


# ---------------------------------------------------------------------------
# Shared state (mutated under a lock by the HTTP handler)
# ---------------------------------------------------------------------------
class CalibrationState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.results: dict[str, dict] = {}  # {"TL": {"xyz": [x,y,z], "image": "TL.jpg"}, ...}
        self.done_event = threading.Event()

    def current_corner(self) -> Optional[str]:
        with self.lock:
            for corner in CORNERS:
                if corner not in self.results:
                    return corner
            return None

    def progress(self) -> dict:
        with self.lock:
            ordered = []
            for corner in CORNERS:
                entry = self.results.get(corner)
                ordered.append({
                    "corner": corner,
                    "captured": entry is not None,
                    "xyz": entry["xyz"] if entry else None,
                })
            current = next((c for c in CORNERS if c not in self.results), None)
            return {
                "current": current,
                "index": (len(self.results) if current is None else CORNERS.index(current)),
                "total": len(CORNERS),
                "done": current is None,
                "corners": ordered,
            }

    def record(self, corner: str, xyz: tuple[float, float, float], image_name: str) -> None:
        with self.lock:
            self.results[corner] = {"xyz": list(xyz), "image": image_name}
            if len(self.results) == len(CORNERS):
                self.done_event.set()

    def clear(self, corner: str) -> None:
        with self.lock:
            self.results.pop(corner, None)
            self.done_event.clear()


STATE = CalibrationState()


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Canvas calibration</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; background: #0c0d10; color: #f4f4f5; display: flex; flex-direction: column; }
    header { padding: 16px 20px; border-bottom: 1px solid #25282e; }
    h1 { margin: 0; font-size: 16px; font-weight: 500; color: #a1a1aa; letter-spacing: 0.05em; text-transform: uppercase; }
    #corner { font-size: 56px; font-weight: 700; margin-top: 4px; letter-spacing: 0.1em; }
    #sub { font-size: 14px; color: #a1a1aa; margin-top: 6px; }
    main { flex: 1; display: flex; flex-direction: column; padding: 16px 20px; gap: 12px; }
    #video-wrap { position: relative; background: #000; border-radius: 12px; overflow: hidden; aspect-ratio: 3/4; }
    video, canvas { width: 100%; height: 100%; object-fit: cover; display: block; }
    canvas { display: none; }
    #status { font-size: 14px; color: #a1a1aa; min-height: 1.2em; }
    #status.err { color: #f87171; }
    #last { background: #16181d; border-radius: 12px; padding: 12px 14px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: #d4d4d8; white-space: pre-wrap; word-break: break-word; }
    .progress { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
    .chip { background: #16181d; border-radius: 8px; padding: 8px 6px; text-align: center; font-size: 12px; color: #a1a1aa; border: 1px solid #25282e; }
    .chip.done { background: #143a25; color: #a7f3d0; border-color: #14532d; }
    .chip.current { background: #1e293b; color: #93c5fd; border-color: #1d4ed8; }
    footer { padding: 16px 20px; border-top: 1px solid #25282e; display: grid; gap: 10px; }
    button { font: inherit; padding: 14px 20px; border-radius: 12px; border: none; cursor: pointer; font-size: 16px; font-weight: 600; }
    button.primary { background: #4ade80; color: #052e16; }
    button.primary:disabled { background: #3f3f46; color: #71717a; cursor: not-allowed; }
    button.secondary { background: #27272a; color: #f4f4f5; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .done-screen { padding: 32px 20px; text-align: center; }
    .done-screen h2 { font-size: 28px; color: #4ade80; }
  </style>
</head>
<body>
  <header>
    <h1>Canvas Calibration</h1>
    <div id="corner">--</div>
    <div id="sub">Position the robot end-effector at this corner and frame its display in view.</div>
  </header>
  <main>
    <div class="progress" id="progress"></div>
    <div id="video-wrap">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas"></canvas>
    </div>
    <div id="status">Initialising camera...</div>
    <div id="last" style="display:none"></div>
  </main>
  <footer>
    <div class="row">
      <button class="secondary" id="retake" disabled>Retake last</button>
      <button class="primary" id="capture" disabled>Capture</button>
    </div>
  </footer>

  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture');
    const retakeBtn = document.getElementById('retake');
    const statusEl = document.getElementById('status');
    const lastEl = document.getElementById('last');
    const cornerEl = document.getElementById('corner');
    const progressEl = document.getElementById('progress');

    let currentCorner = null;
    let lastCaptured = null;
    let busy = false;

    function setStatus(msg, isErr) {
      statusEl.textContent = msg;
      statusEl.classList.toggle('err', !!isErr);
    }

    function renderProgress(state) {
      progressEl.innerHTML = '';
      state.corners.forEach(c => {
        const div = document.createElement('div');
        div.className = 'chip' + (c.captured ? ' done' : '') + (c.corner === state.current ? ' current' : '');
        div.textContent = c.corner + (c.captured ? ' \u2713' : '');
        progressEl.appendChild(div);
      });
    }

    async function refreshState() {
      const r = await fetch('/state');
      const state = await r.json();
      renderProgress(state);
      currentCorner = state.current;
      if (state.done) {
        document.body.innerHTML = '<div class="done-screen"><h2>All 4 corners captured.</h2><p>You can close this tab. Calibration JSON has been written.</p></div>';
        return;
      }
      cornerEl.textContent = state.current + '  (' + (state.index + 1) + '/' + state.total + ')';
      captureBtn.disabled = busy;
      retakeBtn.disabled = busy || state.index === 0;
    }

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        video.srcObject = stream;
        setStatus('Camera ready.');
      } catch (err) {
        setStatus('Camera error: ' + err.message, true);
      }
    }

    function captureFrame() {
      const w = video.videoWidth, h = video.videoHeight;
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, w, h);
      return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    }

    async function doCapture() {
      if (busy || !currentCorner) return;
      busy = true;
      captureBtn.disabled = true;
      retakeBtn.disabled = true;
      setStatus('Capturing ' + currentCorner + ' ...');
      try {
        const blob = await captureFrame();
        const fd = new FormData();
        fd.append('photo', blob, currentCorner + '.jpg');
        fd.append('corner', currentCorner);
        const r = await fetch('/capture', { method: 'POST', body: fd });
        const result = await r.json();
        if (!r.ok || !result.ok) throw new Error(result.error || ('HTTP ' + r.status));
        lastCaptured = result.corner;
        lastEl.style.display = 'block';
        lastEl.textContent =
          result.corner + ' captured.\\n' +
          'x = ' + result.xyz[0].toFixed(5) + '\\n' +
          'y = ' + result.xyz[1].toFixed(5) + '\\n' +
          'z = ' + result.xyz[2].toFixed(5);
        setStatus(result.next ? ('Next: ' + result.next) : 'Done.');
      } catch (err) {
        setStatus('Error: ' + err.message, true);
      } finally {
        busy = false;
        await refreshState();
      }
    }

    async function doRetake() {
      if (!lastCaptured || busy) return;
      busy = true;
      captureBtn.disabled = true;
      retakeBtn.disabled = true;
      try {
        await fetch('/retake', { method: 'POST' });
        lastEl.style.display = 'none';
        setStatus('Retake ' + lastCaptured + '.');
      } finally {
        busy = false;
        await refreshState();
      }
    }

    captureBtn.addEventListener('click', doCapture);
    retakeBtn.addEventListener('click', doRetake);
    startCamera();
    refreshState();
    setInterval(refreshState, 2000);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class CalibrationHandler(BaseHTTPRequestHandler):
    server_version = "CanvasCalibration/0.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter than default
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # -- GET --------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return
        if self.path == "/state":
            self._send_json(200, STATE.progress())
            return
        self.send_error(404, "Not found")

    # -- POST -------------------------------------------------------------
    def do_POST(self) -> None:
        if self.path == "/capture":
            self._handle_capture()
            return
        if self.path == "/retake":
            self._handle_retake()
            return
        self.send_error(404, "Not found")

    def _handle_retake(self) -> None:
        # Clear the most-recent captured corner so the user can redo it.
        with STATE.lock:
            for corner in reversed(CORNERS):
                if corner in STATE.results:
                    STATE.results.pop(corner)
                    STATE.done_event.clear()
                    break
        self._send_json(200, STATE.progress())

    def _handle_capture(self) -> None:
        corner = STATE.current_corner()
        if corner is None:
            self._send_json(400, {"ok": False, "error": "All corners already captured."})
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send_json(400, {"ok": False, "error": f"Bad Content-Length: {length}"})
            return

        body = self.rfile.read(length)
        try:
            jpeg_bytes = _extract_multipart_jpeg(body, self.headers.get("Content-Type", ""))
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        image_path = CAPTURES_DIR / f"{corner}.jpg"
        image_path.write_bytes(jpeg_bytes)

        print(f"[{corner}] saved {image_path.relative_to(REPO_ROOT)} ({len(jpeg_bytes)} bytes)")
        try:
            xyz = extract_xyz_from_image(image_path)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": f"Extraction failed: {exc}"})
            return

        STATE.record(corner, xyz, image_path.name)
        progress = STATE.progress()
        print(f"[{corner}] xyz = {xyz}  ({len(STATE.results)}/{len(CORNERS)} done)")

        self._send_json(200, {
            "ok": True,
            "corner": corner,
            "xyz": list(xyz),
            "next": progress["current"],
            "done": progress["done"],
        })


# ---------------------------------------------------------------------------
# Multipart parsing (tiny: we only ever receive one JPEG part)
# ---------------------------------------------------------------------------
def _extract_multipart_jpeg(body: bytes, content_type: str) -> bytes:
    if "multipart/form-data" not in content_type:
        raise ValueError("Content-Type must be multipart/form-data")
    boundary = content_type.split("boundary=", 1)[-1].strip()
    if not boundary:
        raise ValueError("Missing multipart boundary")
    sep = ("--" + boundary).encode()
    parts = body.split(sep)
    for part in parts:
        part = part.lstrip(b"\r\n")
        if not part or part.startswith(b"--"):
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_blob = part[:header_end].lower()
        if b"image/jpeg" not in headers_blob and b'name="photo"' not in headers_blob:
            continue
        payload = part[header_end + 4:]
        payload = payload.rstrip(b"\r\n-")
        if payload:
            return payload
    raise ValueError("No JPEG part found in upload")


# ---------------------------------------------------------------------------
# Cert + network helpers (self-contained so calibration/ stays standalone)
# ---------------------------------------------------------------------------
def ensure_self_signed_cert() -> None:
    """Mirror the cert behavior in computer-vision/phone_camera_server.py.

    The cert + key files are shared with the phone camera server, so on a
    machine where that server has been run at least once the files already
    exist and this function is a no-op.

    If the cert is missing, we shell out to openssl the same way the phone
    camera server does. (On Git Bash for Windows that command can fail
    because MSYS rewrites the ``/CN=...`` arg into a Windows path; if you
    hit that, run the phone camera server once from PowerShell -- which
    does not do path conversion -- to generate the cert, then come back
    here. Or use the workaround in the error message below.)
    """
    if CERT_PATH.exists() and KEY_PATH.exists():
        return

    CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
        "-days", "3650", "-nodes",
        "-keyout", str(KEY_PATH),
        "-out", str(CERT_PATH),
        "-subj", "/CN=picasso-phone-camera",
    ]
    # Tell Git Bash / MSYS not to rewrite the /CN= argument into a path.
    env = {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}
    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not create a self-signed HTTPS cert. Install openssl, "
            "or run with --http (camera access on phones may not work over HTTP). "
            "On Git Bash for Windows, you can also generate the cert by running "
            "the phone camera server once from PowerShell."
        ) from exc


def default_route_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def list_local_ipv4_addresses() -> list[str]:
    addrs: list[str] = []
    seen: set[str] = set()
    primary = default_route_ip()
    if primary and primary != "127.0.0.1":
        addrs.append(primary)
        seen.add(primary)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = info[4][0]
            if ip in seen or ip.startswith("127."):
                continue
            addrs.append(ip)
            seen.add(ip)
    except socket.gaierror:
        pass
    return addrs or ["127.0.0.1"]


# ---------------------------------------------------------------------------
# Write calibration JSON
# ---------------------------------------------------------------------------
def write_canvas_calibration() -> Path:
    """Write robot/canvas_calibration.json, preserving existing fields."""
    payload: dict
    if CANVAS_CALIBRATION_PATH.exists():
        try:
            payload = json.loads(CANVAS_CALIBRATION_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    payload["units"] = payload.get("units", "meters")
    payload.setdefault(
        "description",
        "Raw canvas corner calibration points in robot/world XYZ coordinates.",
    )
    payload["raw_corners_xyz"] = {
        corner: list(STATE.results[corner]["xyz"])
        for corner in CORNERS
    }
    payload.setdefault("stroke_plane_offset_xyz_m", [-0.018, 0.0, 0.0])
    payload["calibration_captured_at"] = datetime.now().isoformat(timespec="seconds")
    payload["calibration_source_images"] = {
        corner: f"calibration/captures/{STATE.results[corner]['image']}"
        for corner in CORNERS
    }

    CANVAS_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CANVAS_CALIBRATION_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(CANVAS_CALIBRATION_PATH)
    return CANVAS_CALIBRATION_PATH


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
def serve(host: str, port: int, use_https: bool, settle_seconds: float = 2.0) -> None:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), CalibrationHandler)

    scheme = "http"
    if use_https:
        ensure_self_signed_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=CERT_PATH, keyfile=KEY_PATH)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    addrs = list_local_ipv4_addresses()
    print("Canvas calibration server running.", flush=True)
    print(f"  Captures dir:  {CAPTURES_DIR}")
    print(f"  Output JSON:   {CANVAS_CALIBRATION_PATH}")
    print(f"  Local:         {scheme}://127.0.0.1:{port}/")
    print("  Phone URLs (try each if the first does not load):")
    for ip in addrs:
        print(f"    {scheme}://{ip}:{port}/")
    if use_https:
        print("  Accept the self-signed cert warning on the phone once.")
    print("")
    print("  Sequence:  INIT -> TL -> TR -> BR -> BL -> DONE")
    print("  This script exits once all four corners are captured.")
    print("", flush=True)

    server_thread = threading.Thread(target=httpd.serve_forever, name="calib-http", daemon=True)
    server_thread.start()

    try:
        STATE.done_event.wait()
    except KeyboardInterrupt:
        print("Interrupted before all corners were captured.")
        httpd.shutdown()
        httpd.server_close()
        sys.exit(130)

    print("All 4 corners captured. Writing calibration JSON ...")
    out_path = write_canvas_calibration()
    print(f"Wrote {out_path}")
    print("  Corners:")
    for corner in CORNERS:
        x, y, z = STATE.results[corner]["xyz"]
        print(f"    {corner}: x={x:+.5f}  y={y:+.5f}  z={z:+.5f}")

    time.sleep(settle_seconds)  # give the phone a beat to show the "done" screen
    httpd.shutdown()
    httpd.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over plain HTTP. Phone browsers may block camera access then.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Seconds to keep the server up after the final capture so the phone UI updates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    serve(args.host, args.port, use_https=not args.http, settle_seconds=args.settle_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
