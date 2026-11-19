"""Video sources exposing the OpenCV capture interface."""
import shutil
import subprocess
import sys
import os
from pathlib import Path
from urllib.parse import urlparse
from collections import deque
from threading import Thread

if sys.platform == "win32":
    import winreg

import cv2
import numpy as np


def corporate_proxy():
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    if proxy:
        return proxy

    if sys.platform != "win32":
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            if enabled and server:
                if "=" in server:
                    for entry in server.split(";"):
                        name, value = entry.split("=", 1)
                        if name.lower() in {"http", "https"}:
                            return value if "://" in value else f"http://{value}"
                    return None
                return server if "://" in server else f"http://{server}"
    except (OSError, ValueError):
        pass
    return None


def stream_source(url: str, webpage: bool = False) -> tuple[str, str]:
    url = url.strip()
    hostname = urlparse(url).hostname or ""
    youtube = hostname in {"youtube.com", "youtu.be"} or hostname.endswith(".youtube.com")
    if webpage or youtube:
        return f"Web · {url}", f"web://{url}"
    return f"Stream · {url}", url


def open_capture(source):
    source = str(source)
    if source.startswith(("youtube://", "web://")):
        return WebStreamCapture(source)
    if source.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        return FfmpegStreamCapture(source)
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        return capture
    capture.release()

    # OpenCV may lack a codec even when FFmpeg can decode the local file.
    if Path(source).is_file() and shutil.which("ffmpeg"):
        return FfmpegStreamCapture(source)
    return cv2.VideoCapture(source)


def stop_process(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stdout:
        process.stdout.close()


class ProcessErrors:
    """Drain stderr continuously so a verbose child cannot block video output."""

    def __init__(self, process):
        self.chunks = deque(maxlen=64)
        self.thread = Thread(target=self._drain, args=(process.stderr,), daemon=True)
        self.thread.start()

    def _drain(self, stream):
        try:
            while chunk := stream.read(1024):
                self.chunks.append(chunk)
        finally:
            stream.close()

    def text(self):
        return b"".join(list(self.chunks)).decode("utf-8", errors="replace").strip()

    def finish(self):
        self.thread.join(timeout=5)


class FfmpegStreamCapture:
    WIDTH = 1280
    HEIGHT = 720
    FPS = 30.0

    def __init__(self, url, *, stdin=None):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Không tìm thấy ffmpeg trong PATH.")
        proxy = corporate_proxy()
        process_env = os.environ.copy()
        if proxy:
            process_env["HTTP_PROXY"] = proxy
            process_env["HTTPS_PROXY"] = proxy
            process_env["http_proxy"] = proxy
            process_env["https_proxy"] = proxy
        is_http = stdin is None and url.startswith(("http://", "https://"))
        is_network = stdin is None and url.startswith(("http://", "https://", "rtsp://", "rtsps://"))
        input_options = ["-rw_timeout", "15000000"] if is_network else []
        if is_http:
            input_options.extend([
                "-reconnect", "1", "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10",
            ])
        if proxy and is_http:
            input_options.extend(["-http_proxy", proxy])
        self.process = subprocess.Popen(
            [
                ffmpeg, "-loglevel", "error",
                *input_options, "-i", url,
                "-vf", f"scale={self.WIDTH}:{self.HEIGHT}:force_original_aspect_ratio=decrease,"
                       f"pad={self.WIDTH}:{self.HEIGHT}:(ow-iw)/2:(oh-ih)/2",
                "-r", str(int(self.FPS)), "-pix_fmt", "bgr24",
                "-f", "rawvideo", "pipe:1",
            ],
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
        )
        self.errors = ProcessErrors(self.process)
        self.frame_size = self.WIDTH * self.HEIGHT * 3

    def isOpened(self):
        return self.process.poll() is None

    def get(self, prop):
        return self.FPS if prop == cv2.CAP_PROP_FPS else 0

    def read(self):
        if self.process.stdout is None:
            return False, None
        data = bytearray()
        while len(data) < self.frame_size:
            chunk = self.process.stdout.read(self.frame_size - len(data))
            if not chunk:
                return False, None
            data.extend(chunk)
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 3))
        return True, frame

    def release(self):
        stop_process(self.process)
        self.errors.finish()

    def error_text(self):
        if self.process.poll() is not None:
            self.errors.finish()
        return self.errors.text()


class WebStreamCapture(FfmpegStreamCapture):
    def __init__(self, url):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Không tìm thấy ffmpeg trong PATH.")
        command = [
            sys.executable, "-m", "yt_dlp", "--no-playlist",
            "--retries", "5", "--fragment-retries", "5",
            "--format", "bestvideo[ext=mp4]/best[ext=mp4]/best",
        ]
        proxy = corporate_proxy()
        if proxy:
            command.extend(["--proxy", proxy])
        node = shutil.which("node")
        if node:
            command.extend(["--js-runtimes", f"node:{node}"])
        source = url.removeprefix("youtube://").removeprefix("web://")
        command.extend(["--output", "-", source])
        process_env = os.environ.copy()
        if proxy:
            process_env["HTTP_PROXY"] = proxy
            process_env["HTTPS_PROXY"] = proxy
            process_env["http_proxy"] = proxy
            process_env["https_proxy"] = proxy
        self.downloader = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=process_env,
        )
        self.download_errors = ProcessErrors(self.downloader)
        try:
            super().__init__("pipe:0", stdin=self.downloader.stdout)
        except Exception:
            stop_process(self.downloader)
            self.download_errors.finish()
            raise
        finally:
            if self.downloader.stdout:
                self.downloader.stdout.close()

    def release(self):
        try:
            super().release()
        finally:
            stop_process(self.downloader)
            self.download_errors.finish()

    def error_text(self):
        parts = []
        ffmpeg_error = super().error_text()
        if ffmpeg_error:
            parts.append(f"FFmpeg: {ffmpeg_error}")
        if self.downloader.poll() is not None:
            self.download_errors.finish()
        yt_error = self.download_errors.text()
        if yt_error:
            parts.append(f"yt-dlp: {yt_error}")
        return "\n".join(parts)
