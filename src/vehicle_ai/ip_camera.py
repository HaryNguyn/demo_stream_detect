"""Build and validate IP-camera stream URLs.

Camera web pages such as ``/doc/page/preview.asp`` are HTML, not video.  The
helpers here keep the UI from sending those pages to FFmpeg and make the RTSP
URL format explicit for common Hikvision/Dahua cameras.
"""
from urllib.parse import quote, urlsplit

WEB_PAGE_SUFFIXES = (".asp", ".aspx", ".html", ".htm", ".php", ".jsp")
WEB_PAGE_PARTS = ("/doc/page/",)
STREAM_SCHEMES = {"rtsp", "rtsps", "http", "https"}


def _host(value):
    host = str(value or "").strip()
    if not host:
        raise ValueError("Nhập IP hoặc host của camera.")
    if "://" in host:
        host = urlsplit(host).hostname or ""
    if not host or "/" in host or "@" in host:
        raise ValueError("IP/host camera không hợp lệ.")
    # IPv6 literals need brackets in a URL.
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _integer(value, default, label, *, minimum, maximum=None):
    try:
        number = int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        raise ValueError(f"{label} phải là số nguyên.") from None
    if number < minimum or (maximum is not None and number > maximum):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{label} phải trong khoảng {limit}.")
    return number


def build_rtsp_url(host, username="", password="", port=554, channel=1, stream="main",
                   brand="hikvision", path=""):
    """Return an RTSP URL, using a common path template when none is supplied."""
    host = _host(host)
    port = _integer(port, 554, "Cổng RTSP", minimum=1, maximum=65535)
    channel = _integer(channel, 1, "Kênh camera", minimum=1)
    stream = str(stream or "main").strip().lower()
    if stream not in {"main", "sub"}:
        raise ValueError("Luồng camera phải là main hoặc sub.")
    brand = str(brand or "hikvision").strip().lower()
    custom_path = str(path or "").strip()
    if custom_path:
        path_value = custom_path if custom_path.startswith("/") else f"/{custom_path}"
    elif brand == "hikvision":
        # 101 = channel 1 main, 102 = channel 1 sub-stream.
        path_value = f"/Streaming/Channels/{channel}0{1 if stream == 'main' else 2}"
    elif brand == "dahua":
        path_value = f"/cam/realmonitor?channel={channel}&subtype={0 if stream == 'main' else 1}"
    else:
        raise ValueError("Chọn mẫu camera Hikvision/Dahua hoặc nhập đường dẫn RTSP.")

    credentials = ""
    if str(username or "").strip():
        credentials = quote(str(username).strip(), safe="")
        if password:
            credentials += f":{quote(str(password), safe='')}"
        credentials += "@"
    return f"rtsp://{credentials}{host}:{port}{path_value}"


def looks_like_web_page(url):
    """True for camera/admin HTML pages that must not be passed to FFmpeg."""
    parts = urlsplit(str(url or "").strip())
    if parts.scheme.lower() not in {"http", "https"}:
        return False
    path = parts.path.lower()
    return (path in {"", "/"} or path.endswith(WEB_PAGE_SUFFIXES)
            or any(part in path for part in WEB_PAGE_PARTS))


def validate_stream_url(url):
    value = str(url or "").strip()
    if not value:
        return ""
    if looks_like_web_page(value):
        raise ValueError(
            "URL này là trang quản trị camera, không phải luồng video. "
            "Dùng RTSP, ví dụ rtsp://user:pass@IP:554/Streaming/Channels/101, "
            "hoặc điền mục Cấu hình IP camera / RTSP."
        )
    parts = urlsplit(value)
    if parts.scheme.lower() not in STREAM_SCHEMES:
        raise ValueError("URL stream phải bắt đầu bằng rtsp://, rtsps://, http:// hoặc https://.")
    if not parts.hostname:
        raise ValueError("URL stream thiếu host camera.")
    return value


def mask_url(url):
    """Hide the password before a URL is shown in the UI or event metadata."""
    value = str(url or "")
    prefix, separator, remainder = value.partition("://")
    if not separator or "@" not in remainder:
        return value
    credentials, at, host = remainder.rpartition("@")
    user, colon, _password = credentials.partition(":")
    if colon:
        credentials = f"{user}:***"
    return f"{prefix}://{credentials}{at}{host}"
