"""Application paths and settings shared by UI and persistence.

Priority: process environment > project .env > Windows user environment > default.
Reading settings never mutates os.environ.
"""
import os
from pathlib import Path
import sys

from dotenv import dotenv_values


def project_dir() -> Path:
    configured = os.environ.get("VEHICLE_AI_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file() and (source_root / "configs").is_dir():
        return source_root
    return Path.cwd().resolve()


def get_setting(name, default=None, *, required=False):
    value = os.environ.get(name)
    if value:
        return value
    value = dotenv_values(project_dir() / ".env").get(name)
    if value:
        return value
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
                if value:
                    return value
        except (OSError, ValueError):
            pass
    if required:
        raise RuntimeError(f"Thiếu cấu hình: {name}. Điền vào .env hoặc biến môi trường.")
    return default


def image_root() -> Path:
    return Path(get_setting("VDTC_IMAGE_ROOT", str(project_dir() / "runtime" / "images")))
