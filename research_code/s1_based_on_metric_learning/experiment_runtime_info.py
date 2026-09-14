"""
@file: experiment_runtime_info.py
@description: 统一采集训练与评估任务的跨平台运行环境信息，并以紧凑文本或JSON形式保存。
              Windows和Linux均不依赖特定命令；CPU、内存和显卡驱动检测均提供降级路径。
@author: Changxin Ye
@created: 2026-08-10
@version: 1.0
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


RUNTIME_SECTION_TITLE = "Runtime Environment"


def _single_line(value: Any, fallback: str = "unknown") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or fallback


def _cpu_model() -> str:
    system = platform.system().lower()
    if system == "windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if str(value).strip():
                return _single_line(value)
        except (ImportError, OSError):
            pass
    elif system == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip().lower() in {"model name", "hardware"}:
                    if value.strip():
                        return _single_line(value)
        except OSError:
            pass
    return _single_line(platform.processor() or platform.machine())


def _memory_and_cores() -> tuple[int | None, int, int | None]:
    logical = int(os.cpu_count() or 0)
    physical: int | None = None
    total_memory: int | None = None
    try:
        import psutil

        physical_value = psutil.cpu_count(logical=False)
        physical = int(physical_value) if physical_value else None
        total_memory = int(psutil.virtual_memory().total)
    except (ImportError, OSError, RuntimeError):
        pass

    if total_memory is None:
        if platform.system().lower() == "windows":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("length", ctypes.c_ulong),
                        ("memory_load", ctypes.c_ulong),
                        ("total_phys", ctypes.c_ulonglong),
                        ("avail_phys", ctypes.c_ulonglong),
                        ("total_page_file", ctypes.c_ulonglong),
                        ("avail_page_file", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("avail_virtual", ctypes.c_ulonglong),
                        ("avail_extended_virtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.length = ctypes.sizeof(MemoryStatus)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    total_memory = int(status.total_phys)
            except (AttributeError, OSError):
                pass
        else:
            try:
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                page_count = int(os.sysconf("SC_PHYS_PAGES"))
                total_memory = page_size * page_count
            except (AttributeError, OSError, ValueError):
                pass
    return physical, logical, total_memory


def _nvidia_driver_version() -> str | None:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": 3,
        "check": False,
    }
    if platform.system().lower() == "windows":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(command, **kwargs)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def _resolved_device_name(
    requested_device: str | torch.device | None,
    resolved_device: str | torch.device | None,
    gpu_id: int | None,
) -> str:
    if resolved_device is not None:
        return str(resolved_device)
    requested = str(requested_device or "unknown").strip().lower()
    if requested == "auto":
        return f"cuda:{int(gpu_id or 0)}" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        return f"cuda:{int(gpu_id or 0)}"
    return requested or "unknown"


def collect_runtime_environment(
    *,
    requested_device: str | torch.device | None = None,
    resolved_device: str | torch.device | None = None,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """采集稳定、适合辨认实验机器的运行环境信息。"""

    physical_cores, logical_cores, total_memory = _memory_and_cores()
    requested = str(requested_device) if requested_device is not None else "unknown"
    resolved = _resolved_device_name(requested_device, resolved_device, gpu_id)
    gpu: dict[str, Any] | None = None
    if resolved.startswith("cuda") and torch.cuda.is_available():
        parsed = torch.device(resolved)
        index = int(parsed.index if parsed.index is not None else torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": _single_line(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }

    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "hostname": _single_line(socket.gethostname()),
        "os": {
            "system": _single_line(platform.system()),
            "release": _single_line(platform.release()),
            "version": _single_line(platform.version()),
            "architecture": _single_line(platform.machine()),
        },
        "cpu": {
            "model": _cpu_model(),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "total_memory_bytes": total_memory,
        },
        "device": {
            "requested": requested,
            "resolved": resolved,
            "gpu_id_argument": gpu_id,
        },
        "gpu": gpu,
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": cudnn_version,
            # 即使本次明确使用 CPU，也记录机器上的 NVIDIA 驱动版本；这样不会把
            # “GPU 未参与本次运行”误写成“机器没有可用驱动”。
            "nvidia_driver": _nvidia_driver_version(),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": sys.executable,
    }


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def format_runtime_environment(info: dict[str, Any]) -> list[str]:
    """生成适合追加到summary文本末尾的紧凑八行环境信息。"""

    os_info = info["os"]
    cpu = info["cpu"]
    device = info["device"]
    software = info["software"]
    physical = cpu["physical_cores"] if cpu["physical_cores"] is not None else "unknown"
    gpu = info["gpu"]
    if gpu is None:
        gpu_line = "GPU: not used"
    else:
        gpu_line = (
            f"GPU: {gpu['name']} | index={gpu['index']} | "
            f"VRAM={_gib(gpu['total_memory_bytes'])}"
        )
    cuda_visible = info["cuda_visible_devices"]
    return [
        RUNTIME_SECTION_TITLE,
        "-------------------",
        f"Host: {info['hostname']}",
        f"OS: {os_info['system']} {os_info['release']} ({os_info['architecture']})",
        (
            f"CPU: {cpu['model']} | {physical} physical / "
            f"{cpu['logical_cores']} logical cores | RAM {_gib(cpu['total_memory_bytes'])}"
        ),
        f"Device: requested={device['requested']} | resolved={device['resolved']}",
        gpu_line,
        f"Software: Python {software['python']} | PyTorch {software['pytorch']}",
        (
            f"CUDA: {software['cuda_runtime'] or 'unavailable'} | "
            f"cuDNN {software['cudnn'] or 'unavailable'} | "
            f"NVIDIA driver {software['nvidia_driver'] or 'unavailable'}"
        ),
        f"CUDA_VISIBLE_DEVICES: {cuda_visible if cuda_visible is not None else 'not set'}",
    ]


def runtime_environment_lines(
    *,
    requested_device: str | torch.device | None = None,
    resolved_device: str | torch.device | None = None,
    gpu_id: int | None = None,
) -> list[str]:
    return format_runtime_environment(
        collect_runtime_environment(
            requested_device=requested_device,
            resolved_device=resolved_device,
            gpu_id=gpu_id,
        )
    )


def append_runtime_environment(
    path: str | Path,
    *,
    requested_device: str | torch.device | None = None,
    resolved_device: str | torch.device | None = None,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """向已存在的summary文本追加环境段；重复调用不会重复追加。"""

    path = Path(path)
    info = collect_runtime_environment(
        requested_device=requested_device,
        resolved_device=resolved_device,
        gpu_id=gpu_id,
    )
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if RUNTIME_SECTION_TITLE not in existing:
        separator = "" if not existing or existing.endswith("\n\n") else "\n"
        text = existing + separator + "\n".join(format_runtime_environment(info)) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return info


__all__ = [
    "RUNTIME_SECTION_TITLE",
    "append_runtime_environment",
    "collect_runtime_environment",
    "format_runtime_environment",
    "runtime_environment_lines",
]
