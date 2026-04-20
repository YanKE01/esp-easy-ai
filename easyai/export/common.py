"""Shared helpers for export."""

from __future__ import annotations

import math
import re
from pathlib import Path


def sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized:
        sanitized = "easyai_model"
    if sanitized[0].isdigit():
        sanitized = f"m_{sanitized}"
    return sanitized


def format_int_array(values, indent: int = 4, per_line: int = 16) -> str:
    prefix = " " * indent
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append(prefix + ", ".join(str(int(v)) for v in chunk))
    return ",\n".join(lines)


def format_byte_array(values, indent: int = 4, per_line: int = 12) -> str:
    prefix = " " * indent
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append(prefix + ", ".join(f"0x{int(v):02x}" for v in chunk))
    return ",\n".join(lines)


def format_float_array(values, indent: int = 4, per_line: int = 8) -> str:
    prefix = " " * indent
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        rendered = []
        for value in chunk:
            text = f"{float(value):.9g}"
            if "." not in text and "e" not in text and "E" not in text:
                text += ".0"
            rendered.append(f"{text}f")
        lines.append(prefix + ", ".join(rendered))
    return ",\n".join(lines)


def format_float_literal(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return f"{text}f"


def parse_info_tensor_section(info_text: str, section_name: str) -> dict:
    marker = f"{section_name}:"
    if marker not in info_text:
        raise ValueError(f"Missing section '{section_name}' in .info file")
    section = info_text.split(marker, 1)[1]
    match = re.search(
        r"(?P<name>%[^\s,]+), shape: \[(?P<shape>[^\]]+)\], exponents: \[(?P<exp>[^\]]+)\],\s*"
        r"value: array\(\[(?P<values>.*?)\],\s*dtype=(?P<dtype>\w+)\)",
        section,
        re.S,
    )
    if not match:
        raise ValueError(f"Failed to parse section '{section_name}' from .info file")
    shape = [
        int(part.strip()) for part in match.group("shape").split(",") if part.strip()
    ]
    exponent = int(match.group("exp").split(",")[0].strip())
    values = [int(token) for token in re.findall(r"-?\d+", match.group("values"))]
    count = math.prod(shape)
    return {
        "name": match.group("name"),
        "shape": shape,
        "exponent": exponent,
        "dtype": match.group("dtype"),
        "values": values[:count],
    }


def write_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
