"""Template rendering for export."""

from __future__ import annotations

from pathlib import Path


TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"


def render_template(relative_path: str, context: dict[str, object]) -> str:
    template_path = TEMPLATE_ROOT / relative_path
    content = template_path.read_text(encoding="utf-8")
    for key, value in context.items():
        content = content.replace(f"{{{{{key}}}}}", str(value))
        content = content.replace(f"{{{{ {key} }}}}", str(value))
    return content
