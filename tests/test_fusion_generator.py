import importlib.util
from pathlib import Path
from typing import Any

from open_webui_openrouter_pipe.streaming.fusion_embed import _FUSION_TEMPLATE_HTML

_ROOT = Path(__file__).resolve().parent.parent


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_fusion_generator", _ROOT / "scripts" / "build_fusion_template.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_template_in_sync_with_scaffold():
    gen = _load_generator()
    template = gen.build_template(gen.SOURCE_HTML.read_text(encoding="utf-8"))
    gen.self_check(template)
    assert template == _FUSION_TEMPLATE_HTML, (
        "fusion_embed.py is stale or hand-edited; run: python scripts/build_fusion_template.py"
    )
