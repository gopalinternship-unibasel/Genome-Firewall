from __future__ import annotations

import ast
import asyncio
import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vercel_configuration_selects_the_asgi_entrypoint() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    vercel = json.loads((PROJECT_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert pyproject["tool"]["vercel"]["entrypoint"] == "asgi:app"
    assert vercel["framework"] == "python"
    assert vercel["fluid"] is True
    assert "asgi.py" in vercel["functions"]
    assert "app.py" in vercel["functions"]["asgi.py"]["includeFiles"]
    assert vercel["functions"]["asgi.py"]["maxDuration"] == 300


def test_vercel_entrypoint_exports_a_callable_asgi_app() -> None:
    module = importlib.import_module("asgi")

    assert callable(module.app)
    assert Path(module.app.script_path).resolve() == PROJECT_ROOT / "app.py"


def test_vercel_entrypoint_has_a_static_top_level_app_export() -> None:
    tree = ast.parse((PROJECT_ROOT / "asgi.py").read_text(encoding="utf-8"))
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert "app" in assigned_names


def test_source_checkout_can_load_demo_without_editable_install() -> None:
    """Mirror Vercel: dependencies are installed, but this project is not."""
    code = """
import json
import sys
from pathlib import Path

project_root = Path.cwd().resolve()
source_root = (project_root / "src").resolve()
sys.path = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != source_root
]

import asgi
from genome_firewall.demo import load_demo_case

case = load_demo_case("marker_fail")
print(json.dumps({
    "sample_id": case["report"]["sample_id"],
    "mode": case["report"]["mode"],
    "source_on_path": str(source_root) in sys.path,
}))
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "sample_id": "DEMO-EC-001",
        "mode": "DEMO",
        "source_on_path": True,
    }


def test_vercel_entrypoint_serves_streamlit_routes() -> None:
    module = importlib.import_module("asgi")

    async def request_routes() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            root = await client.get("/")
            health = await client.get("/_stcore/health")
        return root, health

    root, health = asyncio.run(request_routes())

    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert health.status_code == 200
    assert health.text == "ok"
