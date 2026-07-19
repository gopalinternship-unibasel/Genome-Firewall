from __future__ import annotations

import ast
import asyncio
import importlib
import json
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
