from __future__ import annotations

import importlib

import model_runtime


def test_model_is_a_singleton():
    first = model_runtime.get_model()
    second = model_runtime.get_model()
    assert first is second


def test_app_imports_and_genuine_request_succeeds():
    app = importlib.import_module("app")
    cleared, history = app.respond("347 + 928", [])
    assert cleared == ""
    assert history[-1]["role"] == "assistant"
    assert "<strong>1275</strong>" in history[-1]["content"]
    assert "5721" in history[-1]["content"]


def test_application_rejects_unsupported_request():
    app = importlib.import_module("app")
    _, history = app.respond("12 * 3", [])
    assert "Unsupported request" in history[-1]["content"]
    assert "addition only" in history[-1]["content"]
