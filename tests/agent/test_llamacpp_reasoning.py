"""llama.cpp reasoning-effort levels from ``GET /props`` and the wire clamp (#152295).

A llama.cpp server publishes the ``reasoning_effort`` values its chat template renders
distinctly under ``chat_template_caps.reasoning_effort_levels``. Hermes probes that once
per (model, base_url) and clamps the requested effort onto it, so "max" on a template
that tops out at "xhigh" sends "xhigh" instead of 400ing — the same shape as the
LM Studio ``allowed_options`` probe.
"""

from __future__ import annotations

import contextlib
import io
import json
import urllib.error

import pytest

from hermes_cli.models_local import llamacpp_reasoning_effort_levels
from providers import get_provider_profile


def _props_response(monkeypatch, payload, *, assert_url=None):
    """Serve *payload* as the ``/props`` JSON body for the probe's urlopen call."""
    seen = {}

    @contextlib.contextmanager
    def fake_urlopen(req, *, timeout, **kwargs):
        seen["url"] = req.full_url
        yield io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("hermes_cli.models._urlopen_model_catalog_request", fake_urlopen)
    return seen


def test_props_levels_returned_lowest_to_highest(monkeypatch):
    seen = _props_response(monkeypatch, {
        "chat_template_caps": {"reasoning_effort_levels": ["low", "medium", "xhigh"]},
    })
    assert llamacpp_reasoning_effort_levels("model", "http://127.0.0.1:8080/v1") == [
        "low", "medium", "xhigh",
    ]
    assert seen["url"] == "http://127.0.0.1:8080/props"


def test_props_without_levels_is_unknown(monkeypatch):
    _props_response(monkeypatch, {"chat_template_caps": {}})
    assert llamacpp_reasoning_effort_levels("model", "http://127.0.0.1:8080/v1") == []
    _props_response(monkeypatch, {})
    assert llamacpp_reasoning_effort_levels("model", "http://127.0.0.1:8080/v1") == []


def test_props_non_ladder_values_are_dropped(monkeypatch):
    _props_response(monkeypatch, {
        "chat_template_caps": {"reasoning_effort_levels": ["low", "bogus", "XHIGH"]},
    })
    assert llamacpp_reasoning_effort_levels("model", "http://127.0.0.1:8080") == ["low", "xhigh"]


def test_props_http_error_is_unknown(monkeypatch):
    def failing_urlopen(req, *, timeout, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)

    monkeypatch.setattr("hermes_cli.models._urlopen_model_catalog_request", failing_urlopen)
    assert llamacpp_reasoning_effort_levels("model", "http://127.0.0.1:8080/v1") == []
    assert llamacpp_reasoning_effort_levels("model", "") == []


def _custom_top_level(effort, options):
    profile = get_provider_profile("custom")
    _extra, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": effort},
        llamacpp_reasoning_options=options,
        base_url="http://127.0.0.1:8080/v1",
    )
    return top_level.get("reasoning_effort")


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("max", "xhigh"), ("ultra", "xhigh"), ("high", "medium"), ("medium", "medium"), ("low", "low")],
)
def test_custom_profile_clamps_onto_probed_levels(effort, expected):
    """A probed llama.cpp template caps the wire level at its top distinct tier."""
    assert _custom_top_level(effort, ["low", "medium", "xhigh"]) == expected


def test_custom_profile_without_probe_keeps_wide_wire_set():
    assert _custom_top_level("max", None) == "max"
    assert _custom_top_level("max", []) == "max"


def test_custom_profile_explicit_disable_is_not_clamped_up():
    profile = get_provider_profile("custom")
    _extra, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False, "effort": "max"},
        llamacpp_reasoning_options=["low", "medium", "xhigh"],
        base_url="http://127.0.0.1:8080/v1",
    )
    assert top_level["reasoning_effort"] == "none"
