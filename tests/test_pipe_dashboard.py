"""Tests for the Pipe Dashboard plugin: commands, auth, formatters, model matching."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import pytest

from open_webui_openrouter_pipe.plugins.base import PluginBase, PluginContext
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import PipeDashboardPlugin
from open_webui_openrouter_pipe.plugins.pipe_dashboard.command_registry import CommandRegistry
from open_webui_openrouter_pipe.plugins.pipe_dashboard.formatters import (
    collapsible,
    format_bytes,
    format_duration,
    humanize_type,
    markdown_table,
    mask_sensitive,
    mermaid_bar,
    mermaid_pie,
)


# ── Fixtures ──


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset both plugin and command registries between tests."""
    original_plugins = PluginRegistry._plugin_classes[:]
    original_valve_fields = dict(PluginRegistry._pending_valve_fields)
    original_user_valve_fields = dict(PluginRegistry._pending_user_valve_fields)
    original_commands = dict(CommandRegistry._commands)
    yield
    PluginRegistry._plugin_classes.clear()
    PluginRegistry._plugin_classes.extend(original_plugins)
    PluginRegistry._pending_valve_fields.clear()
    PluginRegistry._pending_valve_fields.update(original_valve_fields)
    PluginRegistry._pending_user_valve_fields.clear()
    PluginRegistry._pending_user_valve_fields.update(original_user_valve_fields)
    CommandRegistry._commands = original_commands


@pytest.fixture(autouse=True)
def _authz_default_allow(monkeypatch):
    """Authorize on_request/command paths by default so command tests exercise
    dispatch, not the OWUI access seam (unit-tested in test_authz.py). Denial
    tests override authz.can_view to False."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import plugin as _ppl

    monkeypatch.setattr(
        _ppl, "resolve_user",
        AsyncMock(return_value=SimpleNamespace(id="admin1", role="admin")),
    )
    monkeypatch.setattr(_ppl, "can_view", AsyncMock(return_value=True))


def _make_mock_pipe():
    """Create a minimal mock Pipe for Pipe Dashboard tests."""
    pipe = Mock()
    pipe.id = "test-pipe"
    pipe.valves = Mock()
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    pipe.valves.model_fields = {}
    pipe._artifact_store = Mock()
    pipe._artifact_store._session_factory = None
    pipe._artifact_store._item_model = None
    pipe._circuit_breaker = Mock()
    pipe._circuit_breaker._threshold = 5
    pipe._circuit_breaker._window_seconds = 60.0
    pipe._circuit_breaker._breaker_records = {}
    pipe._circuit_breaker._tool_breakers = {}
    pipe._active_pipes_calls = 0
    pipe._video_global_semaphore = None
    pipe._video_global_limit = 0
    pipe._video_active_tasks = {}
    pipe._redis_client = None
    pipe._redis_enabled = False
    pipe._request_queue = None
    pipe._catalog_manager = None
    pipe._http_session = None
    return pipe


def _make_plugin(pipe: object = None) -> PipeDashboardPlugin:
    """Create an initialized PipeDashboardPlugin."""
    import logging

    if pipe is None:
        pipe = _make_mock_pipe()
    ctx = PluginContext(pipe=pipe, logger=logging.getLogger("test"))
    plugin = PipeDashboardPlugin()
    plugin.on_init(ctx)
    return plugin


def _make_body(message: str, model: str = "pipe-dashboard") -> dict:
    """Create a minimal request body."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": message}],
    }


# ── Formatter Tests ──


class TestFormatters:
    def test_markdown_table(self):
        result = markdown_table(["A", "B"], [["1", "2"], ["3", "4"]])
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result
        assert "| 3 | 4 |" in result

    def test_markdown_table_empty(self):
        assert markdown_table([], []) == ""

    def test_mermaid_pie(self):
        result = mermaid_pie("Test", {"A": 10, "B": 20})
        assert "```mermaid" in result
        assert "pie title Test" in result
        assert '"A" : 10' in result

    def test_mermaid_pie_empty(self):
        assert mermaid_pie("Test", {}) == ""

    def test_mermaid_bar(self):
        result = mermaid_bar("Title", "X", "Y", ["a", "b"], [1, 2])
        assert "xychart-beta" in result
        assert '"a"' in result

    def test_mermaid_bar_empty(self):
        assert mermaid_bar("Title", "X", "Y", [], []) == ""

    def test_collapsible(self):
        result = collapsible("Summary", "Content")
        assert "<details>" in result
        assert "<summary>Summary</summary>" in result
        assert "Content" in result

    def test_mask_sensitive(self):
        assert mask_sensitive("sk-or-v1-abc123xyz") == "***3xyz"
        assert mask_sensitive("") == ""
        assert mask_sensitive("abc") == "***"

    def test_format_duration(self):
        assert format_duration(5.2) == "5.2s"
        assert format_duration(125) == "2.1m"
        assert format_duration(7200) == "2.0h"

    def test_format_bytes(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(512) == "512 B"
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(2_621_440) == "2.5 MB"
        assert format_bytes(5_368_709_120) == "5.0 GB"

    def test_humanize_type_known(self):
        assert humanize_type("function_call") == "Function Call"
        assert humanize_type("function_call_output") == "Function Call Output"
        assert humanize_type("reasoning") == "Reasoning"
        assert humanize_type("image_generation_call") == "Image Generation"

    def test_humanize_type_unknown(self):
        assert humanize_type("some_new_type") == "Some New Type"


# ── Command Registry Tests ──


class TestCommandRegistry:
    def test_resolve_exact_match(self):
        entry, args = CommandRegistry.resolve("help")
        assert entry is not None
        assert entry.name == "help"
        assert args == ""

    def test_resolve_with_args(self):
        entry, args = CommandRegistry.resolve("help dashboard")
        assert entry is not None
        assert entry.name == "help"
        assert args == "dashboard"

    def test_resolve_unknown(self):
        entry, args = CommandRegistry.resolve("nonexistent command")
        assert entry is None
        assert args == ""

    def test_resolve_empty(self):
        entry, args = CommandRegistry.resolve("")
        assert entry is None

    def test_all_commands(self):
        commands = CommandRegistry.all_commands()
        names = [c.name for c in commands]
        assert "help" in names
        assert "dashboard" in names


# ── Model ID Matching Tests ──


class TestModelIdMatching:
    def test_plain_pipe_dashboard(self):
        plugin = _make_plugin()
        assert plugin._is_our_model("pipe-dashboard") is True

    def test_pipe_prefixed(self):
        plugin = _make_plugin()
        # Mock pipe has id="test-pipe", so prefix must match
        assert plugin._is_our_model("test-pipe.pipe-dashboard") is True

    def test_other_model(self):
        plugin = _make_plugin()
        assert plugin._is_our_model("openai/gpt-4o") is False

    def test_empty_model(self):
        plugin = _make_plugin()
        assert plugin._is_our_model("") is False

    def test_case_insensitive(self):
        plugin = _make_plugin()
        assert plugin._is_our_model("Pipe-Dashboard") is True


# ── Task Stub Tests ──


class TestTaskStubs:
    @pytest.mark.asyncio
    async def test_title_generation(self):
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("hello"), {}, {}, None, "title_generation",
        )
        assert isinstance(result, dict)
        assert '"title"' in result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_tags_generation(self):
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("hello"), {}, {}, None, "tags_generation",
        )
        assert isinstance(result, dict)
        assert '"tags"' in result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_emoji_generation(self):
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("hello"), {}, {}, None, "emoji_generation",
        )
        assert isinstance(result, dict)
        assert '"emoji"' in result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_follow_up_generation(self):
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("hello"), {}, {}, None, "follow_up_generation",
        )
        assert isinstance(result, dict)
        assert '"follow_ups"' in result["choices"][0]["message"]["content"]


# ── Authorization Tests ──


class TestAuthorization:
    @pytest.mark.asyncio
    async def test_admin_allowed(self):
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("help"), {"role": "admin"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Access Denied" not in content

    @pytest.mark.asyncio
    async def test_non_admin_denied(self, monkeypatch):
        from unittest.mock import AsyncMock

        from open_webui_openrouter_pipe.plugins.pipe_dashboard import plugin as _ppl
        monkeypatch.setattr(_ppl, "can_view", AsyncMock(return_value=False))
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("help"), {"role": "user", "id": "u1"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Access Denied" in content

    @pytest.mark.asyncio
    async def test_no_view_grant_denied(self, monkeypatch):
        from unittest.mock import AsyncMock

        from open_webui_openrouter_pipe.plugins.pipe_dashboard import plugin as _ppl
        monkeypatch.setattr(_ppl, "can_view", AsyncMock(return_value=False))
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("help"), {}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Access Denied" in content


# ── Command Dispatch Tests ──


class TestCommandDispatch:
    @pytest.mark.asyncio
    async def test_help_command_single_table(self):
        """Help output should be a single flat table, no category headings."""
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("help"), {"role": "admin"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Pipe Dashboard" in content
        assert "| Command | Description |" in content
        # Should NOT have category headings like ### Configuration
        assert "### Configuration" not in content
        assert "### Storage" not in content
        assert "### General" not in content

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("foobar"), {"role": "admin"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Unknown command" in content

    @pytest.mark.asyncio
    async def test_default_to_help(self):
        """Empty message defaults to help."""
        plugin = _make_plugin()
        result = await plugin.on_request(
            {"model": "pipe-dashboard", "messages": [{"role": "user", "content": ""}]},
            {"role": "admin"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Pipe Dashboard" in content

    @pytest.mark.asyncio
    async def test_not_our_model_returns_none(self):
        """Requests for other models return None (passthrough)."""
        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("help", model="openai/gpt-4o"),
            {"role": "admin"}, {}, None, None,
        )
        assert result is None


# ── on_models Tests ──


class TestOnModels:
    def test_adds_pipe_dashboard_model(self):
        plugin = _make_plugin()
        models = [{"id": "gpt-4o", "name": "GPT-4o"}]
        asyncio.run(plugin.on_models(models))
        ids = [m["id"] for m in models]
        assert "pipe-dashboard" in ids
        assert "gpt-4o" in ids
        assert len(models) == 2  # original + pipe-dashboard

    def test_always_injects_when_plugin_system_enabled(self):
        """Pipe Dashboard always injects its model — no per-plugin valve needed."""
        plugin = _make_plugin()
        models = [{"id": "gpt-4o", "name": "GPT-4o"}]
        asyncio.run(plugin.on_models(models))
        ids = [m["id"] for m in models]
        assert "pipe-dashboard" in ids
        assert models[-1]["name"] == "Pipe Dashboard"


# ── Message Extraction Tests ──


class TestMessageExtraction:
    def test_extract_text_message(self):
        body = {"messages": [{"role": "user", "content": "  help  "}]}
        assert PipeDashboardPlugin._extract_user_message(body) == "help"

    def test_extract_multimodal_message(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                        {"type": "text", "text": "dashboard"},
                    ],
                }
            ]
        }
        assert PipeDashboardPlugin._extract_user_message(body) == "dashboard"

    def test_extract_last_user_message(self):
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second"},
            ]
        }
        assert PipeDashboardPlugin._extract_user_message(body) == "second"

    def test_empty_messages(self):
        assert PipeDashboardPlugin._extract_user_message({"messages": []}) == ""
        assert PipeDashboardPlugin._extract_user_message({}) == ""


# ── Command Error Handling Tests ──


class TestCommandErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_exception_returns_error_message(self):
        """When a command handler raises, plugin returns markdown error, not propagation."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.command_registry import register_command

        @register_command("boom", summary="Exploding command", category="Test", usage="boom")
        async def handle_boom(ctx):
            raise ValueError("test explosion")

        plugin = _make_plugin()
        result = await plugin.on_request(
            _make_body("boom"), {"role": "admin"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Command Error" in content
        assert "boom" in content
        assert "test explosion" in content


# ── Task Name Extraction Tests ──


class TestTaskNameExtraction:
    def test_dict_type_key(self):
        assert PipeDashboardPlugin._extract_task_name({"type": "title_generation"}) == "title_generation"

    def test_dict_task_key(self):
        assert PipeDashboardPlugin._extract_task_name({"task": "tags_generation"}) == "tags_generation"

    def test_dict_name_key(self):
        assert PipeDashboardPlugin._extract_task_name({"name": "emoji_generation"}) == "emoji_generation"

    def test_dict_empty(self):
        assert PipeDashboardPlugin._extract_task_name({}) == ""

    def test_string_task(self):
        assert PipeDashboardPlugin._extract_task_name("title_generation") == "title_generation"

    def test_none_task(self):
        assert PipeDashboardPlugin._extract_task_name(None) == ""


# ── build_response Edge Case Tests ──


class TestBuildResponse:
    def test_normal_model(self):
        plugin = _make_plugin()
        result = plugin.ctx.build_response(model="test-model", content="hello")
        assert result["model"] == "test-model"
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["object"] == "chat.completion"

    def test_empty_model(self):
        plugin = _make_plugin()
        result = plugin.ctx.build_response(model="", content="hello")
        assert result["model"] == "pipe"

    def test_none_model(self):
        plugin = _make_plugin()
        result = plugin.ctx.build_response(model=None, content="hello")
        assert result["model"] == "pipe"


# ── Alias Resolution Tests ──


class TestAliasResolution:
    def test_resolve_through_alias(self):
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.command_registry import register_command

        @register_command("mytest", summary="Test", category="Test", usage="mytest", aliases=["mt"])
        async def handle_mytest(ctx):
            return "mytest result"

        entry, args = CommandRegistry.resolve("mt")
        assert entry is not None
        assert entry.name == "mytest"


# ── Registry Integration Tests ──


class TestRegistryIntegration:
    def test_pipe_dashboard_through_registry_dispatch(self):
        """PipeDashboardPlugin injects model when dispatched through PluginRegistry."""
        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        models = [{"id": "gpt-4o", "name": "GPT-4o"}]
        asyncio.run(registry.dispatch_on_models(models))
        ids = [m["id"] for m in models]
        assert "pipe-dashboard" in ids

    @pytest.mark.asyncio
    async def test_pipe_dashboard_request_through_registry(self):
        """PipeDashboardPlugin handles requests when dispatched through PluginRegistry."""
        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request(
            _make_body("help"), {"role": "admin"}, {}, None, None,
        )
        assert result is not None
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Pipe Dashboard" in content


# ── Case Preservation Tests (4.3) ──


class TestResolvesCasePreservation:
    def test_args_preserve_original_case(self):
        """resolve() lowercases for matching but preserves arg casing."""
        entry, args = CommandRegistry.resolve("HELP Dashboard")
        assert entry is not None
        assert entry.name == "help"
        assert args == "Dashboard"  # NOT "dashboard"

    def test_args_preserve_mixed_case(self):
        """Arguments with mixed case are not normalized."""
        entry, args = CommandRegistry.resolve("help MyCommand")
        assert entry is not None
        assert args == "MyCommand"


# ── _build_task_fallback Unknown Task Tests (4.7) ──


class TestBuildTaskFallbackUnknown:
    def test_unknown_task_returns_empty_json_object(self):
        """Unknown task type returns '{}' as safe fallback."""
        assert PipeDashboardPlugin._build_task_fallback("search_generation") == "{}"
        assert PipeDashboardPlugin._build_task_fallback("some_future_task") == "{}"

    def test_known_tasks_still_work(self):
        """Existing task types still return correct JSON."""
        import json
        assert "title" in json.loads(PipeDashboardPlugin._build_task_fallback("title_generation"))
        assert "tags" in json.loads(PipeDashboardPlugin._build_task_fallback("tags_generation"))
        assert "emoji" in json.loads(PipeDashboardPlugin._build_task_fallback("emoji_generation"))
        assert "follow_ups" in json.loads(PipeDashboardPlugin._build_task_fallback("follow_up_generation"))


# ── _is_our_model Rejects Arbitrary Prefix (4.8) ──


class TestIsOurModelPrefixValidation:
    def test_rejects_arbitrary_prefix(self):
        """_is_our_model rejects 'evil.pipe-dashboard' — only known pipe prefix accepted."""
        plugin = _make_plugin()
        assert plugin._is_our_model("evil.pipe-dashboard") is False

    def test_rejects_nested_prefix(self):
        """_is_our_model rejects 'x.y.pipe-dashboard'."""
        plugin = _make_plugin()
        assert plugin._is_our_model("x.y.pipe-dashboard") is False

    def test_accepts_correct_pipe_prefix(self):
        """_is_our_model accepts '<pipe_id>.pipe-dashboard'."""
        pipe = _make_mock_pipe()
        pipe.id = "my-pipe"
        plugin = _make_plugin(pipe)
        assert plugin._is_our_model("my-pipe.pipe-dashboard") is True

    def test_rejects_wrong_pipe_prefix(self):
        """_is_our_model rejects '<wrong_pipe_id>.pipe-dashboard'."""
        pipe = _make_mock_pipe()
        pipe.id = "my-pipe"
        plugin = _make_plugin(pipe)
        assert plugin._is_our_model("other-pipe.pipe-dashboard") is False


# ── Pipe Character in Markdown Table (4.9) ──


class TestMarkdownTablePipeEscape:
    def test_pipe_in_cell_escaped(self):
        """Pipe characters in cell values are escaped."""
        result = markdown_table(["A"], [["x|y"]])
        assert "x\\|y" in result
        assert "| x\\|y |" in result

    def test_pipe_in_header_escaped(self):
        """Pipe characters in header values are escaped."""
        result = markdown_table(["A|B"], [["val"]])
        assert "A\\|B" in result

    def test_no_pipe_unchanged(self):
        """Values without pipes are not affected."""
        result = markdown_table(["Header"], [["normal"]])
        assert "| normal |" in result


# ── Plugin-Exported Valves Tests ──


class TestPipeDashboardValveGuards:
    """Tests for PIPE_DASHBOARD_ENABLE valve guards in on_models and on_request."""

    def test_on_models_skips_when_disabled(self):
        """on_models does not inject pipe-dashboard when PIPE_DASHBOARD_ENABLE is False."""
        pipe = _make_mock_pipe()
        pipe.valves.PIPE_DASHBOARD_ENABLE = False
        plugin = _make_plugin(pipe)
        models = [{"id": "gpt-4o", "name": "GPT-4o"}]
        asyncio.run(plugin.on_models(models))
        ids = [m["id"] for m in models]
        assert "pipe-dashboard" not in ids

    def test_on_models_injects_when_enabled(self):
        """on_models injects pipe-dashboard when PIPE_DASHBOARD_ENABLE is True."""
        pipe = _make_mock_pipe()
        pipe.valves.PIPE_DASHBOARD_ENABLE = True
        plugin = _make_plugin(pipe)
        models = [{"id": "gpt-4o", "name": "GPT-4o"}]
        asyncio.run(plugin.on_models(models))
        ids = [m["id"] for m in models]
        assert "pipe-dashboard" in ids

    @pytest.mark.asyncio
    async def test_on_request_returns_none_when_disabled(self):
        """on_request returns None for pipe-dashboard model when PIPE_DASHBOARD_ENABLE is False."""
        pipe = _make_mock_pipe()
        pipe.valves.PIPE_DASHBOARD_ENABLE = False
        plugin = _make_plugin(pipe)
        result = await plugin.on_request(
            _make_body("help"), {"role": "admin"}, {}, None, None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_on_request_responds_when_enabled(self):
        """on_request handles pipe-dashboard model when PIPE_DASHBOARD_ENABLE is True."""
        pipe = _make_mock_pipe()
        pipe.valves.PIPE_DASHBOARD_ENABLE = True
        plugin = _make_plugin(pipe)
        result = await plugin.on_request(
            _make_body("help"), {"role": "admin"}, {}, None, None,
        )
        assert isinstance(result, dict)
        content = result["choices"][0]["message"]["content"]
        assert "Pipe Dashboard" in content


class TestPipeDashboardValvesDeclaration:
    """Tests for PipeDashboardPlugin's plugin_valves declaration."""

    def test_declares_system_valves(self):
        """PipeDashboardPlugin declares PIPE_DASHBOARD_ENABLE."""
        assert "PIPE_DASHBOARD_ENABLE" in PipeDashboardPlugin.plugin_valves

    def test_no_user_valves(self):
        """PipeDashboardPlugin declares no user valves."""
        assert len(PipeDashboardPlugin.plugin_user_valves) == 0

    def test_system_valve_types(self):
        """System valve field specs have correct types."""
        enable_spec = PipeDashboardPlugin.plugin_valves["PIPE_DASHBOARD_ENABLE"]
        assert enable_spec[0] is bool


# ── Dashboard Shell Tests ──


class TestDashboardShell:
    """Tests for the fully dynamic dashboard shell emitted by the stats command."""

    def test_dashboard_shell_containers(self):
        """All section container IDs are present in the shell HTML."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        sid = "dash-abc123"
        html = _build_dashboard_shell(sid)
        # Key section IDs (prefixed with dash_id)
        for suffix in [
            "req-val",
            "tool-val",
            "sess-val",
            "rq-val",
            "up-val",
            "rl-section",
            "models-section",
            "storage-section",
            "health-section",
            "config-section",
            "plugins-section",
        ]:
            section_id = f"{sid}-{suffix}"
            assert section_id in html, f"Missing section: {section_id}"

    def test_dashboard_shell_no_static_data(self):
        """Shell should not contain any hardcoded stats values — all dynamic."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-test")
        # No pre-rendered static data tables in the body markup
        body_markup = html.split("</head>")[1].split("<script")[0]
        assert "<table" not in body_markup
        # Live grid values should start with placeholder "-"
        assert 'id="dash-test-req-val"' in html

    def test_dashboard_shell_socket_connection(self):
        """Shell JS opens an authenticated OWUI socket.io connection."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-sock")
        assert "io(" in html
        assert '"/ws/socket.io"' in html
        assert 'localStorage.getItem("token")' in html
        assert "openrouter:pipe_dashboard" in html
        # No SSE remnants; the HTTP action route IS present
        assert "EventSource" not in html
        assert "/api/pipe/dashboard/action" in html

    @pytest.mark.asyncio
    async def test_dashboard_command_returns_fallback_text(self):
        """The chat bubble is never empty — embeds-disabled deployments get a hint."""
        from unittest.mock import Mock as _Mock

        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import handle_dashboard
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.context import CommandContext

        out = await handle_dashboard(CommandContext(pipe=_Mock(), args="", user={}, metadata={}, event_emitter=None))
        assert out.strip()
        assert "iframe" in out

    def test_dashboard_shell_truthful_states(self):
        """Guards the lifecycle/render fixes: server-disconnect handling,
        visible loaders, stale dimming, honest labels, uptime rounding."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-truth")
        assert '"io server disconnect"' in html
        assert "load-system" in html and "load-storage" in html and "load-about" in html
        assert "classList.add('stale')" in html
        assert "In-flight calls" in html
        assert "Redis (pipe)" in html
        assert "Redis cache (artifacts)" not in html
        assert "Least recent" in html
        assert "Users w/ fail" in html
        assert "-degraded" in html
        assert "Math.round(s%60)" not in html
        assert "emitter pid" in html
        assert "refreshes about every 60s" in html
        assert "Active video gens" in html
        assert "db-section" in html

    def test_dashboard_shell_call_action_and_denied(self):
        """Shell exposes callAction (header-only bearer) + a denied listener."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-act")
        assert "callAction" in html
        assert "/api/pipe/dashboard/action" in html
        assert "Authorization" in html and "Bearer" in html
        assert "openrouter:pipe_dashboard:denied" in html
        assert ".textContent" in html

    def test_dashboard_shell_sub_in_user_join_ack(self):
        """The stats subscribe is emitted inside the user-join ACK (race-safe)."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-ack")
        assert 'sock.emit("user-join"' in html
        assert 'sock.emit("openrouter:pipe_dashboard:sub")' in html
        assert html.index('sock.emit("user-join"') < html.index(
            'sock.emit("openrouter:pipe_dashboard:sub")'
        )

    def test_dashboard_shell_embeds_socketio_client(self):
        """The vendored socket.io UMD client is inlined in the head."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard._socketio_client import (
            SOCKETIO_UMD,
        )
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-umd")
        assert SOCKETIO_UMD in html
        assert html.index(SOCKETIO_UMD) < html.index("connectDashboard")

    def test_dashboard_shell_connect_disconnect_buttons(self):
        """Disconnect/Connect buttons and their handlers are present."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-btn")
        assert "dash-btn-btn-disconnect" in html
        assert "dash-btn-btn-connect" in html
        assert "sock.disconnect()" in html
        assert "sock.connect()" in html

    def test_dashboard_shell_token_absent_note(self):
        """Without a readable token the shell degrades to a static notice."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-static")
        assert "iframe same-origin" in html
        assert "setStatic" in html

    def test_dashboard_shell_renamed_headings(self):
        """Section headings use admin-friendly names."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-h")
        assert "Live" in html
        assert "User Circuit Breakers" in html
        assert "Storage" in html
        assert "Health" in html
        assert "Models" in html
        assert "Configuration" in html
        assert "Plugins" in html
        assert "Workers" in html
        # Should NOT use developer-facing names
        assert "Artifact Storage" not in html
        assert "Worker Diagnostics" not in html

    def test_dashboard_shell_workers_section(self):
        """Workers section container and JS updater are present."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-w")
        assert "dash-w-workers-section" in html
        assert "dash-w-workers-panel" in html
        assert "updateWorkers" in html

    def test_dashboard_shell_embeds_ids(self):
        """Shell embeds the dash_id in JavaScript."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-unique99")
        assert "dash-unique99" in html

    def test_dashboard_shell_tab_bar(self):
        """Dashboard shell has a tab bar with Live, System, Storage, About tabs."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-tabs")
        assert 'data-tab="live"' in html
        assert 'data-tab="system"' in html
        assert 'data-tab="storage"' in html
        assert 'data-tab="about"' in html
        assert "tab-pane" in html
        # Live tab is active by default
        assert 'class="tab-btn active" data-tab="live"' in html

    def test_dashboard_shell_queue_fields(self):
        """Dashboard shows log queue and archive queue fields."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
            _build_dashboard_shell,
        )
        html = _build_dashboard_shell("dash-q")
        assert "dash-q-lq-val" in html
        assert "dash-q-aq-val" in html
        assert "Log queue" in html
        assert "Archive queue" in html

    def test_unique_dash_ids(self):
        """Each dashboard instance gets a unique ID prefix."""
        import secrets
        ids = {"dash-" + secrets.token_hex(4) for _ in range(20)}
        assert len(ids) == 20


@pytest.mark.asyncio
async def test_update_service_constructed_and_auto_task_started():
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.update_service import UpdateService

    plugin = _make_plugin()
    assert isinstance(plugin.update_service, UpdateService)
    task = plugin._auto_update_task
    assert task is not None and not task.done()
    result = plugin.on_shutdown()
    if result is not None:
        import asyncio as _aio

        await _aio.gather(result, return_exceptions=True)
    assert task.cancelled() or task.done()


def test_update_plugin_valves_declared_with_defaults():
    pv = PipeDashboardPlugin.plugin_valves
    assert pv["PIPE_DASHBOARD_UPDATE_ENABLE"][1].default is True
    assert pv["PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP"][1].default == 3
    assert pv["PIPE_DASHBOARD_UPDATE_REPO"][1].default == "rbb-dev/Open-WebUI-OpenRouter-pipe"
    assert pv["PIPE_DASHBOARD_UPDATE_AUTO"][1].default is False
    assert pv["PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS"][1].default == 168
