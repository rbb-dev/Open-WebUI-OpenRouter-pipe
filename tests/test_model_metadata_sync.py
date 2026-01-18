import pytest
from unittest.mock import AsyncMock, Mock, patch

from open_webui_openrouter_pipe import Pipe


def test_build_icon_mapping_success(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "anthropic/claude-3.5-sonnet",
                "endpoint": {
                    "provider_info": {
                        "icon": {"url": "https://example.com/claude.png"},
                    }
                },
            },
            {
                "slug": "openai/gpt-4o",
                "endpoint": {
                    "provider_info": {
                        "icon": {
                            "url": "/images/icons/OpenAI.svg",  # relative URL
                            "className": "something",
                        },
                        "baseUrl": "https://api.openai.com/v1",
                    }
                },
            },
            {
                "slug": "mistral/mistral-small",
                "endpoint": {"provider_info": {"icon": {"url": "/images/icons/Mistral.png"}}},
            },
            {
                "slug": "meta/llama-3.1-70b",
                "endpoint": {"provider_info": {"icon": None}},  # no icon
            },
        ]
    }

    icon_mapping = pipe._build_icon_mapping(frontend_data)

    assert len(icon_mapping) == 3
    assert icon_mapping["anthropic/claude-3.5-sonnet"] == "https://example.com/claude.png"
    assert icon_mapping["openai/gpt-4o"] == "https://openrouter.ai/images/icons/OpenAI.svg"
    assert icon_mapping["mistral/mistral-small"] == "https://openrouter.ai/images/icons/Mistral.png"


def test_build_icon_mapping_empty(pipe_instance):
    pipe = pipe_instance

    assert pipe._build_icon_mapping(None) == {}
    assert pipe._build_icon_mapping({"data": []}) == {}
    assert pipe._build_icon_mapping({}) == {}


def test_extract_openrouter_og_image():
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://openrouter.ai/openai/opengraph-image-abc123?token=xyz"/>'
        "</head></html>"
    )
    assert Pipe._extract_openrouter_og_image(html) == "https://openrouter.ai/openai/opengraph-image-abc123?token=xyz"


def test_guess_image_mime_type_svg_and_png():
    assert (
        Pipe._guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI.svg",
            content_type=None,
            data=b"<svg></svg>",
        )
        == "image/svg+xml"
    )
    assert (
        Pipe._guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI",
            content_type="image/svg+xml; charset=utf-8",
            data=b"<svg></svg>",
        )
        == "image/svg+xml"
    )

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    assert (
        Pipe._guess_image_mime_type(
            "https://openrouter.ai/images/icons/OpenAI.png",
            content_type=None,
            data=png_bytes,
        )
        == "image/png"
    )


def test_build_icon_mapping_uses_first_provider_icon(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "dup/model",
                "endpoint": {"provider_info": {"icon": {"url": "https://example.com/first.png"}}},
            },
            {
                "slug": "dup/model",
                "endpoint": {"provider_info": {"icon": {"url": "https://example.com/second.png"}}},
            },
        ]
    }

    icon_mapping = pipe._build_icon_mapping(frontend_data)
    assert icon_mapping["dup/model"] == "https://example.com/first.png"


def test_build_web_search_support_mapping_detects_multiple_signals(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "x-ai/grok-4",
                "endpoint": {
                    "features": {"supports_native_web_search": True},
                    "supported_parameters": [],
                    "pricing": {"web_search": "0"},
                },
            },
            {
                "slug": "openai/gpt-4o",
                "endpoint": {
                    "features": {},
                    "supported_parameters": ["web_search_options"],
                    "pricing": {"web_search": "0"},
                },
            },
            {
                "slug": "anthropic/claude-opus-4.5",
                "endpoint": {
                    "features": {"supports_native_web_search": False},
                    "supported_parameters": [],
                    "pricing": {"web_search": "0.01"},
                },
            },
            {
                "slug": "nope/nope",
                "endpoint": {"features": {}, "supported_parameters": [], "pricing": {"web_search": "0"}},
            },
        ]
    }

    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {
        "x-ai/grok-4": True,
        "openai/gpt-4o": True,
        "anthropic/claude-opus-4.5": True,
    }


def test_build_web_search_support_mapping_unions_duplicates(pipe_instance):
    pipe = pipe_instance

    frontend_data = {
        "data": [
            {
                "slug": "dup/model",
                "endpoint": {"features": {}, "supported_parameters": [], "pricing": {"web_search": "0"}},
            },
            {
                "slug": "dup/model",
                "endpoint": {
                    "features": {},
                    "supported_parameters": ["web_search_options"],
                    "pricing": {"web_search": "0"},
                },
            },
        ]
    }

    mapping = pipe._build_web_search_support_mapping(frontend_data)
    assert mapping == {"dup/model": True}


@pytest.mark.asyncio
async def test_sync_model_metadata_prefixes_pipe_id_and_prefers_icon_mapping(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    models = [
        {
            "id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
            "capabilities": {"vision": True},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(
        return_value={
            "data": [
                {
                    "slug": "openai/gpt-4o",
                    "endpoint": {
                        "provider_info": {
                            "icon": {"url": "/images/icons/OpenAI.svg"},
                            "baseUrl": "https://api.openai.com/v1",
                        }
                    },
                }
            ]
        }
    )

    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={})
    pipe._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.pipe.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[0] == "open_webui_openrouter_pipe.openai.gpt-4o"
    assert args[1] == "GPT-4o"
    assert args[2] == {"vision": True, "web_search": True}
    assert args[3].startswith("data:image/")

@pytest.mark.asyncio
async def test_sync_model_metadata_sets_web_search_from_frontend(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_IMAGES = False
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True

    models = [
        {
            "id": "x-ai.grok-4",
            "name": "Grok 4",
            "original_id": "x-ai/grok-4",
            "capabilities": {"web_search": False},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(
        return_value={
            "data": [
                {
                    "slug": "x-ai/grok-4",
                    "endpoint": {
                        "features": {"supports_native_web_search": True},
                        "supported_parameters": [],
                        "pricing": {"web_search": "0"},
                    },
                }
            ]
        }
    )

    pipe._ensure_catalog_manager()


    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.pipe.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[0] == "open_webui_openrouter_pipe.x-ai.grok-4"
    assert args[2] == {"web_search": True}
    assert args[3] is None
    assert args[4] is True
    assert args[5] is False


@pytest.mark.asyncio
async def test_sync_model_metadata_falls_back_to_maker_image_mapping(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_IMAGES = True
    pipe.valves.UPDATE_MODEL_CAPABILITIES = False

    models = [
        {
            "id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
            "capabilities": {"vision": True},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={"data": [{"slug": "openai/gpt-4o"}]})
    pipe._build_maker_profile_image_mapping = AsyncMock(return_value={"openai": "https://example.com/openai.png"})
    pipe._fetch_image_as_data_url = AsyncMock(return_value="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.pipe.run_in_threadpool", new=fake_run_in_threadpool):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    args = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args[0]
    assert args[0] == "open_webui_openrouter_pipe.openai.gpt-4o"
    assert args[2] is None
    assert args[3].startswith("data:image/")


@pytest.mark.asyncio
async def test_sync_model_metadata_includes_description_when_enabled(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves = pipe.Valves(
        UPDATE_MODEL_IMAGES=False,
        UPDATE_MODEL_CAPABILITIES=False,
        UPDATE_MODEL_DESCRIPTIONS=True,
        AUTO_ATTACH_ORS_FILTER=False,
        AUTO_INSTALL_ORS_FILTER=False,
        AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=False,
        AUTO_ATTACH_DIRECT_UPLOADS_FILTER=False,
        AUTO_INSTALL_DIRECT_UPLOADS_FILTER=False,
    )

    models = [
        {
            "id": "openai.gpt-4o",
            "norm_id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
        }
    ]

    pipe._ensure_catalog_manager()


    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.registry.ModelFamily._lookup_spec", return_value={"description": "Catalog description"}), patch(
        "open_webui_openrouter_pipe.pipe.run_in_threadpool",
        new=fake_run_in_threadpool,
    ):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    kwargs = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args.kwargs
    assert kwargs["update_descriptions"] is True
    assert kwargs["description"] == "Catalog description"


@pytest.mark.asyncio
async def test_sync_model_metadata_skips_description_when_disabled(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves = pipe.Valves(
        UPDATE_MODEL_IMAGES=False,
        UPDATE_MODEL_CAPABILITIES=True,
        UPDATE_MODEL_DESCRIPTIONS=False,
        AUTO_ATTACH_ORS_FILTER=False,
        AUTO_INSTALL_ORS_FILTER=False,
        AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=False,
        AUTO_ATTACH_DIRECT_UPLOADS_FILTER=False,
        AUTO_INSTALL_DIRECT_UPLOADS_FILTER=False,
    )

    models = [
        {
            "id": "openai.gpt-4o",
            "norm_id": "openai.gpt-4o",
            "name": "GPT-4o",
            "original_id": "openai/gpt-4o",
            "capabilities": {"vision": True},
        }
    ]

    pipe._fetch_frontend_model_catalog = AsyncMock(return_value={"data": []})
    pipe._ensure_catalog_manager()

    pipe._catalog_manager._update_or_insert_model_with_metadata = Mock()

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("open_webui_openrouter_pipe.registry.ModelFamily._lookup_spec", return_value={"description": "Catalog description"}), patch(
        "open_webui_openrouter_pipe.pipe.run_in_threadpool",
        new=fake_run_in_threadpool,
    ):
        await pipe._sync_model_metadata_to_owui(models, pipe_identifier="open_webui_openrouter_pipe")

    pipe._catalog_manager._update_or_insert_model_with_metadata.assert_called_once()
    kwargs = pipe._catalog_manager._update_or_insert_model_with_metadata.call_args.kwargs
    assert kwargs["update_descriptions"] is False
    assert kwargs["description"] is None
