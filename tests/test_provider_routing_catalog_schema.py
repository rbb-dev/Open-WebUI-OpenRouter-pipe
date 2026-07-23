import json

import pytest
from aioresponses import aioresponses


FRONTEND_CATALOG_SAMPLE_JSON = r"""
{
  "data": [
    {
      "slug": "openai/gpt-5.1",
      "hf_slug": "",
      "updated_at": "2025-11-13T18:58:25.56227+00:00",
      "created_at": "2025-11-13T18:58:25+00:00",
      "hf_updated_at": null,
      "name": "OpenAI: GPT-5.1",
      "short_name": "GPT-5.1",
      "author": "openai",
      "description": "GPT-5.1 is the latest frontier-grade model in the GPT-5 series, offering stronger general-purpose reasoning, improved instruction adherence, and a more natural conversational style compared to GPT-5. It uses adaptive reasoning to allocate computation dynamically, responding quickly to simple queries while spending more depth on complex tasks. The model produces clearer, more grounded explanations with reduced jargon, making it easier to follow even on technical or multi-step problems.\n\nBuilt for broad task coverage, GPT-5.1 delivers consistent gains across math, coding, and structured analysis workloads, with more coherent long-form answers and improved tool-use reliability. It also features refined conversational alignment, enabling warmer, more intuitive responses without compromising precision. GPT-5.1 serves as the primary full-capability successor to GPT-5",
      "model_version_group_id": null,
      "context_length": 400000,
      "input_modalities": [
        "image",
        "text",
        "file"
      ],
      "output_modalities": [
        "text"
      ],
      "has_text_output": true,
      "group": "GPT",
      "instruct_type": null,
      "default_system": null,
      "default_stops": [],
      "hidden": false,
      "router": null,
      "warning_message": "",
      "promotion_message": "",
      "routing_error_message": null,
      "permaslug": "openai/gpt-5.1-20251113",
      "supports_reasoning": true,
      "reasoning_config": {
        "start_token": null,
        "end_token": null,
        "system_prompt": null
      },
      "features": {
        "reasoning_config": {
          "start_token": null,
          "end_token": null,
          "system_prompt": null
        },
        "chat_template_config": {}
      },
      "default_parameters": {
        "temperature": null,
        "top_p": null,
        "frequency_penalty": null
      },
      "default_order": [],
      "quick_start_example_type": "reasoning",
      "is_trainable_text": null,
      "is_trainable_image": null,
      "endpoint": {
        "id": "764eb97f-8bab-4326-b29b-7a8799b00a70",
        "name": "OpenAI | openai/gpt-5.1-20251113",
        "context_length": 400000,
        "model": {
          "slug": "openai/gpt-5.1",
          "hf_slug": "",
          "updated_at": "2025-11-13T18:58:25.56227+00:00",
          "created_at": "2025-11-13T18:58:25+00:00",
          "hf_updated_at": null,
          "name": "OpenAI: GPT-5.1",
          "short_name": "GPT-5.1",
          "author": "openai",
          "description": "GPT-5.1 is the latest frontier-grade model in the GPT-5 series, offering stronger general-purpose reasoning, improved instruction adherence, and a more natural conversational style compared to GPT-5. It uses adaptive reasoning to allocate computation dynamically, responding quickly to simple queries while spending more depth on complex tasks. The model produces clearer, more grounded explanations with reduced jargon, making it easier to follow even on technical or multi-step problems.\n\nBuilt for broad task coverage, GPT-5.1 delivers consistent gains across math, coding, and structured analysis workloads, with more coherent long-form answers and improved tool-use reliability. It also features refined conversational alignment, enabling warmer, more intuitive responses without compromising precision. GPT-5.1 serves as the primary full-capability successor to GPT-5",
          "model_version_group_id": null,
          "context_length": 400000,
          "input_modalities": [
            "image",
            "text",
            "file"
          ],
          "output_modalities": [
            "text"
          ],
          "has_text_output": true,
          "group": "GPT",
          "instruct_type": null,
          "default_system": null,
          "default_stops": [],
          "hidden": false,
          "router": null,
          "warning_message": "",
          "promotion_message": "",
          "routing_error_message": null,
          "permaslug": "openai/gpt-5.1-20251113",
          "supports_reasoning": true,
          "reasoning_config": {
            "start_token": null,
            "end_token": null,
            "system_prompt": null
          },
          "features": {
            "reasoning_config": {
              "start_token": null,
              "end_token": null,
              "system_prompt": null
            },
            "chat_template_config": {}
          },
          "default_parameters": {
            "temperature": null,
            "top_p": null,
            "frequency_penalty": null
          },
          "default_order": [],
          "quick_start_example_type": "reasoning",
          "is_trainable_text": null,
          "is_trainable_image": null
        },
        "model_variant_slug": "openai/gpt-5.1",
        "model_variant_permaslug": "openai/gpt-5.1-20251113",
        "adapter_name": "OpenAIResponsesAdapter",
        "provider_name": "OpenAI",
        "provider_info": {
          "name": "OpenAI",
          "displayName": "OpenAI",
          "slug": "openai/default",
          "baseUrl": "https://api.openai.com/v1",
          "dataPolicy": {
            "training": false,
            "trainingOpenRouter": false,
            "retainsPrompts": true,
            "canPublish": false,
            "termsOfServiceURL": "https://openai.com/policies/row-terms-of-use/",
            "privacyPolicyURL": "https://openai.com/policies/privacy-policy/",
            "requiresUserIDs": true
          },
          "headquarters": "US",
          "regionOverrides": {},
          "hasChatCompletions": true,
          "hasCompletions": true,
          "isAbortable": true,
          "moderationRequired": true,
          "editors": [
            "{}"
          ],
          "owners": [
            "{}"
          ],
          "adapterName": "OpenAIResponsesAdapter",
          "isMultipartSupported": true,
          "statusPageUrl": "https://status.openai.com/",
          "byokEnabled": true,
          "icon": {
            "url": "/images/icons/OpenAI.svg",
            "className": "invert-0 dark:invert"
          },
          "ignoredProviderModels": [],
          "sendClientIp": false,
          "pricingStrategy": "openai_responses"
        },
        "provider_display_name": "OpenAI",
        "provider_slug": "openai/default",
        "provider_model_id": "gpt-5.1-2025-11-13",
        "quantization": "unknown",
        "variant": "standard",
        "is_free": false,
        "can_abort": true,
        "max_prompt_tokens": 272000,
        "max_completion_tokens": 128000,
        "max_tokens_per_image": null,
        "supported_parameters": [
          "reasoning",
          "include_reasoning",
          "structured_outputs",
          "response_format",
          "seed",
          "max_tokens",
          "tools",
          "tool_choice"
        ],
        "is_byok": false,
        "moderation_required": true,
        "data_policy": {
          "training": false,
          "trainingOpenRouter": false,
          "retainsPrompts": true,
          "canPublish": false,
          "termsOfServiceURL": "https://openai.com/policies/row-terms-of-use/",
          "privacyPolicyURL": "https://openai.com/policies/privacy-policy/",
          "requiresUserIDs": true
        },
        "pricing": {
          "prompt": "0.00000125",
          "completion": "0.00001",
          "input_cache_read": "0.000000125",
          "web_search": "0.01",
          "discount": 0
        },
        "variable_pricings": [],
        "pricing_json": {
          "openai_responses:prompt_tokens": 1.25e-06,
          "openai_responses:web_search_calls": 0.01,
          "openai_responses:completion_tokens": 1e-05,
          "openai_responses:cached_prompt_tokens": 1.25e-07
        },
        "pricing_version_id": "177855b5-4b6e-4f70-a823-4eb03aa1322b",
        "is_hidden": false,
        "is_deranked": false,
        "is_disabled": false,
        "supports_tool_parameters": true,
        "supports_reasoning": true,
        "supports_multipart": true,
        "limit_rpm": null,
        "limit_rpd": null,
        "limit_rpm_cf": null,
        "has_completions": true,
        "has_chat_completions": true,
        "features": {
          "supports_implicit_caching": true,
          "supports_file_urls": true,
          "supports_native_web_search": true,
          "supports_tool_choice": {
            "literal_none": true,
            "literal_auto": true,
            "literal_required": true,
            "type_function": true
          },
          "supported_parameters": {
            "response_format": true,
            "structured_outputs": true
          },
          "is_mandatory_reasoning": false,
          "supports_input_audio": false
        },
        "provider_region": null,
        "deprecation_date": null
      }
    },
    {
      "slug": "anthropic/claude-3.5-sonnet",
      "hf_slug": null,
      "updated_at": "2025-11-10T16:00:38.246665+00:00",
      "created_at": "2024-10-22T00:00:00+00:00",
      "hf_updated_at": null,
      "name": "Anthropic: Claude 3.5 Sonnet",
      "short_name": "Claude 3.5 Sonnet",
      "author": "anthropic",
      "description": "New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal",
      "model_version_group_id": "30636d20-cda3-4a59-aa0c-1a5b6efba072",
      "context_length": 200000,
      "input_modalities": [
        "text",
        "image",
        "file"
      ],
      "output_modalities": [
        "text"
      ],
      "has_text_output": true,
      "group": "Claude",
      "instruct_type": null,
      "default_system": null,
      "default_stops": [],
      "hidden": false,
      "router": null,
      "warning_message": null,
      "promotion_message": null,
      "routing_error_message": null,
      "permaslug": "anthropic/claude-3.5-sonnet",
      "supports_reasoning": false,
      "reasoning_config": {
        "start_token": null,
        "end_token": null,
        "system_prompt": null
      },
      "features": {
        "reasoning_config": {
          "start_token": null,
          "end_token": null,
          "system_prompt": null
        }
      },
      "default_parameters": {},
      "default_order": [],
      "quick_start_example_type": null,
      "is_trainable_text": null,
      "is_trainable_image": null,
      "endpoint": {
        "id": "d4fb79bd-9786-4932-af81-b83040e9f4e4",
        "name": "Amazon Bedrock | anthropic/claude-3.5-sonnet",
        "context_length": 200000,
        "model": {
          "slug": "anthropic/claude-3.5-sonnet",
          "hf_slug": null,
          "updated_at": "2025-11-10T16:00:38.246665+00:00",
          "created_at": "2024-10-22T00:00:00+00:00",
          "hf_updated_at": null,
          "name": "Anthropic: Claude 3.5 Sonnet",
          "short_name": "Claude 3.5 Sonnet",
          "author": "anthropic",
          "description": "New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal",
          "model_version_group_id": "30636d20-cda3-4a59-aa0c-1a5b6efba072",
          "context_length": 200000,
          "input_modalities": [
            "text",
            "image",
            "file"
          ],
          "output_modalities": [
            "text"
          ],
          "has_text_output": true,
          "group": "Claude",
          "instruct_type": null,
          "default_system": null,
          "default_stops": [],
          "hidden": false,
          "router": null,
          "warning_message": null,
          "promotion_message": null,
          "routing_error_message": null,
          "permaslug": "anthropic/claude-3.5-sonnet",
          "supports_reasoning": false,
          "reasoning_config": {
            "start_token": null,
            "end_token": null,
            "system_prompt": null
          },
          "features": {
            "reasoning_config": {
              "start_token": null,
              "end_token": null,
              "system_prompt": null
            }
          },
          "default_parameters": {},
          "default_order": [],
          "quick_start_example_type": null,
          "is_trainable_text": null,
          "is_trainable_image": null
        },
        "model_variant_slug": "anthropic/claude-3.5-sonnet",
        "model_variant_permaslug": "anthropic/claude-3.5-sonnet",
        "adapter_name": "AmazonBedrockInvokeAnthropicAdapter",
        "provider_name": "Amazon Bedrock",
        "provider_info": {
          "name": "Amazon Bedrock",
          "displayName": "Amazon Bedrock",
          "slug": "amazon-bedrock",
          "baseUrl": "not_used",
          "dataPolicy": {
            "training": false,
            "trainingOpenRouter": false,
            "retainsPrompts": false,
            "canPublish": false,
            "termsOfServiceURL": "https://aws.amazon.com/service-terms/",
            "privacyPolicyURL": "https://aws.amazon.com/privacy"
          },
          "headquarters": "US",
          "regionOverrides": {
            "europe": {
              "baseUrl": "dummy-value"
            }
          },
          "hasChatCompletions": true,
          "hasCompletions": false,
          "isAbortable": false,
          "moderationRequired": true,
          "editors": [
            "{}"
          ],
          "owners": [
            "{}"
          ],
          "adapterName": "AmazonBedrockConverseAdapter",
          "isMultipartSupported": true,
          "statusPageUrl": "https://health.aws.amazon.com/health/status",
          "byokEnabled": true,
          "icon": {
            "url": "/images/icons/Bedrock.svg"
          },
          "ignoredProviderModels": [],
          "sendClientIp": false,
          "pricingStrategy": "anthropic"
        },
        "provider_display_name": "Amazon Bedrock",
        "provider_slug": "amazon-bedrock",
        "provider_model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "quantization": "unknown",
        "variant": "standard",
        "is_free": false,
        "can_abort": false,
        "max_prompt_tokens": null,
        "max_completion_tokens": 8192,
        "max_tokens_per_image": null,
        "supported_parameters": [
          "max_tokens",
          "temperature",
          "top_p",
          "top_k",
          "stop",
          "tools",
          "tool_choice"
        ],
        "is_byok": false,
        "moderation_required": true,
        "data_policy": {
          "training": false,
          "trainingOpenRouter": false,
          "retainsPrompts": false,
          "canPublish": false,
          "termsOfServiceURL": "https://aws.amazon.com/service-terms/",
          "privacyPolicyURL": "https://aws.amazon.com/privacy"
        },
        "pricing": {
          "prompt": "0.000006",
          "completion": "0.00003",
          "discount": 0
        },
        "variable_pricings": [],
        "pricing_json": {
          "anthropic:prompt_tokens": 6e-06,
          "anthropic:completion_tokens": 3e-05
        },
        "pricing_version_id": "eb96ddba-9cd5-4a4b-bb70-4f88db92ecb1",
        "is_hidden": false,
        "is_deranked": false,
        "is_disabled": false,
        "supports_tool_parameters": true,
        "supports_reasoning": false,
        "supports_multipart": true,
        "limit_rpm": null,
        "limit_rpd": null,
        "limit_rpm_cf": null,
        "has_completions": false,
        "has_chat_completions": true,
        "features": {
          "supports_tool_choice": {
            "literal_none": true,
            "literal_auto": true,
            "literal_required": true,
            "type_function": true
          }
        },
        "provider_region": null,
        "deprecation_date": null
      }
    }
  ]
}
"""

FRONTEND_CATALOG_SAMPLE = json.loads(FRONTEND_CATALOG_SAMPLE_JSON)


def test_frontend_catalog_sample_includes_variant_slugs() -> None:
    for item in FRONTEND_CATALOG_SAMPLE["data"]:
        endpoint = item["endpoint"]
        assert endpoint["model_variant_slug"]
        assert endpoint["model_variant_slug"] == item["slug"]


def test_frontend_fallback_map_uses_single_featured_endpoint(pipe_instance) -> None:
    """Pins the DEGRADED frontend-only fallback, not the desired end state.

    The frontend catalog returns one row per model with a single featured
    endpoint, so this map yields at most ONE provider per model. The full
    provider list comes from the per-model endpoints API overlay
    (test_overlay_widens_frontend_fallback_providers); this frontend-derived
    map is only the fallback when that fetch fails.
    """
    catalog_manager = pipe_instance._ensure_catalog_manager()
    provider_map = catalog_manager._build_model_provider_map(FRONTEND_CATALOG_SAMPLE)
    assert "openai/gpt-5.1" in provider_map
    assert "anthropic/claude-3.5-sonnet" in provider_map
    assert provider_map["openai/gpt-5.1"]["providers"] == ["openai/default"]
    assert provider_map["anthropic/claude-3.5-sonnet"]["providers"] == ["amazon-bedrock"]


ENDPOINTS_API_SAMPLE_JSON = r"""
{
 "data": {
  "id": "deepseek/deepseek-v3.2",
  "name": "DeepSeek: DeepSeek V3.2",
  "created": 1764594642,
  "endpoints": [
   {
    "name": "StreamLake | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "StreamLake",
    "tag": "streamlake/fp8",
    "quantization": "fp8",
    "status": 0,
    "context_length": 128000,
    "supports_implicit_caching": false
   },
   {
    "name": "Baidu | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Baidu",
    "tag": "baidu/fp8",
    "quantization": "fp8",
    "status": 0,
    "context_length": 131072,
    "supports_implicit_caching": false
   },
   {
    "name": "SiliconFlow | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "SiliconFlow",
    "tag": "siliconflow/fp8",
    "quantization": "fp8",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "DeepInfra | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "DeepInfra",
    "tag": "deepinfra/fp4",
    "quantization": "fp4",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "AtlasCloud | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "AtlasCloud",
    "tag": "atlas-cloud/fp8",
    "quantization": "fp8",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "Novita | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Novita",
    "tag": "novita/fp8",
    "quantization": "fp8",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "GMICloud | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "GMICloud",
    "tag": "gmicloud/fp8",
    "quantization": "fp8",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "Venice | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Venice",
    "tag": "venice",
    "quantization": "unknown",
    "status": 0,
    "context_length": 160000,
    "supports_implicit_caching": false
   },
   {
    "name": "Alibaba | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Alibaba",
    "tag": "alibaba",
    "quantization": "unknown",
    "status": -5,
    "context_length": 131072,
    "supports_implicit_caching": false
   },
   {
    "name": "DigitalOcean | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "DigitalOcean",
    "tag": "digitalocean",
    "quantization": "unknown",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "Friendli | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Friendli",
    "tag": "friendli",
    "quantization": "unknown",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "Google | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Google",
    "tag": "google-vertex",
    "quantization": "unknown",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "Phala | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "Phala",
    "tag": "phala",
    "quantization": "unknown",
    "status": 0,
    "context_length": 163840,
    "supports_implicit_caching": false
   },
   {
    "name": "SambaNova | deepseek/deepseek-v3.2-20251201",
    "model_id": "deepseek/deepseek-v3.2",
    "provider_name": "SambaNova",
    "tag": "sambanova",
    "quantization": "unknown",
    "status": 0,
    "context_length": 32768,
    "supports_implicit_caching": false
   }
  ]
 }
}
"""

ENDPOINTS_API_SAMPLE = json.loads(ENDPOINTS_API_SAMPLE_JSON)

EXPECTED_DEEPSEEK_PROVIDER_SLUGS = [
    "alibaba",
    "atlas-cloud",
    "baidu",
    "deepinfra",
    "digitalocean",
    "friendli",
    "gmicloud",
    "google-vertex",
    "novita",
    "phala",
    "sambanova",
    "siliconflow",
    "streamlake",
    "venice",
]

_DEEPSEEK_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/deepseek/deepseek-v3.2/endpoints"


@pytest.mark.asyncio
async def test_overlay_parses_live_endpoints_shape(pipe_instance_async) -> None:
    """The overlay builder must parse the real per-model endpoints API response.

    Fixture is a trimmed copy of the live response for deepseek/deepseek-v3.2
    (captured 2026-07-23): composite tags like "streamlake/fp8", bare tags like
    "venice", a deranked endpoint (status=-5), and "unknown" quantizations.
    """
    pipe = pipe_instance_async
    manager = pipe._ensure_catalog_manager()
    with aioresponses() as mocked:
        mocked.get(_DEEPSEEK_ENDPOINTS_URL, payload=ENDPOINTS_API_SAMPLE)
        session = pipe._create_http_session()
        try:
            overlay = await manager._build_routed_provider_overlay(
                session, ["deepseek/deepseek-v3.2"]
            )
        finally:
            await session.close()

    entry = overlay["deepseek/deepseek-v3.2"]
    assert entry["providers"] == EXPECTED_DEEPSEEK_PROVIDER_SLUGS
    assert entry["provider_names"]["streamlake"] == "StreamLake"
    assert entry["provider_names"]["atlas-cloud"] == "AtlasCloud"
    assert list(entry["provider_names"].keys()) == sorted(entry["provider_names"].keys())
    assert entry["quantizations"] == ["fp4", "fp8", "unknown"]
    assert entry["short_name"] == "DeepSeek: DeepSeek V3.2"


@pytest.mark.asyncio
async def test_overlay_widens_frontend_fallback_providers(pipe_instance_async) -> None:
    """Valve-listed models get the FULL provider list even though the frontend
    feed only carries one featured endpoint per model row."""
    pipe = pipe_instance_async
    manager = pipe._ensure_catalog_manager()
    endpoints_payload = {
        "data": {
            "id": "openai/gpt-5.1",
            "name": "OpenAI: GPT-5.1",
            "endpoints": [
                {"provider_name": "OpenAI", "tag": "openai", "quantization": "unknown", "status": 0},
                {"provider_name": "Azure", "tag": "azure/eastus", "quantization": "unknown", "status": 0},
                {"provider_name": "Together", "tag": "together/fp8", "quantization": "fp8", "status": 0},
            ],
        }
    }
    with aioresponses() as mocked:
        mocked.get(
            "https://openrouter.ai/api/v1/models/openai/gpt-5.1/endpoints",
            payload=endpoints_payload,
        )
        session = pipe._create_http_session()
        try:
            provider_map = await manager._build_provider_map_with_overlay(
                session, FRONTEND_CATALOG_SAMPLE, "openai/gpt-5.1", ""
            )
        finally:
            await session.close()

    assert provider_map["openai/gpt-5.1"]["providers"] == ["azure", "openai", "together"]
    assert provider_map["openai/gpt-5.1"]["short_name"] == "GPT-5.1"
    assert provider_map["anthropic/claude-3.5-sonnet"]["providers"] == ["amazon-bedrock"]
