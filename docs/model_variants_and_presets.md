# Model Variants & Presets

OpenRouter's model variants allow you to access specialized routing behaviors by appending variant suffixes to model IDs. OpenRouter presets let you save and reuse LLM configurations (system prompts, provider routing, parameters) without changing code.

This pipe supports creating virtual model catalog entries that make both variants and presets fully discoverable in the Open WebUI model selector.

---

## What Are Model Variants?

Model variants are specialized versions of base models that modify routing behavior, performance characteristics, or capabilities. OpenRouter provides six variant types:

| Variant | Tag | Purpose | Availability |
|---------|-----|---------|--------------|
| **Free** | `:free` | Access free/community versions | Model-specific |
| **Thinking** | `:thinking` | Extended reasoning capabilities | Model-specific |
| **Online** | `:online` | Real-time web search | Universal |
| **Nitro** | `:nitro` | High-speed inference | Universal |
| **Exacto** | `:exacto` | Better tool-calling accuracy | Select models only |
| **Extended** | `:extended` | Extended context windows | Model-specific |

---

## Configuration

### VARIANT_MODELS Valve

The `VARIANT_MODELS` valve accepts a comma-separated list of variant model entries in the format `base_id:variant_tag`.

**Location:** Pipe Settings → VARIANT_MODELS

**Format:**
```
base_model_id:variant_tag,another_model_id:variant_tag
```

**Example:**
```
openai/gpt-4o:exacto,anthropic/claude-sonnet-4.5:extended,deepseek/deepseek-r1:thinking
```

### How It Works

When you configure variant models:

1. **Parsing:** The pipe splits the CSV and extracts base model IDs and variant tags
2. **Lookup:** For each entry, it finds the matching base model in the catalog
3. **Cloning:** Creates a virtual model entry that inherits all metadata from the base:
   - Description
   - Icon
   - Capabilities (vision, audio, video, tools)
   - Pricing information
   - Context length
4. **Naming:** Appends the variant tag to the display name
5. **Catalog:** Adds the virtual model to the catalog alongside the base model

**Result:** Both the base model and its variant(s) appear as separate, fully-functional entries in the model selector.

---

## Examples

### Development & Testing (Free Tier)

Access free versions of models for development and testing:

```
VARIANT_MODELS = "meta-llama/llama-3.2-3b-instruct:free,qwen/qwen-2.5-7b-instruct:free"
```

**Result in Model Selector:**
- Meta Llama 3.2 3b Instruct
- **Meta Llama 3.2 3b Instruct Free** ← New virtual entry
- Qwen 2.5 7b Instruct
- **Qwen 2.5 7b Instruct Free** ← New virtual entry

### Production (Speed + Accuracy)

Optimize for low latency and tool-calling reliability:

```
VARIANT_MODELS = "openai/gpt-4o:nitro,anthropic/claude-opus:exacto"
```

**Result:**
- GPT-4o
- **GPT-4o Nitro** ← Prioritizes speed
- Claude Opus
- **Claude Opus Exacto** ← Better tool accuracy

### Research (Extended Context + Reasoning)

Enable larger context windows and advanced reasoning:

```
VARIANT_MODELS = "anthropic/claude-sonnet-4.5:extended,deepseek/deepseek-r1:thinking"
```

**Result:**
- Claude Sonnet 4.5
- **Claude Sonnet 4.5 Extended** ← Larger context
- DeepSeek R1
- **DeepSeek R1 Thinking** ← Extended reasoning

---

## Variant Details

### Static Variants (Model-Specific)

#### `:free` - Free/Community Versions
- **Purpose:** Access free tiers of models
- **Considerations:** May have different rate limits or availability windows
- **Best for:** Development, testing, experimentation
- **Example:** `meta-llama/llama-3.2-3b-instruct:free`
- **Reference:** [OpenRouter Free Variant Docs](https://openrouter.ai/docs/guides/routing/model-variants/free)

#### `:thinking` - Extended Reasoning
- **Purpose:** Enables extended reasoning capabilities for complex problem-solving
- **Considerations:** May have higher latency due to additional processing
- **Best for:** Complex analysis, step-by-step problem solving, research tasks
- **Example:** `deepseek/deepseek-r1:thinking`
- **Reference:** [OpenRouter Thinking Variant Docs](https://openrouter.ai/docs/guides/routing/model-variants/thinking)

#### `:exacto` - Tool-Calling Accuracy
- **Purpose:** Routes to providers with measurably better tool-calling success rates
- **How it works:** Uses curated provider allowlist based on tool-calling telemetry
- **Best for:** Agentic workflows, function calling, structured outputs
- **Currently supported models:**
  - Kimi K2 (`moonshotai/kimi-k2-0905:exacto`)
  - DeepSeek v3.1 Terminus (`deepseek/deepseek-v3.1-terminus:exacto`)
  - GLM 4.6 (`z-ai/glm-4.6:exacto`)
  - GPT-OSS 120B (`openai/gpt-oss-120b:exacto`)
  - Qwen3 Coder (`qwen/qwen3-coder:exacto`)
- **Example:** `openai/gpt-oss-120b:exacto`
- **Reference:** [OpenRouter Exacto Variant Docs](https://openrouter.ai/docs/guides/routing/model-variants/exacto)

#### `:extended` - Extended Context
- **Purpose:** Access larger context window versions of models
- **Considerations:** May have higher costs per token
- **Best for:** Long document analysis, multi-document reasoning, extensive context retention
- **Example:** `anthropic/claude-sonnet-4.5:extended`
- **Reference:** [OpenRouter Extended Variant Docs](https://openrouter.ai/docs/guides/routing/model-variants/extended)

### Dynamic Variants (Universal)

#### `:online` - Real-Time Web Search
- **Purpose:** Enables real-time web search capabilities
- **Availability:** Works with all models
- **Equivalent to:** OpenRouter Search filter/toggle
- **Best for:** Current events, fact-checking, research requiring fresh data
- **Example:** `openai/gpt-5.2:online`
- **Reference:** [OpenRouter Online Variant Docs](https://openrouter.ai/docs/guides/routing/model-variants/online)

#### `:nitro` - High-Speed Inference
- **Purpose:** Prioritizes speed and low latency
- **Availability:** Works with all models
- **Trade-offs:** May sacrifice some quality for speed (provider-dependent)
- **Best for:** Real-time applications, interactive experiences, latency-sensitive workloads
- **Example:** `anthropic/claude-opus:nitro`
- **Reference:** [OpenRouter Nitro Variant Docs](https://openrouter.ai/docs/guides/routing/model-variants/nitro)

---

## User Experience

### In the Model Selector

When you configure variant models, they appear as fully-discoverable entries:

```
@ [type to search models]

Search results:
  🤖 GPT-4o                          [Base model]
  🤖 GPT-4o Exacto                   [Virtual variant - same icon]
  🧠 Claude Sonnet 4.5               [Base model]
  🧠 Claude Sonnet 4.5 Extended      [Virtual variant - same icon]
  🔬 DeepSeek R1                     [Base model]
  🔬 DeepSeek R1 Thinking            [Virtual variant - same icon]
```

### Metadata Inheritance

Variant models automatically inherit all properties from their base model:

| Property | Inherited? | Notes |
|----------|-----------|-------|
| **Icon** | ✅ Yes | Same visual representation |
| **Description** | ✅ Yes | Identical model description |
| **Capabilities** | ✅ Yes | Vision, audio, video, tools - all preserved |
| **Pricing** | ✅ Yes | Same pricing structure (unless variant modifies it) |
| **Context Length** | ✅ Yes | Base context (`:extended` may increase this) |
| **Display Name** | ⚙️ Modified | Appends variant tag (e.g., "Exacto", "Free") |
| **Model ID** | ⚙️ Modified | Includes `:variant` suffix for API routing |

### API Behavior

When you select a variant model:

1. **User selects:** "GPT-4o Exacto" from model selector
2. **Internal ID:** `openai.gpt-4o:exacto` (sanitized for Open WebUI)
3. **API request:** `{"model": "openai/gpt-4o:exacto"}` (sent to OpenRouter)
4. **OpenRouter routing:** Applies exacto-specific provider filtering
5. **Response:** Returns normally through the pipe

---

## Integration with Existing Features

### Model Selection (MODEL_ID Valve)

Variant models work seamlessly with existing model selection:

**Auto selection (`MODEL_ID = "*"`):**
- Base models AND their variants both appear in catalog
- Example: Both "GPT-4o" and "GPT-4o Exacto" are available

**Specific selection (`MODEL_ID = "openai/gpt-4o,anthropic/claude-opus"`):**
- If base model is selected, its configured variants are added automatically
- Example: Selecting `gpt-4o` enables both "GPT-4o" and "GPT-4o Exacto"

**Pattern matching (`MODEL_ID = "gpt*"`):**
- Variants inherit pattern matching from base models
- Example: `gpt*` matches both "GPT-4o" and "GPT-4o Exacto"

### Model Filters

Variant models respect all existing filters:

**FREE_MODEL_FILTER:**
- `:free` variants are tagged as free models
- Filter settings apply to both base and variant

**TOOL_CALLING_FILTER:**
- `:exacto` variants inherit tool-calling capability from base
- Filter applies based on base model's capabilities

**ENABLE_REASONING:**
- `:thinking` variants work with reasoning token extraction
- All reasoning features apply automatically

### Direct Upload Functionality

Variant models inherit file/audio/video capabilities:

```
VARIANT_MODELS = "openai/gpt-4o:exacto"
```

If base `gpt-4o` supports vision:
- ✅ "GPT-4o Exacto" also supports vision
- ✅ Direct Upload filter works identically
- ✅ Capability checks use base model's metadata

### Tools & Function Calling

Variant models inherit tool-calling capabilities:

- **Base model supports tools** → **Variant supports tools**
- `:exacto` variants are specifically optimized for tool accuracy
- Open WebUI tool integrations work unchanged
- OpenRouter native tools work unchanged

---

## Troubleshooting

### Variant Model Not Appearing

**Symptom:** Configured variant doesn't show up in model selector

**Possible causes:**

1. **Base model not in catalog:**
   ```
   VARIANT_MODELS = "nonexistent/model:exacto"
   ```
   **Fix:** Check pipe logs for warning: "Variant model base not found"
   **Solution:** Verify base model ID is correct and model is available in your catalog

2. **Malformed CSV entry:**
   ```
   VARIANT_MODELS = "openai/gpt-4o-exacto"  ❌ Missing colon
   ```
   **Fix:** Use correct format with colon separator:
   ```
   VARIANT_MODELS = "openai/gpt-4o:exacto"  ✅
   ```

3. **Base model filtered out:**
   ```
   MODEL_ID = "claude*"
   VARIANT_MODELS = "openai/gpt-4o:exacto"
   ```
   **Issue:** Base GPT-4o is filtered out by MODEL_ID pattern
   **Solution:** Include base model in MODEL_ID selection

### Variant Tag Capitalization

**Symptom:** Display name shows lowercase tag

This should not occur - the pipe automatically capitalizes tags. If you see:
- ❌ "GPT-4o exacto"

Expected behavior:
- ✅ "GPT-4o Exacto"

**Debug:** Check pipe logs for variant expansion messages

### API Request Issues

**Symptom:** Variant suffix not being sent to OpenRouter

**Diagnosis:**
Check orchestrator logs for model ID in API request. Should see:
```
api_model_id: openai/gpt-4o:exacto
```

If you see `openai/gpt-4o` (no suffix), this indicates a bug in `api_model_id()` mapping.

**Verification:**
Unit test `test_api_model_id_with_variant()` in [tests/test_variant_models.py](../tests/test_variant_models.py) should pass.

---

## Best Practices

### Organizing Variants

**For development teams:**
```
VARIANT_MODELS = "meta-llama/llama-3.2-3b-instruct:free,qwen/qwen-2.5-7b-instruct:free"
```
↳ Provides cost-effective options for testing

**For production deployments:**
```
VARIANT_MODELS = "openai/gpt-4o:nitro,anthropic/claude-opus:exacto"
```
↳ Optimizes for speed and tool reliability

**For research workloads:**
```
VARIANT_MODELS = "anthropic/claude-sonnet-4.5:extended,deepseek/deepseek-r1:thinking"
```
↳ Maximizes context and reasoning capabilities

### Performance Considerations

**Catalog size:**
- Each variant adds one entry to the catalog
- 10 base models + 10 variants = 20 total entries
- No significant performance impact (catalog is loaded once)

**Memory overhead:**
- Variant models use shallow copy (minimal memory)
- Shared metadata references (icon, description)
- Estimated: ~1KB per variant entry

### Cost Management

**Free tier variants:**
```
VARIANT_MODELS = "meta-llama/llama-3.2-3b-instruct:free"
```
↳ Good for:
- Development and testing
- High-volume, low-stakes queries
- Cost-conscious deployments

**Premium variants:**
```
VARIANT_MODELS = "openai/gpt-4o:exacto"
```
↳ Consider for:
- Production tool-calling workflows
- High-accuracy requirements
- Critical business applications

**Monitor costs:**
- Check OpenRouter dashboard for per-model spending
- Variant costs may differ from base model
- Use COST_DISPLAY valve to show per-request costs

---

## Advanced Use Cases

### Combining with Model Selection Patterns

**Scenario:** Offer free variants for all Llama models

```
MODEL_ID = "meta-llama/*"
VARIANT_MODELS = "meta-llama/llama-3.2-3b-instruct:free,meta-llama/llama-3.2-1b-instruct:free"
```

**Result:**
- All Llama models appear in catalog
- Free variants appear for configured models
- Users can choose between paid and free versions

### Multi-Variant Strategy

**Scenario:** Provide multiple routing options for key models

```
VARIANT_MODELS = "openai/gpt-4o:exacto,openai/gpt-4o:nitro,openai/gpt-4o:online"
```

**Result:**
- GPT-4o (standard routing)
- GPT-4o Exacto (tool accuracy)
- GPT-4o Nitro (speed)
- GPT-4o Online (web search)

Users can choose the routing behavior that fits their task.

### Tool-Calling Optimization

**Scenario:** Maximize tool-calling reliability for agentic workflows

```
TOOL_CALLING_FILTER = "only"
VARIANT_MODELS = "moonshotai/kimi-k2-0905:exacto,deepseek/deepseek-v3.1-terminus:exacto"
```

**Result:**
- Only tool-capable models appear
- Exacto variants provide best-in-class tool accuracy
- Perfect for agents, function calling, structured extraction

---

## Presets

[OpenRouter Presets](https://openrouter.ai/docs/guides/features/presets) allow you to separate your LLM configuration from your code. Create and manage presets through the OpenRouter web application to control provider routing, model selection, system prompts, and other parameters, then reference them in API requests.

### What Are Presets?

Presets are named configurations that encapsulate all the settings needed for a specific use case. For example, you might create:

- An **email-copywriter** preset for generating marketing copy
- An **inbound-classifier** preset for categorizing customer inquiries
- A **code-reviewer** preset for analyzing pull requests

Each preset can manage:

- Provider routing preferences (sort by price, latency, etc.)
- Model selection (specific model or array of models with fallbacks)
- System prompts
- Generation parameters (temperature, top_p, etc.)
- Provider inclusion/exclusion rules

### Supported Preset Methods

OpenRouter provides three ways to use presets. **This pipe supports two of them:**

| Method | Syntax | Supported | Recommended | Notes |
|--------|--------|-----------|-------------|-------|
| **Combined Model and Preset** | `model@preset/slug` | ✅ Yes | ⭐ **Yes** | Use in `VARIANT_MODELS` valve |
| **Preset Field** | `"preset": "slug"` | ⚠️ Partial | No | OpenRouter may not honor; see below |
| **Direct Model Reference** | `@preset/slug` | ❌ No | — | Requires major architectural changes |

#### Method 1: Combined Model and Preset (VARIANT_MODELS) — Recommended

Create virtual model entries that combine a base model with a preset, making them discoverable in the model selector. **This is the recommended approach** because it embeds the preset in the model ID, allowing the pipe to use OpenRouter's more feature-rich `/responses` endpoint.

**Configuration:**
```
VARIANT_MODELS = "openai/gpt-4o@preset/email-copywriter,anthropic/claude-sonnet-4@preset/code-reviewer"
```

**Syntax:** `base_model_id@preset/preset-slug`

**Result in Model Selector:**
- OpenAI: GPT-4o
- **OpenAI: GPT-4o Preset: email-copywriter** ← New virtual entry
- Anthropic: Claude Sonnet 4
- **Anthropic: Claude Sonnet 4 Preset: code-reviewer** ← New virtual entry

#### Method 2: Preset Field (Custom Parameter)

Set the `preset` parameter in a model's Advanced Settings to apply a preset to all requests using that model.

> **⚠️ Known Issue:** During testing, OpenRouter was not honoring the `preset` field when sent via `/chat/completions`. The pipe correctly sends the `preset` parameter in the request payload, but OpenRouter may not apply the preset configuration. **We recommend using Method 1 (VARIANT_MODELS) instead**, which embeds the preset in the model ID and works reliably. If you need to use this method, contact [OpenRouter Support](https://openrouter.ai/docs) for assistance.

> **⚠️ Endpoint Limitation:** This method forces the pipe to use OpenRouter's `/chat/completions` endpoint instead of the more feature-rich `/responses` endpoint. This is because the `preset` field is only supported on `/chat/completions`. For this reason, **Method 1 (VARIANT_MODELS) is recommended** when possible.

**Configuration:**
1. Go to **Admin → Models → [Model Name] → Settings**
2. Under **Advanced Parameters**, add: `preset: your-preset-slug`
3. Save the model settings

**How it works:**
- The pipe detects the `preset` parameter in the request body
- Automatically routes the request to `/chat/completions` (required for preset field support)
- The pipe sends the correct `preset` field to OpenRouter

**When to use:**
- When Method 1 (VARIANT_MODELS) is not available
- When you can't modify the `VARIANT_MODELS` valve
- When OpenRouter resolves the preset field issue

### Preset Display Names

Preset models use a distinctive display format to differentiate them from built-in variants:

| Type | Input | Display Name |
|------|-------|--------------|
| Variant | `openai/gpt-4o:nitro` | OpenAI: GPT-4o **Nitro** |
| Preset | `openai/gpt-4o@preset/email-copywriter` | OpenAI: GPT-4o **Preset: email-copywriter** |

The `Preset:` label makes it clear which models are using user-defined presets vs OpenRouter's built-in variants.

### Mixing Variants and Presets

You can configure both variants and presets in the same `VARIANT_MODELS` valve:

```
VARIANT_MODELS = "openai/gpt-4o:nitro,openai/gpt-4o@preset/email-copywriter,anthropic/claude-opus:exacto"
```

**Result:**
- OpenAI: GPT-4o Nitro (variant - high speed)
- OpenAI: GPT-4o Preset: email-copywriter (preset - custom config)
- Anthropic: Claude Opus Exacto (variant - tool accuracy)

### Preset API Behavior

When you select a preset model:

1. **User selects:** "GPT-4o Preset: email-copywriter" from model selector
2. **Internal ID:** `openai.gpt-4o:preset/email-copywriter` (sanitized for Open WebUI)
3. **API request:** `{"model": "openai/gpt-4o@preset/email-copywriter"}` (sent to OpenRouter)
4. **OpenRouter:** Applies the preset configuration before processing

Note: The internal separator (`:`) is converted to `@` when sending to OpenRouter, matching their API expectations.

### Creating Presets

Presets are created and managed in your [OpenRouter dashboard](https://openrouter.ai/settings/presets):

1. Go to **Settings → Presets** in OpenRouter
2. Click **Create Preset**
3. Configure your desired settings (model, system prompt, temperature, etc.)
4. Save with a memorable slug (e.g., `email-copywriter`)
5. Reference it in this pipe using `@preset/your-slug`

### Preset Limitations

1. **Presets are account-specific:**
   - Presets are tied to your OpenRouter account
   - Team members need access to the same OpenRouter organization to use shared presets

2. **No preset discovery:**
   - The pipe cannot auto-discover your available presets
   - You must manually configure preset entries in `VARIANT_MODELS`

3. **Direct Model Reference not supported:**
   - Using `@preset/slug` as the entire model ID is not supported
   - Presets must be combined with a base model

---

## Limitations & Future Enhancements

### Current Limitations

1. **No variant combination:**
   - Cannot use multiple variants simultaneously
   - Example: `:exacto:nitro` is not supported
   - Workaround: Choose the primary routing behavior needed

2. **Manual configuration:**
   - Variants must be manually listed in VARIANT_MODELS
   - No auto-discovery of available variants per model

3. **No per-user variant preferences:**
   - VARIANT_MODELS is global (applies to all users)
   - Users cannot configure individual variant preferences

### Planned Enhancements

**Auto-discovery (future):**
```
VARIANT_MODELS = "*:free"  # Create free variants for all models
```

**Per-user preferences (future):**
```python
class UserValves:
    PREFERRED_VARIANT: str = "free"  # Auto-apply :free to all models
```

**Variant-specific metadata (future):**
- Different pricing for variants (if OpenRouter provides)
- Variant-specific capability differences
- Provider allowlist visibility

---

## Reference

### OpenRouter Documentation

- [Model Variants Overview](https://openrouter.ai/docs/guides/routing/model-variants)
- [Presets Guide](https://openrouter.ai/docs/guides/features/presets)
- [Exacto Variant Details](.external/openrouter_docs/guides/routing/model-variants/exacto.md)
- [Thinking Variant Details](.external/openrouter_docs/guides/routing/model-variants/thinking.md)

### Related Pipe Documentation

- [Configuration Reference](valves_and_configuration_atlas.md) - All valve settings
- [Model Selection](valves_and_configuration_atlas.md#model_id) - MODEL_ID valve details
- [Tool Integration](tooling_and_integrations.md) - Tool-calling features
- [Cost Tracking](openrouter_integrations_and_telemetry.md) - Billing and attribution

### Test Coverage

Unit tests: [tests/test_variant_models.py](../tests/test_variant_models.py)
- `test_expand_single_variant()` - Basic expansion
- `test_expand_multiple_variants()` - Multiple variants
- `test_expand_variant_capitalization()` - Tag formatting
- `test_api_model_id_with_variant()` - API ID preservation
- `test_expand_variant_preserves_metadata()` - Metadata inheritance

---

## Support

If you encounter issues with model variants:

1. **Check pipe logs** for variant expansion and API request details
2. **Verify base model** is available in your catalog
3. **Review test suite** for expected behavior examples
4. **Report issues** at [GitHub Issues](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues)

---

*This feature enables full discoverability and usability of OpenRouter's variant routing system and user-defined presets within Open WebUI's native interface.*
