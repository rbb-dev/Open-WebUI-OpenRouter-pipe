# OpenRouter Routers

A **router** is an OpenRouter model that chooses the model that answers. You pick the router in the model picker, and for each request it picks a model, from a pool OpenRouter maintains or from a list you give it. The response names the model that answered, and so does the status line under the reply when it differs from the one you picked. The pipe gives each of the routers below its own settings in Open WebUI, so an admin can limit what a router may pick and what it may spend, and decide which of those limits users may change in their own chats. The settings are generated from the options OpenRouter publishes, so a new option of one of these routers appears without a pipe release.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Provider Routing](openrouter_provider_routing.md) · [Model Variants & Presets](model_variants_and_presets.md) · [Telemetry](openrouter_integrations_and_telemetry.md)

---

## Which routers get settings

| Router | Model id | What OpenRouter says it does |
| --- | --- | --- |
| Auto Router | `openrouter/auto` | Picks a model for each prompt from what the OpenRouter community spends on for that kind of task. |
| Auto Router (Beta) | `openrouter/auto-beta` | The early-access track of the Auto Router. It reads its own settings, not the Auto Router's. |
| Pareto Code Router | `openrouter/pareto-code` | Picks a coding model that meets a minimum coding score. |
| NVIDIA: Switchyard | `nvidia/switchyard` | Switches between a cheaper and a stronger model to keep the cost of a request down. |
| TypeSafe: Jev Router | `typesafe/jev-router` | Lets Jev, TypeSafe's decision model, pick the model and the reasoning effort for each request, inside lists you control. |

Other models that route get no router settings from this feature. OpenRouter's published schema has no per-request settings for the free models router or the `~...-latest` aliases, and Fusion has its own controls (see [OpenRouter Fusion](openrouter_fusion.md)). A routing variant of one of the routers above, such as a `:nitro` row, gets the same settings as the router.

## What an admin gets

When the model list refreshes, the pipe reads OpenRouter's published request schema (`https://openrouter.ai/openapi.json`, a public document read without your API key) and gives each router above an entry named **Router settings: <router name>** in **Admin Panel → Functions**. The entry is attached to the router's model, and it acts only on chats with that model.

Open the entry's valves (the gear icon) to set defaults for every chat with that router:

- Each option OpenRouter publishes is one setting, shown with OpenRouter's own description. A list of choices is a drop-down, a list of models is comma-separated text, and a number carries the minimum and maximum OpenRouter publishes.
- A setting left empty sends nothing. OpenRouter's own default applies, including any default saved on your OpenRouter account (see [Defaults saved at OpenRouter](#defaults-saved-at-openrouter)).
- Next to each setting is a switch named **Users may set: <setting>**. It starts off.

Two OpenRouter settings are not offered. The one that switches a router off has no use here: on the Auto Router, a request that switched the router off failed with OpenRouter's "No endpoints available" error. The one that reuses the model of the previous reply needs the answering model to be remembered from turn to turn, which the pipe does not do.

The entry has no per-chat switch. Once attached it runs in every chat with its router and sends only what has been set, and nothing when nothing is.

The pipe's two settings for this feature are `Install router settings` (`AUTO_INSTALL_ROUTER_FILTERS`) and `Attach router settings to models` (`AUTO_ATTACH_ROUTER_FILTERS`), both on by default. With the first off the pipe stops creating and updating the entries and switches the ones it created off; with the second off it takes them off the models. A single model can opt out of being given its entry with the `disable_router_filter_auto_attach` advanced parameter in its model settings.

## What each router offers

The settings come from OpenRouter. This table is what OpenRouter publishes at the time of writing; the form always shows the current options.

| Router | Settings |
| --- | --- |
| Auto Router and Auto Router (Beta) | Allowed models, Excluded models, Cost tier |
| Pareto Code Router | Max price, Min coding score, Price source |
| NVIDIA: Switchyard | Algorithm, Judge model, Models |
| TypeSafe: Jev Router | Models, Allowed models, Excluded models, Cost tier |

On the Auto routers and the Jev Router, the lists of models take model ids and wildcard patterns, for example `acme/*` for every model from one maker, and an excluded model wins over an allowed one. On Switchyard, **Models** names the candidates it chooses between. OpenRouter limits how many, and its Switchyard guide gives the number. The pipe sends them as the request's fallback list, so each one must also be a model the pipe lets the person use.

## Letting users change a setting

Switch on **Users may set: <setting>** for the setting. A user then opens a chat with that router, opens **Controls**, then **Valves**, chooses **Functions**, picks **Router settings: <router name>** and sets a value. It applies to that user's chats with the router until they change it. Open WebUI keeps these values per user, not per chat.

For each setting the rule is the same:

1. A user's value counts only if the admin switched on **Users may set** for that setting and the user filled it in.
2. Otherwise the admin's value is sent.
3. Otherwise nothing is sent.

A user's value for a setting the admin has not opened is stored and ignored. An empty value keeps the admin's, so a user cannot ask for "no limit" where the admin has set one.

## Costs to watch

A router chooses for you, so what a request costs depends on the model it picks.

- **Pareto Code Router.** With no setting it asks for the top coding tier. In one measurement in October 2026, a one-word reply to a chat that carried about 8,700 tokens of context was answered by a top-tier model and cost $0.087. That is one measurement, not a price list. Set **Min coding score** to move down a tier, or **Max price** to cap the input price in dollars per million tokens; with **Max price** set, OpenRouter selects by price and **Min coding score** is not used.
- **Auto Router and Auto Router (Beta).** **Cost tier** chooses a cost band. OpenRouter's guide says the lowest band favours the cheapest capable models and the highest the most capable ones, whatever the price. The lists of allowed and excluded models narrow what may be picked.
- **TypeSafe: Jev Router.** According to OpenRouter's guide, Jev picks the cheapest candidate that meets the bar it sets for the request. The lists of models and excluded models narrow the pool it picks from.
- **NVIDIA: Switchyard.** The algorithms that can use a judge model (OpenRouter's guide names `capability`, `stage` and `composite`) can make a separate call to a small model when they need one, and OpenRouter bills it to the same account as an ordinary request. The cost OpenRouter reports for the request does not include that call, so the final status line and the dashboard show less than the account is charged. This was observed by comparing the reported cost of judge-backed requests with the rise of the key's usage counter. It is how OpenRouter reports the request, and the pipe does not correct it. `auto`, `random` and `passthrough` make no judge call.

## Defaults saved at OpenRouter

OpenRouter lets an account save default settings for some routers: the Auto Router on the Routing page of the workspace settings, and the Pareto Code Router under Settings, Plugins. Saved values apply when the pipe sends nothing for a setting. When an account has switched on "prevent overrides" for a router, a value the pipe sends does not replace the saved one. An empty setting therefore sends nothing, and never a value of the pipe's own.

## What the settings do not cover

- **Titles, tags and follow-ups.** Open WebUI sends these straight to the model, without running the entries attached to it, so they go out without router settings. So do the models of a Fusion panel.
- **A value OpenRouter does not accept.** OpenRouter refuses it, an unknown choice for instance, and its refusal shows in the chat's error card. Where OpenRouter itself is lenient, so is the request: its Jev Router guide says an include list that matches no model is ignored, so a misspelled entry in **Models** lifts the restriction, while an exclusion is never ignored and a list that leaves no model fails with a 404.
- **Free text.** **Judge model** on Switchyard and **Cost tier** on the Jev Router are free text in OpenRouter's schema. The pipe offers no list of choices for them and sends what is typed.

## Keeping the settings current

- The schema is read again on the model-list refresh once `MODEL_CATALOG_REFRESH_SECONDS` (an hour by default) has passed. A new option appears in the form after that read, and an option OpenRouter drops disappears. A value stored for a dropped option, or one outside the choices and bounds OpenRouter currently publishes, is ignored, and the rest are kept.
- If OpenRouter's schema cannot be read, the entries already installed stay exactly as they are, the options the last good read produced stay in use, and the read is retried after a short delay that grows with each failure. A line in the log says so.
- If OpenRouter stops publishing settings for a router, that router's entry is taken off its model; the entry itself stays in the Functions list.
- A router that OpenRouter adds later gets settings once a pipe release pairs it with its model. Until then the log names the router settings the pipe has no model for.
- An entry switched off by hand in the Functions list stays off. The pipe keeps its code up to date but does not switch it back on.

## Troubleshooting

- **No "Router settings" entry in the Functions list.** Check that `Install router settings` and `Attach router settings to models` are on, that the router is in the model list (see `MODEL_ID`), and that the log has an INFO line saying the pipe read the settings of the routers from OpenRouter's schema. Until the schema has been read once, nothing is installed.
- **A user's value has no effect.** The admin has to switch on **Users may set** for that setting, and the user has to fill the setting in on the entry for the router they are chatting with.
- **The reply came from a model you excluded.** The status line under the reply names the model that answered (see [Telemetry](openrouter_integrations_and_telemetry.md#final-usage-status-line)). Check that the entry is attached to the model you chatted with, that the setting is saved on the entry, and that the OpenRouter account has no saved default that applies.
