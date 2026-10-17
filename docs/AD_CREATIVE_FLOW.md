# Competitor-ad selection, analysis, and ad prompts

> **Audience:** Engineers touching the competitor-ad path, and whoever is asked
> "can it copy this competitor's ad?".
> **Purpose:** The five-step flow from ad-library search to a copyable *original*
> prompt, the vision-model configuration it depends on, and the safety rules that
> keep a competitor's logo and copy out of the output.

The assistant flow keeps competitor research and vision calls separated:

1. An explicit competitor-ad request searches the public Meta Ad Library.
2. Up to six safe image previews are staged in a workspace-scoped cache for 24 hours. The chat presents those previews through the authenticated `GET /api/v1/assistant/ad-previews/{preview_id}` route; a browser `<img>` never receives a Meta snapshot URL or token.
3. The assistant does not send any preview to a model before the user selects one. On selection, only that image is sent to the isolated vision model.
4. The visual analysis is stored in the conversation's existing JSON message blocks (no DB migration), then the user's requested changes are turned into an original, copyable ad prompt. Competitor text, logos, and identifying assets are not meant to be copied.
5. Image generation is optional and requires a separate, explicit user request. The capability is available both from Layla's creative graph and after competitor-ad prompt generation; it creates a draft only and does not publish or change a campaign. Layla's general image adapter uses server-side image settings without constructing a ToolContext; the selected-competitor path continues to use its isolated context factory.

## Vision model configuration

The default is isolated from general chat and remains Gemini 3.5 Flash-Lite:

```dotenv
AI_ADS_VLM_PROVIDER=gemini
# Optional override; otherwise gemini-3.5-flash-lite is used.
AI_ADS_VLM_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=                 # set only in the backend's secret environment
```

For an optional Ollama-compatible vision server reached through ngrok, configure the dedicated ad-VLM endpoint and model:

```dotenv
AI_ADS_VLM_PROVIDER=ollama
AI_ADS_VLM_MODEL=qwen3-vl:8b                 # exact vision-capable Ollama tag
AI_ADS_VLM_BASE_URL=https://<vlm-ngrok-host> # optional dedicated VLM server URL
```

`AI_ADS_VLM_BASE_URL` is used only by competitor-ad image analysis. If it is unset, the VLM continues to use the existing shared `OLLAMA_BASE_URL` (or its legacy aliases `NGROK_URL` / `OLLAMA_URL`) for backward compatibility. This permits a separate general Ollama host and VLM host without changing `AI_LLM_PROVIDER`, `OLLAMA_MODEL`, or the general chat model. Set `AI_ADS_VLM_MODEL` explicitly to the model tag installed on the VLM server; do not rely on the general text-model fallback for image analysis. No ngrok URL or model id is assumed by the code.

## Optional image generation

The existing image tool supports an OpenAI-compatible image-generation endpoint. Configure it only if in-app image drafts are wanted:

```dotenv
IMAGE_PROVIDER=openai
IMAGE_BASE_URL=                # operator-provided OpenAI-compatible API root
IMAGE_MODEL=                   # operator-provided image model id
IMAGE_API_KEY=                 # set only in the backend/workspace secret store
```

The endpoint must expose the standard `/images/generations` path under the configured API root. Leave the image provider on `mock` when no real image service is configured; the assistant labels mock output as a placeholder. Prompt generation itself remains available independently of image generation.

## Safety and tests

- The preview route requires a valid user session and scopes bytes to the active workspace.
- Preview ids are opaque, expire after 24 hours, and are not filesystem paths.
- Unselected image bytes never enter an LLM request.
- The Meta connector stays isolated from ordinary chat and the general agent tool lists.
- All local flow tests should use mocks; do not run live Meta, Gemini, or image-generation calls to validate this feature.
