# Ollama as LLM provider (per-stage choice) — design

Date: 2026-09-22. Status: approved by user (Approach 1).

## Goal

Per-stage choice of LLM backend for the two LLM stages — `llm` (text:
translation/glossary/remix) and `vision` (frame analysis) — between
local Gemma (llama-server sidecar), Ollama, and OpenRouter. The two
stages are independent (e.g. text via Ollama + vision via OpenRouter).

## Background (verified in repo + Context7 `/websites/ollama_api`)

- `crates/dub-llm/src/client.rs` `ChatClient` already speaks
  `POST {base}/v1/chat/completions` with
  `{model, messages, temperature, top_p, [top_k], [repeat_penalty],
  max_tokens, stream:false}`. `is_remote()` = `model.is_some()` skips
  the llama-only `chat_template_kwargs`.
- Ollama exposes the same OpenAI-compatible endpoint at
  `{ollama_url}/v1/chat/completions` (default
  `http://localhost:11434`), auth dummy (`api_key='ollama'`, ignored),
  vision via the same `image_url` parts the client already sends.
- `crates/dub-server/src/llm_provider.rs::open()` picks Remote
  (OpenRouter, needs `or_key` + model) or Local (llama-server); all
  5 call-sites use `LlmProvider::client()`.
- Conclusion: Ollama needs NO protocol work — it is "remote with
  model + custom base_url, no auth". Work is selection + settings + UI.

## Decisions (user answers)

1. Scope: choice between OpenRouter and Ollama, usable simultaneously
   in their own sections (per-stage independence).
2. Connection: custom URL + per-stage models.
3. Selection: 3-way choice per stage (option A).

## Changes

### 1. Settings (`crates/dub-server/src/models.rs`)

New selection keys (add to allowlist + helpers):

- `llm_provider`, `vision_provider` ∈ `local | ollama | openrouter`,
  default `local`.
- `ollama_url` (default `http://localhost:11434`),
  `ollama_llm`, `ollama_vision` (model names, e.g. `gemma3`,
  `qwen3-vl:8b`).
- Back-compat: when `*_provider` is absent, map legacy
  `or_*_on == "1"` → `openrouter`, else `local`. Existing users
  unaffected, no data migration.

### 2. Client (`crates/dub-llm/src/client.rs`)

- Add `ChatClient::ollama(base_url, model)` (~10 lines): sets
  `base_url` + `model`, no auth, no referer/title headers. Retry,
  timeout, body building, response parse
  (`/choices/0/message/content`) untouched.
- `input_audio` parts pass through as-is; a 4xx from Ollama surfaces
  to the caller (text/vision paths that matter never send audio).

### 3. Provider switch (`crates/dub-server/src/llm_provider.rs`)

`open()` matches the stage provider string:

- `openrouter` → existing key + model path (unchanged).
- `ollama` → read `ollama_url` + stage model; empty model = explicit
  `Err` ("Ollama выбран, но модель не задана"); else
  `ChatClient::ollama(...)` in the existing `Remote{client}` variant
  (no sidecar to hold; `is_remote()` keeps its meaning).
- `local` (or unknown value → fail-safe `local`) → unchanged
  llama-server path.
- All existing call-sites untouched (they use `.client()`).

### 4. Error handling

Explicit choice = explicit error, no silent fallback to local when
the user picked `ollama`/`openrouter` with a bad/empty config
(the old empty-model → local fallback stays only for legacy
`or_*` keys). Connection/timeout/4xx propagate as today's
human-readable `Err` strings.

### 5. Frontend (`frontend/src/App.tsx` + locales)

- Replace the Local/Cloud `EngineTabs` toggle with a 3-way select per
  section (LLM-text, vision independently).
- `ollama` selected → URL input + model text input; `openrouter`
  selected → existing `OrModelSelect`.
- New strings into `frontend/src/locales/` (all 6) + `docs/index.html`
  dict, per repo locale rule.

## Non-goals (upgrade path)

- `GET /engine/ollama/models` proxying Ollama `/api/tags` + model
  dropdown + connection check: skipped; add when typing model names
  proves annoying. (`ponytail:` text inputs now, listing later.)
- Embeddings/ASR/TTS via Ollama: out of scope.
- No new crates/deps; no changes to translate/vision stage logic.

## Verification

1. `cargo test -p dub-llm`, `cargo build -p dub-server`,
   `cd frontend && npm run build` (typecheck) + `npm run lint`.
2. Real short-clip E2E per repo rules: 20s `ffmpeg -t 20` cut of a
   `test_media/` file → throwaway pid → analyze with
   llm/vision on Ollama → verify in the preview UI (translated
   subs readable, vision context present). Never mutate a real
   `workspace/<pid>`.
3. If no Ollama daemon is available locally, verify the HTTP shape
   against a stub speaking the same OpenAI-compatible contract.
