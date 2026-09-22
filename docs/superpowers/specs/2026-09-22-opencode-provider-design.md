# OpenCode as LLM provider (per-stage choice) — design

Date: 2026-09-22. Status: approved by user (Approach A, native, no new binaries).

## Goal

Per-stage choice of LLM backend for `llm` (text: translation/glossary/remix),
`vision` (frame analysis), and cloud `asr` (transcription) between local
engines, Ollama, OpenRouter, and **OpenCode** — usable simultaneously per stage
(e.g. text via OpenCode + vision via local). No cloud TTS (explicit non-goal
per user scope).

OpenCode runs in two sub-modes (per user answers):

- **Cloud (BYOK + Go subscription):** `POST https://opencode.ai/zen/v1/chat/completions`
  (OpenAI-compatible, Context7 `/websites/opencode_ai_v2`), Bearer API key
  (Zen credits key or OpenCode Go subscription key — both are Bearer keys to
  the same endpoint), model id format `opencode/<model-id>`.
- **CLI on PC:** configurable local URL (default `http://localhost:4096`),
  keys already configured in `opencode.json` / local serve; Dub Studio sends
  the key only if `opencode_key` is set (covers password-protected
  `opencode serve`).

Model dropdowns for OpenCode stages are powered by the public
`https://models.dev/catalog.json?type=all` catalog (per user instruction),
backend-cached, OpenRouter/Ollama UI untouched.

## Background (verified in repo + Context7)

- `crates/dub-llm/src/client.rs` `ChatClient` speaks
  `POST {base}/v1/chat/completions` with
  `{model, messages, temperature, top_p, [top_k], [repeat_penalty],
  max_tokens, stream:false}`. `is_remote()` = `model.is_some()` skips the
  llama-only `chat_template_kwargs`. `openrouter()` = fixed base +
  Bearer + model + referer/title; `ollama()` = custom base + model, no auth.
- Conclusion: OpenCode chat (llm/vision) needs NO protocol work — it is
  "remote with model + base_url + optional Bearer". Work is selection +
  settings + models.dev catalog + UI.
- Cloud ASR/TTS do NOT go through `ChatClient`: `cloud_asr.rs` posts the wav
  file via the Go sidecar (`stt` op, `verbose_json` segments). OpenCode ASR
  therefore gets a small native multipart POST to
  `{base}/v1/audio/transcriptions` in Rust (no sidecar), parsing the same
  `segments[] -> (start,end,text)` shape with single-segment fallback.
- `crates/dub-server/src/llm_provider.rs::open()` picks Remote vs Local;
  all 5 call-sites use `LlmProvider::client()`.
- `GET /engine/openrouter/models?kind=` serves dropdowns via the Go sidecar
  (requires `or_key`). The models.dev catalog is public (no key) and must
  work behind the existing `proxy_on`/`proxy_url` mechanism.
- Frontend `ProviderTabs` (local|ollama|openrouter) per llm/vision stage;
  ASR group is Parakeet|Whisper|OpenRouter; Ollama uses text inputs,
  OpenRouter uses `OrModelSelect` dropdowns.

## Decisions (user answers)

1. Endpoints: Zen cloud + local CLI server (answer: "Zen + GO + local";
   GO = OpenCode Go subscription = Bearer key to Zen, confirmed via
   Context7 `/websites/opencode_ai_v2` providers/Go docs).
2. Stages: LLM + Vision + cloud ASR only, without cloud TTS.
3. models.dev listing: OpenCode dropdowns only (new endpoint, other
   providers untouched).

## Changes

### 1. Settings (`crates/dub-server/src/models.rs`)

New selection keys (add to `is_selection_key` allowlist + helpers):

- `opencode_key` — Zen/Go-sub Bearer key (local-only in `active.json`,
  never logged; same treatment as `or_key`).
- `opencode_mode` — `cloud | local` (default `cloud` once
  `*_provider=opencode` / `opencode_asr_on=1`).
- `opencode_url` — local server URL, default `http://localhost:4096`
  (trimmed of trailing `/`, same rule as `ollama_url`).
- `opencode_llm`, `opencode_vision` (empty vision → falls back to
  `opencode_llm`, same rule as Ollama/OpenRouter), `opencode_asr`.
- `opencode_asr_on` — `"1"` enables cloud ASR via OpenCode (mirrors
  `or_asr_on`; TTS keys untouched).
- `llm_provider`, `vision_provider` gain the `opencode` value;
  `llm_provider_kind()` returns `"opencode"` for stages `llm`/`vision`.
- New helpers (mirroring `ollama_*`/`openrouter_*`):
  `opencode_mode()`, `opencode_url()`, `opencode_model(stage)`,
  `opencode_key()`, `opencode_asr_on()`.
- Back-compat unchanged: absent `*_provider` still maps legacy
  `or_*_on == "1"` → `openrouter`, else `local`.

### 2. Client (`crates/dub-llm/src/client.rs`)

- Add `ChatClient::opencode(base_url, model, api_key: Option<String>)`
  (~12 lines): sets `base_url` + `model` + optional Bearer, no
  referer/title headers. Retry, timeout, body building, response parse
  (`/choices/0/message/content`) untouched; `is_remote()` stays true via
  `model`, so no `chat_template_kwargs`.
- Cloud constant: `base_url=https://opencode.ai/zen`, so the client posts
  `https://opencode.ai/zen/v1/chat/completions` (the documented Zen path).
  Model id stored as-is (expected `opencode/<id>` for Zen; server-side ids
  for local) — no auto-prefixing.
- `input_audio` parts pass through as-is (same as Ollama).
- Unit test mirroring `ollama_sends_model_without_llama_kwargs`:
  stub server asserts `model` present, no `chat_template_kwargs`, Bearer
  present (cloud) / absent (local without key).

### 3. Provider switch (`crates/dub-server/src/llm_provider.rs`)

`open()` matches the stage provider string:

- `opencode` + mode `cloud` → require key + model; else explicit `Err`
  naming the missing piece (`opencode_key` / `opencode_llm`/`opencode_vision`).
- `opencode` + mode `local` → require model + URL (key optional, sent when
  set); else explicit `Err`. Assumption: the local base speaks
  OpenAI-compatible `POST /v1/chat/completions` (verified at E2E against
  `opencode serve`; otherwise the explicit error surfaces and the user
  points `opencode_url` at a compatible base).
- Both return the existing `Remote{client}` variant (no sidecar to hold).
- `openrouter` / `ollama` / `local` paths unchanged; unknown → fail-safe
  `local`.
- All existing call-sites untouched (they use `.client()`).

### 4. Cloud ASR (`crates/dub-server/src/opencode_asr.rs`, new)

- `transcribe(models_root, wav, src_lang)` — native multipart file POST to
  `{base}/v1/audio/transcriptions` (`response_format=verbose_json`,
  `language` unless `auto`/empty), Bearer when key set, base+mode resolved
  like §3. Parses `segments[] -> (start,end,text)` (skip empties), fallback
  to single segment from `text`/`duration` (same contract as `cloud_asr.rs`).
- `analyze.rs` branches: OpenCode-ASR-on → this; else existing
  OpenRouter/local paths. Zen/local without transcription support →
  explicit error (user falls back to OpenRouter/local ASR).
- `ponytail:` no word-level timestamps beyond what the endpoint returns;
  per-word refinement later if a real gap appears.

### 5. models.dev catalog (`crates/dub-server/src/endpoints.rs` + cache)

- New `GET /engine/opencode/models?kind=llm|vision|asr` (public, no key):
  fetch `https://models.dev/catalog.json?type=all` via the existing
  proxy-aware `ureq` agent, cache raw bytes to
  `models/models-dev-catalog.json` (24h TTL, stale-on-error), filter and
  return `{models:[{id,name,context}]}` sorted by id.
- Filter: prefer `providers["opencode"].models` keys when present, else all
  catalog models; modality from provider-agnostic `models{}` metadata —
  `llm`: output∋`text`, `vision`: input∋`image`, `asr`: input∋`audio`.
  `context` from `limit.context` when present.
- Fetch failure with fresh cache → serve stale; without cache → `502`
  with reason (dropdown shows empty + hint).
- Unit tests on a fixture catalog (no network).

### 6. Frontend (`frontend/src/App.tsx` + `lib/api.ts` + locales)

- `ProviderTabs` gains 4th tab `OpenCode` (llm/vision sections); when
  selected, a cloud/local mini-toggle + key input (cloud) / URL input
  (local) + model dropdown appear. Vision empty → same-model note.
- ASR group gains 4th button `OpenCode` with the same cloud/local
  controls + STT dropdown (`opencode_asr_on` + `opencode_asr`).
- `api.opencodeModels(kind)` → new endpoint; fetched when an OpenCode
  stage is selected (no key gate — catalog is public).
- New strings into `frontend/src/locales/` (all 6) + `docs/index.html`
  dict if it carries settings strings (the Ollama plan verified it does
  not — same check here).

## Non-goals (upgrade path)

- Cloud TTS via OpenCode: out of scope (user said no).
- Extending the Go sidecar: skipped — Zen is plain OpenAI-compatible
  (docs), plain HTTP suffices. (`ponytail:` revisit only if Zen needs
  SDK-only headers.)
- Embeddings, `verify`/credits endpoint, usage accounting: skipped; add
  when a real billing-display need appears.
- Unifying OpenRouter/Ollama dropdowns onto models.dev: skipped (user
  chose OpenCode-only).
- No new crates/deps; no changes to translate/vision stage logic.

## Verification

1. `cargo test -p dub-llm`, `cargo test -p dub-server --lib`,
   `cd frontend && npm run build` (typecheck) + `npm run lint`.
2. Real short-clip E2E per repo rules: 20s `ffmpeg -t 20` cut of a
   `test_media/` file → throwaway pid → analyze with llm/vision (+ASR)
   on OpenCode → verify in the preview UI (translated subs readable,
   vision context present). Never mutate a real `workspace/<pid>`.
3. If no OpenCode key/daemon is available locally, verify the HTTP shape
  against a stub speaking the same OpenAI-compatible contract (Ollama-plan
  precedent).
