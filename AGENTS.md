# AGENTS.md

## Project overview

**Dub Studio** — free, offline AI video dubbing studio for Windows. One native app (Rust + C++/CUDA engines, zero Python at runtime): drop a video → auto-pass builds a first draft (separation → ASR → diarization → translation + vision → OCR) → live editor → export.

- **Stack:** Tauri 2 (Rust) shell spawns `dub-server` (axum, local port) and opens a window onto the SPA — React 19 + Vite + Tailwind + react-konva over JASSUB.
- **Engines (all native):** Parakeet-TDT or Whisper (ASR) · Sortformer (diarization) · Gemma-4-12B GGUF via llama.cpp (translation + vision) · Higgs Audio v3 (TTS) · Mel-Band Roformer via BSRoformer.cpp (separation) · PP-OCR via ONNX · ffmpeg/NVENC.
- **Modes:** Dub · Voice-over · Subtitles · Funny remix · Transcript. Switchable on the fly from one analyzed Project.
- **Fully portable:** everything lives inside the app folder (`models/`, `voices/`, `workspace/`, engines, ffmpeg). Delete the folder = no trace.

## Repo layout

- `frontend/` — React 19 + Vite + Tailwind SPA. Source in `src/` (`App.tsx`, `store.ts` (zustand), `lib/api.ts`, `components/`). Builds to `frontend/dist` (`base: "./"` — must stay relative for portable serving).
- `crates/` — Rust workspace (edition 2021, resolver 2): `dub-server` (axum REST/SSE API + static SPA serving), `dub-core`, `dub-asr`, `dub-llm`, `dub-translate`, `dub-sep`, `dub-captions`, `dub-ocr`, `dub-faces`, `audiocpp`.
- `desktop/src-tauri/` — Tauri shell. **Single source of version: `tauri.conf.json`** (frontend reads it at build time as `__APP_VERSION__`). `staging/` is generated at build time, never committed.
- `backend/app.py` — legacy Python/FastAPI server (reference only; the contract lives in `docs/PORT-CONTRACT.md`).
- `docs/` — `PORT-CONTRACT.md` (REST/SSE contract — read before touching API), screenshots, landing page source.
- `dub-engine/`, `tools/`, `fonts/` — legacy engine / sidecars / bundled fonts.
- Runtime dirs (gitignored, never commit): `workspace/`, `models/`, `voices/`, `downloads/`, `temp/`, `cache/`, `output/`, `test_media/`, `target/`, `desktop/src-tauri/staging/`, `scratchpad/`, `logs/`.

## Setup commands

Needs Node 20+, Rust (MSVC toolchain), WebView2. Native engines ship as prebuilt binaries downloaded by the app — no rebuild needed.

```bash
# 1) SPA
cd frontend && npm install && npm run build && cd ..
# 2) native server (axum)
cargo build --release -p dub-server
# 3) desktop shell
cd desktop && npm install && npx tauri build
```

Dev loop (UI hot-reload, no full build):

```bash
cd frontend && npm run dev   # Vite :5173 -> backend :8765
```

## Build and test commands

- Frontend typecheck + build: `cd frontend && npm run build` (`tsc -b && vite build`)
- Frontend lint: `cd frontend && npm run lint` (`eslint .`)
- Frontend preview: `cd frontend && npm run preview`
- Rust build: `cargo build --release -p dub-server` (full workspace `cargo build --workspace` excludes `desktop/src-tauri` by design)
- Rust tests: `cargo test -p <crate>` or `cargo test --workspace --exclude <heavy>`; unit tests live inline (`#[test]`) + `crates/*/tests/`
- No JS unit-test runner is configured — verification is via typecheck/lint + real UI flow (see below).
- Single source of version: bump only `desktop/src-tauri/tauri.conf.json` (`productVersion`); workspace `Cargo.toml` and `frontend/package.json` follow.

## Code style

- Rust: edition 2021, `cargo fmt` clean, no new warnings. Errors via `anyhow`/`thiserror`; shared tracing. Keep FFI/ONNX deps minimal — prefer already-present workspace deps over new crates.
- TypeScript: strict mode (`tsconfig.app.json`), functional components, no semicolons drift — follow existing files. `eslint` (react-hooks + react-refresh + typescript-eslint) must pass.
- API client: extend `frontend/src/lib/api.ts` (`getJson`/`postJson` wrappers + serialized `_chain` for PATCH) — don't add parallel fetch helpers.
- State: single zustand store in `frontend/src/store.ts`. Reuse `progress/activities/exports/past/future/rev` — don't invent parallel state channels.
- Comments in code are Russian-leaning; keep terse. Mark deliberate shortcuts with `ponytail:` + ceiling + upgrade path.
- Reuse built-in mechanisms (benchmark `bench.rs` `Bench::start/stage/finish`, JPG avatar convention, existing `models/faces/` paths) — never invent a parallel timer/format/path.

## API / contract rules

- The SPA must work unchanged against `dub-server` and `backend/app.py`. Before changing any endpoint, read `docs/PORT-CONTRACT.md`.
- `analyze()` is a fixed first pass: separation → ASR (word timings) → diarization → translation + vision → OCR → editable **Project**. Edits are patches; export re-runs only dirtied stages; preview is ~0.17 s/frame.
- New artifacts follow the agreed convention (native JPG for faces/avatars — `image` crate and ffmpeg infer format from extension).

## Verification rules (hard requirements)

- **Verify through the preview UI as a user, never via backend files/curl.** Casting → look at avatars in the editor; dubbing → play audio + karaoke in the player; transcript/subs → read in the UI. Backend reads are timing-only, never the reported proof. Rebuild frontend + hard-reset cache before any browser re-test.
- **Verify the whole feature end-to-end on a real short clip** before declaring done or releasing. Cut one: `ffmpeg -y -i test_media/<file>.mp4 -t 20 -c copy <scratch>/clip.mp4`, then createProject → analyze → feature. "Too many segments / too heavy" is not an excuse. Never mutate the user's real `workspace/<pid>` while testing — use a throwaway pid.
- **Read the compact code yourself before patching.** Relevant crates are a handful of files — read the function AND its callers end-to-end, reproduce on real cached data, then fix the root cause once where all callers route through. Do not spawn subagents to investigate this repo.
- Casting avatar gates (never regress): sharpest frame wins (`score² · sharpness · frontality · area_gate`), min face size 96px (`min_face_px()`), bbox fully in frame (reject edge-touching). Confirm visually in the UI.
- Casting `content_type`: sample a real frame (`ffmpeg -ss <t> -i in.mp4 -frames:v 1 f.png`) and look — real photo → `real` (SCRFD + LVFace), drawn/cartoon → `anime`. Never infer from the title (e.g. Avatar exists as both). Model paths resolve directly under `models/faces/`.

## Git / push discipline

- `test_media/*.mp4` (60–180 MB each), `workspace/`, `models/`, `staging/`, `scratchpad/` are NEVER committed (GitHub 100 MB limit). Stage with `git add <specific paths>` — never `git add -A` / `git add .`.
- If test assets slip in: strip with `git filter-branch ... 'git rm -r --cached --ignore-unmatch test_media'`; do not force-push master.
- Locales: app ships 6 languages — new strings go into `frontend/src/locales/` (+ `docs/index.html` dict) with all locales updated.
- Never `gh release create` / publish / deploy until the user explicitly says go.

## Release builds

Do NOT hand-write a build script — reuse the newest `scratchpad/pack_portable_<ver>.ps1` (copy → bump). Then: kill the running app (exe lock → `os error 5`), build staging per `desktop/src-tauri/STAGING.md`, `npx tauri build`. Updater signing needs `TAURI_SIGNING_PRIVATE_KEY` + password or `.sig`/`latest.json` break. Changelog diffs against the last **published** GitHub tag, lists user-facing features only, and carries `beta` labels to every surface.
