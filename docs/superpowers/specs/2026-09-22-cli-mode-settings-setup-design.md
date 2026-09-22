# CLI mode with settings setup — design (Option 1, in-process)

Date: 2026-09-22. Scope: full headless dub. Entry: subcommands in `dub-server`.
Style: wizard + flags. Limits: exclude visual/file-dialog only.

## 1. Architecture

New `crates/dub-server/src/cli.rs` + `clap` v4 derive (`Parser`/`Subcommand`,
Context7 `/clap-rs/clap`). No new crate, no HTTP.

- `dub-server [serve]` — no args = today (axum :8765, backward compat).
- `dub-server setup [--preset ID] [--models-dir DIR] [--yes] [--only a,b] [--tts q8_0|...] [--mt ...] [--asr-engine parakeet|whisper] [--sep-backend auto|gpu|cpu] ... [--or-key KEY] [--proxy-url URL]`
- `dub-server doctor` — read-only (`/setup/status` + `/engine/capabilities` + `/engine/presets` equivalent).
- `dub-server run --input ... --tgt-lang ... [...]` — headless analyze+render in-process.
- Deps: `clap 4 + derive` in `crates/dub-server/Cargo.toml` only. Prompts via `std::io`
  stdin; no dialoguer/inquire.

Reuse (same source of truth as GUI, no drift): `setup::manifest()`,
`models::{load_selection,set_selection,component_selection,is_selection_key}`,
`presets::{recommend,apply}`, existing analyze/render job fns via `AppState`
(progress callback to stderr, not SSE). Disk state stays `models/active.json` +
manifest markers only, so GUI and CLI interoperate.

## 2. Setup wizard (GUI FirstRun order)

1. HW + preset (`PresetsSection`): table + recommended ★; `--preset ID`.
   Cloud presets skip heavy locals (same GUI note).
2. Quant radio per family: higgs / gemma / parakeet / roformer / whisper
   (active family only; default = installed or GUI default);
   `--tts/--mt/--asr/--sep/--whisper-model`.
3. Engine/backend/provider tabs: `asr_engine`, `sep/diar/asr_backend`,
   `llm/vision_provider`; prompted only when relevant (no openrouter without key);
   flags mirror keys 1:1.
4. Keys: OpenRouter / OpenCode / proxy prompts (`[skip]` = skip, as GUI);
   verify OpenRouter only if network (warn, not block);
   `--or-key/--opencode-key/--proxy-url`.
5. Checklist with GUI preselect rule (`download && !installed && !=optional`):
   table + `Y/n` + extras; `--yes/--only/--skip`.
6. Download via server download-loop; stderr progress; Ctrl-C cancels;
   end with `ready` re-check (covers SSE-drop false-failure).
7. Excluded: avatar pick, canvas `sub_y`, native folder browse
   (`--models-dir` path arg instead), driver site-open (print URL),
   per-seg overrides, `llama_ubatch`/`higgs_ref_secs` (flags with defaults).

## 3. Run data flow

`run --input --tgt-lang [--mode dub] [--src-lang auto] [--subs auto] [--out] [--yes] [--json]`:
preflight required-components gate (fail fast with `run setup first` + missing list);
create `workspace/<pid>/` (copy input + `source.txt`); analyze
(`probe/extract_audio/diarize/asr/vision/translate`, translation fail-safe kept);
render dirty-only (`probe/.../mux`); copy to `--out`; final line prints
`project_id/output/segments` (`--json` for scripts). No subs-file import in v1.

## 4. Errors + verification

Exit non-zero + `error: <what> [<hint>]`. Unknown preset/key lists valid values
(`is_selection_key` whitelist); empty values reuse server 400 texts; partial
downloads resume via marker sizes. Fallbacks stay server-identical
(whisper-without-bin → parakeet via `resolve_*`).

Verify: `cargo build -p dub-server`, `cargo test -p dub-server`,
`doctor` vs Settings screen, wizard on throwaway `DUB_STUDIO_ROOT`,
one ~20s clip `run` → playable output with subs (never touch real workspace).
