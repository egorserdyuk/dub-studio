# Ollama per-stage LLM provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ollama as a per-stage LLM backend (text + vision independently) alongside local Gemma and OpenRouter, reusing the existing OpenAI-compatible `ChatClient`.

**Architecture:** Ollama's `POST {url}/v1/chat/completions` takes the exact body `ChatClient::chat()` already sends, so the backend change is one ~8-line constructor plus a provider switch; settings live in `models/active.json` with legacy `or_*_on` fallback; the MT settings group gets 3-way tabs per stage.

**Tech Stack:** Rust (reqwest blocking, serde_json, std-only tests), React 19 + TS strict, existing `getJson/postJson` api client and zustand store untouched.

---

### Task 1: `ChatClient::ollama()` + stub-server unit test

**Files:**
- Modify: `crates/dub-llm/src/client.rs` (constructor ~line 162, tests at end of file)

- [ ] **Step 1: Write the failing test** — append to the test module at the end of `crates/dub-llm/src/client.rs` (create `#[cfg(test)] mod tests` if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn ollama_sends_model_without_llama_kwargs() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let seen2 = seen.clone();
        let h = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 65536];
            let n = s.read(&mut buf).unwrap();
            *seen2.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            let body = r#"{"choices":[{"message":{"content":"hi"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            s.write_all(resp.as_bytes()).unwrap();
        });
        let c = ChatClient::ollama(format!("http://127.0.0.1:{port}"), "gemma3").unwrap();
        let out = c.chat(&[Message::user_text("ping")], &Sampling::new(0.0, 1.0, 32)).unwrap();
        h.join().unwrap();
        assert_eq!(out, "hi");
        let req = seen.lock().unwrap().clone();
        let start = req.find('{').expect("request has json body");
        let v: serde_json::Value = serde_json::from_str(&req[start..]).unwrap();
        assert_eq!(v["model"], "gemma3");
        assert!(v.get("chat_template_kwargs").is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dub-llm ollama_sends_model_without_llama_kwargs`
Expected: FAIL with `error[E0599]: no function or associated item named 'ollama' found for struct 'ChatClient'`

- [ ] **Step 3: Write minimal implementation** — insert after `openrouter()` (after line 162) in `crates/dub-llm/src/client.rs`:

```rust
    /// Клиент к Ollama (OpenAI-совместимый /v1/chat/completions): base_url настраивается
    /// (дефолт http://localhost:11434), обязательное поле `model`, без ключа (Ollama Bearer
    /// игнорирует) и без llama-специфичного chat_template_kwargs (`is_remote()` true по model).
    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self, LlmError> {
        let mut c = Self::new(base_url)?;
        c.model = Some(model.into());
        Ok(c)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dub-llm`
Expected: all tests PASS (existing `strip_think` tests + new test)

- [ ] **Step 5: Commit**

```bash
git add crates/dub-llm/src/client.rs
git commit -m "feat(llm): ChatClient::ollama for Ollama OpenAI-compat endpoint"
```

---

### Task 2: Settings keys + helpers in `models.rs` (+ unit tests)

**Files:**
- Modify: `crates/dub-server/src/models.rs` (allowlist ~line 92, helpers after `openrouter_model` ~line 200, tests at end of file)

- [ ] **Step 1: Write the failing tests** — append at end of `crates/dub-server/src/models.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dub-ollama-{}-{}", tag, std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let _ = std::fs::remove_file(d.join("active.json"));
        d
    }

    #[test]
    fn provider_kind_defaults_local_with_legacy_fallback() {
        let d = tmp_root("kind");
        assert_eq!(llm_provider_kind(&d, "llm"), "local");
        assert_eq!(llm_provider_kind(&d, "vision"), "local");
        set_selection(&d, "or_llm_on", "1").unwrap();
        assert_eq!(llm_provider_kind(&d, "llm"), "openrouter");
        assert_eq!(llm_provider_kind(&d, "vision"), "local");
        set_selection(&d, "llm_provider", "ollama").unwrap();
        assert_eq!(llm_provider_kind(&d, "llm"), "ollama");
        assert!(!llm_provider_explicit(&d, "vision"));
        assert!(llm_provider_explicit(&d, "llm"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn ollama_url_default_and_trims_slash() {
        let d = tmp_root("url");
        assert_eq!(ollama_url(&d), "http://localhost:11434");
        set_selection(&d, "ollama_url", "http://srv:11434/").unwrap();
        assert_eq!(ollama_url(&d), "http://srv:11434");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn ollama_model_vision_falls_back_to_llm() {
        let d = tmp_root("model");
        assert_eq!(ollama_model(&d, "llm"), "");
        set_selection(&d, "ollama_llm", "gemma3").unwrap();
        assert_eq!(ollama_model(&d, "llm"), "gemma3");
        assert_eq!(ollama_model(&d, "vision"), "gemma3");
        set_selection(&d, "ollama_vision", "qwen3-vl:8b").unwrap();
        assert_eq!(ollama_model(&d, "vision"), "qwen3-vl:8b");
        std::fs::remove_dir_all(&d).ok();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dub-server --lib models::tests`
Expected: FAIL with `cannot find function 'llm_provider_kind' in module 'models'` (and friends)

- [ ] **Step 3a: Add allowlist keys** — in `is_selection_key`, after line 103 (`| "or_concurrency" ...`), insert:

```rust
            | "llm_provider"    // провайдер текста перевода: "local" (Gemma) | "ollama" | "openrouter"
            | "vision_provider" // провайдер vision-анализа кадров: "local" | "ollama" | "openrouter"
            | "ollama_url"      // базовый URL Ollama (дефолт http://localhost:11434)
            | "ollama_llm"      // id текстовой модели Ollama (напр. "gemma3")
            | "ollama_vision"   // id vision-модели Ollama (пусто -> берём ollama_llm)
```

- [ ] **Step 3b: Add helpers** — after `openrouter_model` (after line 200), insert:

```rust
/// Провайдер стадии ("llm"|"vision"): "local" | "ollama" | "openrouter".
/// Back-compat: если ключ *_provider не задан, маппим legacy or_*_on ("1" -> openrouter, иначе local).
pub fn llm_provider_kind(mroot: &Path, stage: &str) -> &'static str {
    let sel = load_selection(mroot);
    let (new_key, legacy_flag) = match stage {
        "llm" => ("llm_provider", "or_llm_on"),
        "vision" => ("vision_provider", "or_vision_on"),
        _ => return "local",
    };
    match pick(&sel, new_key) {
        Some("ollama") => "ollama",
        Some("openrouter") => "openrouter",
        Some("local") => "local",
        _ => {
            if pick(&sel, legacy_flag) == Some("1") {
                "openrouter"
            } else {
                "local"
            }
        }
    }
}

/// true, если провайдер стадии задан явно новым ключом (а не legacy or_*_on).
/// Явный выбор = явная ошибка при плохой конфигурации; legacy = тихий fallback на локаль.
pub fn llm_provider_explicit(mroot: &Path, stage: &str) -> bool {
    let sel = load_selection(mroot);
    let new_key = match stage {
        "llm" => "llm_provider",
        "vision" => "vision_provider",
        _ => return false,
    };
    pick(&sel, new_key).is_some()
}

/// Базовый URL Ollama без хвостового слэша. Дефолт http://localhost:11434.
pub fn ollama_url(mroot: &Path) -> String {
    let u = pick(&load_selection(mroot), "ollama_url").unwrap_or("http://localhost:11434");
    u.trim_end_matches('/').to_string()
}

/// id модели Ollama для стадии ("llm"|"vision"). Vision: ollama_vision, пусто -> ollama_llm.
/// Пусто, если ничего не задано (вызывающий обязан упасть с понятной ошибкой).
pub fn ollama_model(mroot: &Path, stage: &str) -> String {
    let sel = load_selection(mroot);
    match stage {
        "llm" => pick(&sel, "ollama_llm").unwrap_or("").to_string(),
        "vision" => pick(&sel, "ollama_vision")
            .or_else(|| pick(&sel, "ollama_llm"))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}
```

Note: `pick` already trims and drops empty strings (`models.rs:36-38`), so no extra emptiness checks needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dub-server --lib models::tests`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/dub-server/src/models.rs
git commit -m "feat(server): ollama settings keys + provider/model/url helpers"
```

---

### Task 3: Ollama branch in `llm_provider.rs::open()` (+ unit tests)

**Files:**
- Modify: `crates/dub-server/src/llm_provider.rs` (whole `open()` body lines 56-84, tests at end of file)

- [ ] **Step 1: Write the failing tests** — append at end of `crates/dub-server/src/llm_provider.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dub-prov-{}-{}", tag, std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let _ = std::fs::remove_file(d.join("active.json"));
        d
    }

    fn open_in(dir: &std::path::Path, mode: LlmMode) -> Result<LlmProvider, String> {
        let fake = dir.join("nope.exe");
        open(
            &LlmOpen { llama_bin: &fake, mt_model: &fake, mmproj: &fake, models_root: dir },
            mode,
        )
    }

    #[test]
    fn ollama_returns_remote_without_sidecar() {
        let d = tmp_root("remote");
        crate::models::set_selection(&d, "llm_provider", "ollama").unwrap();
        crate::models::set_selection(&d, "ollama_llm", "gemma3").unwrap();
        let p = open_in(&d, LlmMode::Text).unwrap();
        assert!(p.is_remote());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn ollama_empty_model_is_explicit_error() {
        let d = tmp_root("empty");
        crate::models::set_selection(&d, "llm_provider", "ollama").unwrap();
        let e = open_in(&d, LlmMode::Text).unwrap_err();
        assert!(e.contains("Ollama"), "unexpected error: {e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn legacy_openrouter_still_opens_remote() {
        let d = tmp_root("legacy");
        crate::models::set_selection(&d, "or_llm_on", "1").unwrap();
        crate::models::set_selection(&d, "or_key", "k").unwrap();
        crate::models::set_selection(&d, "or_llm", "google/gemini-2.5-flash").unwrap();
        let p = open_in(&d, LlmMode::Text).unwrap();
        assert!(p.is_remote());
        std::fs::remove_dir_all(&d).ok();
    }
}
```

Note: `fake` paths never touched on the ollama/openrouter paths (no sidecar spawn, no network — constructors only build the HTTP client), so nonexistent paths are fine.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dub-server --lib llm_provider::tests`
Expected: FAIL — `ollama_returns_remote_without_sidecar` errors with "llama-server не найден" (falls through to local), `ollama_empty_model_is_explicit_error` fails on `unwrap_err` (returns Ok local or Err about llama binary)

- [ ] **Step 3: Replace `open()` body** — replace lines 56-68 of `crates/dub-server/src/llm_provider.rs` (the openrouter `if` block) with:

```rust
pub fn open(o: &LlmOpen, mode: LlmMode) -> Result<LlmProvider, String> {
    let stage = if mode == LlmMode::Vision { "vision" } else { "llm" };
    match crate::models::llm_provider_kind(o.models_root, stage) {
        "ollama" => {
            let model = crate::models::ollama_model(o.models_root, stage);
            if model.trim().is_empty() {
                return Err("Ollama выбран, но модель не задана (ollama_llm/ollama_vision)".to_string());
            }
            let url = crate::models::ollama_url(o.models_root);
            let client = ChatClient::ollama(url, model).map_err(|e| e.to_string())?;
            return Ok(LlmProvider::Remote { client });
        }
        "openrouter" => {
            let explicit = crate::models::llm_provider_explicit(o.models_root, stage);
            let key = crate::models::openrouter_key(o.models_root);
            let model = crate::models::openrouter_model(o.models_root, stage);
            match (key, model.trim().is_empty()) {
                (Some(k), false) => {
                    let client = ChatClient::openrouter(k, model).map_err(|e| e.to_string())?;
                    return Ok(LlmProvider::Remote { client });
                }
                _ if explicit => {
                    return Err("OpenRouter выбран, но ключ или модель не заданы".to_string())
                }
                _ => {} // legacy (or_*_on без ключа/модели): тихо на локаль, как раньше
            }
        }
        _ => {} // "local" и неизвестные значения -> локальный llama-server (fail-safe)
    }

    if !o.llama_bin.is_file() {
```

The rest of the function (local path, lines 70-83) stays byte-identical. Also update the module doc comment line 5-6: replace `Облако требует НЕПУСТОЙ or_key —` line with `Явный выбор облака/Ollama с плохой конфигурацией — явная ошибка; legacy or_*_on без ключа/модели — тихий fallback на локаль.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dub-server --lib llm_provider::tests && cargo test -p dub-server --lib models::tests`
Expected: all PASS (3 + 3)

- [ ] **Step 5: Commit**

```bash
git add crates/dub-server/src/llm_provider.rs
git commit -m "feat(server): route ollama/openrouter/local per-stage via llm_provider_kind"
```

---

### Task 4: Frontend 3-way provider UI + locales

**Files:**
- Modify: `frontend/src/App.tsx` (MT group lines 346-366)
- Modify: `frontend/src/locales/en.json`, `ru.json`, `es.json`, `fr.json`, `pt.json`, `zh.json` (insert after the `"roleMt"` line in each file's `settings` object)

`docs/index.html` needs nothing: verified via grep — it contains no `or_llm`/`roleMt` settings strings (landing page only).

- [ ] **Step 1: Add locale keys** — after the `"roleMt"` line in each locale file, insert (exact strings per language):

en.json:
```json
    "needOrKey": "Enter your OpenRouter key below (Cloud settings)",
    "ollamaUrl": "Ollama URL",
    "ollamaLlm": "Ollama text model",
    "ollamaVision": "Ollama vision model (empty = same as text)",
    "visionTitle": "Frame vision analysis",
    "emptyVisionOr": "same as translation model",
```
ru.json:
```json
    "needOrKey": "Введите ключ OpenRouter ниже (Облачные настройки)",
    "ollamaUrl": "URL Ollama",
    "ollamaLlm": "Текстовая модель Ollama",
    "ollamaVision": "Vision-модель Ollama (пусто = как текстовая)",
    "visionTitle": "Vision-анализ кадров",
    "emptyVisionOr": "как модель перевода",
```
es.json:
```json
    "needOrKey": "Introduzca su clave de OpenRouter abajo (ajustes en la nube)",
    "ollamaUrl": "URL de Ollama",
    "ollamaLlm": "Modelo de texto de Ollama",
    "ollamaVision": "Modelo de visión de Ollama (vacío = igual que el de texto)",
    "visionTitle": "Análisis visual de fotogramas",
    "emptyVisionOr": "igual que el modelo de traducción",
```
fr.json:
```json
    "needOrKey": "Saisissez votre clé OpenRouter ci-dessous (paramètres cloud)",
    "ollamaUrl": "URL Ollama",
    "ollamaLlm": "Modèle texte Ollama",
    "ollamaVision": "Modèle vision Ollama (vide = comme le texte)",
    "visionTitle": "Analyse vision des images",
    "emptyVisionOr": "comme le modèle de traduction",
```
pt.json:
```json
    "needOrKey": "Introduza sua chave OpenRouter abaixo (configurações de nuvem)",
    "ollamaUrl": "URL do Ollama",
    "ollamaLlm": "Modelo de texto do Ollama",
    "ollamaVision": "Modelo de visão do Ollama (vazio = igual ao de texto)",
    "visionTitle": "Análise visual de quadros",
    "emptyVisionOr": "igual ao modelo de tradução",
```
zh.json:
```json
    "needOrKey": "请在下方输入 OpenRouter 密钥（云设置）",
    "ollamaUrl": "Ollama 地址",
    "ollamaLlm": "Ollama 文本模型",
    "ollamaVision": "Ollama 视觉模型（留空则与文本模型相同）",
    "visionTitle": "画面视觉分析",
    "emptyVisionOr": "与翻译模型相同",
```

- [ ] **Step 2: Replace the MT group** — replace lines 346-366 of `frontend/src/App.tsx` with:

```tsx
      <Group label={t("settings.roleMt")}>
        {(() => {
          const cur = selv("llm_provider") || (selv("or_llm_on") === "1" ? "openrouter" : "local");
          const vcur = selv("vision_provider") || (selv("or_vision_on") === "1" ? "openrouter" : "local");
          const tabs = (c: string, k: string, localLabel: string) => (
            <div className="flex gap-1 mb-1.5">
              {[{ id: "local", label: localLabel }, { id: "ollama", label: "Ollama" }, { id: "openrouter", label: "OpenRouter" }].map((p) => {
                const dis = p.id === "openrouter" && !hasOrKey;
                const active = c === p.id;
                return (
                  <button key={p.id} disabled={dis} title={dis ? t("settings.needOrKey") : ""} onClick={() => setSel(k, p.id)}
                    className={`flex-1 px-2 py-1.5 rounded-md text-[12px] font-medium border transition-colors ${active ? "border-[var(--color-accent)] bg-[color-mix(in_oklab,var(--color-accent)_14%,transparent)] text-[var(--color-text)]" : "border-[var(--color-border)] text-[var(--color-muted)] hover:text-[var(--color-text)]"} disabled:opacity-40`}>
                    {p.label}
                  </button>
                );
              })}
            </div>
          );
          return (
            <>
              {tabs(cur, "llm_provider", "Gemma-4 12B")}
              {cur === "openrouter" ? (
                <div className={`${orRowCls} space-y-2`}>
                  <OrModelSelect kind="llm" k="or_llm" empty="— выбрать модель перевода —" />
                </div>
              ) : cur === "ollama" ? (
                <div className={`${orRowCls} space-y-2`}>
                  <input value={selv("ollama_url") || "http://localhost:11434"} onChange={(e) => setSel("ollama_url", e.target.value)} className={orSelectCls} placeholder={t("settings.ollamaUrl")} />
                  <input value={selv("ollama_llm")} onChange={(e) => setSel("ollama_llm", e.target.value)} className={orSelectCls} placeholder={t("settings.ollamaLlm")} />
                </div>
              ) : (
                <>
                  <VariantPicker base="Gemma-4 12B QAT + vision" ids={["gemma", "gemma-q5_0", "gemma-q6_k", "gemma-q8_0"]} />
                  {rowOf("llama")}
                </>
              )}
              <div className="text-[12px] text-[var(--color-muted)] pt-1">{t("settings.visionTitle")}</div>
              {tabs(vcur, "vision_provider", "Gemma-4 12B")}
              {vcur === "openrouter" ? (
                <div className={`${orRowCls} space-y-2`}>
                  <OrModelSelect kind="vision" k="or_vision" empty={t("settings.emptyVisionOr")} />
                </div>
              ) : vcur === "ollama" ? (
                <div className={`${orRowCls} space-y-2`}>
                  <input value={selv("ollama_vision")} onChange={(e) => setSel("ollama_vision", e.target.value)} className={orSelectCls} placeholder={t("settings.ollamaVision")} />
                </div>
              ) : null}
            </>
          );
        })()}
      </Group>
```

`EngineTabs` stays (still used by the TTS group). `orRowCls`, `orSelectCls`, `OrModelSelect`, `VariantPicker`, `rowOf` are reused as-is.

- [ ] **Step 3: Typecheck + lint**

Run: `cd frontend && npm run build`
Expected: `tsc -b` clean + `vite build` succeeds

Run: `cd frontend && npm run lint`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx frontend/src/locales/en.json frontend/src/locales/ru.json frontend/src/locales/es.json frontend/src/locales/fr.json frontend/src/locales/pt.json frontend/src/locales/zh.json
git commit -m "feat(ui): per-stage local/ollama/openrouter provider tabs + locales"
```

---

### Task 5: Full verification (build + E2E on a short clip via preview UI)

**Files:** none (verification only)

- [ ] **Step 1: Rust build + tests**

Run: `cargo test -p dub-llm && cargo test -p dub-server --lib models::tests && cargo test -p dub-server --lib llm_provider::tests`
Expected: all PASS

Run: `cargo build -p dub-server`
Expected: success, no new warnings (`cargo fmt` clean — run `cargo fmt --check` first)

- [ ] **Step 2: Cut a throwaway test clip** (never touch a real `workspace/<pid>`)

Run: `ffmpeg -y -i test_media/<file>.mp4 -t 20 -c copy scratchpad/clip-ollama.mp4`
Expected: 20s clip created under `scratchpad/` (gitignored)

- [ ] **Step 3: E2E in the preview UI as a user** — rebuild frontend, hard-reset browser cache, then: create project on the throwaway clip → set LLM-text to Ollama (+ model) and vision to Ollama → analyze → verify in the player/editor UI: translated subtitles readable, scene context present, karaoke plays. Repeat with text=Ollama/vision=OpenRouter (or vice versa) to prove stage independence. Backend logs may be glanced at for timing only — proof is what the UI shows.
- [ ] **Step 4: If no Ollama daemon is available**, verify the HTTP shape against a stub speaking the same OpenAI-compatible contract (Task 1's stub pattern) and do the UI pass with `ollama_url` pointed at it.

---

## Self-review

- **Spec coverage:** §1 settings keys → Task 2; §2 `ChatClient::ollama` → Task 1; §3 provider switch (incl. explicit-error vs legacy-fallback) → Task 3; §4 error handling → Tasks 2-3 tests; §5 frontend 3-way + locales → Task 4; §6 verification → Task 5. Non-goals (`/api/tags` listing, embeddings, new deps) appear in no task. No gaps.
- **Placeholder scan:** all steps carry exact code/commands/expected output; translations provided verbatim for all 6 locales; no TBD/TODO; no "similar to Task N" (Task 4 reuses named existing helpers, each named explicitly).
- **Type consistency:** `llm_provider_kind`/`llm_provider_explicit`/`ollama_url`/`ollama_model` signatures identical in Tasks 2 and 3; `ChatClient::ollama(base_url, model)` identical in Tasks 1 and 3; frontend selection keys (`llm_provider`, `vision_provider`, `ollama_url`, `ollama_llm`, `ollama_vision`) identical in Tasks 2 and 4.
