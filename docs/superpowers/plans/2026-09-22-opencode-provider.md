# OpenCode Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenCode as a 4th per-stage LLM provider (`local | ollama | openrouter | opencode`) for llm + vision + cloud ASR, with model dropdowns backed by models.dev.

**Architecture:** Native extension of the existing Ollama seam — one new `ChatClient::opencode()` constructor, new `active.json` keys + helpers in `models.rs`, one new match arm in `llm_provider::open()`, one new `opencode_asr.rs` (multipart to `/v1/audio/transcriptions`), one new cached catalog module + endpoint, frontend tabs/dropdowns. No new crates, no sidecar changes.

**Tech Stack:** Rust (reqwest blocking, ureq, axum, serde_json), React 19 + TypeScript (existing `api.ts` wrappers, `ProviderTabs`), models.dev `catalog.json?type=all`.

**Spec:** `docs/superpowers/specs/2026-09-22-opencode-provider-design.md`

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/dub-llm/src/client.rs` | New `ChatClient::opencode(base_url, model, Option<key>)` constructor + tests |
| `crates/dub-server/src/models.rs` | New selection keys + `opencode_*` helpers + tests |
| `crates/dub-server/src/llm_provider.rs` | New `"opencode"` match arm in `open()` + tests |
| `crates/dub-server/src/opencode_asr.rs` | NEW: native `transcribe()` + pure `parse_verbose_json()` + tests |
| `crates/dub-server/src/models_catalog.rs` | NEW: models.dev fetch/cache/filter + tests |
| `crates/dub-server/src/endpoints.rs` | New `opencode_models` handler |
| `crates/dub-server/src/lib.rs` | Register `mod opencode_asr; mod models_catalog;` + route |
| `crates/dub-server/src/analyze.rs` | New `opencode_asr_on` branch (mirrors OpenRouter branch) |
| `crates/dub-server/Cargo.toml` | Add `"multipart"` to existing reqwest features (one word, no new crate) |
| `frontend/src/lib/api.ts` | New `opencodeModels()` wrapper |
| `frontend/src/App.tsx` | 4th provider tab, cloud/local toggles, dropdowns, ASR button |
| `frontend/src/locales/{en,es,fr,pt,ru,zh}.json` | New UI strings (all 6 locales) |

---

### Task 1: `ChatClient::opencode()` + stub-server unit tests

**Files:**
- Modify: `crates/dub-llm/src/client.rs` (insert after `ollama()`, lines 173-177)
- Test: inline `#[cfg(test)] mod tests` at end of same file

- [ ] **Step 1: Write the failing tests** — append to `mod tests` in `crates/dub-llm/src/client.rs`:

```rust
#[test]
fn opencode_cloud_sends_model_and_bearer_without_llama_kwargs() {
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
    let c = ChatClient::opencode(
        format!("http://127.0.0.1:{port}"),
        "opencode/gpt-5.5",
        Some("sk-test".to_string()),
    )
    .unwrap();
    let out = c
        .chat(&[Message::user_text("ping")], &Sampling::new(0.0, 1.0, 32))
        .unwrap();
    h.join().unwrap();
    assert_eq!(out, "hi");
    let req = seen.lock().unwrap().clone();
    assert!(req.to_lowercase().contains("authorization: bearer sk-test"));
    let start = req.find('{').expect("request has json body");
    let v: serde_json::Value = serde_json::from_str(&req[start..]).unwrap();
    assert_eq!(v["model"], "opencode/gpt-5.5");
    assert!(v.get("chat_template_kwargs").is_none());
}

#[test]
fn opencode_local_without_key_sends_no_auth() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let seen2 = seen.clone();
    let h = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let mut buf = [0u8; 65536];
        let n = s.read(&mut buf).unwrap();
        *seen2.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
        let body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body
        );
        s.write_all(resp.as_bytes()).unwrap();
    });
    let c = ChatClient::opencode(format!("http://127.0.0.1:{port}"), "local-model", None).unwrap();
    let out = c
        .chat(&[Message::user_text("ping")], &Sampling::new(0.0, 1.0, 32))
        .unwrap();
    h.join().unwrap();
    assert_eq!(out, "ok");
    let req = seen.lock().unwrap().clone();
    assert!(!req.to_lowercase().contains("authorization:"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dub-llm opencode_`
Expected: FAIL with `error[E0599]: no function or associated item named 'opencode' found for struct 'ChatClient'`

- [ ] **Step 3: Write minimal implementation** — insert after `ollama()` (after line 177) in `crates/dub-llm/src/client.rs`:

```rust
/// Клиент к OpenCode (OpenAI-совместимый /v1/chat/completions): cloud —
/// base_url https://opencode.ai/zen + Bearer-ключ (Zen/Go-sub), local —
/// настраиваемый URL (`opencode serve`, дефолт http://localhost:4096),
/// ключ опционален (пароль serve). Обязательное поле `model`
/// (`opencode/<id>` для Zen), без llama-специфичного chat_template_kwargs
/// (`is_remote()` true по model).
pub fn opencode(
    base_url: impl Into<String>,
    model: impl Into<String>,
    api_key: Option<String>,
) -> Result<Self, LlmError> {
    let mut c = Self::new(base_url)?;
    c.model = Some(model.into());
    c.auth = api_key.filter(|k| !k.trim().is_empty());
    Ok(c)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dub-llm`
Expected: all PASS (including existing `ollama_sends_model_without_llama_kwargs`)

- [ ] **Step 5: Commit**

```bash
git add crates/dub-llm/src/client.rs
git commit -m "feat(llm): ChatClient::opencode for Zen cloud + local serve"
```

---

### Task 2: `models.rs` — selection keys + helpers + tests

**Files:**
- Modify: `crates/dub-server/src/models.rs` (allowlist ~lines 91-113, `llm_provider_kind` ~line 209, helpers after `ollama_model` ~line 261)
- Test: `mod tests` at end of same file (reuse existing `tmp_root`)

- [ ] **Step 1: Write the failing tests** — append to `mod tests` in `crates/dub-server/src/models.rs`:

```rust
#[test]
fn opencode_provider_kind_and_helpers() {
    let d = tmp_root("opencode");
    // дефолт: local (новый ключ не задан)
    assert_eq!(llm_provider_kind(&d, "llm"), "local");
    set_selection(&d, "llm_provider", "opencode").unwrap();
    assert_eq!(llm_provider_kind(&d, "llm"), "opencode");
    assert!(llm_provider_explicit(&d, "llm"));
    // cloud-дефолт базы
    assert_eq!(opencode_mode(&d), "cloud");
    assert_eq!(opencode_base_url(&d), "https://opencode.ai/zen");
    // local-режим: URL + trim слэша
    set_selection(&d, "opencode_mode", "local").unwrap();
    assert_eq!(opencode_mode(&d), "local");
    assert_eq!(opencode_base_url(&d), "http://localhost:4096");
    set_selection(&d, "opencode_url", "http://srv:4096/").unwrap();
    assert_eq!(opencode_base_url(&d), "http://srv:4096");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn opencode_model_vision_falls_back_to_llm() {
    let d = tmp_root("ocmodel");
    assert_eq!(opencode_model(&d, "llm"), "");
    set_selection(&d, "opencode_llm", "opencode/gpt-5.5").unwrap();
    assert_eq!(opencode_model(&d, "llm"), "opencode/gpt-5.5");
    assert_eq!(opencode_model(&d, "vision"), "opencode/gpt-5.5");
    set_selection(&d, "opencode_vision", "opencode/gemini-3").unwrap();
    assert_eq!(opencode_model(&d, "vision"), "opencode/gemini-3");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn opencode_asr_on_gates_flag_model_and_cloud_key() {
    let d = tmp_root("ocasr");
    assert!(!opencode_asr_on(&d));
    set_selection(&d, "opencode_asr_on", "1").unwrap();
    assert!(!opencode_asr_on(&d)); // нет модели
    set_selection(&d, "opencode_asr", "opencode/whisper-large-v3").unwrap();
    assert!(!opencode_asr_on(&d)); // cloud без ключа
    set_selection(&d, "opencode_key", "sk-test").unwrap();
    assert!(opencode_asr_on(&d));
    set_selection(&d, "opencode_mode", "local").unwrap();
    set_selection(&d, "opencode_key", "").unwrap();
    // local: ключ не нужен — но пустая строка в active.json остаётся Some("").
    // pick() фильтрует пустые, поэтому opencode_key() -> None и всё равно on:
    assert!(opencode_asr_on(&d));
    std::fs::remove_dir_all(&d).ok();
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dub-server --lib models::tests::opencode_`
Expected: FAIL with `cannot find function 'opencode_mode' in module 'models'` (and friends)

- [ ] **Step 3a: Allowlist + kind** — in `crates/dub-server/src/models.rs`:
  1. Update the comments on the `llm_provider`/`vision_provider` allowlist lines to `"local" (Gemma) | "ollama" | "openrouter" | "opencode"`.
  2. Append after the `"ollama_vision"` allowlist line:

```rust
            | "opencode_key"      // API-ключ OpenCode Zen/Go-sub (Bearer; хранится локально, не логируется)
            | "opencode_mode"     // "cloud" (Zen) | "local" (`opencode serve` на ПК)
            | "opencode_url"      // базовый URL локального OpenCode (дефолт http://localhost:4096)
            | "opencode_llm"      // id текстовой модели (Zen: "opencode/<id>")
            | "opencode_vision"   // id vision-модели (пусто -> берём opencode_llm)
            | "opencode_asr"      // id STT-модели OpenCode
            | "opencode_asr_on"   // "1" -> транскрипция через OpenCode вместо локального ASR
```

  3. In `llm_provider_kind`, add the arm `Some("opencode") => "opencode",` next to the ollama/openrouter arms.

- [ ] **Step 3b: Helpers** — insert after `ollama_model()` (after line 261) in `crates/dub-server/src/models.rs`:

```rust
/// Режим OpenCode: "local" (`opencode serve` на ПК) или "cloud" (Zen). Дефолт cloud.
pub fn opencode_mode(mroot: &Path) -> &'static str {
    match pick(&load_selection(mroot), "opencode_mode") {
        Some("local") => "local",
        _ => "cloud",
    }
}

/// Базовый URL OpenCode без хвостового слэша. Cloud -> https://opencode.ai/zen
/// (ChatClient допишет /v1/chat/completions); local -> настройка opencode_url.
pub fn opencode_base_url(mroot: &Path) -> String {
    if opencode_mode(mroot) == "local" {
        let sel = load_selection(mroot);
        pick(&sel, "opencode_url").unwrap_or("http://localhost:4096").trim_end_matches('/').to_string()
    } else {
        "https://opencode.ai/zen".to_string()
    }
}

/// API-ключ OpenCode Zen/Go-sub из active.json (локальное хранение, десктоп). Пусто/нет -> None.
pub fn opencode_key(mroot: &Path) -> Option<String> {
    pick(&load_selection(mroot), "opencode_key").map(str::to_string)
}

/// id модели OpenCode для стадии ("llm"|"vision"|"asr"). Vision: opencode_vision,
/// пусто -> opencode_llm. Пусто, если ничего не задано (вызывающий падает с понятной ошибкой).
pub fn opencode_model(mroot: &Path, stage: &str) -> String {
    let sel = load_selection(mroot);
    match stage {
        "llm" => pick(&sel, "opencode_llm").unwrap_or("").to_string(),
        "vision" => pick(&sel, "opencode_vision")
            .or_else(|| pick(&sel, "opencode_llm"))
            .unwrap_or("")
            .to_string(),
        "asr" => pick(&sel, "opencode_asr").unwrap_or("").to_string(),
        _ => String::new(),
    }
}

/// Включена ли транскрипция через OpenCode: флаг + модель; в cloud-режиме ещё и ключ.
/// Local-режим ключей в Dub Studio не требует (ключи уже в opencode.json на ПК).
pub fn opencode_asr_on(mroot: &Path) -> bool {
    let sel = load_selection(mroot);
    if pick(&sel, "opencode_asr_on") != Some("1") {
        return false;
    }
    let model_ok = pick(&sel, "opencode_asr").is_some_and(|m| !m.trim().is_empty());
    if !model_ok {
        return false;
    }
    if pick(&sel, "opencode_mode") != Some("local") && pick(&sel, "opencode_key").is_none() {
        return false;
    }
    true
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dub-server --lib models::tests`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/dub-server/src/models.rs
git commit -m "feat(server): opencode settings keys + mode/url/model/key/asr helpers"
```

---

### Task 3: `llm_provider.rs` — `"opencode"` branch + tests

**Files:**
- Modify: `crates/dub-server/src/llm_provider.rs` (`open()`, insert `"opencode"` arm after the `"ollama"` arm, lines 59-69)
- Test: `mod tests` at end of same file (reuse `tmp_root`/`open_in`)

- [ ] **Step 1: Write the failing tests** — append to `mod tests`:

```rust
#[test]
fn opencode_cloud_returns_remote_without_sidecar() {
    let d = tmp_root("oc-cloud");
    crate::models::set_selection(&d, "llm_provider", "opencode").unwrap();
    crate::models::set_selection(&d, "opencode_llm", "opencode/gpt-5.5").unwrap();
    crate::models::set_selection(&d, "opencode_key", "sk-test").unwrap();
    let p = open_in(&d, LlmMode::Text).unwrap();
    assert!(p.is_remote());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn opencode_empty_model_is_explicit_error() {
    let d = tmp_root("oc-empty");
    crate::models::set_selection(&d, "llm_provider", "opencode").unwrap();
    crate::models::set_selection(&d, "opencode_key", "sk-test").unwrap();
    let e = match open_in(&d, LlmMode::Text) {
        Ok(_) => panic!("ожидалась ошибка пустой opencode-модели"),
        Err(e) => e,
    };
    assert!(e.contains("OpenCode"), "unexpected error: {e}");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn opencode_cloud_without_key_is_explicit_error() {
    let d = tmp_root("oc-nokey");
    crate::models::set_selection(&d, "llm_provider", "opencode").unwrap();
    crate::models::set_selection(&d, "opencode_llm", "opencode/gpt-5.5").unwrap();
    let e = match open_in(&d, LlmMode::Text) {
        Ok(_) => panic!("ожидалась ошибка отсутствующего ключа"),
        Err(e) => e,
    };
    assert!(e.contains("opencode_key"), "unexpected error: {e}");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn opencode_local_without_key_opens_remote() {
    let d = tmp_root("oc-local");
    crate::models::set_selection(&d, "vision_provider", "opencode").unwrap();
    crate::models::set_selection(&d, "opencode_mode", "local").unwrap();
    crate::models::set_selection(&d, "opencode_vision", "my-local-model").unwrap();
    let p = open_in(&d, LlmMode::Vision).unwrap();
    assert!(p.is_remote());
    std::fs::remove_dir_all(&d).ok();
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p dub-server --lib llm_provider::tests::opencode_`
Expected: FAIL — `opencode_cloud_returns_remote_without_sidecar` falls through to local and errors with "llama-server не найден"; error tests fail on `Ok(_) => panic!`

- [ ] **Step 3: Replace `open()` — insert the `"opencode"` arm after the `"ollama"` arm** (after line 69) in `crates/dub-server/src/llm_provider.rs`:

```rust
"opencode" => {
    let model = crate::models::opencode_model(o.models_root, stage);
    if model.trim().is_empty() {
        return Err("OpenCode выбран, но модель не задана (opencode_llm/opencode_vision)".to_string());
    }
    // Cloud (Zen/Go-sub) без ключа невозможен; local ключ не требует
    // (опционален — пароль `opencode serve`).
    let key = if crate::models::opencode_mode(o.models_root) == "cloud" {
        match crate::models::opencode_key(o.models_root) {
            Some(k) => Some(k),
            None => return Err("OpenCode cloud выбран, но ключ не задан (opencode_key)".to_string()),
        }
    } else {
        crate::models::opencode_key(o.models_root)
    };
    let base = crate::models::opencode_base_url(o.models_root);
    let client = ChatClient::opencode(base, model, key).map_err(|e| e.to_string())?;
    return Ok(LlmProvider::Remote { client });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p dub-server --lib llm_provider::tests && cargo test -p dub-server --lib models::tests`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add crates/dub-server/src/llm_provider.rs
git commit -m "feat(server): route opencode cloud/local per-stage via llm_provider_kind"
```

---

### Task 4: `opencode_asr.rs` — native transcription + `analyze.rs` wiring

**Files:**
- Modify: `crates/dub-server/Cargo.toml` (reqwest line ~35: add `"multipart"` to features)
- Create: `crates/dub-server/src/opencode_asr.rs`
- Modify: `crates/dub-server/src/lib.rs` (add `mod opencode_asr;` after the `mod llm_provider;` line)
- Modify: `crates/dub-server/src/analyze.rs` (insert `opencode_asr_on` branch before the `openrouter_asr_on` branch at line 701)

- [ ] **Step 1: Write the failing test for the pure parser** — create `crates/dub-server/src/opencode_asr.rs` with ONLY this content:

```rust
//! Облачная транскрипция через OpenCode (Zen/local): нативный multipart-POST файла на
//! {base}/v1/audio/transcriptions (response_format=verbose_json). Без сайдкара —
//! reqwest уже в дереве. Возвращает сегменты (start, end, text), как cloud_asr.rs.

/// Разобрать verbose_json-ответ: segments[] -> (start,end,text); пусто ->
/// fallback на один сегмент из text/duration; совсем пусто -> Err.
pub fn parse_verbose_json(v: &serde_json::Value) -> Result<Vec<(f64, f64, String)>, String> {
    let mut out = Vec::new();
    if let Some(arr) = v.get("segments").and_then(|s| s.as_array()) {
        for s in arr {
            let st = s.get("start").and_then(|x| x.as_f64());
            let en = s.get("end").and_then(|x| x.as_f64());
            let tx = s.get("text").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
            if let (Some(st), Some(en)) = (st, en) {
                if !tx.is_empty() {
                    out.push((st, en, tx));
                }
            }
        }
    }
    if out.is_empty() {
        let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
        if text.is_empty() {
            return Err("облачный STT вернул пустой транскрипт".into());
        }
        let dur = v.get("duration").and_then(|x| x.as_f64()).unwrap_or(0.0);
        out.push((0.0, dur.max(0.1), text));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_segments_and_skips_empties() {
        let v = serde_json::json!({
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " привет "},
                {"start": 1.5, "end": 2.0, "text": "   "},
                {"start": 2.0, "end": 3.0, "text": "мир"}
            ]
        });
        let out = parse_verbose_json(&v).unwrap();
        assert_eq!(out, vec![(0.0, 1.5, "привет".to_string()), (2.0, 3.0, "мир".to_string())]);
    }

    #[test]
    fn falls_back_to_single_segment_from_text() {
        let v = serde_json::json!({"text": "целый кусок", "duration": 5.0});
        let out = parse_verbose_json(&v).unwrap();
        assert_eq!(out, vec![(0.0, 5.0, "целый кусок".to_string())]);
    }

    #[test]
    fn empty_response_is_error() {
        let v = serde_json::json!({"segments": []});
        assert!(parse_verbose_json(&v).is_err());
    }
}
```

Also add `mod opencode_asr;` to `crates/dub-server/src/lib.rs` now (so the test compiles).

- [ ] **Step 2: Run test to verify it fails** (compile error — `serde_json` not imported? `serde_json` IS a workspace dep used across the crate, so this compiles; instead the failure is the missing `transcribe` used nowhere yet — skip: run to confirm the new tests PASS as pure logic)

Run: `cargo test -p dub-server --lib opencode_asr`
Expected: PASS (parser is self-contained; TDD here guards the contract before wiring)

- [ ] **Step 3: Add `multipart` + `transcribe()`** —
  1. In `crates/dub-server/Cargo.toml`, change the reqwest line to:

```toml
reqwest = { version = "0.12", default-features = false, features = ["blocking", "rustls-tls", "multipart"] }
```

  2. Append to `crates/dub-server/src/opencode_asr.rs`:

```rust
use std::path::Path;

/// Транскрибировать wav через OpenCode -> сегменты (start, end, text). Нет модели ->
/// Err. `src_lang` — ISO-639-1 или "auto"/"" (авто-детект). Cloud без ключа -> Err
/// (проверяется раньше в opencode_asr_on, здесь — честный повторный гард).
pub fn transcribe(models_root: &Path, wav: &Path, src_lang: &str) -> Result<Vec<(f64, f64, String)>, String> {
    let base = crate::models::opencode_base_url(models_root);
    let model = crate::models::opencode_model(models_root, "asr");
    if model.trim().is_empty() {
        return Err("STT-модель OpenCode не выбрана в настройках".into());
    }
    if crate::models::opencode_mode(models_root) == "cloud" && crate::models::opencode_key(models_root).is_none() {
        return Err("облачный ASR включён, но ключ OpenCode не задан".into());
    }
    let key = crate::models::opencode_key(models_root);
    let lang = src_lang.trim();
    let mut form = reqwest::blocking::multipart::Form::new()
        .text("model", model)
        .text("response_format", "verbose_json")
        .file("file", wav)
        .map_err(|e| format!("чтение wav: {e}"))?;
    if !lang.is_empty() && !lang.eq_ignore_ascii_case("auto") {
        form = form.text("language", lang.to_string());
    }
    // Клиент строим явно (600с — длинная транскрипция); прокси подхватывается из env,
    // который apply_proxy_env прописал на старте из active.json (как у остальных клиентов).
    let mut req = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("http: {e}"))?
        .post(format!("{base}/v1/audio/transcriptions"))
        .multipart(form);
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    let v: serde_json::Value = req
        .send()
        .map_err(|e| format!("связь с OpenCode: {e}"))?
        .json()
        .map_err(|e| format!("ответ OpenCode не JSON: {e}"))?;
    parse_verbose_json(&v)
}
```

  3. In `crates/dub-server/src/analyze.rs`, insert BEFORE the `} else if crate::models::openrouter_asr_on(...)` branch (line 701):

```rust
} else if crate::models::opencode_asr_on(&paths.models_root) {
    // Облачная транскрипция (OpenCode STT): сегменты из облака/локального serve,
    // спикеров раздаём диаризацией (как локальный ASR и OpenRouter STT).
    bench.stage("asr");
    emit(progress, "asr", "транскрипция через облако (OpenCode STT)");
    let raw = crate::opencode_asr::transcribe(&paths.models_root, &asr_wav, &args.src_lang)
        .map_err(|e| format!("облачный STT: {e}"))?;
    let turns: &[dub_asr::Turn] = diar.as_ref().map(|d| d.turns.as_slice()).unwrap_or(&[]);
    let nsp = diar.as_ref().map(|d| d.n_speakers).unwrap_or(1).max(1);
    let segs: Vec<Segment> = raw
        .into_iter()
        .enumerate()
        .map(|(i, (st, en, tx))| Segment {
            id: format!("s{i}"),
            start: st,
            end: en,
            speaker: Some(if turns.is_empty() { "0".to_string() } else { speaker_for(st, en, turns) }),
            src_text: tx,
            tgt_text: String::new(),
            voice: None,
            dirty: false,
            ckpt: None,
            extra: Default::default(),
        })
        .collect();
    emit(progress, "asr", &format!("облако: {} реплик, {} спикер(ов)", segs.len(), nsp));
    (segs, nsp)
```

- [ ] **Step 4: Run tests + build**

Run: `cargo test -p dub-server --lib opencode_asr && cargo build -p dub-server`
Expected: tests PASS, build clean (no new warnings)

- [ ] **Step 5: Commit**

```bash
git add crates/dub-server/Cargo.toml crates/dub-server/src/opencode_asr.rs crates/dub-server/src/lib.rs crates/dub-server/src/analyze.rs
git commit -m "feat(server): OpenCode cloud ASR transcribe + analyze branch"
```

---

### Task 5: models.dev catalog — `models_catalog.rs` + endpoint + route

**Files:**
- Create: `crates/dub-server/src/models_catalog.rs`
- Modify: `crates/dub-server/src/endpoints.rs` (append `opencode_models` handler after `openrouter_models`, ~line 57)
- Modify: `crates/dub-server/src/lib.rs` (add `mod models_catalog;`, add route after line 327)

- [ ] **Step 1: Write the failing tests for the pure filter** — create `crates/dub-server/src/models_catalog.rs` with ONLY the types + filter + tests:

```rust
//! Каталог моделей OpenCode через публичный models.dev (без ключа).
//! `catalog.json?type=all` кешируем на 24ч в models/models-dev-catalog.json
//! (stale-on-error), фильтруем по модальности из provider-agnostic metadata,
//! предпочитаем список провайдера "opencode". Только для OpenCode-дропдаунов.

use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CATALOG_URL: &str = "https://models.dev/catalog.json?type=all";
pub const CACHE_FILE: &str = "models-dev-catalog.json";
/// TTL кэша каталога: 24ч (каталог меняется редко, вес — мегабайты).
pub const CACHE_TTL_SECS: u64 = 24 * 3600;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<u64>,
}

fn has_mod(meta: &serde_json::Value, ptr: &str, m: &str) -> bool {
    meta.pointer(ptr)
        .and_then(|v| v.as_array())
        .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(m)))
}

/// Модальность стадии: llm — output text; vision — input image; asr — input audio.
pub fn modality_ok(meta: &serde_json::Value, kind: &str) -> bool {
    match kind {
        "vision" => has_mod(meta, "/modalities/input", "image"),
        "asr" => has_mod(meta, "/modalities/input", "audio"),
        _ => has_mod(meta, "/modalities/output", "text"),
    }
}

/// Отфильтровать каталог: id-кандидаты (модели провайдера "opencode", иначе все),
/// затем фильтр модальности; сортировка по id. Чистая функция — тестируется на фикстуре.
pub fn filter_models(catalog: &serde_json::Value, kind: &str) -> Vec<ModelEntry> {
    let ids: Vec<&str> = catalog
        .pointer("/providers/opencode/models")
        .and_then(|v| v.as_object())
        .map(|m| m.keys().map(|k| k.as_str()).collect())
        .unwrap_or_else(|| {
            catalog
                .pointer("/models")
                .and_then(|v| v.as_object())
                .map(|m| m.keys().map(|k| k.as_str()).collect())
                .unwrap_or_default()
        });
    let mut out: Vec<ModelEntry> = ids
        .into_iter()
        .filter_map(|id| {
            let meta = catalog.pointer(&format!("/models/{id}"))?;
            if !modality_ok(meta, kind) {
                return None;
            }
            Some(ModelEntry {
                id: id.to_string(),
                name: meta.get("name").and_then(|n| n.as_str()).unwrap_or(id).to_string(),
                context: meta.pointer("/limit/context").and_then(|c| c.as_u64()),
            })
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::json!({
            "providers": {"opencode": {"models": {
                "opencode/gpt-5.5": {}, "opencode/qwen-vl": {}, "opencode/whisper": {}, "opencode/tts-only": {}
            }}},
            "models": {
                "opencode/gpt-5.5": {"name": "GPT 5.5", "modalities": {"input": ["text"], "output": ["text"]}, "limit": {"context": 200000}},
                "opencode/qwen-vl": {"name": "Qwen VL", "modalities": {"input": ["text", "image"], "output": ["text"]}},
                "opencode/whisper": {"name": "Whisper", "modalities": {"input": ["audio"], "output": ["text"]}},
                "opencode/tts-only": {"name": "TTS", "modalities": {"input": ["text"], "output": ["audio"]}}
            }
        })
    }

    #[test]
    fn llm_lists_text_output_only() {
        let ids: Vec<String> = filter_models(&fixture(), "llm").into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["opencode/gpt-5.5", "opencode/qwen-vl", "opencode/whisper"]);
    }

    #[test]
    fn vision_lists_image_input_only() {
        let ids: Vec<String> = filter_models(&fixture(), "vision").into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["opencode/qwen-vl"]);
    }

    #[test]
    fn asr_lists_audio_input_only() {
        let ids: Vec<String> = filter_models(&fixture(), "asr").into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["opencode/whisper"]);
    }

    #[test]
    fn falls_back_to_all_models_without_opencode_provider() {
        let mut c = fixture();
        c.as_object_mut().unwrap().get_mut("providers").unwrap()
            .as_object_mut().unwrap().remove("opencode");
        let ids: Vec<String> = filter_models(&c, "llm").into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["opencode/gpt-5.5", "opencode/qwen-vl", "opencode/whisper"]);
    }

    #[test]
    fn context_comes_from_limit() {
        let all = filter_models(&fixture(), "llm");
        let g = all.iter().find(|m| m.id == "opencode/gpt-5.5").unwrap();
        assert_eq!(g.context, Some(200000));
        assert_eq!(g.name, "GPT 5.5");
    }
}
```

Also add `mod models_catalog;` to `crates/dub-server/src/lib.rs` now.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p dub-server --lib models_catalog`
Expected: PASS (pure logic, no network)

- [ ] **Step 3: Add fetch/cache + endpoint + route** —
  1. Append to `crates/dub-server/src/models_catalog.rs`:

```rust
fn cache_is_fresh(mroot: &Path) -> bool {
    std::fs::metadata(mroot.join(CACHE_FILE))
        .and_then(|m| m.modified())
        .map(|t| std::time::SystemTime::now().duration_since(t).map(|d| d.as_secs() < CACHE_TTL_SECS).unwrap_or(false))
        .unwrap_or(false)
}

/// Загрузить каталог: свежий кэш -> с диска; иначе скачать (прокси-aware ureq,
/// как endpoints::proxy_test), положить в кэш, распарсить. Ошибка скачки при
/// наличии ЛЮБОГО кэша -> stale-on-error (пусть и протухший).
pub fn load_catalog(mroot: &Path) -> Result<serde_json::Value, String> {
    let cache = mroot.join(CACHE_FILE);
    if cache_is_fresh(mroot) {
        if let Ok(v) = std::fs::read_to_string(&cache)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        {
            return Ok(v);
        }
    }
    let agent: ureq::Agent = match crate::models::proxy_url(mroot) {
        Some(url) => {
            let proxy = ureq::Proxy::new(&url).map_err(|e| format!("прокси: {e}"))?;
            ureq::Agent::config_builder().proxy(Some(proxy)).build().into()
        }
        None => ureq::Agent::config_builder().build().into(),
    };
    match agent.get(CATALOG_URL).call() {
        Ok(mut resp) => {
            let body = resp.body_mut().read_to_string().map_err(|e| format!("чтение каталога: {e}"))?;
            let v: serde_json::Value =
                serde_json::from_str(&body).map_err(|e| format!("парсинг каталога: {e}"))?;
            let _ = std::fs::create_dir_all(mroot);
            let _ = std::fs::write(&cache, body.as_bytes());
            Ok(v)
        }
        Err(e) => {
            if let Ok(v) = std::fs::read_to_string(&cache)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            {
                return Ok(v); // stale-on-error
            }
            Err(format!("скачивание каталога: {e}"))
        }
    }
}

/// Каталог для дропдауна стадии: load + filter одним вызовом.
pub fn load_opencode_models(mroot: &Path, kind: &str) -> Result<Vec<ModelEntry>, String> {
    load_catalog(mroot).map(|c| filter_models(&c, kind))
}
```

  2. `serde` derive needs `serde` with `derive` feature — check: `endpoints.rs` uses `serde_json`, handlers use `Json`; `dub-server/Cargo.toml` — serde present? `models.rs` uses `serde_json::Value`; derive for Serialize/Deserialize needs `serde` crate with derive. If missing, add `serde = { version = "1", features = ["derive"] }` — verify with `cargo build` in Step 4 and add only if the compiler asks (`cannot find derive macro`). Prefer: check first via the grep in Step 4; the plan allows this single conditional add.
  3. Append handler to `crates/dub-server/src/endpoints.rs` (after `openrouter_models`, ~line 57):

```rust
// ─── GET /engine/opencode/models?kind=llm|vision|asr ─────────────────────────
// Каталог моделей OpenCode через публичный models.dev (БЕЗ ключа): catalog.json
// кешируем на 24ч в models/models-dev-catalog.json (stale-on-error), фильтруем
// по модальности; предпочитаем список провайдера "opencode". Только для OpenCode.
pub async fn opencode_models(State(st): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let kind = q.get("kind").cloned().unwrap_or_else(|| "llm".to_string());
    let mroot = st.models_root.clone();
    let res = tokio::task::spawn_blocking(move || crate::models_catalog::load_opencode_models(&mroot, &kind))
        .await
        .unwrap_or_else(|e| Err(e.to_string()));
    match res {
        Ok(models) => Json(json!({ "models": models })).into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("каталог models.dev: {e}")).into_response(),
    }
}
```

  4. In `crates/dub-server/src/lib.rs`, add route after line 327:

```rust
.route("/engine/opencode/models", get(endpoints::opencode_models))
```

- [ ] **Step 4: Run tests + build**

Run: `cargo test -p dub-server --lib models_catalog && cargo build -p dub-server`
Expected: tests PASS, build clean. (If `serde derive` missing: `cargo add` NOT needed — add `serde = { version = "1", features = ["derive"] }` to `crates/dub-server/Cargo.toml` deps, matching the workspace's serde version, then rebuild.)

- [ ] **Step 5: Commit**

```bash
git add crates/dub-server/src/models_catalog.rs crates/dub-server/src/endpoints.rs crates/dub-server/src/lib.rs crates/dub-server/Cargo.toml
git commit -m "feat(server): models.dev-backed /engine/opencode/models with 24h cache"
```

---

### Task 6a: Frontend — `api.ts` + `ProviderTabs` + llm/vision sections

**Files:**
- Modify: `frontend/src/lib/api.ts` (after `openrouterModels`, line 118)
- Modify: `frontend/src/App.tsx` (`ProviderTabs` lines 71-84; llm section lines 364-396)

- [ ] **Step 1: `api.ts` wrapper** — insert after line 118:

```ts
opencodeModels: (kind: "llm" | "vision" | "asr") => getJson<{ models: { id: string; name: string; context?: number }[] }>(`/engine/opencode/models?kind=${kind}`),
```

- [ ] **Step 2: `ProviderTabs` 4th tab** — replace the array in `ProviderTabs` (line 73):

```tsx
{[{ id: "local", label: localLabel }, { id: "ollama", label: "Ollama" }, { id: "openrouter", label: "OpenRouter" }, { id: "opencode", label: "OpenCode" }].map((p) => {
```

and extend the disable rule (line 74): OpenCode-cloud needs the key, local does not — handle per-section instead of here. Keep `orDisabled` applying ONLY to `openrouter`:

```tsx
const dis = p.id === "openrouter" && orDisabled;
```

(No change needed — the new tab is never disabled by `orDisabled`. `hasOrKey` stays OpenRouter-only; add `hasOcKey` next to it in the section component: `const hasOcKey = (cap?.selection?.opencode_key ?? "").trim().length > 0;`)

- [ ] **Step 3: OpenCode controls in the llm section** — after the `llmProv === "ollama"` block (lines 370-373), insert the `opencode` branch (mirror it, plus cloud/local toggle). Required state (top of `ModelsSection`, next to `orModels` line 108):

```tsx
const [ocModels, setOcModels] = useState<Record<string, { id: string }[]>>({});
const ocMode = selv("opencode_mode") || "cloud";
```

Fetch effect (next to the OpenRouter catalog effect, lines 125-129):

```tsx
// Каталог OpenCode — публичный models.dev (без ключа), как только секция выбрана.
useEffect(() => {
  if (llmProv !== "opencode" && visProv !== "opencode" && selv("opencode_asr_on") !== "1") return;
  (["llm", "vision", "asr"] as const).forEach((kind) =>
    api.opencodeModels(kind).then((r) => setOcModels((m) => ({ ...m, [kind]: r.models }))).catch(() => {}));
}, [llmProv, visProv, cap?.selection?.opencode_asr_on]);
```

`OcModelSelect` (next to `OrModelSelect`, line 264):

```tsx
const OcModelSelect = ({ kind, k, empty }: { kind: "llm" | "vision" | "asr"; k: string; empty: string }) => (
  <select value={selv(k)} onChange={(e) => setSel(k, e.target.value)} className={orSelectCls}>
    <option value="">{empty}</option>
    {(ocModels[kind] ?? []).map((m) => <option key={m.id} value={m.id}>{m.id}</option>)}
  </select>
);
```

llm branch JSX (after the ollama block):

```tsx
) : llmProv === "opencode" ? (
  <div className={`${orRowCls} space-y-2`}>
    <div className="flex gap-1">
      {(["cloud", "local"] as const).map((m) => (
        <button key={m} onClick={() => setSel("opencode_mode", m)}
          className={`flex-1 px-2 py-1 rounded-md text-[11px] font-medium border transition-colors ${ocMode === m ? "border-[var(--color-accent)] text-[var(--color-text)]" : "border-[var(--color-border)] text-[var(--color-muted)]"}`}>
          {m === "cloud" ? t("settings.opencodeCloud") : t("settings.opencodeLocal")}
        </button>
      ))}
    </div>
    {ocMode === "cloud" ? (
      <input value={selv("opencode_key")} onChange={(e) => setSel("opencode_key", e.target.value)} type="password" className={orSelectCls} placeholder={t("settings.opencodeKey")} />
    ) : (
      <input value={selv("opencode_url") || "http://localhost:4096"} onChange={(e) => setSel("opencode_url", e.target.value)} className={orSelectCls} placeholder={t("settings.opencodeUrl")} />
    )}
    <OcModelSelect kind="llm" k="opencode_llm" empty={t("settings.opencodeLlm")} />
  </div>
```

Same pattern for the vision section (`opencode_vision`, `kind="vision"`, `empty={t("settings.opencodeVision")}`).

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npm run build`
Expected: clean (tsc + vite)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/App.tsx
git commit -m "feat(ui): OpenCode provider tabs + cloud/local + models.dev dropdowns (llm/vision)"
```

---

### Task 6b: Frontend — ASR section 4th button

**Files:**
- Modify: `frontend/src/App.tsx` (ASR engine tabs, lines 320-332; ASR cloud block, lines 336-340)

- [ ] **Step 1: 4th ASR button** — extend the engine array (line 321):

```tsx
{[{ id: "parakeet", label: "Parakeet-TDT", cloud: false }, { id: "whisper", label: "Whisper", cloud: false }, { id: "openrouter", label: "OpenRouter", cloud: true }, { id: "opencode", label: "OpenCode", cloud: true }].map((e) => {
```

and rework the active/click logic: the section currently tracks a single `or_asr_on` flag. Introduce `const ocAsrOn = selv("opencode_asr_on") === "1";` next to `asrCloud`, with mutual exclusion on click:

```tsx
const asrCloud = selv("or_asr_on") === "1";
const ocAsrOn = selv("opencode_asr_on") === "1";
const active = e.id === "openrouter" ? asrCloud : e.id === "opencode" ? ocAsrOn : (!asrCloud && !ocAsrOn && asrEngine === e.id);
const dis = e.cloud && (e.id === "openrouter" ? !hasOrKey : ocMode === "cloud" && !hasOcKey);
onClick={() => {
  if (e.id === "openrouter") { setSel("opencode_asr_on", "0"); setSel("or_asr_on", "1"); }
  else if (e.id === "opencode") { setSel("or_asr_on", "0"); setSel("opencode_asr_on", "1"); }
  else { setSel("or_asr_on", "0"); setSel("opencode_asr_on", "0"); setAsrEngine(e.id); api.setSelection("asr_engine", e.id).catch(() => {}); }
}}
```

- [ ] **Step 2: OpenCode ASR block** — after the OpenRouter ASR block (lines 336-340), add:

```tsx
) : ocAsrOn ? (
  <div className={`${orRowCls} space-y-2`}>
    <div className="flex gap-1">
      {(["cloud", "local"] as const).map((m) => (
        <button key={m} onClick={() => setSel("opencode_mode", m)}
          className={`flex-1 px-2 py-1 rounded-md text-[11px] font-medium border transition-colors ${ocMode === m ? "border-[var(--color-accent)] text-[var(--color-text)]" : "border-[var(--color-border)] text-[var(--color-muted)]"}`}>
          {m === "cloud" ? t("settings.opencodeCloud") : t("settings.opencodeLocal")}
        </button>
      ))}
    </div>
    {ocMode === "cloud" ? (
      <input value={selv("opencode_key")} onChange={(e) => setSel("opencode_key", e.target.value)} type="password" className={orSelectCls} placeholder={t("settings.opencodeKey")} />
    ) : (
      <input value={selv("opencode_url") || "http://localhost:4096"} onChange={(e) => setSel("opencode_url", e.target.value)} className={orSelectCls} placeholder={t("settings.opencodeUrl")} />
    )}
    <OcModelSelect kind="asr" k="opencode_asr" empty={t("settings.opencodeAsr")} />
    <div className="text-[11px] text-[var(--color-muted)]">{t("settings.opencodeAsrHint")}</div>
  </div>
```

and change the local-ASR guards: `{selv("or_asr_on") !== "1" && <BackendTabs k="asr_backend" />}` → `{selv("or_asr_on") !== "1" && selv("opencode_asr_on") !== "1" && <BackendTabs k="asr_backend" />}`.

- [ ] **Step 3: Typecheck + lint**

Run: `cd frontend && npm run build && npm run lint`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(ui): OpenCode ASR engine button + cloud/local + STT dropdown"
```

---

### Task 6c: Locales (all 6) + landing dict check

**Files:**
- Modify: `frontend/src/locales/en.json`, `es.json`, `fr.json`, `pt.json`, `ru.json`, `zh.json` (settings block, after the `ollamaVision` line)

- [ ] **Step 1: Add strings** — after the `"ollamaVision"` line in each locale file, insert (en shown; other locales below):

en:
```json
"opencodeKey": "OpenCode API key (Zen / Go subscription)",
"opencodeCloud": "Cloud (Zen)",
"opencodeLocal": "Local CLI",
"opencodeUrl": "OpenCode local URL",
"opencodeLlm": "OpenCode text model",
"opencodeVision": "OpenCode vision model (empty = same as text)",
"opencodeAsr": "OpenCode STT model",
"opencodeAsrHint": "Transcription via OpenCode — no heavy local ASR download needed.",
```

ru:
```json
"opencodeKey": "API-ключ OpenCode (Zen / Go-подписка)",
"opencodeCloud": "Облако (Zen)",
"opencodeLocal": "Локальный CLI",
"opencodeUrl": "URL локального OpenCode",
"opencodeLlm": "Текстовая модель OpenCode",
"opencodeVision": "Vision-модель OpenCode (пусто = как текстовая)",
"opencodeAsr": "STT-модель OpenCode",
"opencodeAsrHint": "Транскрипция через OpenCode — тяжёлые локальные ASR-модели качать не нужно.",
```

es:
```json
"opencodeKey": "Clave API de OpenCode (Zen / suscripción Go)",
"opencodeCloud": "Nube (Zen)",
"opencodeLocal": "CLI local",
"opencodeUrl": "URL local de OpenCode",
"opencodeLlm": "Modelo de texto de OpenCode",
"opencodeVision": "Modelo de visión de OpenCode (vacío = igual que el de texto)",
"opencodeAsr": "Modelo STT de OpenCode",
"opencodeAsrHint": "Transcripción vía OpenCode — sin descargar ASR locales pesados.",
```

fr:
```json
"opencodeKey": "Clé API OpenCode (Zen / abonnement Go)",
"opencodeCloud": "Cloud (Zen)",
"opencodeLocal": "CLI local",
"opencodeUrl": "URL OpenCode local",
"opencodeLlm": "Modèle texte OpenCode",
"opencodeVision": "Modèle vision OpenCode (vide = comme le texte)",
"opencodeAsr": "Modèle STT OpenCode",
"opencodeAsrHint": "Transcription via OpenCode — pas de gros ASR local à télécharger.",
```

pt:
```json
"opencodeKey": "Chave API do OpenCode (Zen / assinatura Go)",
"opencodeCloud": "Nuvem (Zen)",
"opencodeLocal": "CLI local",
"opencodeUrl": "URL local do OpenCode",
"opencodeLlm": "Modelo de texto do OpenCode",
"opencodeVision": "Modelo de visão do OpenCode (vazio = igual ao de texto)",
"opencodeAsr": "Modelo STT do OpenCode",
"opencodeAsrHint": "Transcrição via OpenCode — sem baixar ASR locais pesados.",
```

zh:
```json
"opencodeKey": "OpenCode API 密钥 (Zen / Go 订阅)",
"opencodeCloud": "云端 (Zen)",
"opencodeLocal": "本地 CLI",
"opencodeUrl": "本地 OpenCode URL",
"opencodeLlm": "OpenCode 文本模型",
"opencodeVision": "OpenCode 视觉模型 (空 = 与文本相同)",
"opencodeAsr": "OpenCode 语音识别模型",
"opencodeAsrHint": "通过 OpenCode 转录 — 无需下载本地 ASR 大模型。",
```

Also check `docs/index.html` for settings strings (the Ollama plan verified via grep it carries none — same check here):

Run: `grep -c "ollamaLlm\|or_llm" docs/index.html || echo "no settings strings in landing"`
Expected: `no settings strings in landing` (nothing to update). If matches exist, mirror them.

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npm run build`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add frontend/src/locales/ docs/index.html
git commit -m "feat(i18n): OpenCode provider strings in all 6 locales"
```

---

### Task 7: Full verification (backend + frontend + E2E)

- [ ] **Step 1: Rust tests (whole workspace light)**

Run: `cargo test -p dub-llm && cargo test -p dub-server --lib`
Expected: all PASS

- [ ] **Step 2: Rust build, no new warnings**

Run: `cargo build --release -p dub-server`
Expected: exit 0, no warnings mentioning `opencode`/`models_catalog`

- [ ] **Step 3: Frontend build + lint**

Run: `cd frontend && npm run build && npm run lint`
Expected: both clean

- [ ] **Step 4: Real short-clip E2E (per repo verification rules)**

```bash
ffmpeg -y -i test_media/<file>.mp4 -t 20 -c copy scratchpad/oc-clip.mp4
```

Then in the app: create project from `scratchpad/oc-clip.mp4` (throwaway pid) → analyze with llm/vision on OpenCode (cloud if key present, else local serve) and ASR on OpenCode → verify **in the preview UI as a user**: translated subs readable, vision context present, transcript sensible. Never mutate a real `workspace/<pid>`. Backend reads are timing-only, never the reported proof.

If no OpenCode key/daemon is available: verify the HTTP shape against a stub speaking the OpenAI-compatible contract (Ollama-plan precedent — stub asserts `model` + Bearer + no `chat_template_kwargs`, and `/v1/audio/transcriptions` returns a `verbose_json` fixture).

- [ ] **Step 5: Commit the plan + push nothing**

```bash
git add docs/superpowers/plans/2026-09-22-opencode-provider.md
git commit -m "docs: OpenCode provider implementation plan"
```

Do NOT `gh release create` / publish / deploy (repo rule — release only on explicit user go).
