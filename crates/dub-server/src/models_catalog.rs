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
            // id содержит `/` (напр. "opencode/gpt-5.5") — экранируем по RFC6901, иначе pointer режет ключ.
            let esc = id.replace('~', "~0").replace('/', "~1");
            let meta = catalog.pointer(&format!("/models/{esc}"))?;
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
        if let Some(v) = std::fs::read_to_string(&cache)
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
            if let Some(v) = std::fs::read_to_string(&cache)
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
