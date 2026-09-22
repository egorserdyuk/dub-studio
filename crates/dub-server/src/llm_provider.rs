//! Выбор LLM-провайдера для одной стадии: локальный llama-server (Gemma+mmproj) ИЛИ облако OpenRouter.
//! Переключение — по настройкам active.json (or_key + or_llm_on/or_vision_on). Абстрагирует 5 call-site'ов
//! (translate/compose/endpoints/analyze), где раньше был прямой `LlamaServer::start + ChatClient::new`.
//!
//! Инвариант: локальный путь неизменен (тот же ServerOpts/ubatch/mmproj). Явный выбор облака/Ollama
//! с плохой конфигурацией — явная ошибка; legacy or_*_on без ключа/модели — тихий fallback на локаль.

use std::path::Path;

use dub_llm::{ChatClient, LlamaServer, ServerOpts};

/// Режим вызова: плоский текст (перевод/ремикс) или мультимодальный (vision-анализ кадров).
#[derive(Clone, Copy, PartialEq)]
pub enum LlmMode {
    Text,
    Vision,
}

/// Готовый провайдер: держит клиент чата + (для локали) живой llama-server, который глушится по Drop.
pub enum LlmProvider {
    /// Локальный сайдкар: сервер держим живым, пока провайдер в скоупе (Drop останавливает процесс).
    Local {
        _server: LlamaServer,
        client: ChatClient,
    },
    /// Облако OpenRouter: только HTTP-клиент, сервер не нужен.
    Remote {
        client: ChatClient,
    },
}

impl LlmProvider {
    pub fn client(&self) -> &ChatClient {
        match self {
            LlmProvider::Local { client, .. } => client,
            LlmProvider::Remote { client } => client,
        }
    }

    /// true, если это облачный путь (для логов/веток, где vision требует multimodal-модель).
    pub fn is_remote(&self) -> bool {
        matches!(self, LlmProvider::Remote { .. })
    }
}

/// Параметры открытия провайдера (пути локальных весов + models_root для чтения настроек).
pub struct LlmOpen<'a> {
    pub llama_bin: &'a Path,
    pub mt_model: &'a Path,
    pub mmproj: &'a Path,
    pub models_root: &'a Path,
}

/// Открыть провайдер для стадии. Облако — если включено в настройках и есть ключ; иначе локальный
/// llama-server (для Vision добавляем mmproj, если файл есть). Возвращает Err с человекочитаемой причиной.
pub fn open(o: &LlmOpen, mode: LlmMode) -> Result<LlmProvider, String> {
    let stage = if mode == LlmMode::Vision { "vision" } else { "llm" };
    match crate::models::llm_provider_kind(o.models_root, stage) {
        "ollama" => {
            let model = crate::models::ollama_model(o.models_root, stage);
            if model.trim().is_empty() {
                return Err(
                    "Ollama выбран, но модель не задана (ollama_llm/ollama_vision)".to_string(),
                );
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
        return Err(format!("llama-server не найден ({})", o.llama_bin.display()));
    }
    if !o.mt_model.is_file() {
        return Err(format!("GGUF Gemma не найден ({})", o.mt_model.display()));
    }
    let mut opts = ServerOpts::new(o.llama_bin, o.mt_model)
        .with_ubatch(crate::models::sel_num(o.models_root, "llama_ubatch").map(|f| f as u32));
    if mode == LlmMode::Vision && o.mmproj.is_file() {
        opts = opts.with_mmproj(o.mmproj);
    }
    let server = LlamaServer::start(&opts).map_err(|e| format!("llama-server: {e}"))?;
    let client = ChatClient::new(server.base_url()).map_err(|e| format!("клиент чата: {e}"))?;
    Ok(LlmProvider::Local { _server: server, client })
}

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
            &LlmOpen {
                llama_bin: &fake,
                mt_model: &fake,
                mmproj: &fake,
                models_root: dir,
            },
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
        // match вместо unwrap_err: LlmProvider не Debug (sidecar/reqwest внутри)
        let e = match open_in(&d, LlmMode::Text) {
            Ok(_) => panic!("ожидалась ошибка пустой ollama-модели"),
            Err(e) => e,
        };
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
