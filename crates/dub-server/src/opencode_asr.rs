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
