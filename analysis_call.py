import os
os.environ["HUGGINGFACE_HUB_CACHE"] = 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm
from faster_whisper import WhisperModel
import librosa

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    pipeline,
)



CALLS_DIR = Path.home() / "Downloads" / "call"
OUT_ROOT  = Path.home() / "Downloads" / "new_out"
MAX_FILES = 10

LANGUAGE = "en"
MODEL_SIZE = "small"
BEAM_SIZE = 1

COMPUTE_TYPE_GPU = "float16"
COMPUTE_TYPE_CPU = "int8"

MAX_AUDIO_EMO_SECONDS = 6.0

PUNCT_MODEL_ID     = "oliverguhr/fullstop-punctuation-multilang-large"
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
AUDIO_EMO_MODEL    = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


USE_FALLBACK_EMO_MODEL = False
FALLBACK_EMO_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

ENABLE_FFMPEG_SPEECH_FILTERS = True

USE_VAD = True
VAD_MIN_SIL_MS    = 700
VAD_SPEECH_PAD_MS = 250

ENABLE_CONTEXT_ANALYSIS = True
CONTEXT_BACKEND    = "ollama"
OLLAMA_MODEL       = "llama3.2:3b"
OLLAMA_TIMEOUT_SEC = 600
MAX_CONTEXT_CHARS  = 6000



def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def format_ts_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hh = ms // 3600000; ms %= 3600000
    mm = ms // 60000;   ms %= 60000
    ss = ms // 1000;    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def ensure_ffmpeg_available() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg not found.") from e

def ensure_ollama_available() -> None:
    try:
        subprocess.run(
            [r, "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    except Exception as e:
        raise RuntimeError("Ollama not found.") from e

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def moving_average(arr, window: int = 7):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    window = min(max(window, 1), len(arr))
    return np.convolve(arr, np.ones(window) / window, mode="same")

# ─── Audeering Model 
class _ModelHead(nn.Module):
    
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AudeEmoModel(Wav2Vec2PreTrainedModel):
 
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2   = Wav2Vec2Model(config)
        # ONE head, 3 outputs
        num_labels      = getattr(config, "num_labels", 3)
        self.classifier = _ModelHead(config, num_labels)
        self.init_weights()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state.mean(dim=1)
        logits  = self.classifier(hidden)          # shape (batch, 3)
        return torch.sigmoid(logits)               # values in [0, 1]


def debug_check_audio_model(model: _AudeEmoModel) -> None:
    """
    Prints classifier weights to confirm they loaded correctly.
    If weights are all near 0 → still a key mismatch.
    If weights are varied non-zero values → loaded correctly.
    """
    w = model.classifier.dense.weight
    sample = w[0, :5].detach().cpu().tolist()
    print("\n" + "="*60)
    print("[AUDIO MODEL DEBUG] classifier.dense.weight[0, :5]:")
    print(f"  {sample}")
    all_near_zero = all(abs(v) < 0.01 for v in sample)
    if all_near_zero:
        print("  all near zero")
        print(" use fall back model")
    else:
        print("weights loaded correctly")
    print("="*60 + "\n")



_CATEGORICAL_VAD = {
    "angry":    (0.15, 0.85, 0.70),   # low valence, high activation, high dominance
    "fearful":  (0.15, 0.80, 0.15),   # low valence, high activation, low dominance
    "disgust":  (0.20, 0.60, 0.55),
    "sad":      (0.20, 0.25, 0.25),   # low valence, low activation, low dominance
    "neutral":  (0.50, 0.40, 0.50),
    "calm":     (0.65, 0.25, 0.55),
    "happy":    (0.80, 0.65, 0.65),
    "surprised":(0.60, 0.75, 0.45),
}

def load_fallback_emotion_pipeline(device_id: int):
    print(f"Loading fallback emotion model: {FALLBACK_EMO_MODEL}")
    return pipeline(
        "audio-classification",
        model=FALLBACK_EMO_MODEL,
        device=device_id,
    )

def get_fallback_audio_emotion(
    chunk: np.ndarray,
    sr: int,
    fallback_pipe,
) -> dict:
    empty = {
        "audio_emotion": None, "audio_emotion_score": None,
        "audio_valence": None, "audio_activation": None,
        "audio_dominance": None, "distress_score": None,
        "distress_level": None,
    }
    if len(chunk) < int(0.25 * sr):
        return empty
    try:
        results = fallback_pipe({"array": chunk, "sampling_rate": sr}, top_k=1)
        label   = results[0]["label"].lower()
        score   = float(results[0]["score"])
        valence, activation, dominance = _CATEGORICAL_VAD.get(
            label, (0.50, 0.50, 0.50)
        )
        ds    = compute_distress_score(valence, activation)
        state = interpret_vad(valence, activation, dominance)
        return {
            "audio_emotion":       state,
            "audio_emotion_score": round(score, 3),
            "audio_valence":       round(valence,    3),
            "audio_activation":    round(activation, 3),
            "audio_dominance":     round(dominance,  3),
            "distress_score":      ds,
            "distress_level":      distress_level(ds),
        }
    except Exception as e:
        print(f"  Fallback emotion error: {e}")
        return empty

# Prosody 

def extract_prosody_features(y: np.ndarray, sr: int) -> dict:
    if y is None or len(y) < int(0.2 * sr):
        return {
            "f0_median_hz": None, "f0_mean_hz": None,
            "f0_std_hz": None,    "f0_cv": None,
            "voiced_frac": 0.0,   "rms": None, "rms_db": None,
        }
    rms    = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    rms_db = float(20.0 * np.log10(rms + 1e-12))
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                frame_length=2048, hop_length=256)
        if f0 is None:
            raise RuntimeError("pyin returned None")
        voiced_mask = ~np.isnan(f0)
        voiced_frac = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0
        f0_voiced   = f0[voiced_mask]
        if f0_voiced.size > 0:
            f0_median = float(np.median(f0_voiced))
            f0_mean   = float(np.mean(f0_voiced))
            f0_std    = float(np.std(f0_voiced))
            f0_cv     = safe_div(f0_std, f0_mean)
        else:
            f0_median = f0_mean = f0_std = f0_cv = None
    except Exception:
        f0_median = f0_mean = f0_std = f0_cv = None
        voiced_frac = 0.0
    return {
        "f0_median_hz": f0_median, "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,       "f0_cv": f0_cv,
        "voiced_frac": voiced_frac, "rms": rms, "rms_db": rms_db,
    }

def estimate_syllables(text: str) -> int:
    text = clean_text(text).lower()
    if not text:
        return 0
    groups = re.findall(r"[aeiouy]+", text)
    return max(1, len(groups)) if groups else 0

def speech_rate_features(text: str, dur_s: float) -> dict:
    dur_s = float(dur_s)
    if dur_s <= 0.0:
        return {"wpm": None, "wps": None, "syllables_per_sec": None}
    words = re.findall(r"\b[\w']+\b", text or "")
    wps   = len(words) / dur_s
    return {
        "wpm": float(wps * 60.0),
        "wps": float(wps),
        "syllables_per_sec": float(estimate_syllables(text or "") / dur_s),
    }

# ─── VAD

def compute_distress_score(valence: float, activation: float) -> float:
    return round(float((1.0 - valence) * activation * 100.0), 1)

def interpret_vad(valence: float, activation: float, dominance: float) -> str:
    if valence < 0.35 and activation > 0.70:
        return "panicked" if dominance < 0.35 else "agitated"
    if valence < 0.40 and activation > 0.55 and dominance < 0.45:
        return "fearful"
    if valence < 0.40 and activation > 0.40:
        return "distressed"
    if valence < 0.40 and activation < 0.25:
        return "emotionally flat"
    if valence < 0.40:
        return "upset"
    if valence > 0.60 and activation > 0.60:
        return "engaged/alert"
    if valence > 0.55 and activation < 0.45:
        return "calm"
    return "neutral"

def distress_level(score: float) -> str:
    if score >= 80: return "CRITICAL"
    if score >= 60: return "HIGH"
    if score >= 35: return "MODERATE"
    return "LOW"

# Plotting
def plot_emotion_metrics(segments: list[dict], out_dir: Path) -> None:
    if not segments:
        return
    plot_path = out_dir / "emotion_metrics.png"
    x = [0.5 * (seg["start"] + seg["end"]) for seg in segments]

    emotion_labels = sorted(
        {k for seg in segments for k in seg.get("text_emotion_scores", {})}
    )
    text_series = {
        label: moving_average(
            [float(seg.get("text_emotion_scores", {}).get(label, 0.0)) * 100.0
             for seg in segments], window=7,
        )
        for label in emotion_labels
    }

    vad_series = {}
    for dim in ("audio_valence", "audio_activation", "audio_dominance"):
        vals = [
            float(seg[dim]) * 100.0 if seg.get(dim) is not None else 50.0
            for seg in segments
        ]
        vad_series[dim.replace("audio_", "")] = moving_average(vals, window=7)

    distress_vals = moving_average(
        [float(seg.get("distress_score", 0.0)) for seg in segments], window=7,
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax1 = axes[0]
    for label, vals in text_series.items():
        ax1.plot(x, vals, label=label)
    ax1.set_title("Text Emotion Scores (smoothed)")
    ax1.set_ylabel("Confidence (%)")
    ax1.set_ylim(0, 100)
    ax1.axhline(50, color="gray", linewidth=0.5, linestyle="--")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    colors = {"valence": "#2196F3", "activation": "#E53935", "dominance": "#43A047"}
    for dim, vals in vad_series.items():
        ax2.plot(x, vals, label=dim, color=colors.get(dim), linewidth=1.5)
    ax2.set_title("Audio VAD Dimensions  —  Valence / Activation / Dominance (smoothed)")
    ax2.set_ylabel("Dimension score (%)")
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color="gray", linewidth=0.5, linestyle="--", label="neutral")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    ax3 = axes[2]
    ax3.fill_between(x, distress_vals, alpha=0.35, color="#E53935")
    ax3.plot(x, distress_vals, color="#B71C1C", linewidth=1.5, label="distress score")
    ax3.axhline(60, color="orange", linewidth=1.0, linestyle="--", label="HIGH threshold (60)")
    ax3.axhline(80, color="red",    linewidth=1.0, linestyle="--", label="CRITICAL threshold (80)")
    ax3.set_title("Caller Distress Score  (0 = calm, 100 = maximum distress)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Distress score")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("PLOT:", plot_path.resolve())

#  LLM context analysis

def build_full_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        text = clean_text(seg.get("text_punct") or seg.get("text") or "")
        if text:
            lines.append(f"[{seg['start']:.1f}s] {text}")
    return "\n".join(lines)

def truncate_for_llm(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = clean_text(text)
    return text if len(text) <= max_chars else text[:max_chars]

def extract_overall_signal_summary(segments: list[dict]) -> dict:
    if not segments:
        return {}
    text_emotions:  dict = {}
    audio_states:   dict = {}
    speech_rates, rms_vals, voiced_vals, f0_vals = [], [], [], []
    valence_vals, activation_vals, dominance_vals, distress_scores = [], [], [], []
    peak_distress_seg   = None
    peak_distress_score = 0.0

    for seg in segments:
        if te := seg.get("text_emotion"):
            text_emotions[te] = text_emotions.get(te, 0) + 1
        if ae := seg.get("audio_emotion"):
            audio_states[ae] = audio_states.get(ae, 0) + 1
        rate = seg.get("speech_rate", {})
        if (wpm := rate.get("wpm")) is not None:
            speech_rates.append(float(wpm))
        pros = seg.get("prosody", {})
        if (v := pros.get("rms_db"))       is not None: rms_vals.append(float(v))
        if (v := pros.get("voiced_frac"))  is not None: voiced_vals.append(float(v))
        if (v := pros.get("f0_median_hz")) is not None: f0_vals.append(float(v))
        if (v := seg.get("audio_valence"))    is not None: valence_vals.append(float(v))
        if (v := seg.get("audio_activation")) is not None: activation_vals.append(float(v))
        if (v := seg.get("audio_dominance"))  is not None: dominance_vals.append(float(v))
        ds = seg.get("distress_score")
        if ds is not None:
            distress_scores.append(float(ds))
            if float(ds) > peak_distress_score:
                peak_distress_score = float(ds)
                peak_distress_seg   = seg

    def top_items(d: dict, k: int = 3):
        return [{"label": k1, "count": v1}
                for k1, v1 in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]
    avg = lambda lst: round(float(np.mean(lst)), 3) if lst else None
    avg_distress  = avg(distress_scores)
    peak_time_str = (
        f"{peak_distress_seg['start']:.1f}s–{peak_distress_seg['end']:.1f}s"
        if peak_distress_seg else "N/A"
    )
    high_count     = sum(1 for s in distress_scores if s >= 60)
    critical_count = sum(1 for s in distress_scores if s >= 80)

    return {
        "dominant_caller_states":      top_items(audio_states),
        "dominant_text_emotions":      top_items(text_emotions),
        "avg_distress_score":          avg_distress,
        "peak_distress_score":         round(peak_distress_score, 1),
        "peak_distress_time":          peak_time_str,
        "overall_distress_level":      distress_level(avg_distress or 0),
        "segments_high_distress":      high_count,
        "segments_critical_distress":  critical_count,
        "avg_valence":                 avg(valence_vals),
        "avg_activation":              avg(activation_vals),
        "avg_dominance":               avg(dominance_vals),
        "avg_speech_rate_wpm":         avg(speech_rates),
        "avg_pitch_median_hz":         avg(f0_vals),
        "avg_voiced_fraction":         avg(voiced_vals),
        "avg_intensity_rms_db":        avg(rms_vals),
        "segment_count":               len(segments),
        "total_duration_sec":          float(segments[-1]["end"] - segments[0]["start"]) if segments else 0.0,
    }

def try_parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {"raw_response": ""}
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    if m := re.search(r"\{.*\}", raw, flags=re.DOTALL):
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"raw_response": raw}

def call_ollama_json(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout_sec: int = OLLAMA_TIMEOUT_SEC,
) -> dict:
    ensure_ollama_available()
    proc = subprocess.run(
        [r"C:\Users\shrey\AppData\Local\Programs\Ollama\ollama.exe", "run", model],
        input=prompt, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout_sec, check=True,
    )
    return try_parse_json(proc.stdout.strip())

def analyze_call_context_with_ollama(segments: list[dict]) -> dict:
    transcript = truncate_for_llm(build_full_transcript(segments), MAX_CONTEXT_CHARS)
    signals    = extract_overall_signal_summary(segments)

    prompt = f"""You are analyzing a police or 911 call recording transcript.
Your job is to extract key information and return ONLY a valid JSON object.
Do not write anything outside the JSON. Do not use markdown. Just return the JSON.

Use this exact JSON structure (fill every field based on the transcript):
{{
  "call_summary": "1-2 sentences describing what happened",
  "incident_type": "one of: traffic emergency, medical emergency, domestic dispute, robbery, assault, noise complaint, welfare check, other",
  "caller_profile": "brief description of who is calling and how they seem",
  "caller_emotional_state": "describe how the caller sounds throughout the call",
  "distress_assessment": "assess caller distress level and what is causing it",
  "key_facts": ["fact 1", "fact 2", "fact 3"],
  "location_mentioned": "any address, road, or landmark mentioned, or unknown",
  "persons_involved": ["person description 1", "person description 2"],
  "vehicles_mentioned": ["vehicle description or none"],
  "weapons_mentioned": ["weapon mentioned or none"],
  "injuries_reported": ["injury description or none"],
  "threat_indicators": ["ongoing threat description or none"],
  "urgency_level": "one of: LOW, MODERATE, HIGH, CRITICAL",
  "recommended_response": "what police unit or response is needed",
  "action_items": ["action 1", "action 2"],
  "resolution_status": "one of: resolved, unresolved, partially resolved, unknown",
  "risk_flags": ["officer safety note or case follow-up note"],
  "confidence": 0.8
}}

Rules:
- confidence must be a number between 0 and 1, not a string
- If something is not mentioned, use "none" or "unknown" as a string value
- For list fields with nothing to report, use ["none"]
- urgency_level must be exactly one of: LOW, MODERATE, HIGH, CRITICAL

Signal summary:
avg_distress_score: {signals.get('avg_distress_score')}
overall_distress_level: {signals.get('overall_distress_level')}
dominant_caller_states: {signals.get('dominant_caller_states')}
avg_speech_rate_wpm: {signals.get('avg_speech_rate_wpm')}

Transcript:
{transcript}

Return ONLY the JSON object now:"""

    result = call_ollama_json(prompt)
    result["signal_summary"] = signals
    return result

def write_context_outputs(context_result: dict, out_dir: Path) -> None:
    context_json_path = out_dir / "call_context.json"
    context_txt_path  = out_dir / "call_context.txt"
    with open(context_json_path, "w", encoding="utf-8") as f:
        json.dump(context_result, f, ensure_ascii=False, indent=2)
    with open(context_txt_path, "w", encoding="utf-8") as f:
        f.write("LAW ENFORCEMENT CALL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        for key, value in context_result.items():
            if isinstance(value, (dict, list)):
                f.write(f"{key}:\n{json.dumps(value, ensure_ascii=False, indent=2)}\n\n")
            else:
                f.write(f"{key}: {value}\n\n")
    print("CONTEXT JSON:", context_json_path.resolve())
    print("CONTEXT TXT :", context_txt_path.resolve())

# Main

def process_one(audio_in: Path, out_dir: Path) -> None:
    if not audio_in.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_in}")
    ensure_ffmpeg_available()
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path  = out_dir / "input_16k_mono.wav"
    txt_path  = out_dir / "transcript.txt"
    srt_path  = out_dir / "transcript.srt"
    json_path = out_dir / "transcript.json"

    print("Processing:", audio_in.name)
    print("Output dir:", out_dir.resolve())
    print("Python:", sys.version.split()[0])

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = COMPUTE_TYPE_GPU if device == "cuda" else COMPUTE_TYPE_CPU
    pipe_device  = 0 if device == "cuda" else -1
    print("Device:", device, "| compute_type:", compute_type)

    # 1) Convert to 16 kHz mono WAV
    print("Converting to 16 kHz mono WAV...")
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(audio_in), "-ac", "1", "-ar", "16000", "-vn"]
    if ENABLE_FFMPEG_SPEECH_FILTERS:
        ffmpeg_cmd += ["-af", "highpass=f=80,aresample=16000,loudnorm"]
    ffmpeg_cmd.append(str(wav_path))
    run(ffmpeg_cmd)
    print("WAV ready:", wav_path)

    # 2) Transcription
    print(f"Loading faster-whisper model: {MODEL_SIZE} ...")
    asr = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
    print("Transcribing...")
    transcribe_kwargs = dict(
        audio=str(wav_path),
        language=LANGUAGE if LANGUAGE else None,
        beam_size=BEAM_SIZE, best_of=1, temperature=0.0,
        word_timestamps=False, condition_on_previous_text=False,
        no_speech_threshold=0.35, log_prob_threshold=-1.0,
    )
    if USE_VAD:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=VAD_MIN_SIL_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )
    else:
        transcribe_kwargs["vad_filter"] = False
    segments_gen, info = asr.transcribe(**transcribe_kwargs)
    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text}
                for s in segments_gen]
    print("Detected language:", info.language, "| segments:", len(segments))

    # 3) Punctuation restoration
    print("Loading punctuation model:", PUNCT_MODEL_ID)
    punct_tokenizer = AutoTokenizer.from_pretrained(PUNCT_MODEL_ID)
    punct_model     = AutoModelForTokenClassification.from_pretrained(PUNCT_MODEL_ID)
    punct_pipe      = pipeline(
        task="token-classification", model=punct_model,
        tokenizer=punct_tokenizer, aggregation_strategy="none", device=pipe_device,
    )
    punct_map = {"COMMA": ",", "PERIOD": ".", "QUESTION": "?",
                 "EXCLAMATION": "!", "COLON": ":", "SEMICOLON": ";"}

    def normalize_label(label: str) -> str:
        if label.startswith("LABEL_"):
            try:
                return punct_model.config.id2label.get(int(label.split("_")[1]), label)
            except Exception:
                return label
        return label

    def restore_punctuation(text: str) -> str:
        text = clean_text(text)
        if not text:
            return ""
        preds = punct_pipe(text)
        insertions: dict[int, str] = {}
        for p in preds:
            end = p.get("end")
            if not end or end <= 0 or end > len(text):
                continue
            label      = normalize_label(p.get("entity", "O"))
            punct_char = punct_map.get(label)
            if not punct_char:
                continue
            if end == len(text) or text[end].isspace():
                if end > 0 and text[end - 1] in ".,?!:;":
                    continue
                insertions[end] = punct_char
        out = text
        for pos in sorted(insertions.keys(), reverse=True):
            out = out[:pos] + insertions[pos] + out[pos:]
        return out

    print("Restoring punctuation...")
    for seg in tqdm(segments):
        seg["text_punct"] = restore_punctuation(seg["text"])

    # 4) Text emotion
    print("Loading text emotion model:", TEXT_EMOTION_MODEL)
    emo_tok   = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL)
    emo_model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL).eval()
    if device == "cuda":
        emo_model = emo_model.to("cuda")
    id2label = emo_model.config.id2label

    @torch.no_grad()
    def get_text_emotions(text: str) -> dict:
        text = clean_text(text)
        if not text:
            return {"top_emotion": None, "top_score": 0.0, "scores": {}}
        inputs = emo_tok(text[:900], return_tensors="pt", truncation=True)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        probs = torch.softmax(emo_model(**inputs).logits, dim=-1).squeeze(0)
        top_i = int(torch.argmax(probs).item())
        return {
            "top_emotion": id2label[top_i],
            "top_score":   float(probs[top_i].detach().cpu()),
            "scores":      {id2label[i]: float(probs[i].detach().cpu())
                            for i in range(probs.numel())},
        }

    print("Scoring text emotions...")
    for seg in tqdm(segments):
        emo = get_text_emotions(seg["text_punct"])
        seg["text_emotion"]        = emo["top_emotion"]
        seg["text_emotion_score"]  = emo["top_score"]
        seg["text_emotion_scores"] = emo["scores"]

    # Load WAV
    audio, sr = sf.read(str(wav_path))
    if sr != 16000:
        raise RuntimeError(f"Expected 16 kHz wav, got {sr}")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    def slice_audio(start_s: float, end_s: float) -> np.ndarray:
        s = max(0, int(start_s * sr))
        e = min(len(audio), int(end_s * sr))
        return audio[s:e].astype(np.float32)

    # 5) Audio emotion
    if USE_FALLBACK_EMO_MODEL:
       
        fallback_pipe = load_fallback_emotion_pipeline(pipe_device)

        print("Scoring audio emotions ")
        for seg in tqdm(segments):
            chunk = slice_audio(seg["start"],
                                min(seg["end"], seg["start"] + MAX_AUDIO_EMO_SECONDS))
            seg.update(get_fallback_audio_emotion(chunk, sr, fallback_pipe))

    else:
        #  audeering VAD model 
        print("Loading audio emotion model:", AUDIO_EMO_MODEL)
        audio_emo_processor = Wav2Vec2Processor.from_pretrained(AUDIO_EMO_MODEL)
        audio_emo_model     = _AudeEmoModel.from_pretrained(AUDIO_EMO_MODEL).eval()
        if device == "cuda":
            audio_emo_model = audio_emo_model.to("cuda")

        # Print debug info — confirms weights loaded vs still random
        debug_check_audio_model(audio_emo_model)

        @torch.no_grad()
        def get_audio_emotion(start_s: float, end_s: float) -> dict:
            empty = {
                "audio_emotion": None, "audio_emotion_score": None,
                "audio_valence": None, "audio_activation": None,
                "audio_dominance": None, "distress_score": None,
                "distress_level": None,
            }
            dur = float(end_s - start_s)
            if dur <= 0.25:
                return empty
            if dur > MAX_AUDIO_EMO_SECONDS:
                end_s = start_s + MAX_AUDIO_EMO_SECONDS
            chunk = slice_audio(start_s, end_s)
            if len(chunk) < int(0.25 * sr):
                return empty
            try:
                inputs = audio_emo_processor(
                    chunk, sampling_rate=16000, return_tensors="pt", padding=True,
                )
                if device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                out  = audio_emo_model(
                    inputs["input_values"],
                    inputs.get("attention_mask"),
                )
                vals = out.squeeze(0).cpu().tolist()

                # Audeering MSP-Podcast output order: [arousal, dominance, valence]
                arousal, dominance, valence = vals[0], vals[1], vals[2]

                ds    = compute_distress_score(valence, arousal)
                label = interpret_vad(valence, arousal, dominance)

                return {
                    "audio_emotion":       label,
                    "audio_emotion_score": round(ds / 100.0, 3),
                    "audio_valence":       round(valence,   3),
                    "audio_activation":    round(arousal,   3),
                    "audio_dominance":     round(dominance, 3),
                    "distress_score":      ds,
                    "distress_level":      distress_level(ds),
                }
            except Exception as e:
                print(f"  Audio emotion error: {e}")
                return empty

        print("Scoring audio emotions and distress...")
        for seg in tqdm(segments):
            seg.update(get_audio_emotion(seg["start"], seg["end"]))

    # 6) Prosody + speech rate
    print("Extracting prosody + speech rate...")
    for seg in tqdm(segments):
        chunk = slice_audio(seg["start"], seg["end"])
        seg["prosody"]     = extract_prosody_features(chunk, sr)
        seg["speech_rate"] = speech_rate_features(
            seg.get("text_punct", seg.get("text", "")),
            float(seg["end"] - seg["start"]),
        )

    # 7) Plot
    print("Plotting emotion metrics...")
    plot_emotion_metrics(segments, out_dir)

    # 8) Context analysis
    if ENABLE_CONTEXT_ANALYSIS and CONTEXT_BACKEND == "ollama":
        print("Analyzing call context with Ollama...")
        try:
            context_result = analyze_call_context_with_ollama(segments)
            write_context_outputs(context_result, out_dir)
        except Exception as e:
            print("Context analysis failed:", repr(e))

    # 9) Write transcript outputs
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            te   = seg.get("text_emotion")
            tes  = float(seg.get("text_emotion_score", 0.0) or 0.0)
            ae   = seg.get("audio_emotion")
            ds   = seg.get("distress_score")
            dl   = seg.get("distress_level")
            val  = seg.get("audio_valence")
            act  = seg.get("audio_activation")
            dom  = seg.get("audio_dominance")
            pros = seg.get("prosody", {})
            rate = seg.get("speech_rate", {})
            ds_str  = "None" if ds  is None else f"{ds:.1f}"
            val_str = "None" if val is None else f"{val:.3f}"
            act_str = "None" if act is None else f"{act:.3f}"
            dom_str = "None" if dom is None else f"{dom:.3f}"
            f.write(
                f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg.get('text_punct', seg['text'])}\n"
                f"  text_emotion={te} ({tes:.3f})"
                f" | caller_state={ae}"
                f" | distress={ds_str}/100 ({dl})\n"
                f"  valence={val_str} activation={act_str} dominance={dom_str}\n"
                f"  pitch_hz(med/mean/std)="
                f"{pros.get('f0_median_hz')}/{pros.get('f0_mean_hz')}/{pros.get('f0_std_hz')}"
                f" voiced_frac={pros.get('voiced_frac')}"
                f" f0cv={pros.get('f0_cv')}"
                f" rms_db={pros.get('rms_db')}"
                f" wpm={rate.get('wpm')}\n\n"
            )

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_ts_srt(seg['start'])} --> {format_ts_srt(seg['end'])}\n")
            f.write(f"{seg.get('text_punct', seg['text'])}\n\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"language": info.language, "segments": segments},
                  f, ensure_ascii=False, indent=2)

    print("DONE")
    print("TXT :", txt_path.resolve())
    print("SRT :", srt_path.resolve())
    print("JSON:", json_path.resolve())


def main() -> None:
    if not CALLS_DIR.exists():
        raise FileNotFoundError(f"Calls folder not found: {CALLS_DIR}")
    exts  = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = [p for p in sorted(CALLS_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in exts][:MAX_FILES]
    if not files:
        raise FileNotFoundError(f"No audio files found in: {CALLS_DIR}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Found", len(files), "audio files in", CALLS_DIR)
    print("Output root:", OUT_ROOT.resolve())
    for audio_in in files:
        out_dir = OUT_ROOT / audio_in.stem
        try:
            process_one(audio_in, out_dir)
        except Exception as e:
            print("\nFAILED:", audio_in.name)
            print("Reason:", repr(e))


if __name__ == "__main__":
    main() 
