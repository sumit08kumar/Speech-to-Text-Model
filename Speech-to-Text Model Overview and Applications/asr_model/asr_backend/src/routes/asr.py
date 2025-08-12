import os
import io
import base64
import logging
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import numpy as np
import librosa

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_asr_pipeline = None

def get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is None:
        from transformers import pipeline
        model_name = os.getenv("ASR_MODEL", "openai/whisper-tiny")  # try "openai/whisper-tiny.en" for English-only
        log.info(f"Loading ASR model: {model_name}")
        _asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=-1  # CPU
        )
    return _asr_pipeline

asr_bp = Blueprint("asr", __name__)

@asr_bp.route("/health", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify({"status": "ok"}), 200

def _read_audio_bytes_from_request():
    # 1) multipart/form-data
    for key in ("file", "audio", "upload"):
        if key in request.files:
            data = request.files[key].read()
            if data:
                return data
    # 2) raw audio body
    ctype = request.headers.get("Content-Type", "")
    if ctype.startswith("audio/") and request.data:
        return request.data
    # 3) JSON base64
    if request.is_json:
        j = request.get_json(silent=True) or {}
        b64 = j.get("audio_base64")
        if b64:
            try:
                return base64.b64decode(b64, validate=True)
            except Exception:
                pass
    return None

@asr_bp.route("/transcribe", methods=["POST"])
@cross_origin()
def transcribe_audio():
    data = _read_audio_bytes_from_request()
    if not data:
        return jsonify({
            "error": "No audio provided. Use form-data field 'file', raw audio (Content-Type: audio/*), or JSON 'audio_base64'."
        }), 400

    # Decode to mono 16k
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
    except Exception as e:
        log.exception("Failed to read audio")
        return jsonify({"error": f"Failed to read audio: {e}"}), 400

    try:
        asr = get_asr_pipeline()
    except Exception as e:
        log.exception("Failed to load ASR pipeline")
        return jsonify({"error": f"Failed to load ASR model: {e}"}), 500

    try:
        result = asr({"array": np.asarray(y), "sampling_rate": 16000})
        text = result["text"] if isinstance(result, dict) and "text" in result else str(result)
        return jsonify({"transcription": text}), 200
    except Exception as e:
        log.exception("ASR inference failed")
        return jsonify({"error": f"ASR inference failed: {e}"}), 500

