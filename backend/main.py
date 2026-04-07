import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import librosa
import soundfile as sf
import numpy as np

app = FastAPI(title="Studio Tools API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.gettempdir()


def quantize_audio(
    y: np.ndarray,
    sr: int,
    bpm: float,
    quantize_strength: float,
    swing: float,
    onset_sensitivity: float,
) -> np.ndarray:
    beat_duration = 60.0 / bpm
    subdivision = beat_duration / 4

    delta = 0.07 + (1.0 - onset_sensitivity) * 0.15
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="samples",
        delta=delta,
        wait=int(sr * 0.05),
    )

    if len(onset_frames) == 0:
        return y

    def grid_position_with_swing(t):
        beat_idx = t / beat_duration
        beat_int = int(beat_idx)
        beat_frac = beat_idx - beat_int
        subdivision_idx = int(beat_frac / 0.25)
        if subdivision_idx % 2 == 1:
            swing_offset = swing * subdivision
        else:
            swing_offset = 0.0
        nearest = round(beat_frac / 0.25) * 0.25
        nearest_time = (beat_int + nearest) * beat_duration + swing_offset
        return nearest_time

    y_out = np.zeros_like(y)
    onset_times = librosa.samples_to_time(onset_frames, sr=sr)

    for i, (onset_sample, onset_time) in enumerate(zip(onset_frames, onset_times)):
        next_onset_sample = onset_frames[i + 1] if i + 1 < len(onset_frames) else len(y)

        grid_time = grid_position_with_swing(onset_time)
        quantized_time = onset_time + quantize_strength * (grid_time - onset_time)
        quantized_sample = int(quantized_time * sr)
        quantized_sample = max(0, min(quantized_sample, len(y) - 1))

        segment = y[onset_sample:next_onset_sample]
        seg_len = len(segment)
        end_pos = min(quantized_sample + seg_len, len(y_out))
        actual_len = end_pos - quantized_sample
        if actual_len > 0:
            y_out[quantized_sample:end_pos] += segment[:actual_len]

    max_val = np.max(np.abs(y_out))
    if max_val > 0:
        y_out = y_out / max_val * np.max(np.abs(y)) * 0.99

    return y_out


@app.post("/quantize-percussion")
async def quantize_percussion(
    file: UploadFile = File(...),
    bpm: float = Form(120.0),
    quantize_strength: float = Form(0.8),
    swing: float = Form(0.0),
    onset_sensitivity: float = Form(0.5),
    auto_bpm: bool = Form(False),
):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .wav")

    job_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{job_id}_input.wav")
    output_path = os.path.join(TEMP_DIR, f"{job_id}_output.wav")

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        y, sr = librosa.load(input_path, sr=None, mono=False)
        is_stereo = y.ndim == 2

        if auto_bpm:
            if is_stereo:
                tempo, _ = librosa.beat.beat_track(y=librosa.to_mono(y), sr=sr)
            else:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)

        if is_stereo:
            left = quantize_audio(y[0], sr, bpm, quantize_strength, swing, onset_sensitivity)
            right = quantize_audio(y[1], sr, bpm, quantize_strength, swing, onset_sensitivity)
            y_out = np.stack([left, right])
        else:
            y_out = quantize_audio(y, sr, bpm, quantize_strength, swing, onset_sensitivity)

        sf.write(output_path, y_out.T if is_stereo else y_out, sr, subtype="PCM_24")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"quantized_{file.filename}",
            headers={"X-Detected-BPM": str(round(bpm, 1))},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.get("/detect-bpm")
async def detect_bpm_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    job_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{job_id}_bpm.wav")
    with open(input_path, "wb") as f:
        f.write(content)
    y, sr = librosa.load(input_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    os.remove(input_path)
    return {"bpm": round(float(tempo), 1)}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Studio Tools API"}
  
