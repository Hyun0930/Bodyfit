"""BodyFit FastAPI 서버.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000

환경변수:
    BODYFIT_CKPT  : CVAE 체크포인트 경로 (기본: checkpoints/cvae/best.pt)
    OPENAI_API_KEY: GPT-4o 피드백용
"""
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(os.environ.get("BODYFIT_DATA", ".")) / ".." / ".env", override=False)
load_dotenv(".env", override=False)

app = FastAPI(title="BodyFit")
app.mount("/static", StaticFiles(directory="static"), name="static")

CKPT = Path(os.environ.get("BODYFIT_CKPT", "checkpoints/cvae/best.pt"))
EXERCISES = ["squat", "bench", "deadlift", "ohp"]
EXERCISE_KO = {"squat": "스쿼트", "bench": "벤치프레스", "deadlift": "데드리프트", "ohp": "오버헤드프레스"}

_engines: dict = {}


def get_engine(exercise: str):
    if exercise not in _engines:
        from src.inference import BodyFitInference
        _engines[exercise] = BodyFitInference(exercise=exercise, ckpt_path=CKPT)
    return _engines[exercise]


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    exercise: str = Form("squat"),
):
    if exercise not in EXERCISES:
        return JSONResponse({"error": f"exercise must be one of {EXERCISES}"}, status_code=400)

    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        engine = get_engine(exercise)
        results = engine.run(tmp_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)

    # heatmap numpy → list (JSON 직렬화)
    for r in results:
        r["heatmap"] = r["heatmap"].tolist()

    return {
        "exercise": exercise,
        "exercise_ko": EXERCISE_KO[exercise],
        "total_reps": len(results),
        "reps": results,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
