"""
Injury Risk Forecaster — FastAPI Backend v2.2
Render-ready: auto-trains model on first boot if outputs missing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging, time, os, subprocess, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Resolve all paths relative to this file (api/main.py) ───────────────────
API_DIR  = os.path.dirname(os.path.abspath(__file__))   # /repo/api
ROOT_DIR = os.path.dirname(API_DIR)                      # /repo
OUTPUTS  = os.path.join(API_DIR, "outputs")             # /repo/api/outputs


def ensure_model():
    model_path = os.path.join(OUTPUTS, "injury_model.joblib")
    if os.path.exists(model_path):
        logger.info("Model artifacts found — skipping training.")
        return

    logger.info("No model found — training now (~30s) ...")
    os.makedirs(OUTPUTS, exist_ok=True)

    # train_model.py lives in repo root, next to api/
    train_script = os.path.join(ROOT_DIR, "train_model.py")
    if not os.path.exists(train_script):
        raise RuntimeError(f"train_model.py not found at {train_script}")

    result = subprocess.run(
        [sys.executable, train_script],
        capture_output=True, text=True,
        cwd=ROOT_DIR,   # run from repo root so it can find data_generator etc.
        env={**os.environ, "OUTPUTS_DIR": OUTPUTS},  # pass output path
    )
    logger.info(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        logger.error(result.stderr[-2000:])
        raise RuntimeError("Model training failed — check logs above.")
    logger.info("Model trained and saved successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model()

    # Override ARTIFACTS path in AppState before loading
    from models.state import AppState
    from pathlib import Path
    AppState._artifacts_path = Path(OUTPUTS)

    logger.info("Loading ML model artifacts ...")
    AppState.load()
    logger.info(f"Model ready: {AppState.meta['model_type']} (AUC={AppState.meta['test_auc']})")

    from services.user_store import seed_admin
    seed_admin()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


from routes.predict       import router as predict_router
from routes.athlete       import router as athlete_router
from routes.history       import router as history_router
from routes.auth          import router as auth_router
from routes.admin         import router as admin_router
from routes.report        import router as report_router
from routes.compare       import router as compare_router
from routes.notifications import router as notifications_router

app = FastAPI(title="Injury Risk Forecaster API", version="2.2.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def timing(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter()-t0)*1000, 1))
    return response

app.include_router(auth_router,          prefix="/api/v1")
app.include_router(admin_router,         prefix="/api/v1")
app.include_router(report_router,        prefix="/api/v1")
app.include_router(compare_router,       prefix="/api/v1")
app.include_router(notifications_router, prefix="/api/v1")
app.include_router(predict_router,       prefix="/api/v1", tags=["Prediction"])
app.include_router(athlete_router,       prefix="/api/v1", tags=["Athlete"])
app.include_router(history_router,       prefix="/api/v1", tags=["History"])

@app.get("/", include_in_schema=False)
async def root(): return {"message": "Injury Risk Forecaster API v2.2", "docs": "/docs"}

@app.get("/health", tags=["System"])
async def health():
    from models.state import AppState
    return {"status": "healthy", "model": AppState.meta.get("model_type"),
            "auc": AppState.meta.get("test_auc"), "version": "2.2.0"}

@app.get("/api/v1/model-info", tags=["System"])
async def model_info():
    from models.state import AppState
    return AppState.meta