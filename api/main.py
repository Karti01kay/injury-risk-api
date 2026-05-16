"""
Injury Risk Forecaster — FastAPI Backend v2.2
Model artifacts are committed to the repo — no training on startup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging, time, os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

API_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS  = os.path.join(API_DIR, "outputs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from models.state import AppState
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