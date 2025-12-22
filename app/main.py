
from fastapi import FastAPI
from app.core.logging.logger import setup_logging
from .routers import rag
from .settings.config import AppSettings
from app.core.rag_pipeline import RAGPipeline
from contextlib import asynccontextmanager





@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------- Startup --------
    logger = setup_logging() # Initialize logging
    app.state.logger = logger
    settings = AppSettings(logger) #validate settings
    rag_pipeline = RAGPipeline(settings)
    app.state.settings = settings
    app.state.rag_pipeline = rag_pipeline
    print("🚀 Application startup complete")
    yield
     # -------- Shutdown --------
    print("🛑 Application shutdown complete")


app = FastAPI(lifespan=lifespan)
app.include_router(rag.router)


@app.get("/")
async def root():
    return {"message": "PaperMind is running!"}
