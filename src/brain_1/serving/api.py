from __future__ import annotations

from fastapi import FastAPI

from brain_1.serving.inference import run_inference


def build_app() -> FastAPI:
    app = FastAPI(
        title="brain-1 API",
        summary="Commercial clean-room brain-encoding scaffold",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/predict")
    async def predict() -> dict[str, object]:
        result = run_inference()
        return {
            "shape": list(result.shape),
            "description": result.description,
        }

    return app
