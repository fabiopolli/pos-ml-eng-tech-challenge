from fastapi import FastAPI
from app.routes import health, predict
from app.middleware.latency import add_latency_header

app = FastAPI()

# Middleware
app.middleware("http")(add_latency_header)

# Rotas
app.include_router(health.router)
app.include_router(predict.router)