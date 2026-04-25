import time
from fastapi import Request

async def add_latency_header(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    print(f"[LATENCY] {request.url.path} - {process_time:.4f}s")

    return response