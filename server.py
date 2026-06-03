"""
Run: .venv/Scripts/python.exe server.py
Open: http://localhost:3001
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from test2 import AudioLoop

HTML = os.path.join(os.path.dirname(__file__), "index.html")

interview_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    global interview_task
    if interview_task and not interview_task.done():
        interview_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(interview_task), timeout=3)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StatusResponse(BaseModel):
    running: bool


class ActionResponse(BaseModel):
    status: str


@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse(HTML, media_type="text/html")


@app.get("/status", response_model=StatusResponse)
def get_status():
    running = interview_task is not None and not interview_task.done()
    return StatusResponse(running=running)


@app.post("/start", response_model=ActionResponse)
async def start_interview():
    global interview_task
    if interview_task and not interview_task.done():
        return ActionResponse(status="already_running")
    loop = AudioLoop()
    interview_task = asyncio.create_task(loop.run())
    return ActionResponse(status="started")


@app.post("/stop", response_model=ActionResponse)
async def stop_interview():
    global interview_task
    if not interview_task or interview_task.done():
        return ActionResponse(status="not_running")
    interview_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(interview_task), timeout=3)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    interview_task = None
    return ActionResponse(status="stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=3001)
