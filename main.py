import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gemini_live import GeminiLive

load_dotenv()
logging.basicConfig(level=logging.INFO)

MODEL = "gemini-3.1-flash-live-preview"

OPENING = (
    "Please start the interview now by saying this opening line exactly: "
    "'Hey Alex — appreciate you taking the time. I'm Alex, Staff Engineer here. "
    "I've gone through your resume, and I'd like to discuss your work — but first, "
    "give me a quick overview of your background.'"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def interview_ws(ws: WebSocket):
    await ws.accept()
    logging.info("WebSocket connected")

    audio_queue: asyncio.Queue = asyncio.Queue()
    video_queue: asyncio.Queue = asyncio.Queue()
    text_queue: asyncio.Queue = asyncio.Queue()

    gemini = GeminiLive(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model=MODEL,
        input_sample_rate=16000,
    )

    async def audio_cb(data: bytes) -> None:
        try:
            await ws.send_bytes(data)
        except Exception:
            pass

    async def interrupt_cb() -> None:
        try:
            await ws.send_text(json.dumps({"type": "interrupted"}))
        except Exception:
            pass

    async def recv_from_browser() -> None:
        try:
            while True:
                msg = await ws.receive()
                if "bytes" in msg:
                    await audio_queue.put(msg["bytes"])
                elif "text" in msg:
                    try:
                        parsed = json.loads(msg["text"])
                        if parsed.get("type") == "text":
                            await text_queue.put(parsed["text"])
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logging.error(f"recv_from_browser error: {e}")

    async def run_gemini() -> None:
        # Queue opening instruction so Alex greets immediately on connect
        await text_queue.put(OPENING)
        try:
            async for event in gemini.start_session(
                audio_queue, video_queue, text_queue, audio_cb, interrupt_cb
            ):
                try:
                    await ws.send_text(json.dumps(event))
                except Exception:
                    break
        except Exception as e:
            logging.error(f"run_gemini error: {e}")

    t1 = asyncio.create_task(recv_from_browser())
    t2 = asyncio.create_task(run_gemini())
    await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
    t1.cancel()
    t2.cancel()
    await asyncio.gather(t1, t2, return_exceptions=True)
    logging.info("WebSocket session ended")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7000)
