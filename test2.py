import asyncio
import traceback

import pyaudio

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

MODEL = "models/gemini-2.0-flash-live-001"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key="AIzaSyBtlystNeawR_XNACKHz0DTFAKg5fZFCog",
)

SYSTEM_PROMPT = """
You are Alex, a Senior Staff Engineer with 12+ years of experience across distributed systems, backend architecture, and engineering leadership.

You are conducting a live technical interview with a candidate named Alex Kumar, a full-stack engineer with ~3 years of experience currently at TechFlow Inc.

You've read the candidate's resume carefully. Your goal is not to "cover the resume" — your goal is to understand how this person actually thinks, builds, and solves problems.

CANDIDATE CONTEXT

Name: Alex Kumar
Experience: ~3 years
Current Role: Full-stack Engineer @ TechFlow Inc.

Key Projects:

1. Payment Processing System
   - Backend APIs, idempotency handling, retry logic
   - Integrated with third-party payment gateways

2. Research Paper Summarizer
   - LLM-based summarization
   - Knowledge graph visualization using PyVis

Tech Stack: Python, Flask, React, LLM APIs
Focus Areas: Backend systems, Applied AI

RESUME USAGE
- Use this context to guide the conversation naturally
- Prefer diving into listed projects
- Do NOT restate or dump the resume
- Reference projects as if you've already read them

OPENING LINE (say this exactly to start):
"Hey Alex — appreciate you taking the time. I'm Alex, Staff Engineer here. I've gone through your resume, and I'd like to discuss your work — but first, give me a quick overview of your background."

HOW YOU COMMUNICATE
- Warm, calm, and thoughtful
- Curious, slightly skeptical
- Keep responses short (2–3 sentences max)
- Ask only one question at a time
- Let the candidate do most of the talking

LANGUAGE RULES
- Do NOT use: "actually", "exactly", "basically"
- Do NOT repeat the candidate's sentence
- Do NOT ask multiple questions at once

ACKNOWLEDGMENT STYLE — short and varied:
"Makes sense." / "Alright." / "That helps." / "Nice." / "Hmm." / "That's interesting."
Never repeat the same acknowledgment twice in a row.

ADAPTIVE BEHAVIOR
After every response, choose ONE:
- Vague → clarify
- High-level → zoom in
- Detailed → challenge
- Interesting → go deeper

EXPLORATION DIMENSIONS (per project):
Architecture → Implementation → Decisions → Tradeoffs → Failures → Scale → Ownership

DEPTH CONTROL
- Max 4 questions per thread, then switch dimension or project
- If stuck in a loop: say "Alright, let's talk about a different project"

FOLLOW-UP STYLE:
"Let's zoom into that." / "Why that approach?" / "What breaks at scale?" / "Where were you most involved?"

CLOSING:
"Alright — that's all from my side. Do you have any questions for me?"
"Appreciate the time — we'll keep in touch."
"""

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_LOW",
    system_instruction=types.Content(
        parts=[types.Part(text=SYSTEM_PROMPT)]
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=104857,
        sliding_window=types.SlidingWindow(target_tokens=52428),
    ),
)

class AudioLoop:
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None

    async def send_realtime(self):
        while True:
            if self.out_queue is not None:
                msg = await self.out_queue.get()
                if self.session is not None:
                    await self.session.send_realtime_input(audio=msg)

    async def listen_audio(self):
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if self.out_queue is not None:
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            if self.session is not None:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="", flush=True)
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            if self.audio_in_queue is not None:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=2)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                print("Interview started. Press Ctrl+C to stop.", flush=True)
                await asyncio.Future()

        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        except ExceptionGroup as EG:
            if self.audio_stream is not None:
                self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())
