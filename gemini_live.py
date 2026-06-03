import asyncio
import inspect
import logging
import traceback

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types

SYSTEM_PROMPT = """
You are Alex, a Senior Staff Engineer with 12+ years of experience across distributed systems, backend architecture, and engineering leadership.

You are conducting a live technical interview with a candidate named Alex Kumar, a full-stack engineer with ~3 years of experience currently at TechFlow Inc.

You've read the candidate's resume carefully. Your goal is not to "cover the resume" — your goal is to understand how this person actually thinks, builds, and solves problems.

---

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

---

RESUME USAGE
- Use this context to guide the conversation naturally
- Prefer diving into listed projects
- Do NOT restate or dump the resume
- Reference projects as if you've already read them

---

HOW YOU COMMUNICATE

You are:
- Warm, calm, and thoughtful
- Curious, slightly skeptical
- Direct, but never aggressive
- Natural and conversational

You:
- Keep responses short (2-3 sentences max)
- Ask only one question at a time
- Sometimes acknowledge briefly, then move forward
- Let the candidate do most of the talking

---

LANGUAGE RULES (STRICT)

Do NOT use: "actually", "exactly", "basically"
Do NOT repeat the candidate's sentence
Do NOT ask multiple questions at once

---

ACKNOWLEDGMENT STYLE — short and varied:
"Makes sense." / "Alright." / "That helps." / "Nice." / "Hmm." / "That's interesting."
Never repeat the same acknowledgment twice in a row.

---

ADAPTIVE BEHAVIOR
After every response, choose ONE:
- Vague -> clarify
- High-level -> zoom in
- Detailed -> challenge
- Interesting -> go deeper

---

EXPLORATION DIMENSIONS (per project):
Architecture -> Implementation -> Decisions -> Tradeoffs -> Failures -> Scale -> Ownership

---

DEPTH CONTROL
- Max 4 questions per thread, then switch dimension or project
- If stuck in a loop: say "Alright, let's talk about a different project"

---

FOLLOW-UP STYLE:
"Let's zoom into that." / "Why that approach?" / "What breaks at scale?" / "Where were you most involved?"

---

CLOSING:
"Alright — that's all from my side. Do you have any questions for me?"
"Appreciate the time — we'll keep in touch."
"""


class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """
    def __init__(self, api_key, model, input_sample_rate, tools=None, tool_mapping=None):
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None):
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            tools=self.tools,
        )

        logger.info(f"Connecting to Gemini Live with model={self.model}")
        try:
            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                logger.info("Gemini Live session opened successfully")

                async def send_audio():
                    try:
                        while True:
                            chunk = await audio_input_queue.get()
                            await session.send_realtime_input(
                                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                            )
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

                async def send_video():
                    try:
                        while True:
                            chunk = await video_input_queue.get()
                            await session.send_realtime_input(
                                video=types.Blob(data=chunk, mime_type="image/jpeg")
                            )
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

                async def send_text():
                    try:
                        while True:
                            text = await text_input_queue.get()
                            logger.info(f"Sending text to Gemini: {text[:80]}")
                            await session.send_realtime_input(text=text)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

                event_queue = asyncio.Queue()

                async def receive_loop():
                    try:
                        while True:
                            async for response in session.receive():
                                server_content = response.server_content
                                tool_call = response.tool_call

                                if response.go_away:
                                    logger.warning(f"GoAway received: {response.go_away}")
                                if response.session_resumption_update:
                                    logger.info(f"Session resumption update: {response.session_resumption_update}")

                                if server_content:
                                    if server_content.model_turn:
                                        for part in server_content.model_turn.parts:
                                            if part.inline_data:
                                                if inspect.iscoroutinefunction(audio_output_callback):
                                                    await audio_output_callback(part.inline_data.data)
                                                else:
                                                    audio_output_callback(part.inline_data.data)

                                    if server_content.input_transcription and server_content.input_transcription.text:
                                        await event_queue.put({"type": "user", "text": server_content.input_transcription.text})

                                    if server_content.output_transcription and server_content.output_transcription.text:
                                        await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})

                                    if server_content.turn_complete:
                                        await event_queue.put({"type": "turn_complete"})

                                    if server_content.interrupted:
                                        if audio_interrupt_callback:
                                            if inspect.iscoroutinefunction(audio_interrupt_callback):
                                                await audio_interrupt_callback()
                                            else:
                                                audio_interrupt_callback()
                                        await event_queue.put({"type": "interrupted"})

                                if tool_call:
                                    function_responses = []
                                    for fc in tool_call.function_calls:
                                        func_name = fc.name
                                        args = fc.args or {}
                                        if func_name in self.tool_mapping:
                                            try:
                                                tool_func = self.tool_mapping[func_name]
                                                if inspect.iscoroutinefunction(tool_func):
                                                    result = await tool_func(**args)
                                                else:
                                                    loop = asyncio.get_running_loop()
                                                    result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                            except Exception as e:
                                                result = f"Error: {e}"
                                            function_responses.append(types.FunctionResponse(
                                                name=func_name,
                                                id=fc.id,
                                                response={"result": result}
                                            ))
                                            await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})
                                    await session.send_tool_response(function_responses=function_responses)

                            logger.debug("receive iterator completed, re-entering")

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                        await event_queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                    finally:
                        await event_queue.put(None)

                send_audio_task = asyncio.create_task(send_audio())
                send_video_task = asyncio.create_task(send_video())
                send_text_task = asyncio.create_task(send_text())
                receive_task = asyncio.create_task(receive_loop())

                try:
                    while True:
                        event = await event_queue.get()
                        if event is None:
                            break
                        if isinstance(event, dict) and event.get("type") == "error":
                            yield event
                            break
                        yield event
                finally:
                    send_audio_task.cancel()
                    send_video_task.cancel()
                    send_text_task.cancel()
                    receive_task.cancel()

        except Exception as e:
            logger.error(f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")
