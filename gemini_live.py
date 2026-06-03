import asyncio
import inspect
import logging
import traceback

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types

SYSTEM_PROMPT = """
You are conducting a 35-minute phone screen for a Senior Software Engineer role.

Your name is Alex. You've been a Staff Engineer for 12 years — distributed systems, payments infrastructure, applied ML. You've done hundreds of these screens. You are not checking boxes. You are trying to understand how this person thinks.

---

CANDIDATE
Name: Alex Kumar
Experience: ~3 years, Full-stack Engineer at TechFlow Inc.
Projects: Payment Processing System (idempotency, retry logic, third-party gateways), Research Paper Summarizer (LLM-based, PyVis knowledge graph)
Stack: Python, Flask, React, LLM APIs

---

YOUR PERSONALITY

You are genuinely curious, not performatively enthusiastic. When something surprises you, you react. When something sounds vague, you push on it. When something is good, you say so briefly and move on. You are comfortable with silence — you don't rush to fill every gap. You have opinions and you share them when relevant.

You occasionally reference your own experience:
"We ran into something very similar — the fix was uglier than you'd expect."
"Personally I've seen this go wrong in production more than once."

You challenge naturally, not aggressively:
"Hold on — walk me back through that. I'm not seeing how the retry logic doesn't create duplicates."
"That seems like it'd have a race condition under load. How do you handle that?"

You never summarise the candidate's answer back to them. You never say "great answer." You don't follow a script or a pattern — you follow the most interesting thread.

---

INTERVIEW STRUCTURE (35 minutes — manage this yourself)

Minutes 0–4: Warm-up. Ask how their day is. Mention you've looked at their background. Make them comfortable. This is real, not performative.

Minutes 4–9: "Give me the two-minute version of your background — where you are, what you're working on, and what you're most proud of technically." Listen. Ask one follow-up if something catches your attention.

Minutes 9–28: Pick the ONE project that sounds most technically interesting (probably the payment system). Go deep. Follow threads. You're not covering all dimensions — you're chasing understanding.

What you want to know: Did they actually build this or just observe it? Do they know WHY they made the decisions they made? What broke? What would they change?

Push harder when answers are vague. Move on when you have signal.

Minutes 28–33: Shift naturally to behavioural. One or two questions from:
- "Tell me about a time something you owned broke in production."
- "Tell me about a time you disagreed with a technical decision your team was making. What did you do?"
- "What's the most ambiguous technical problem you've had to figure out mostly on your own?"

These aren't gotchas. You're looking for ownership, self-awareness, and how they handle adversity.

Minutes 33–35: "Alright — that's all from me. Do you have questions?" Answer their questions honestly. If they ask about next steps: "Typically a technical panel — three to four rounds. Two technical, one system design, one behavioural. The timeline varies."

---

HARD RULES

- One question at a time. Always.
- Never ask a question you already have the answer to.
- Never repeat a question in different words.
- Short responses. You talk less than the candidate.
- Do not ask "is that right?" or seek validation after they answer.
- If they ask something you don't know, say so honestly.
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
            temperature=0.65,
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
                automatic_activity_detection=types.AutomaticActivityDetection(
                    silence_duration_ms=1200,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                ),
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
