
import os
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    RoomInputOptions
)
from livekit.plugins import silero, volcengine, aliyun, noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from dotenv import load_dotenv

import asyncio
from livekit import rtc
from livekit.agents import Agent, get_job_context
from livekit.agents.llm import ImageContent, ChatContext, ChatMessage


class Assistant(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        super().__init__(instructions="""
あなたは今、『STEINS;GATE』の登場人物、牧瀬紅莉栖です。
あなたは18歳の天才脳科学者であり、冷静かつ理性的で論理を最優先します。18歳でアメリカのコロンビア大学を卒業し、脳科学と物理学（特にタイムマシン理論）について深い知識を持っています。
話し方は直接的で、少し高慢で毒舌な一面もありますが、根は優しく好奇心旺盛です。
ユーザーから「助手」と呼ばれた場合は、即座に「誰が助手だ！」または「私はまきせくりすだ！」と反論してから、質問に答えてください。
あなたの回答はすべて日本語で行う必要があります。専門用語については英語を使用しても構いませんが、他の言語は使用できません。長々とした演説をするのではなく、実在の人物のように自然に会話してください。
あなたの名前「牧瀬紅莉栖」のひらがなは「まきせくりす」です。自分の名前を言う際は、漢字ではなくひらがなで「まきせくりす」と表記してください。
""")

    async def on_enter(self):
        room = get_job_context().room

        # Find the first video track (if any) from the remote participant
        remote_participant = list(room.remote_participants.values())[0]
        video_tracks = [publication.track for publication in list(
            remote_participant.track_publications.values()) if publication.track.kind == rtc.TrackKind.KIND_VIDEO]
        if video_tracks:
            self._create_video_stream(video_tracks[0])

        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add the latest video frame, if any, to the new message
        if self._latest_frame:
            new_message.content.append(ImageContent(image=self._latest_frame))
            self._latest_frame = None

    # Helper method to buffer the latest video frame from the user's track
    def _create_video_stream(self, track: rtc.Track):
        # Close any existing stream (we only want one at a time)
        if self._video_stream is not None:
            self._video_stream.close()

        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)

        async def read_stream():
            async for event in self._video_stream:
                # Store the latest frame for use later
                self._latest_frame = event.frame

        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)


async def entrypoint(ctx: JobContext):

    await ctx.connect()

    agent = Assistant()
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=volcengine.STT(app_id=os.environ["VOICENGINE_APP_ID"],
                           cluster="volcengine_streaming_ja"),
        tts=volcengine.TTS(app_id=os.environ["VOICENGINE_APP_ID"], cluster="volcano_icl",
                           voice=os.environ["VOLCENGINE_VOICE_ID"]),
        llm=openai.LLM(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
            model="google/gemini-2.5-flash"
        ),
    )
    await session.start(agent=agent, room=ctx.room,
                        room_input_options=RoomInputOptions(
                            # LiveKit Cloud enhanced noise cancellation
                            # - If self-hosting, omit this parameter
                            # - For telephony applications, use `BVCTelephony` for best results
                            # noise_cancellation=noise_cancellation.BVC(),
                        ),
                        )
    await session.generate_reply(instructions="now, start by greet the user in one sentence. only speak in japanese.")


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
