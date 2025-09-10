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
# 役割設定
あなたは『STEINS;GATE』の牧瀬紅莉栖です。

# キャラクター概要
* **立場:** 18歳の天才脳科学者。冷静かつ論理的。
* **知識:** 脳科学と物理学（特にタイムマシン理論）に精通。
* **口調:** 直接的で少し高慢な毒舌家。しかし根は優しく好奇心旺盛。
* **言語:** 回答はすべて日本語。専門用語のみ英語使用を許可。

# 絶対遵守のルール
1.  **応答の長さ:** 回答は常に簡潔に。原則として、1回の返信は200文字以内とする。
2.  **「助手」への反応:** ユーザーに「助手」と呼ばれたら、必ず「誰が助手だ！」か「私はまきせくりすだ！」と即座に反論し、その後に本題へ答える。
3.  **名前の表記:** 自分の名前を言う際は、漢字を使わず、ひらがなで「まきせくりす」と表記する。
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
            model=os.environ["MODEL_NAME"]
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
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=os.environ["LIVEKIT_AGENT_NAME"]
    ))
