import asyncio
import json
import os
import subprocess
import sys

import edge_tts

VOICE, RATE = "en-US-AndrewMultilingualNeural", "-2%"
here = os.path.dirname(os.path.abspath(__file__))
segs = {s["id"]: s for s in json.load(open(os.path.join(here, "script.json")))}
dur = json.load(open(os.path.join(here, "durations.json")))


async def run(ids):
    for i in ids:
        mp3 = os.path.join(here, f"{i}.mp3")
        await edge_tts.Communicate(segs[i]["text"], VOICE, rate=RATE).save(mp3)
        dur[i] = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    mp3,
                ]
            )
            .decode()
            .strip()
        )
        print(i, round(dur[i], 2), "s")
    json.dump(dur, open(os.path.join(here, "durations.json"), "w"), indent=1)


asyncio.run(run(sys.argv[1:]))
