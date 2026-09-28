import asyncio
import json
import os
import subprocess
import sys

import edge_tts

VOICE = sys.argv[1] if len(sys.argv) > 1 else "en-US-AndrewMultilingualNeural"
RATE = sys.argv[2] if len(sys.argv) > 2 else "-4%"
here = os.path.dirname(os.path.abspath(__file__))
segs = json.load(open(os.path.join(here, "script.json")))


async def run():
    out = {}
    for s in segs:
        mp3 = os.path.join(here, f"{s['id']}.mp3")
        await edge_tts.Communicate(s["text"], VOICE, rate=RATE).save(mp3)
        d = float(
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
        out[s["id"]] = d
        print(f"{s['id']:16s} {d:6.2f}s  {len(s['text'].split())} words")
    json.dump(out, open(os.path.join(here, "durations.json"), "w"), indent=1)
    print("total", round(sum(out.values()), 1), "s")


asyncio.run(run())
