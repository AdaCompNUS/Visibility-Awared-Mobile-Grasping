import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mg import sources as S

D = os.path.join(S.DROPBOX, "real robot demo", "all demos")
JOBS = [
    ("Video 2026-1-21, 14 17 02.mov", 37.0, 50.0),
    ("Video 2026-1-26, 10 25 31.mov", 43.0, 94.0),
    ("Video 2026-1-26, 10 11 50.mov", 39.0, 80.0),
    ("Video 2026-1-26, 11 15 51.mov", 32.0, 78.0),
    ("Video 2026-1-26, 09 32 09.mov", 58.0, 132.0),
    ("Video 2026-1-21, 13 46 34.mov", 55.0, 86.0),
]
for f, a, b in JOBS:
    t0 = time.time()
    tr = S.person_track(os.path.join(D, f), a, b, step=2)
    n = sum(1 for v in tr.values() if v)
    print(
        f"{f}: {len(tr)} frames scanned, {n} with people, {time.time()-t0:.0f}s",
        flush=True,
    )
