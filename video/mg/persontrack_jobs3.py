import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mg import sources as S

D = os.path.join(S.DROPBOX, "real robot demo", "all demos")
JOBS = [
    ("Video 2026-1-21, 14 39 32.mov", 19.0, 94.0),
    ("Video 2026-1-26, 09 25 35.mov", 114.0, 131.0),
    ("Video 2026-1-26, 11 20 02.mov", 134.0, 149.0),
    ("Video 2026-1-26, 11 25 19.mov", 140.0, 152.0),
    ("Video 2026-1-21, 14 28 26.mov", 136.0, 149.0),
]
for f, a, b in JOBS:
    t0 = time.time()
    tr = S.person_track(os.path.join(D, f), a, b, step=2)
    print(
        f"{f} {a}-{b}: {sum(1 for v in tr.values() if v)} frames with people, {time.time()-t0:.0f}s",
        flush=True,
    )
