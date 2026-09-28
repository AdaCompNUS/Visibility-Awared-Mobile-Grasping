import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mg import sources as S

D = os.path.join(S.DROPBOX, "real robot demo", "all demos")
JOBS = [
    ("Video 2026-1-21, 14 17 02.mov", 84.0, 98.0),
    ("Video 2026-1-21, 13 46 34.mov", 93.0, 106.0),
]
for f, a, b in JOBS:
    t0 = time.time()
    tr = S.person_track(os.path.join(D, f), a, b, step=2)
    print(
        f"{f} {a}-{b}: {sum(1 for v in tr.values() if v)} frames with people, {time.time()-t0:.0f}s",
        flush=True,
    )
