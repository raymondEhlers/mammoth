from __future__ import annotations

from pathlib import Path

import IPython


def main(filename: Path) -> list[str]:
    with filename.open() as f:
        names = [line.strip("\n").strip() for line in f]

    return list(set(names))


if __name__ == "__main__":
    # NOTE: I removed names of folks who are listed but never took an assignment
    names = main(filename=Path("people_running_jobs.txt"))

    # This is what I used to print in the end
    print(", ".join(sorted(names)))
    # But can explore some more if needed
    IPython.start_ipython(user_ns={**globals(), **locals()})
