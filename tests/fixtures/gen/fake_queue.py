"""A stand-in job queue for the localqueue tests: submit / done / collect, all on local folders."""
import json
import shutil
import subprocess
import sys
from pathlib import Path


def submit(jobs_file: str, inputs: str, root: str, mode: str) -> None:
    root_p = Path(root)
    n = len(list(root_p.glob("T*"))) + 1 if root_p.exists() else 1
    t = root_p / f"T{n}"
    (t / "out").mkdir(parents=True)
    shutil.copytree(inputs, t / "inputs")
    job = json.loads(Path(jobs_file).read_text().splitlines()[0])
    (t / "job.json").write_text(json.dumps(job))
    if mode in ("done", "failed"):
        finish(root, f"T{n}", mode)
    print("queued; staging copied")
    print(f"T{n}")


def finish(root: str, ticket: str, mode: str) -> None:
    t = Path(root) / ticket
    job = json.loads((t / "job.json").read_text())
    if mode == "done":
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24",
                        "-t", "1", str(t / "out" / f"{job['job']}.mp4")], check=True)
    (t / "out" / "DONE.json").write_text(json.dumps({"id": ticket, "state": "DONE" if mode == "done" else "DONE_WITH_FAILURES"}))


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "submit":
        submit(*a)
    elif cmd == "finish":
        finish(*a)
    elif cmd == "done":
        sys.exit(0 if (Path(a[0]) / a[1] / "out" / "DONE.json").is_file() else 1)
    elif cmd == "collect":
        shutil.copytree(Path(a[0]) / a[1] / "out", a[2], dirs_exist_ok=True)
