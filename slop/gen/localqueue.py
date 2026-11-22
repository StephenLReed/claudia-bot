"""`localqueue`: a video provider that hands a locked clip to a job queue you run yourself (a home GPU, a lab cluster),
through three shell commands named in a JSON config. Nothing here knows the queue: the config says how to submit a job
bundle, how to tell that it is done, and how to bring its outputs back.

A take costs $0 in the ledger (the GPU is yours); the config may give an estimate of GPU minutes per clip instead.
The spend gate still applies: a clip is submitted only when it is locked and the owner said yes (--yes or the gate).

Config (JSON; path in $SLOP_LOCALQUEUE, else project.json "localqueue_config"):

    {
      "name": "my-queue",                      # shown in messages
      "model": "ltx-2.5-a2v",                  # recorded on the clip and the take
      "work_root": "/abs/folder",              # each take's bundle goes to work_root/<project>/<clip>-t<take>/
      "fps": 24, "frames_multiple": 8, "frames_offset": 1,      # frames = multiple*k + offset, at most duration*fps
      "frames_round": "down",                  # or "nearest" (may run past the audio window)
      "inputs_dir": "inputs", "jobs_file": "jobs.jsonl",
      "job": {"job": "{job_name}", "image": "{start_name}", "audio": "{audio_name}",
              "frames": "{frames}", "seed": "{seed}", "prompt": "{prompt}"},
      "submit": ["/abs/submit.sh", "--jobs", "{jobs_file}", "--inputs", "{inputs_dir}", "--set", "{set_name}"],
      "ticket": "last_line",                   # or {"prefix": "ticket: "}: the line that names the job id
      "done": ["sh", "-c", "test -f /queue/{ticket}/out/DONE.json"],     # exit 0 = finished (ok or not)
      "collect": ["rsync", "-a", "host:/queue/{ticket}/out/", "{collect_dir}/"],
      "status_file": "DONE.json", "status_key": "state", "status_ok": ["DONE"],
      "output_glob": "**/{job_name}.mp4",
      "gpu_min_per_clip": 6
    }

Placeholders: project, clip, take, job_name, set_name, work_dir, inputs_dir, jobs_file, collect_dir, ticket,
start_name, audio_name, frames, seed, prompt, duration, fps. A string that is exactly one placeholder keeps the value's
type (frames and seed stay numbers); "audio" is dropped from the job when the clip sends no audio.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ..config import SlopError
from . import store

PROVIDER = "localqueue"
REQUIRED = ("name", "model", "work_root", "job", "submit", "done", "collect", "output_glob")
DEFAULTS = {"fps": 24, "frames_multiple": 8, "frames_offset": 1, "frames_round": "down", "inputs_dir": "inputs", "jobs_file": "jobs.jsonl",
            "ticket": "last_line", "status_file": None, "status_key": "state", "status_ok": None, "gpu_min_per_clip": None,
            "timeout_s": 600}
PROMPT_SUBS = (("@Audio1", "the music"), ("@Image1", "the first frame"))


def config_path(pdir: Path) -> Path | None:
    env = os.environ.get("SLOP_LOCALQUEUE")
    if env:
        return Path(env).expanduser()
    pj = pdir / "project.json"
    if pj.is_file():
        v = (json.loads(pj.read_text(encoding="utf-8")) or {}).get("localqueue_config")
        if v:
            return Path(v).expanduser()
    return None


def load_config(pdir: Path) -> dict:
    p = config_path(pdir)
    if not p:
        raise SlopError("localqueue needs a config: set $SLOP_LOCALQUEUE or project.json localqueue_config to its path "
                        "(fields: slop/gen/localqueue.py)")
    if not p.is_file():
        raise SlopError(f"localqueue config not found: {p}")
    cfg = {**DEFAULTS, **json.loads(p.read_text(encoding="utf-8"))}
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        raise SlopError(f"localqueue config {p} lacks: {', '.join(missing)}")
    for k in ("submit", "done", "collect"):
        if not (isinstance(cfg[k], list) and cfg[k] and all(isinstance(x, str) for x in cfg[k])):
            raise SlopError(f"localqueue config {k} must be a command as a list of strings (no shell string)")
    cfg["_path"] = str(p)
    return cfg


def frames_for(duration: float, fps: int, multiple: int, offset: int, rounding: str = "down") -> int:
    """multiple*k + offset frames for a duration. "down" (default) never runs past the clip's audio window, which a
    model that holds the window fixed needs (its audio must be at least frames/fps long); "nearest" may run past it."""
    x = (float(duration) * fps - offset) / multiple
    k = max(1, int(x + 1e-9) if rounding == "down" else round(x))
    return multiple * k + offset


def seed_for(cid: str, take: int) -> int:
    return int(hashlib.sha1(f"{cid}#{take}".encode()).hexdigest()[:8], 16) % (2 ** 31)


def prompt_for(c: dict) -> str:
    p = (c.get("body") or {}).get("prompt") or ""
    for a, b in PROMPT_SUBS:
        p = p.replace(a, b)
    return p


def fill(obj, v: dict):
    """the config's templates with the take's values; "{x}" alone keeps x's type."""
    if isinstance(obj, dict):
        return {k: fill(x, v) for k, x in obj.items()}
    if isinstance(obj, list):
        return [fill(x, v) for x in obj]
    if isinstance(obj, str):
        if obj.startswith("{") and obj.endswith("}") and obj[1:-1] in v and obj.count("{") == 1:
            return v[obj[1:-1]]
        try:
            return obj.format_map(v)
        except KeyError as e:
            raise SlopError(f"localqueue config uses an unknown placeholder {e}") from None
    return obj


def take_values(pdir: Path, c: dict, cfg: dict, take: int, *, audio: Path | None, ticket: str = "") -> dict:
    cid = c["id"]
    work = Path(cfg["work_root"]).expanduser() / pdir.name / f"{cid}-t{take}"
    start = store.absp(pdir, c.get("start"))
    dur = float((c.get("body") or {}).get("duration") or 4)
    job = f"{pdir.name}-{cid}-t{take}"
    return {"project": pdir.name, "clip": cid, "take": take, "job_name": job, "set_name": job, "work_dir": str(work),
            "inputs_dir": str(work / cfg["inputs_dir"]), "jobs_file": str(work / cfg["jobs_file"]),
            "collect_dir": str(work / "collected"), "ticket": ticket,
            "start_name": f"{job}-start{start.suffix if start else '.png'}",
            "audio_name": f"{job}-audio{audio.suffix}" if audio else "",
            "frames": frames_for(dur, int(cfg["fps"]), int(cfg["frames_multiple"]), int(cfg["frames_offset"]),
                                 cfg.get("frames_round") or "down"),
            "seed": int((c.get("body") or {}).get("seed") or seed_for(cid, take)), "prompt": prompt_for(c),
            "duration": dur, "fps": int(cfg["fps"])}


def build(pdir: Path, c: dict, cfg: dict, take: int, *, audio: Path | None) -> dict:
    """everything a submit would write and run, without writing or running it."""
    v = take_values(pdir, c, cfg, take, audio=audio)
    job = fill(cfg["job"], v)
    if not audio:
        job = {k: x for k, x in job.items() if x != ""}
    start = store.absp(pdir, c.get("start"))
    files = [(start, v["start_name"])] + ([(audio, v["audio_name"])] if audio else [])
    return {"values": v, "job": job, "files": files, "submit": fill(cfg["submit"], v)}


def write_bundle(b: dict) -> Path:
    v = b["values"]
    work, inputs = Path(v["work_dir"]), Path(v["inputs_dir"])
    if work.exists():
        shutil.rmtree(work)
    inputs.mkdir(parents=True)
    for src, name in b["files"]:
        if not src or not Path(src).is_file():
            raise SlopError(f"localqueue: input missing: {src}")
        shutil.copy2(src, inputs / name)
    Path(v["jobs_file"]).write_text(json.dumps(b["job"], ensure_ascii=False) + "\n", encoding="utf-8")
    return work


def run(argv: list[str], cfg: dict, *, check: bool = True) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=int(cfg.get("timeout_s") or 600))
    except FileNotFoundError:
        raise SlopError(f"localqueue: command not found: {argv[0]}") from None
    except subprocess.TimeoutExpired:
        raise SlopError(f"localqueue: {argv[0]} did not finish in {cfg.get('timeout_s')} s") from None
    if check and r.returncode != 0:
        raise SlopError(f"localqueue: {' '.join(argv[:2])}… exited {r.returncode}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r


def ticket_from(stdout: str, cfg: dict) -> str:
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    rule = cfg.get("ticket") or "last_line"
    if isinstance(rule, dict) and rule.get("prefix"):
        hits = [ln[len(rule["prefix"]):].strip() for ln in lines if ln.startswith(rule["prefix"])]
        if hits:
            return hits[-1]
    elif rule == "last_line" and lines:
        return lines[-1]
    raise SlopError(f"localqueue: no ticket in the submit output: {(stdout or '').strip()[-300:]!r}")


def describe(cfg: dict | None) -> str:
    if not cfg:
        return "$0 (local queue)"
    g = cfg.get("gpu_min_per_clip")
    return f"$0 on {cfg['name']}" + (f" (~{g:g} GPU-min)" if g else "")


def dry_text(c: dict, b: dict, cfg: dict) -> list[str]:
    v = b["values"]
    return [f"{c['id']} DRY RUN (nothing written, run or spent): {cfg['name']} {cfg['model']}, {v['duration']:g} s, "
            f"{v['frames']} frames at {v['fps']} fps, seed {v['seed']}, {describe(cfg)}",
            "inputs: " + ", ".join(f"{name} <- {src}" for src, name in b["files"]),
            "job: " + json.dumps({k: x for k, x in b["job"].items() if k != "prompt"}, ensure_ascii=False),
            "submit: " + " ".join(b["submit"]), "prompt:", v["prompt"]]


def collect(pdir: Path, c: dict, tk: dict, cfg: dict) -> dict:
    """poll one in-flight take: {"state": "pending"} or {"state": "delivered", "file"} or {"state": "failed", "error"}."""
    v = {**tk.get("values", {}), "ticket": tk.get("id") or ""}
    if run(fill(cfg["done"], v), cfg, check=False).returncode != 0:
        return {"state": "pending"}
    cdir = Path(v["collect_dir"])
    cdir.mkdir(parents=True, exist_ok=True)
    run(fill(cfg["collect"], v), cfg)
    status = None
    if cfg.get("status_file"):
        sf = next(iter(sorted(cdir.rglob(cfg["status_file"]))), None)
        if sf:
            status = (json.loads(sf.read_text(encoding="utf-8")) or {}).get(cfg.get("status_key") or "state")
    vids = sorted(cdir.glob(fill(cfg["output_glob"], v)))
    ok = cfg.get("status_ok")
    if not vids or (ok and status is not None and status not in ok):
        return {"state": "failed", "error": f"queue state {status or 'unknown'}; "
                                            f"{len(vids)} output(s) matching {fill(cfg['output_glob'], v)}"}
    out = store.take_path(pdir, c["id"], tk["n"])
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vids[0], out)
    return {"state": "delivered", "file": out, "queue_state": status}
