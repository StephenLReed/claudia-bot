"""localqueue: a locked clip goes to a job queue you run yourself, through three commands named in a config."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

_H = Path(__file__).parent / "fixtures" / "gen" / "helpers.py"
_spec = importlib.util.spec_from_file_location("gen_test_helpers", _H)
H = importlib.util.module_from_spec(_spec); sys.modules.setdefault("gen_test_helpers", H); _spec.loader.exec_module(H)
FAKE = Path(__file__).parent / "fixtures" / "gen" / "fake_queue.py"

from slop import ledger  # noqa: E402
from slop.cli import main  # noqa: E402
from slop.gen import localqueue as lq, store, templates as tpl  # noqa: E402

pytestmark = pytest.mark.skipif(not H.FFMPEG, reason="ffmpeg is required")


def config(tmp_path: Path, mode: str) -> Path:
    q = tmp_path / "queue"
    cfg = {"name": "test-queue", "model": "test-a2v", "work_root": str(tmp_path / "work"), "gpu_min_per_clip": 5,
           "job": {"job": "{job_name}", "image": "{start_name}", "audio": "{audio_name}", "frames": "{frames}",
                   "seed": "{seed}", "prompt": "{prompt}"},
           "submit": [sys.executable, str(FAKE), "submit", "{jobs_file}", "{inputs_dir}", str(q), mode],
           "done": [sys.executable, str(FAKE), "done", str(q), "{ticket}"],
           "collect": [sys.executable, str(FAKE), "collect", str(q), "{ticket}", "{collect_dir}"],
           "status_file": "DONE.json", "status_ok": ["DONE"], "output_glob": "**/{job_name}.mp4"}
    p = tmp_path / f"lq-{mode}.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SLOP_FREE_ONLY", "1")                          # a local queue spends nothing: the lock must not bite
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SLOP_PROJECT", raising=False)
    H.write_cache(tmp_path / "home")
    return H.make_project(tmp_path / "projects")


def run(pdir, *argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["--json", "--project", str(pdir), *argv])
    return rc, json.loads(buf.getvalue())


def ready(env, tmp_path, monkeypatch, mode="done", *extra):
    monkeypatch.setenv("SLOP_LOCALQUEUE", str(config(tmp_path, mode)))
    rc, out = run(env, "clip", "new", "s004", "--kind", "motion", "--provider", "localqueue", "--frame", "first", *extra)
    assert rc == 0, out
    cid = out["clip_id"]
    c = store.load(env, cid)
    assert c["provider"] == "localqueue" and c["model"] == "test-a2v"
    f = tmp_path / f"{cid}.txt"
    f.write_text(tpl.SLOT_RE.sub("they dance", c["body"]["prompt"]), encoding="utf-8")
    assert run(env, "clip", "set", cid, "--prompt-file", str(f))[0] == 0
    rc, out = run(env, "clip", "lock", cid)
    assert rc == 0, out
    assert "$0 (local queue)" in out["text"]
    return cid


def test_frames_rule():
    assert lq.frames_for(4, 24, 8, 1) == 89                      # never past the 4 s window (97 frames = 4.04 s)
    assert lq.frames_for(4, 24, 8, 1, "nearest") == 97
    assert lq.frames_for(5, 24, 8, 1) == 113
    for d in (4, 4.2, 5, 7.5, 12):
        f = lq.frames_for(d, 24, 8, 1)
        assert (f - 1) % 8 == 0 and f / 24 <= d


def test_fill_keeps_types_and_refuses_unknown():
    assert lq.fill({"a": "{n}", "b": "x-{n}"}, {"n": 7}) == {"a": 7, "b": "x-7"}
    with pytest.raises(Exception, match="unknown placeholder"):
        lq.fill("{nope}", {})


def test_dry_run_writes_and_runs_nothing(env, tmp_path, monkeypatch):
    cid = ready(env, tmp_path, monkeypatch)
    rc, out = run(env, "clip", "submit", cid, "--dry-run")
    assert rc == 0, out
    assert out["provider"] == "localqueue" and out["est_usd"] == 0.0
    job = out["job"]
    dur = store.load(env, cid)["body"]["duration"]
    assert job["frames"] == lq.frames_for(dur, 24, 8, 1) and (job["frames"] - 1) % 8 == 0 and isinstance(job["seed"], int) and "@Audio1" not in job["prompt"]
    assert job["image"].endswith("-start.png") and job["audio"].endswith("-audio.wav")
    assert not (tmp_path / "work").exists() and not (tmp_path / "queue").exists()
    assert f"clip submit {cid} --yes" in out["text"]


def test_submit_poll_delivers_a_take_at_no_cost(env, tmp_path, monkeypatch):
    cid = ready(env, tmp_path, monkeypatch)
    rc, out = run(env, "clip", "submit", cid)                      # no --yes: the spend gate holds
    assert rc != 0
    rc, out = run(env, "clip", "submit", cid, "--yes")
    assert rc == 0, out
    assert out["ticket"] == "T1"
    sent = tmp_path / "queue" / "T1" / "inputs"
    assert sorted(p.name.split("-t1-")[1] for p in sent.iterdir()) == ["audio.wav", "start.png"]
    c = store.load(env, cid)
    assert c["status"] == "submitted" and c["takes"][-1]["id"] == "T1"
    rows = [r for r in ledger.rows(env) if r.get("provider") == "localqueue"] if hasattr(ledger, "rows") else []
    assert all(float(r.get("est_usd") or 0) == 0 for r in rows)
    rc, out = run(env, "clip", "poll", cid)
    assert rc == 0, out
    c = store.load(env, cid)
    tk = c["takes"][-1]
    assert c["status"] == "delivered" and tk["status"] == "delivered" and tk["queue_state"] == "DONE"
    assert (env / tk["file"]).is_file() and tk["duration_s"] > 0.5


def test_pending_then_done(env, tmp_path, monkeypatch):
    cid = ready(env, tmp_path, monkeypatch, "slow")
    assert run(env, "clip", "submit", cid, "--yes")[0] == 0
    rc, out = run(env, "clip", "poll", cid)
    assert rc == 0 and "still in the queue" in out["text"]
    assert store.load(env, cid)["status"] == "submitted"
    subprocess.run([sys.executable, str(FAKE), "finish", str(tmp_path / "queue"), "T1", "done"], check=True)
    rc, out = run(env, "clip", "poll", cid)
    assert rc == 0 and store.load(env, cid)["status"] == "delivered"


def test_failed_ticket_unlocks_for_a_retry(env, tmp_path, monkeypatch):
    cid = ready(env, tmp_path, monkeypatch, "failed")
    assert run(env, "clip", "submit", cid, "--yes")[0] == 0
    rc, out = run(env, "clip", "poll", cid)
    assert rc == 0 and "failed" in out["text"]
    c = store.load(env, cid)
    assert c["status"] == "locked" and c["takes"][-1]["status"] == "failed"


def test_missing_config_is_a_lock_time_problem(env, tmp_path, monkeypatch):
    cid = ready(env, tmp_path, monkeypatch)
    monkeypatch.setenv("SLOP_LOCALQUEUE", str(tmp_path / "gone.json"))
    rc, out = run(env, "clip", "submit", cid, "--dry-run")
    assert rc == 0 and "localqueue config not found" in out["text"]
