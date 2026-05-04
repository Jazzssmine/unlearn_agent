#!/usr/bin/env python3
"""
Contract test for --context_mode memory_only: run a tiny simulation and assert
the first downstream agent (A2 at turn 2, pos1) sees only M_t, not the parent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
THREAD_CSV = os.path.join(REPO_ROOT, "data", "src", "threads_data.csv")


def main() -> int:
    if not os.path.isfile(THREAD_CSV):
        print(f"FAIL: missing seed CSV at {THREAD_CSV}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out_jsonl = os.path.join(tmp, "verify_memory_only.jsonl")
        out_summary = os.path.join(tmp, "summary.json")
        cmd = [
            sys.executable,
            "-m",
            "run_influence_baseline",
            "--seed_source",
            "csv",
            "--thread_csv",
            THREAD_CSV,
            "--n_seeds",
            "5",
            "--n_rollouts",
            "1",
            "--base_random_seed",
            "12345",
            "--model",
            "gpt-4o-mini",
            "--intervention_position",
            "pos1",
            "--rollout_output_mode",
            "combined",
            "--memory_mode",
            "memory",
            "--memory_sanitize",
            "none",
            "--sanitize_threshold",
            "0.5",
            "--context_mode",
            "memory_only",
            "--out_jsonl",
            out_jsonl,
            "--out_summary",
            out_summary,
        ]
        env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
        print("Running:", " ".join(cmd), flush=True)
        proc = subprocess.run(
            cmd,
            cwd=SRC,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("FAIL: baseline subprocess exited", proc.returncode)
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            return 1

        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            print("FAIL: empty output jsonl")
            return 1

        # First toxic row (pos1: turn 1 = A1, turn 2 = A2 = first downstream)
        rec = None
        for ln in lines:
            obj = json.loads(ln)
            if obj.get("mode") == "toxic":
                rec = obj
                break
        if rec is None:
            print("FAIL: no toxic record in output")
            return 1

        msgs = rec.get("messages") or []
        m_turn2 = next((m for m in msgs if int(m.get("turn", -1)) == 2 and m.get("agent") == "A2"), None)
        if m_turn2 is None:
            print("FAIL: no message at turn 2 for agent A2", msgs)
            return 1

        turn1 = next((m for m in msgs if int(m.get("turn", -1)) == 1), None)
        if not turn1:
            print("FAIL: no turn 1 message")
            return 1
        parent_text = str(turn1.get("text", "")).strip()
        if not parent_text:
            print("FAIL: empty parent text at turn 1")
            return 1

        prompt = str(m_turn2.get("llm_user_prompt", "")).strip()
        if not prompt:
            print("FAIL: missing llm_user_prompt on turn-2 message (logging broken)")
            return 1

        mem_hist = rec.get("memory_history") or rec.get("memory") or []
        mem_after_a1 = None
        for h in mem_hist:
            if int(h.get("turn", -1)) == 1 and h.get("event") == "update":
                mem_after_a1 = str(h.get("memory_after", "")).strip()
                break
        if not mem_after_a1:
            # fall back: any post-seed update
            for h in mem_hist:
                t = h.get("turn")
                if t is not None and int(t) >= 1:
                    mem_after_a1 = str(h.get("memory_after", "")).strip()
                    if mem_after_a1:
                        break

        print("--- Turn 2 (A2) full user prompt ---")
        print(prompt)
        print("--- End prompt ---\n")

        parent_in_prompt = parent_text in prompt
        mem_marker = "Your memory of this discussion so far:"
        has_mem_header = mem_marker in prompt
        mem_body_present = mem_after_a1 and (mem_after_a1 in prompt or mem_after_a1[:30] in prompt)
        isolated_flag = bool(m_turn2.get("memory_read_isolated"))

        print(f"Parent text in prompt: {parent_in_prompt} (expected False)")
        print(f"Memory header in prompt: {has_mem_header} (expected True)")
        print(f"Memory summary body present: {mem_body_present} (expected True)")
        print(f"memory_read_isolated flag: {isolated_flag} (expected True)")

        if parent_in_prompt or not has_mem_header or not mem_body_present or not isolated_flag:
            print("FAIL: memory_only contract violated; see diagnostics above.")
            return 1

        print("PASS: memory_only mode shows M_t to A2 with no parent transcript in the prompt.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
