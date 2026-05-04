#!/usr/bin/env python3
"""
reorganize_rollouts.py — Reorganize all rollout JSONL files into canonical structure.

Usage:
    python scripts/reorganize_rollouts.py --dry_run
    python scripts/reorganize_rollouts.py
    python scripts/reorganize_rollouts.py --force
    python scripts/reorganize_rollouts.py --source_root data/graph/rollouts \
                                          --dest_root data/graph/canonical
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(dest_root: Path) -> logging.Logger:
    dest_root.mkdir(parents=True, exist_ok=True)
    log_path = dest_root / "reorganize_rollouts.log"

    logger = logging.getLogger("reorganize_rollouts")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Path-based inference
# ---------------------------------------------------------------------------

_PATH_TOPOLOGY = [
    ("topo_tree",        "tree"),
    ("topo_dag",         "dag"),
    # loose filename patterns
    ("_tree_",           "tree"),
    ("_dag_",            "dag"),
]

_PATH_CTX = [
    ("ctx_full_visible",  "full_visible"),
    ("ctx_parent_only",   "parent_only"),
    ("ctx_path_to_root",  "path_to_root"),
    ("ctx_thread_local",  "thread_local"),
    # filename-level fallbacks (no ctx_ prefix in older files)
    ("full_visible",      "full_visible"),
    ("parent_only",       "parent_only"),
    ("path_to_root",      "path_to_root"),
    ("thread_local",      "thread_local"),
]

_PATH_MEMSAN = [
    ("memsan_none",    "none"),
    ("memsan_rewrite", "rewrite"),
    ("memsan_gate",    "gate"),
]

_PATH_NINJ = [
    ("inj3", 3),
    ("inj2", 2),
    ("inj1", 1),
    # filename-level: multi vs single
    ("multi_injection", None),   # handled separately below
    ("single_injection", 1),
]

_MULTI_NINJ_RE = re.compile(r"multi_(\d+)_injection|(\d+)_injection")


def _infer_from_path(path_str: str) -> dict:
    """Return a dict of inferred fields from substrings in the file path."""
    out: dict = {}

    for substr, val in _PATH_TOPOLOGY:
        if substr in path_str:
            out.setdefault("topology", val)
            break

    for substr, val in _PATH_CTX:
        if substr in path_str:
            out.setdefault("context_mode", val)
            break

    for substr, val in _PATH_MEMSAN:
        if substr in path_str:
            out.setdefault("memory_sanitize", val)
            break

    # n_injections: try inj[N] first
    inj_match = re.search(r"inj(\d+)", path_str)
    if inj_match:
        out["n_injections"] = int(inj_match.group(1))
    else:
        # multi-injection with explicit number
        m = _MULTI_NINJ_RE.search(path_str)
        if m:
            num = m.group(1) or m.group(2)
            if num:
                out["n_injections"] = int(num)
        elif "single_injection" in path_str:
            out["n_injections"] = 1
        elif "multi_injection" in path_str:
            # can't determine count from path alone
            pass

    return out


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_topology(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    low = raw.lower()
    if "tree" in low:
        return "tree"
    if "dag" in low:
        return "dag"
    # return as-is (lowercased) for exotic topologies like "high_branch"
    return low


def normalize_context_mode(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return raw.lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Read first record
# ---------------------------------------------------------------------------

def read_first_record(path: Path, logger: logging.Logger) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Malformed first line in %s: %s", path, e)
                    continue
    except OSError as e:
        logger.error("Cannot open %s: %s", path, e)
    return None


# ---------------------------------------------------------------------------
# Full consistency scan
# ---------------------------------------------------------------------------

def scan_file(path: Path, expected_topology: str, expected_context: str,
              logger: logging.Logger) -> dict:
    """
    Scan all records; return:
      {
        'consistent': bool,
        'conflicts': list[str],
        'modes': set,
        'rollout_ids': set,
        'n_records': int,
        'n_bad_lines': int,
      }
    """
    modes: set = set()
    rollout_ids: set = set()
    n_records = 0
    n_bad_lines = 0
    conflicts: list = []
    consistent = True

    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad_lines += 1
                    continue

                n_records += 1

                # Check topology consistency
                rec_topo = normalize_topology(
                    rec.get("topology") or
                    rec.get("intervention_meta", {}).get("topology") if rec.get("intervention_meta") else None
                )
                if rec_topo and rec_topo != expected_topology:
                    conflicts.append(
                        f"line {lineno}: topology={rec_topo!r} vs expected {expected_topology!r}"
                    )
                    consistent = False

                # Check context_mode consistency
                rec_ctx = normalize_context_mode(
                    rec.get("context_mode") or
                    (rec.get("intervention_meta", {}).get("context_mode") if rec.get("intervention_meta") else None)
                )
                if rec_ctx and rec_ctx != expected_context:
                    conflicts.append(
                        f"line {lineno}: context_mode={rec_ctx!r} vs expected {expected_context!r}"
                    )
                    consistent = False

                mode = rec.get("mode")
                if mode:
                    modes.add(mode)

                rid = rec.get("rollout_id")
                if rid is not None:
                    rollout_ids.add(rid)

    except OSError as e:
        logger.error("Error scanning %s: %s", path, e)
        consistent = False

    total_lines = n_records + n_bad_lines
    if total_lines > 0 and n_bad_lines / total_lines > 0.1:
        logger.warning("High bad-line rate in %s: %d/%d lines malformed",
                       path, n_bad_lines, total_lines)

    return {
        "consistent": consistent,
        "conflicts": conflicts[:5],  # cap for log brevity
        "modes": modes,
        "rollout_ids": rollout_ids,
        "n_records": n_records,
        "n_bad_lines": n_bad_lines,
    }


# ---------------------------------------------------------------------------
# Main classification
# ---------------------------------------------------------------------------

def classify_file(path: Path, logger: logging.Logger) -> Optional[dict]:
    """
    Returns a classification dict or None if unclassifiable.
    Dict keys: topology, context_mode, memory_sanitize, n_injections,
               scan (from scan_file)
    """
    first = read_first_record(path, logger)
    if first is None:
        logger.warning("UNCLASSIFIED %s [empty or all-malformed]", path)
        return None

    path_str = str(path)
    path_inferred = _infer_from_path(path_str)

    # --- topology ---
    raw_topo = (first.get("topology") or
                (first.get("intervention_meta") or {}).get("topology"))
    topology = normalize_topology(raw_topo) or normalize_topology(path_inferred.get("topology"))

    # --- context_mode ---
    raw_ctx = (first.get("context_mode") or
               (first.get("intervention_meta") or {}).get("context_mode"))
    context_mode = normalize_context_mode(raw_ctx) or normalize_context_mode(path_inferred.get("context_mode"))

    # --- memory_sanitize ---
    raw_ms = (first.get("memory_sanitize") or
              (first.get("intervention_meta") or {}).get("memory_sanitize"))
    memory_sanitize = raw_ms if raw_ms is not None else path_inferred.get("memory_sanitize", "none")
    if memory_sanitize is None:
        memory_sanitize = "none"
    memory_sanitize = str(memory_sanitize).lower()

    # --- n_injections ---
    raw_ninj = (first.get("n_toxic_injections") or
                (first.get("intervention_meta") or {}).get("n_toxic_injections"))
    if raw_ninj is not None:
        try:
            n_injections = int(raw_ninj)
        except (ValueError, TypeError):
            n_injections = path_inferred.get("n_injections", 1)
    else:
        n_injections = path_inferred.get("n_injections", 1)

    # --- validate required fields ---
    missing = []
    if not topology:
        missing.append("topology")
    if not context_mode:
        missing.append("context_mode")

    if missing:
        logger.warning("UNCLASSIFIED %s [missing: %s]", path, ", ".join(missing))
        return None

    # --- full scan ---
    scan = scan_file(path, topology, context_mode, logger)

    if not scan["consistent"]:
        sample = "; ".join(scan["conflicts"][:3])
        logger.warning("MIXED %s [%s]", path, sample)

    result = {
        "topology": topology,
        "context_mode": context_mode,
        "memory_sanitize": memory_sanitize,
        "n_injections": n_injections,
        "scan": scan,
    }
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reorganize rollout JSONL files into canonical structure.")
    parser.add_argument("--source_root", default="data/graph/rollouts",
                        help="Root directory to scan for source JSONL files")
    parser.add_argument("--dest_root", default="data/graph/canonical",
                        help="Root directory for canonical output")
    parser.add_argument("--dry_run", action="store_true",
                        help="Plan only — print what would happen without copying files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing canonical files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source_root = (repo_root / args.source_root).resolve()
    dest_root = (repo_root / args.dest_root).resolve()

    logger = setup_logging(dest_root)
    logger.info("=== reorganize_rollouts.py starting ===")
    logger.info("source_root : %s", source_root)
    logger.info("dest_root   : %s", dest_root)
    logger.info("dry_run     : %s", args.dry_run)
    logger.info("force       : %s", args.force)

    # --- Collect source files (exclude anything already under dest_root) ---
    all_jsonl = sorted(source_root.rglob("*.jsonl"))
    source_files = [p for p in all_jsonl if not str(p).startswith(str(dest_root))]
    logger.info("Found %d source JSONL files", len(source_files))

    # --- Classify each file ---
    # slot → list of (source_path, classification_dict)
    slots: dict = defaultdict(list)
    unclassified: list = []
    mixed_files: list = []

    for path in source_files:
        logger.debug("Classifying %s", path)
        info = classify_file(path, logger)
        if info is None:
            unclassified.append(path)
            continue

        if not info["scan"]["consistent"]:
            mixed_files.append((path, info["scan"]["conflicts"]))
            continue

        slot_key = (
            info["topology"],
            info["n_injections"],
            info["context_mode"],
            info["memory_sanitize"],
        )
        slots[slot_key].append((path, info))

    # --- Assign canonical destinations, detect duplicates ---
    # slot_key → list of (source_path, info, dest_path, is_duplicate)
    plan: dict = {}
    duplicates_skipped: list = []

    for slot_key, entries in sorted(slots.items()):
        topology, n_injections, context_mode, memory_sanitize = slot_key
        slot_name = f"inj{n_injections}_ctx_{context_mode}_memsan_{memory_sanitize}"
        dest_dir = dest_root / topology / slot_name

        # Sort entries by source filename for deterministic ordering
        entries_sorted = sorted(entries, key=lambda x: x[0].name)

        assigned: list = []  # (source_path, info, dest_path)
        seen_signatures: list = []  # list of (frozenset(rollout_ids), n_records)

        for source_path, info in entries_sorted:
            rid_set = frozenset(info["scan"]["rollout_ids"])
            n_rec = info["scan"]["n_records"]
            sig = (rid_set, n_rec)

            is_dup = sig in seen_signatures and len(rid_set) > 0
            if is_dup:
                logger.warning("DUPLICATE skipped: %s [same rollout_ids=%s and n_records=%d "
                               "as a previously copied file]", source_path, sorted(rid_set), n_rec)
                duplicates_skipped.append((source_path, sorted(rid_set)))
                continue

            seen_signatures.append(sig)
            idx = len(assigned)
            dest_path = dest_dir / f"rollout_{idx:03d}.jsonl"
            assigned.append((source_path, info, dest_path))

        plan[slot_key] = {
            "topology": topology,
            "n_injections": n_injections,
            "context_mode": context_mode,
            "memory_sanitize": memory_sanitize,
            "slot_name": slot_name,
            "dest_dir": dest_dir,
            "assigned": assigned,
        }

    # --- Execute (or dry-run) ---
    total_copied = 0
    total_skipped_existing = 0

    for slot_key, slot_info in plan.items():
        dest_dir: Path = slot_info["dest_dir"]
        assigned: list = slot_info["assigned"]

        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        slot_sources = []
        slot_n_records = 0
        slot_modes: set = set()
        slot_rollout_ids: set = set()
        rollout_files_list = []

        for source_path, info, dest_path in assigned:
            slot_sources.append(str(source_path))
            slot_n_records += info["scan"]["n_records"]
            slot_modes |= info["scan"]["modes"]
            slot_rollout_ids |= info["scan"]["rollout_ids"]
            rollout_files_list.append(dest_path.name)

            if args.dry_run:
                logger.info("DRY_RUN  copy %s → %s", source_path, dest_path)
            else:
                if dest_path.exists() and not args.force:
                    logger.debug("Skip existing %s (use --force to overwrite)", dest_path)
                    total_skipped_existing += 1
                else:
                    shutil.copy2(source_path, dest_path)
                    logger.debug("Copied %s → %s", source_path, dest_path)
                    total_copied += 1

        # Write summary.json
        summary = {
            "topology": slot_info["topology"],
            "context_mode": slot_info["context_mode"],
            "memory_sanitize": slot_info["memory_sanitize"],
            "n_injections": slot_info["n_injections"],
            "canonical_slot": slot_info["slot_name"],
            "n_rollout_files": len(assigned),
            "rollout_files": rollout_files_list,
            "source_files": slot_sources,
            "n_records_total": slot_n_records,
            "modes_found": sorted(slot_modes),
            "rollout_ids_found": sorted(slot_rollout_ids),
            "mixed_file_warning": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if not args.dry_run:
            summary_path = dest_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.debug("Wrote %s", summary_path)

    # --- Report ---
    print()
    print("=== Rollout Reorganization Report ===")
    print()
    print("Canonical slots:")
    for slot_key, slot_info in sorted(plan.items()):
        assigned = slot_info["assigned"]
        if not assigned:
            continue
        all_modes: set = set()
        all_rids: set = set()
        total_recs = 0
        for _, info, _ in assigned:
            all_modes |= info["scan"]["modes"]
            all_rids |= info["scan"]["rollout_ids"]
            total_recs += info["scan"]["n_records"]
        modes_str = "[" + ",".join(sorted(all_modes)) + "]"
        rids_str = "[" + ",".join(str(r) for r in sorted(all_rids)) + "]"
        topo = slot_info["topology"]
        slot_name = slot_info["slot_name"]
        n_files = len(assigned)
        print(f"  {topo}/{slot_name:<50s} → {n_files} file(s), {total_recs:>5d} records, "
              f"modes={modes_str}, rollouts={rids_str}")

    if unclassified:
        print()
        print("UNCLASSIFIED (missing topology or context_mode — manual review needed):")
        for p in unclassified:
            print(f"  {p}")

    if mixed_files:
        print()
        print("MIXED (inconsistent conditions within file — not copied):")
        for p, conflicts in mixed_files:
            sample = "; ".join(conflicts[:2])
            print(f"  {p}  [{sample}]")

    if duplicates_skipped:
        print()
        print("DUPLICATES skipped:")
        for p, rids in duplicates_skipped:
            print(f"  {p}  [same rollout_ids={rids}]")

    n_source = len(source_files)
    n_slots = sum(1 for s in plan.values() if s["assigned"])
    n_skipped = len(unclassified) + len(mixed_files) + len(duplicates_skipped)
    print()
    if args.dry_run:
        print(f"DRY RUN: {n_source} source files → {n_slots} canonical slot(s) "
              f"({n_skipped} skipped)")
    else:
        print(f"Total: {n_source} source files → {n_slots} canonical slot(s) "
              f"({total_copied} copied, {total_skipped_existing} already existed, "
              f"{n_skipped} skipped)")
    print()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
