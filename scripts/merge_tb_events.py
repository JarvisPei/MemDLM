#!/usr/bin/env python3
"""
Merge two TensorBoard event runs into one continuous run.

Typical use case:
- You resumed training but produced a second run with overlapping/reset steps.
- You want one clean curve in TensorBoard.
- By default, run1 is truncated to steps strictly smaller than min_step(run2).

Example:
    python dllm/tools/merge_tb_events.py \
      --run1 /path/to/tb/run_part1 \
      --run2 /path/to/tb/run_part2 \
      --out  /path/to/tb/run_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


ScalarPoint = Tuple[float, int, float]  # (wall_time, step, value)


def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        return path.parent
    return path


def _load_scalars(run_dir: Path) -> Dict[str, List[ScalarPoint]]:
    # size_guidance=0 means "load all events" for each type.
    acc = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tag_to_points: Dict[str, List[ScalarPoint]] = {}
    for tag in acc.Tags().get("scalars", []):
        points = acc.Scalars(tag)
        tag_to_points[tag] = [(p.wall_time, int(p.step), float(p.value)) for p in points]
    return tag_to_points


def _max_step(tag_to_points: Dict[str, List[ScalarPoint]]) -> int:
    max_step = -1
    for points in tag_to_points.values():
        for _, step, _ in points:
            if step > max_step:
                max_step = step
    return max_step


def _min_step(tag_to_points: Dict[str, List[ScalarPoint]]) -> int:
    min_step = 2**63 - 1
    found = False
    for points in tag_to_points.values():
        for _, step, _ in points:
            found = True
            if step < min_step:
                min_step = step
    if not found:
        return 0
    return min_step


def _truncate_run1_by_run2_min_step(
    run1: Dict[str, List[ScalarPoint]], run2_min_step: int
) -> Dict[str, List[ScalarPoint]]:
    # Keep run1 strictly before run2 starts to avoid overlap.
    return {
        tag: [(wt, step, val) for wt, step, val in points if step < run2_min_step]
        for tag, points in run1.items()
    }


def _iter_merged_points(
    run1: Dict[str, List[ScalarPoint]],
    run2: Dict[str, List[ScalarPoint]],
    step_offset: int,
) -> Iterable[Tuple[str, ScalarPoint]]:
    tags = sorted(set(run1.keys()) | set(run2.keys()))
    for tag in tags:
        for wall_time, step, value in run1.get(tag, []):
            yield tag, (wall_time, step, value)
        for wall_time, step, value in run2.get(tag, []):
            yield tag, (wall_time, step + step_offset, value)


def _write_events(
    out_dir: Path, merged_points: Iterable[Tuple[str, ScalarPoint]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(out_dir))
    try:
        for tag, (wall_time, step, value) in merged_points:
            event = event_pb2.Event(
                wall_time=wall_time,
                step=step,
                summary=summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)]
                ),
            )
            writer.add_event(event)
        writer.flush()
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two TensorBoard runs.")
    parser.add_argument(
        "--run1",
        required=True,
        help="First run dir (earlier training segment).",
    )
    parser.add_argument(
        "--run2",
        required=True,
        help="Second run dir (resumed segment).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for merged event file.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Step offset added to run2. Default: auto = max_step(run1) + 1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run1_dir = _resolve_run_dir(args.run1)
    run2_dir = _resolve_run_dir(args.run2)
    out_dir = Path(args.out).expanduser().resolve()

    run1_scalars = _load_scalars(run1_dir)
    run2_scalars = _load_scalars(run2_dir)
    run2_min = _min_step(run2_scalars)
    run1_scalars = _truncate_run1_by_run2_min_step(run1_scalars, run2_min)
    run1_max = _max_step(run1_scalars)
    offset = args.offset if args.offset is not None else run1_max + 1

    merged = _iter_merged_points(run1_scalars, run2_scalars, step_offset=offset)
    _write_events(out_dir, merged)

    print(f"run1: {run1_dir}")
    print(f"run2: {run2_dir}")
    print(f"out : {out_dir}")
    print(f"run1 truncated to steps < run2_min_step ({run2_min})")
    print(f"offset applied to run2: {offset}")
    print("Done. Launch TensorBoard with --logdir pointing to the output directory.")


if __name__ == "__main__":
    main()

