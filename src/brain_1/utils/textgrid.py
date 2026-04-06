from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_BAD_TOKENS = frozenset(
    {
        "",
        "sp",
        "br",
        "lg",
        "ls",
        "ns",
        "{br}",
        "{lg}",
        "{ls}",
        "{ns}",
        "{sp}",
        "sentence_start",
        "sentence_end",
    }
)


@dataclass(slots=True)
class Interval:
    start: float
    stop: float
    text: str


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('""', '"')


def parse_interval_tier(path: str | Path, tier_name: str) -> list[Interval]:
    target_name = tier_name.lower()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    intervals: list[Interval] = []
    current_name: str | None = None
    current_interval: dict[str, float | str] | None = None

    def flush() -> None:
        nonlocal current_interval
        if current_name != target_name or current_interval is None:
            current_interval = None
            return
        if {"start", "stop", "text"} <= set(current_interval):
            intervals.append(
                Interval(
                    start=float(current_interval["start"]),
                    stop=float(current_interval["stop"]),
                    text=str(current_interval["text"]),
                )
            )
        current_interval = None

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("item ["):
            flush()
            current_name = None
            continue
        if line.startswith("name = "):
            flush()
            current_name = _unquote(line.split("=", 1)[1])
            continue
        if current_name != target_name:
            continue
        if line.startswith("intervals ["):
            flush()
            current_interval = {}
            continue
        if current_interval is None:
            continue
        if line.startswith("xmin = ") and "start" not in current_interval:
            current_interval["start"] = float(line.split("=", 1)[1])
        elif line.startswith("xmax = ") and "stop" not in current_interval:
            current_interval["stop"] = float(line.split("=", 1)[1])
        elif line.startswith("text = "):
            current_interval["text"] = _unquote(line.split("=", 1)[1]).strip().lower()

    flush()
    return intervals


def words_per_tr(
    intervals: list[Interval],
    tr_count: int,
    tr_seconds: float,
    bad_tokens: frozenset[str] = DEFAULT_BAD_TOKENS,
) -> list[str]:
    buckets: list[list[str]] = [[] for _ in range(tr_count)]
    for interval in intervals:
        token = interval.text.strip().lower()
        if token in bad_tokens:
            continue
        tr_index = int(interval.start // tr_seconds)
        if 0 <= tr_index < tr_count:
            buckets[tr_index].append(token)
    return [" ".join(bucket) for bucket in buckets]
