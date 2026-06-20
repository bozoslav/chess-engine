#!/usr/bin/env python3

import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO


UCI_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$")


def normalize_fen(fen: str, append_counters: bool = True) -> str:
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError("FEN must contain at least placement, side, castling, and ep fields")
    if append_counters and len(fields) == 4:
        fields.extend(["0", "1"])
    return " ".join(fields[:6])


def side_to_move(fen: str) -> str:
    fields = fen.split()
    if len(fields) < 2 or fields[1] not in ("w", "b"):
        raise ValueError(f"bad side-to-move in FEN: {fen}")
    return fields[1]


def convert_score_pov(score: int, fen: str, input_pov: str, target_pov: str) -> int:
    if input_pov == target_pov:
        return score
    if {input_pov, target_pov} != {"white", "side-to-move"}:
        raise ValueError(f"unsupported score POV conversion: {input_pov} -> {target_pov}")
    return score if side_to_move(fen) == "w" else -score


def stable_sample_keep(key: str, rate: float, seed: str) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.blake2b(f"{seed}\0{key}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value < int(rate * ((1 << 64) - 1))


@contextlib.contextmanager
def open_text_input(path: Path) -> Iterator[TextIO]:
    if str(path) == "-":
        yield sys.stdin
        return

    if path.suffix == ".zst":
        process = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None:
            raise RuntimeError("failed to open zstd stdout")

        text = io.TextIOWrapper(process.stdout, encoding="utf-8")
        try:
            yield text
        finally:
            text.close()
            _, stderr = process.communicate()
            message = stderr.decode("utf-8", errors="replace").strip()
            broken_pipe = process.returncode in (141, -13) or "Broken pipe" in message
            if process.returncode != 0 and not broken_pipe:
                raise RuntimeError(f"zstd failed for {path}: {message}")
        return

    with path.open("r", encoding="utf-8") as source:
        yield source


def integer_field(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def first_pv(eval_record: dict[str, Any]) -> dict[str, Any] | None:
    pvs = eval_record.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        return None
    pv = pvs[0]
    return pv if isinstance(pv, dict) else None


def select_eval(record: dict[str, Any]) -> dict[str, Any] | None:
    evals = record.get("evals")
    if isinstance(evals, list) and evals:
        candidates = [entry for entry in evals if isinstance(entry, dict)]
    else:
        candidates = [record]

    candidates = [entry for entry in candidates if first_pv(entry) is not None]
    if not candidates:
        return None

    def strength(entry: dict[str, Any]) -> tuple[int, int]:
        depth = integer_field(entry, "depth")
        knodes = integer_field(entry, "knodes")
        return (depth if depth is not None else -1, knodes if knodes is not None else -1)

    return max(candidates, key=strength)


def moves_from_pv(pv: dict[str, Any]) -> list[str]:
    moves = pv.get("moves")
    if moves is None:
        moves = pv.get("line")
    if not isinstance(moves, str):
        return []
    return [move for move in moves.split() if move]


def normalize_standard_castling(move: str) -> str:
    # Lichess evaluated-position PVs can encode standard castling as king-to-rook
    # moves. Stockfish's training parser expects king destinations on files c/g.
    if move == "e1h1":
        return "e1g1"
    if move == "e1a1":
        return "e1c1"
    if move == "e8h8":
        return "e8g8"
    if move == "e8a8":
        return "e8c8"
    return move


def select_move(pv: dict[str, Any]) -> str | None:
    moves = moves_from_pv(pv)
    if not moves:
        return None
    move = normalize_standard_castling(moves[0].lower())
    return move if UCI_MOVE_RE.fullmatch(move) is not None else None


def ply_from_fen(fen: str, default_ply: int) -> int:
    fields = fen.split()
    if len(fields) < 6:
        return default_ply
    try:
        fullmove = max(1, int(fields[5]))
    except ValueError:
        return default_ply
    return max(0, 2 * (fullmove - 1) + (1 if fields[1] == "b" else 0))


def select_score(pv: dict[str, Any], fen: str, args: argparse.Namespace) -> int | None:
    cp = integer_field(pv, "cp")
    mate = integer_field(pv, "mate")
    if cp is None and mate is None:
        return None
    if cp is not None and args.max_abs_cp > 0 and abs(cp) > args.max_abs_cp:
        return None
    if mate is not None and cp is None:
        if args.skip_mates:
            return None
        sign = 1 if mate > 0 else -1
        cp = sign * max(0, args.mate_score - min(abs(mate), args.mate_score))
    assert cp is not None
    cp = convert_score_pov(cp, fen, args.input_score_pov, "side-to-move")
    return max(-args.score_clip, min(args.score_clip, cp))


def write_entry(target: TextIO, fen: str, move: str, score: int, ply: int, result: int) -> None:
    target.write(f"fen {fen}\n")
    target.write(f"move {move}\n")
    target.write(f"score {score}\n")
    target.write(f"ply {ply}\n")
    target.write(f"result {result}\n")
    target.write("e\n")


def convert(args: argparse.Namespace) -> dict[str, int]:
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"{args.output} already exists; pass --overwrite")

    counts: dict[str, int] = {
        "read": 0,
        "written": 0,
        "blank": 0,
        "json": 0,
        "sampled_out": 0,
        "bad_fen": 0,
        "missing_eval": 0,
        "missing_pv": 0,
        "missing_move": 0,
        "missing_score": 0,
        "depth": 0,
        "missing_depth": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_text_input(args.input) as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if args.limit > 0 and counts["written"] >= args.limit:
                break

            line = line.strip()
            if not line:
                counts["blank"] += 1
                continue

            counts["read"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["json"] += 1
                continue
            if not isinstance(record, dict):
                counts["json"] += 1
                continue

            raw_fen = record.get("fen")
            if not isinstance(raw_fen, str) or not raw_fen.strip():
                counts["bad_fen"] += 1
                continue
            try:
                fen = normalize_fen(raw_fen, append_counters=not args.keep_four_field_fen)
            except ValueError:
                counts["bad_fen"] += 1
                continue

            if not stable_sample_keep(fen, args.sample_rate, args.seed):
                counts["sampled_out"] += 1
                continue

            selected = select_eval(record)
            if selected is None:
                counts["missing_eval"] += 1
                continue

            depth = integer_field(selected, "depth")
            if depth is None:
                if args.require_depth:
                    counts["missing_depth"] += 1
                    continue
            elif depth < args.min_depth:
                counts["depth"] += 1
                continue

            pv = first_pv(selected)
            if pv is None:
                counts["missing_pv"] += 1
                continue

            move = select_move(pv)
            if move is None:
                counts["missing_move"] += 1
                continue

            score = select_score(pv, fen, args)
            if score is None:
                counts["missing_score"] += 1
                continue

            ply = args.default_ply if args.force_default_ply else ply_from_fen(fen, args.default_ply)
            ply = max(0, min(args.max_ply, ply))
            write_entry(target, fen, move, score, ply, args.default_result)
            counts["written"] += 1

            if args.report_every > 0 and counts["written"] % args.report_every == 0:
                print(
                    f"wrote {counts['written']} entries after reading {counts['read']} "
                    f"(line={line_number})",
                    flush=True,
                )

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the Lichess evaluated-position JSONL(.zst) dump into "
            "Stockfish nnue-pytorch plain training format."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--min-depth", type=int, default=0)
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--max-abs-cp", type=int, default=0)
    parser.add_argument(
        "--input-score-pov",
        choices=("white", "side-to-move"),
        default="white",
        help="POV used by Lichess cp/mate values. The output is always side-to-move.",
    )
    parser.add_argument("--default-ply", type=int, default=30)
    parser.add_argument("--force-default-ply", action="store_true")
    parser.add_argument("--max-ply", type=int, default=16383)
    parser.add_argument("--score-clip", type=int, default=32000)
    parser.add_argument("--mate-score", type=int, default=30000)
    parser.add_argument("--skip-mates", action="store_true", default=True)
    parser.add_argument("--keep-mates", dest="skip_mates", action="store_false")
    parser.add_argument("--default-result", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--keep-four-field-fen", action="store_true")
    parser.add_argument("--report-every", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if not 0.0 <= args.sample_rate <= 1.0:
        parser.error("--sample-rate must be in [0, 1]")
    if args.min_depth < 0 or args.max_abs_cp < 0:
        parser.error("--min-depth and --max-abs-cp cannot be negative")
    if args.default_ply < 0 or args.max_ply < 0:
        parser.error("ply values cannot be negative")
    if args.default_ply > args.max_ply:
        parser.error("--default-ply cannot exceed --max-ply")
    if args.score_clip <= 0 or args.score_clip > 32767:
        parser.error("--score-clip must be in [1, 32767]")
    if args.mate_score <= 0 or args.mate_score > args.score_clip:
        parser.error("--mate-score must be positive and <= --score-clip")
    if args.report_every < 0:
        parser.error("--report-every cannot be negative")

    counts = convert(args)
    skipped = sum(value for key, value in counts.items() if key not in ("read", "written"))
    print(
        f"wrote {counts['written']} entries to {args.output} "
        f"(read={counts['read']} skipped={skipped})"
    )
    for key in sorted(counts):
        if key not in ("read", "written") and counts[key]:
            print(f"skip_{key}={counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
