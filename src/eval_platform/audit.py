from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..infra.config import get_config
from .audit_pipeline import (
    _update_human_review_from_csv_row,
    build_sampling_watermark,
    build_human_review_queue,
    build_production_audit_batches,
    build_promotion_candidates,
    build_review_records_from_batch,
    export_review_records_to_csv,
    import_review_csv_rows,
    load_interactions,
    review_csv_fields,
    save_production_audit_batches,
    utc_now_iso,
    yaml_dump,
    yaml_load,
)


def _status(message: str) -> None:
    print(message, flush=True)


def _timestamp_token() -> str:
    ts = utc_now_iso().replace(":", "").replace("-", "")
    return ts


def _default_batch_dir() -> Path:
    return Path(".cache") / "eval" / "production_audit" / "batches" / _timestamp_token()


def _default_run_dir() -> Path:
    return Path(".cache") / "eval" / "production_audit" / "runs" / _timestamp_token()


def _default_step_dir(name: str) -> Path:
    return Path(".cache") / "eval" / "production_audit" / name


def _default_state_path() -> Path:
    return Path(".state") / "eval" / "production_audit" / "sampling_state.json"


def _load_sampling_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_sampling_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production audit closed-loop utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample", help="Sample interactions into audit batch files."
    )
    sample.add_argument("--limit", type=int, default=50)
    sample.add_argument("--days", type=int, default=30)
    sample.add_argument("--out-dir", default=str(_default_batch_dir()))
    sample.add_argument("--state-file", default=str(_default_state_path()))
    sample.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Ignore saved sampling cursor and bootstrap from the date window again.",
    )

    run_latest = subparsers.add_parser(
        "run-latest",
        help="Run sample -> judge -> review-queue for the latest production audit batch.",
    )
    run_latest.add_argument("--limit", type=int, default=50)
    run_latest.add_argument("--days", type=int, default=30)
    run_latest.add_argument("--out-dir", default=str(_default_run_dir()))
    run_latest.add_argument("--state-file", default=str(_default_state_path()))
    run_latest.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Ignore saved sampling cursor and bootstrap from the date window again.",
    )
    run_latest.add_argument("--auto-pass-confidence", type=float, default=0.9)

    judge = subparsers.add_parser("judge", help="Run AI judge on an audit batch.")
    judge.add_argument("--batch", action="append", required=True)
    judge.add_argument("--out-dir", default=str(_default_step_dir("reviews")))

    queue = subparsers.add_parser(
        "review-queue", help="Build a human review queue from review files."
    )
    queue.add_argument("--review", action="append", required=True)
    queue.add_argument("--out-dir", default=str(_default_step_dir("queues")))
    queue.add_argument("--auto-pass-confidence", type=float, default=0.9)

    promote = subparsers.add_parser(
        "promote", help="Export expert promotion candidates from reviewed audit files."
    )
    promote.add_argument("--review", action="append", required=True)
    promote.add_argument("--out-dir", default=str(_default_step_dir("promotions")))

    export_csv = subparsers.add_parser(
        "export-csv",
        help="Export review or queue records to Excel-friendly CSV for human review.",
    )
    export_csv.add_argument("--review", action="append", default=[])
    export_csv.add_argument("--queue", action="append", default=[])
    export_csv.add_argument("--out-dir", default=str(_default_step_dir("csv")))

    import_csv = subparsers.add_parser(
        "import-csv",
        help="Import human-reviewed CSV back into review YAML files.",
    )
    import_csv.add_argument("--csv", action="append", required=True)
    return parser


def _sync_adjacent_queue(review_path: Path, payload: dict) -> None:
    candidate_paths = [
        review_path.parent.parent / "queues" / f"{review_path.stem}.queue.yaml",
        review_path.parent / f"{review_path.stem}.queue.yaml",
    ]
    for queue_path in candidate_paths:
        if queue_path.exists():
            queue = build_human_review_queue(payload)
            yaml_dump(queue, queue_path)


def _cmd_sample(args) -> int:
    cfg = get_config()
    state_path = Path(args.state_file)
    state = {} if args.reset_cursor else _load_sampling_state(state_path)
    _status(
        f"[sample] loading interactions limit={args.limit} days={args.days} store={cfg.interaction_store}"
    )
    rows = load_interactions(
        limit=args.limit,
        days=args.days,
        after_created_at=state.get("last_created_at"),
        after_id=state.get("last_id"),
    )
    _status(f"[sample] loaded_rows={len(rows)}")
    _status("[sample] building audit batches")
    batches = build_production_audit_batches(rows, store_name=cfg.interaction_store)
    saved_paths = save_production_audit_batches(batches, Path(args.out_dir))
    for path in saved_paths:
        _status(f"[sample] wrote {path}")
    watermark = build_sampling_watermark(rows)
    if watermark:
        _save_sampling_state(state_path, watermark)
        _status(f"[sample] state_file={state_path}")
    return 0


def _cmd_run_latest(args) -> int:
    batch_dir = Path(args.out_dir)
    review_dir = batch_dir / "reviews"
    queue_dir = batch_dir / "queues"
    cfg = get_config()
    state_path = Path(args.state_file)
    state = {} if args.reset_cursor else _load_sampling_state(state_path)
    _status(
        f"[run-latest 1/4] loading interactions limit={args.limit} days={args.days} store={cfg.interaction_store}"
    )
    rows = load_interactions(
        limit=args.limit,
        days=args.days,
        after_created_at=state.get("last_created_at"),
        after_id=state.get("last_id"),
    )
    _status(f"[run-latest 1/4] loaded_rows={len(rows)}")
    _status("[run-latest 1/4] building and writing audit batches")
    batches = build_production_audit_batches(rows, store_name=cfg.interaction_store)
    batch_paths = save_production_audit_batches(batches, batch_dir)
    _status(f"[run-latest 1/4] batch_files={len(batch_paths)}")
    watermark = build_sampling_watermark(rows)
    if watermark:
        _save_sampling_state(state_path, watermark)
    review_paths: List[Path] = []
    _status("[run-latest 2/4] running AI judge")
    for index, batch_path in enumerate(batch_paths, start=1):
        _status(
            f"[run-latest 2/4] judging {index}/{len(batch_paths)} {batch_path.name}"
        )
        payload = build_review_records_from_batch(batch_path)
        out_path = review_dir / f"{batch_path.stem}.review.yaml"
        yaml_dump(payload, out_path)
        review_paths.append(out_path)
        _status(f"[run-latest 2/4] wrote {out_path}")
    _status("[run-latest 3/4] building human review queues")
    for index, review_path in enumerate(review_paths, start=1):
        _status(
            f"[run-latest 3/4] queue {index}/{len(review_paths)} {review_path.name}"
        )
        payload = yaml_load(review_path)
        queue = build_human_review_queue(
            payload, max_confidence_auto_pass=args.auto_pass_confidence
        )
        out_path = queue_dir / f"{review_path.stem}.queue.yaml"
        yaml_dump(queue, out_path)
        _status(f"[run-latest 3/4] wrote {out_path}")
    _status("[run-latest 4/4] completed")
    print(f"batch_dir={batch_dir}")
    print(f"review_dir={review_dir}")
    print(f"queue_dir={queue_dir}")
    if watermark:
        print(f"state_file={state_path}")
    return 0


def _cmd_judge(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _status(f"[judge] batch_files={len(args.batch)} out_dir={out_dir}")
    for index, batch_arg in enumerate(args.batch, start=1):
        batch_path = Path(batch_arg)
        _status(f"[judge] processing {index}/{len(args.batch)} {batch_path.name}")
        payload = build_review_records_from_batch(batch_path)
        out_path = out_dir / f"{batch_path.stem}.review.yaml"
        yaml_dump(payload, out_path)
        print(out_path)
    return 0


def _cmd_review_queue(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _status(f"[review-queue] review_files={len(args.review)} out_dir={out_dir}")
    for index, review_arg in enumerate(args.review, start=1):
        review_path = Path(review_arg)
        _status(
            f"[review-queue] processing {index}/{len(args.review)} {review_path.name}"
        )
        payload = yaml_load(review_path)
        queue = build_human_review_queue(
            payload, max_confidence_auto_pass=args.auto_pass_confidence
        )
        queue["source_review_file"] = str(review_path)
        out_path = out_dir / f"{review_path.stem}.queue.yaml"
        yaml_dump(queue, out_path)
        print(out_path)
    return 0


def _cmd_promote(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _status(f"[promote] review_files={len(args.review)} out_dir={out_dir}")
    for index, review_arg in enumerate(args.review, start=1):
        review_path = Path(review_arg)
        _status(f"[promote] processing {index}/{len(args.review)} {review_path.name}")
        payload = yaml_load(review_path)
        grouped = build_promotion_candidates(payload)
        _status(f"[promote] task_groups={len(grouped)} from {review_path.name}")
        exported_at = utc_now_iso()
        task_to_promotion_file: dict[str, str] = {}
        for task, candidate_payload in grouped.items():
            out_path = out_dir / f"{review_path.stem}.{task}.promotion.yaml"
            yaml_dump(candidate_payload, out_path)
            task_to_promotion_file[task] = str(out_path)
            print(out_path)
        if task_to_promotion_file:
            updated_records = []
            for record in payload.get("records") or []:
                human_review = dict(record.get("human_review") or {})
                if (
                    str(human_review.get("status") or "") == "promote_to_expert"
                    and str(record.get("task") or "") in task_to_promotion_file
                ):
                    human_review["resolved_at"] = human_review.get("resolved_at") or exported_at
                    human_review["promotion_exported_at"] = exported_at
                    human_review["promotion_file"] = task_to_promotion_file[str(record.get("task") or "")]
                updated_record = dict(record)
                updated_record["human_review"] = human_review
                updated_records.append(updated_record)
            payload["records"] = updated_records
            yaml_dump(payload, review_path)
            _sync_adjacent_queue(review_path, payload)
    return 0


def _cmd_export_csv(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [("review", Path(path)) for path in args.review] + [
        ("queue", Path(path)) for path in args.queue
    ]
    if not inputs:
        raise SystemExit("At least one --review or --queue path is required.")
    _status(f"[export-csv] files={len(inputs)} out_dir={out_dir}")
    for index, (kind, path) in enumerate(inputs, start=1):
        _status(f"[export-csv] processing {index}/{len(inputs)} {kind} {path.name}")
        payload = yaml_load(path)
        source_review_file = str(payload.get("source_review_file") or "")
        if kind == "review":
            source_review_file = str(path if kind == "review" else "")
        elif not source_review_file:
            raise ValueError(
                f"Queue payload missing source_review_file, cannot export/import safely: {path}"
            )
        out_path = out_dir / f"{path.stem}.csv"
        export_review_records_to_csv(
            payload,
            out_path=out_path,
            source_review_file=source_review_file,
        )
        print(out_path)
    print(f"csv_columns={', '.join(review_csv_fields())}")
    return 0


def _cmd_import_csv(args) -> int:
    updated_reviews: set[str] = set()
    _status(f"[import-csv] csv_files={len(args.csv)}")
    for index, csv_arg in enumerate(args.csv, start=1):
        csv_path = Path(csv_arg)
        _status(f"[import-csv] processing {index}/{len(args.csv)} {csv_path.name}")
        grouped_rows = import_review_csv_rows(csv_path)
        for review_file, rows in grouped_rows.items():
            review_path = Path(review_file)
            _status(
                f"[import-csv] updating {review_path.name} rows={len(rows)}"
            )
            payload = yaml_load(review_path)
            rows_by_id = {
                str(row.get("case_id") or "").strip(): row
                for row in rows
                if str(row.get("case_id") or "").strip()
            }
            updated_records = []
            for record in payload.get("records") or []:
                case_id = str(record.get("id") or "").strip()
                row = rows_by_id.get(case_id)
                if row:
                    updated_record = dict(record)
                    updated_record["human_review"] = _update_human_review_from_csv_row(
                        dict(record.get("human_review") or {}),
                        row,
                    )
                    updated_records.append(updated_record)
                else:
                    updated_records.append(record)
            payload["records"] = updated_records
            yaml_dump(payload, review_path)
            _sync_adjacent_queue(review_path, payload)
            updated_reviews.add(str(review_path))
    for review_path in sorted(updated_reviews):
        print(review_path)
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sample":
        return _cmd_sample(args)
    if args.command == "run-latest":
        return _cmd_run_latest(args)
    if args.command == "judge":
        return _cmd_judge(args)
    if args.command == "review-queue":
        return _cmd_review_queue(args)
    if args.command == "promote":
        return _cmd_promote(args)
    if args.command == "export-csv":
        return _cmd_export_csv(args)
    if args.command == "import-csv":
        return _cmd_import_csv(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
