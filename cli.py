"""Command-line interface: `python -m rag_assistant.cli <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from rag_assistant.config import get_settings
from rag_assistant.generation.pipeline import RAGPipeline
from rag_assistant.ingestion.chunker import chunk_documents
from rag_assistant.ingestion.loader import load_directory


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    directory = Path(args.directory) if args.directory else settings.documents_dir

    documents = load_directory(directory)
    if not documents:
        print(f"No supported documents found in {directory}", file=sys.stderr)
        return 1

    chunks = chunk_documents(documents)
    pipeline = RAGPipeline()

    if args.reset:
        pipeline.retriever.store.clear()

    indexed = pipeline.retriever.index(chunks)
    print(
        f"Ingested {len(documents)} document(s) -> {len(chunks)} chunk(s); "
        f"{indexed} indexed, {pipeline.retriever.store.count()} vectors total."
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline()
    if pipeline.retriever.store.count() == 0:
        print("Index is empty. Run `ingest` first.", file=sys.stderr)
        return 1

    result = pipeline.answer(args.question, top_k=args.top_k)

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0

    print(f"\n{result.answer}\n")
    if result.citations:
        print("Sources:")
        for citation in result.citations:
            print(f"  [{citation.marker}] {citation.label}  (score {citation.score})")
    else:
        print("No sources cited -- treat this answer as ungrounded.")
    print(f"\n({result.metadata['latency_ms']} ms, {result.metadata['reason']})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from evals.run_eval import run_evaluation

    report = run_evaluation(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output) if args.output else None,
    )
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_assistant", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load and index documents.")
    ingest.add_argument("-d", "--directory", help="Defaults to DOCUMENTS_DIR.")
    ingest.add_argument("--reset", action="store_true", help="Clear the index first.")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question against the index.")
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", type=int, default=None)
    ask.add_argument("--json", action="store_true", help="Emit the full structured answer.")
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("eval", help="Run the regression evaluation suite.")
    evaluate.add_argument("-d", "--dataset", default="evals/regression_set.json")
    evaluate.add_argument("-o", "--output", default=None, help="Write the JSON report here.")
    evaluate.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
