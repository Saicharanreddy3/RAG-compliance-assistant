"""Run the frozen regression set and gate on aggregate thresholds.

Usage:
    python -m evals.run_eval
    python -m evals.run_eval --output reports/eval.json --fail-under 0.9

Exits non-zero when a threshold is breached, so CI blocks the merge.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evals.metrics import CaseResult, score_case
from rag_assistant.config import get_settings
from rag_assistant.generation import prompts
from rag_assistant.generation.pipeline import RAGPipeline
from rag_assistant.ingestion.chunker import chunk_documents
from rag_assistant.ingestion.loader import load_directory

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "pass_rate": 0.90,
    "retrieval_hit_rate": 0.95,
    "mrr": 0.75,
    "citation_precision": 0.95,
}


@dataclass
class EvalReport:
    dataset_version: str
    prompt_version: str
    timestamp: str
    config: dict
    aggregate: dict
    thresholds: dict
    breaches: list[str]
    cases: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.breaches

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _ensure_index(pipeline: RAGPipeline) -> None:
    """Index the corpus if the store is empty, so the eval is self-contained."""
    if pipeline.retriever.store.count() > 0:
        return
    settings = get_settings()
    documents = load_directory(settings.documents_dir)
    if not documents:
        raise RuntimeError(f"No documents found in {settings.documents_dir}; cannot evaluate.")
    pipeline.retriever.index(chunk_documents(documents))


def _aggregate(results: list[CaseResult]) -> dict:
    def mean(key: str, subset: list[CaseResult] | None = None) -> float:
        pool = subset if subset is not None else results
        values = [r.scores[key] for r in pool]
        return round(statistics.fmean(values), 4) if values else 0.0

    answerable = [r for r in results if r.category != "abstention"]
    abstention = [r for r in results if r.category == "abstention"]
    latencies = sorted(r.latency_ms for r in results)

    return {
        "total_cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 4) if results else 0.0,
        "retrieval_hit_rate": mean("retrieval_hit", answerable),
        "mrr": mean("reciprocal_rank", answerable),
        "keyword_recall": mean("keyword_recall", answerable),
        "citation_precision": mean("citation_precision", answerable),
        "citation_correctness": mean("citation_correctness", answerable),
        "abstention_rate": round(
            sum(r.scores["abstained"] for r in abstention) / len(abstention), 4
        )
        if abstention
        else 0.0,
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 2) if latencies else 0.0,
    }


def run_evaluation(
    dataset_path: Path,
    output_path: Path | None = None,
    thresholds: dict | None = None,
    pipeline: RAGPipeline | None = None,
) -> EvalReport:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    pipeline = pipeline or RAGPipeline()
    _ensure_index(pipeline)

    results: list[CaseResult] = []
    for case in dataset["cases"]:
        answer = pipeline.answer(case["question"])
        results.append(score_case(case, answer))

    aggregate = _aggregate(results)
    breaches = [
        f"{metric}: {aggregate[metric]:.4f} < {minimum:.4f}"
        for metric, minimum in thresholds.items()
        if metric in aggregate and aggregate[metric] < minimum
    ]

    settings = get_settings()
    report = EvalReport(
        dataset_version=dataset.get("dataset_version", "unknown"),
        prompt_version=prompts.PROMPT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config={
            "embedding_provider": settings.embedding.provider,
            "llm_provider": settings.generation.provider,
            "llm_model": getattr(pipeline.llm, "model", "unknown"),
            "vector_backend": settings.retrieval.backend,
            "top_k": settings.retrieval.top_k,
            "chunk_size": settings.chunking.chunk_size,
            "chunk_overlap": settings.chunking.chunk_overlap,
        },
        aggregate=aggregate,
        thresholds=thresholds,
        breaches=breaches,
        cases=[asdict(r) for r in results],
    )

    _print_summary(report, results)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_json(), encoding="utf-8")
        print(f"\nReport written to {output_path}")

    return report


def _print_summary(report: EvalReport, results: list[CaseResult]) -> None:
    agg = report.aggregate
    print(f"\n{'=' * 68}")
    print(f"Regression set {report.dataset_version} | prompt {report.prompt_version}")
    print(f"{report.config['llm_provider']}/{report.config['llm_model']} | "
          f"embeddings={report.config['embedding_provider']} | "
          f"backend={report.config['vector_backend']}")
    print(f"{'=' * 68}")
    print(f"  Cases              {agg['passed']}/{agg['total_cases']} passed "
          f"({agg['pass_rate']:.1%})")
    print(f"  Retrieval hit rate {agg['retrieval_hit_rate']:.1%}")
    print(f"  MRR                {agg['mrr']:.3f}")
    print(f"  Keyword recall     {agg['keyword_recall']:.1%}")
    print(f"  Citation precision {agg['citation_precision']:.1%}")
    print(f"  Abstention rate    {agg['abstention_rate']:.1%}")
    print(f"  Latency p50/p95    {agg['latency_p50_ms']:.1f} / {agg['latency_p95_ms']:.1f} ms")

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  {len(failures)} failing case(s):")
        for result in failures:
            print(f"    [{result.case_id}] {result.question}")
            for failure in result.failures:
                print(f"        - {failure}")
            print(f"        got: {result.answer[:140]}")

    print()
    if report.breaches:
        print("THRESHOLD BREACHED:")
        for breach in report.breaches:
            print(f"  - {breach}")
    else:
        print("All thresholds met.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", default="evals/regression_set.json")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--fail-under", type=float, default=None,
                        help="Override the minimum pass rate.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    overrides = {"pass_rate": args.fail_under} if args.fail_under is not None else None
    report = run_evaluation(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output) if args.output else None,
        thresholds=overrides,
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
