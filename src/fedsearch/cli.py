"""Command-line entry point: ``fedsearch --help``."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from fedsearch.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("-v")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def seed(sql_url: Annotated[str | None, typer.Option(help="CRM database URL (default: SQLite in data dir)")] = None,
         seed: int = 7) -> None:
    """Generate the synthetic multi-system company and labelled queries."""
    from fedsearch.seed.write import write_all

    s = get_settings()
    console.print_json(data=write_all(Path(s.data_dir), sql_url, seed))


@app.command()
def ingest(only: Annotated[list[str] | None, typer.Option(help="Restrict to these sources")] = None) -> None:
    """Extract every source, stream records through the bus (Kafka or memory), and build the index."""
    from fedsearch.ingest.pipeline import ingest as run
    from fedsearch.streaming.bus import make_bus

    s = get_settings()
    counts, idx = run(s, make_bus(s), only)
    console.print(f"[bold]bus[/bold]={s.bus} extracted={counts} indexed_total={len(idx.docs)} -> {s.index_dir}")


@app.command()
def extract(only: Annotated[list[str] | None, typer.Option()] = None) -> None:
    """Producer half only: connectors -> Kafka topics (use with ``index`` running elsewhere)."""
    from fedsearch.connectors.base import load_sources_config
    from fedsearch.ingest.pipeline import extract as run
    from fedsearch.streaming.bus import make_bus

    s = get_settings()
    console.print(run(make_bus(s), load_sources_config(Path(s.sources_config)), only,
                      Path(s.index_dir) / "connector_state.json"))


@app.command()
def index(idle_timeout: float = 10.0) -> None:
    """Consumer half only: Kafka topics -> normalise -> link entities -> hybrid index."""
    from fedsearch.connectors.base import load_sources_config
    from fedsearch.ingest.pipeline import index_from_bus
    from fedsearch.streaming.bus import make_bus

    s = get_settings()
    idx = index_from_bus(make_bus(s), load_sources_config(Path(s.sources_config)), s, idle_timeout)
    console.print(f"indexed_total={len(idx.docs)}")


@app.command()
def search(query: str, k: Annotated[int, typer.Option("-k", "--k")] = 8, as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Ask a question across all systems."""
    from fedsearch.search.engine import SearchEngine

    resp = asyncio.run(SearchEngine.from_settings(get_settings()).search(query, k))
    if as_json:
        console.print_json(resp.model_dump_json())
        return
    console.print(f"[bold]plan[/bold] {escape(resp.plan.model_dump_json())}")
    t = Table("#", "source", "doc", "score", "why")
    for i, h in enumerate(resp.hits, 1):
        t.add_row(str(i), h.doc.source, f"{h.doc.id}\n[dim]{h.doc.title}[/dim]", f"{h.score:.4f}", ", ".join(h.reasons))
    console.print(t)
    if resp.answer:
        console.print(f"[bold]answer[/bold] {escape(resp.answer.answer)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the search HTTP API."""
    import uvicorn

    uvicorn.run("fedsearch.api:app", host=host, port=port)


@app.command("mock-support-api")
def mock_support_api(port: int = 8081) -> None:
    """Serve seeded tickets as a Zendesk-style REST API (the 'support' system)."""
    import uvicorn

    uvicorn.run("fedsearch.mocks.support_api:app", host="127.0.0.1", port=port)


@app.command("eval")
def eval_cmd(split: str = "test", limit: Annotated[int | None, typer.Option()] = None, k: int = 10,
             answers: bool = True, out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Evaluate planner accuracy, retrieval (Recall/Precision@K, MRR, nDCG) and answer groundedness."""
    from fedsearch.eval.run import evaluate, load_split
    from fedsearch.search.engine import SearchEngine

    s = get_settings()
    examples = load_split(Path(s.data_dir) / "queries" / f"{split}.jsonl", limit)
    report = asyncio.run(evaluate(SearchEngine.from_settings(s), examples, k, answers))
    report["config"] = {"planner": s.planner, "answerer": s.answerer, "llm_model": s.llm_model,
                        "embedder": s.embedder, "split": split, "n": len(examples)}
    out = out or Path("artifacts/eval") / f"{split}-{s.planner}-{s.answerer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    o = report["overall"]
    t = Table("layer", "metric", "value")
    for layer in ("planner", "retrieval", "retrieval_oracle_plan", "answers"):
        for m, v in o.get(layer, {}).items():
            if layer.startswith("retrieval") and not any(m.startswith(p) for p in ("mrr", "recall@", "precision@5", "ndcg@10")):
                continue
            t.add_row(layer, m, f"{v}")
    console.print(t)
    console.print(f"full report -> {out}")


if __name__ == "__main__":
    app()
