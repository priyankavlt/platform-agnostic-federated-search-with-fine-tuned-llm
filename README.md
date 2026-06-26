
# Platform-Agnostic Federated Search with Fine-Tuned LLM

Ask one natural-language question; get a ranked, cited answer drawn from **several unrelated systems** —
without knowing which system holds the data.

**Stack:** Qwen3.8-27B · QLoRA (Unsloth) · vLLM · PydanticAI · Airbyte-protocol connectors · Kafka

```
           ┌──────────── sources (heterogeneous) ────────────┐
           │ CRM (Postgres)  Support (REST)  Docs (MD)  Chat │  + any Airbyte source via PyAirbyte
           └───────┬──────────────┬────────────┬─────────┬───┘
                   │  Airbyte-protocol RECORDs (connectors/)  │
                   ▼                                          ▼
         Kafka topics  fedsearch.raw.<source>.<stream>   (streaming/bus.py)
                   │
                   ▼
   indexer: normalise (sources.yaml) → entity linking → hybrid BM25+dense index
                   │                     (CRM teaches "Globex" = globex.com = CUST-0001)
                   ▼
 query ─► Planner (fine-tuned Qwen3.8 on vLLM, PydanticAI → QueryPlan schema)
                   │   intent · target systems · canonical entities · rewritten query
                   ▼
          Federated executor: per-system hybrid retrieval + entity-anchored retrieval
                   + cross-reference expansion (ticket ids → chat threads) → RRF fusion
                   ▼
          Answerer (PydanticAI → GroundedAnswer; validator rejects citations whose quote
                    isn't verbatim in the cited doc and re-asks the model)
```

### Why the data is hard on purpose
The synthetic company (`seed/world.py`) is fragmented like real enterprise data: support tickets carry a
`customer_id` but never the company name; chat messages mention ticket ids but not the customer or product.
"*Is Globex still having trouble with SSO?*" can only be answered by resolving Globex in the CRM,
finding that customer's SSO tickets, then following those ticket ids into chat. Test queries include
typos ("Glboex"), nicknames ("okta login" → Single Sign-On) and phrasings never seen in training.

## Quickstart (laptop, no GPU)

```bash
uv sync
uv run fedsearch seed                       # 40 customers, 105 tickets, 18 docs, 157 chats, 1.4k labelled queries
uv run fedsearch mock-support-api &         # the "support" system is a real REST API on :8081
uv run fedsearch ingest                     # connectors → bus → index (in-memory bus)
uv run fedsearch search "What problems has Globex had with SSO?"
uv run fedsearch eval                       # planner / retrieval / answer metrics → artifacts/eval/
uv run pytest
```

### With Kafka + Postgres
```bash
docker compose up -d
export FEDSEARCH_BUS=kafka FEDSEARCH_CRM_URL=postgresql+psycopg://crm:crm@localhost:5432/crm
uv run fedsearch seed --sql-url $FEDSEARCH_CRM_URL
uv run fedsearch extract    # producers: each source → its own Kafka topic
uv run fedsearch index      # consumer group: topics → index (re-run any time; incremental)
uv run fedsearch serve      # POST /search {"query": "..."}
```

### Adding a source
Add a block to `configs/sources.yaml`: connector (`sql`, `http`, `files`, or `airbyte` for any of Airbyte's
source connectors via PyAirbyte), streams, and how to map fields to title/text/entities. No code changes.
In production you can run the Airbyte platform with its Kafka destination writing to the same topics.

## Fine-tuning the planner (Google Colab — no local GPU needed)

The model's job is structured: emit a schema-valid `QueryPlan`. The SFT data is generated from the same
synthetic world with gold plans, and the prompts are imported from the serving code so training and
serving can't drift apart.

[![Fine-tune in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyankavlt/platform-agnostic-federated-search-with-fine-tuned-llm/blob/main/notebooks/finetune_planner_colab.ipynb) **fine-tune** &nbsp;
[![Serve in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyankavlt/platform-agnostic-federated-search-with-fine-tuned-llm/blob/main/notebooks/serve_vllm_colab.ipynb) **serve**

1. Click **Fine-tune in Colab** above.
2. **Runtime → Change runtime type → A100** (Colab Pro). A free T4 runs a `qwen3.5-4b-smoke` profile
   automatically so you can test the pipeline first.
3. Optional: add an `HF_TOKEN` Colab secret (🔑) to push the adapter to the Hugging Face Hub.
4. **Run all.** The notebook generates the data, loads `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`, scores the
   **base** model on the held-out test questions, runs QLoRA (LoRA r=16 on the language layers, loss on the
   assistant turn only), scores the **fine-tuned** model on the same questions, shows a before/after table
   broken down by intent and seen vs. unseen phrasing, and saves the adapter + report to Google Drive (and
   optionally the HF Hub).

To use the model from the search engine, open
[`notebooks/serve_vllm_colab.ipynb`](notebooks/serve_vllm_colab.ipynb). It serves the 4-bit base + adapter with
vLLM (fits an A100 40 GB), checks a schema-constrained `QueryPlan` response, and opens a temporary
Cloudflare tunnel. Then on your laptop:

```bash
export FEDSEARCH_PLANNER=llm FEDSEARCH_ANSWERER=llm
export FEDSEARCH_LLM_BASE_URL=https://<tunnel>.trycloudflare.com/v1
uv run fedsearch eval                                   # fine-tuned
FEDSEARCH_LLM_MODEL=base uv run fedsearch eval          # untuned base model, same server
FEDSEARCH_PLANNER=rules uv run fedsearch eval           # deterministic baseline
```

The same training code (`training/sft_lib.py`) also runs from the CLI on any CUDA box:
`python training/train_qlora.py --profile qwen3.8-27b-qlora`. Profiles live in `configs/train_qlora.yaml`.

Structured output defaults to `native` (JSON-schema `response_format`, enforced by vLLM guided decoding);
`tool` and `prompted` modes exist for servers without it. PydanticAI validates every response and retries on
schema or citation failures.

## Evaluation

| layer | metrics |
|---|---|
| planner | schema validity, intent accuracy, exact-plan accuracy, source & entity precision/recall/F1 |
| retrieval | Recall@K, Precision@K, MRR, nDCG@K — end-to-end *and* with the gold plan (isolates retrieval from planning) |
| answers | citation validity (no invented doc ids), quote support (verbatim), citation precision, groundedness, fact recall |

Current numbers on the 577-query test split (479 unseen phrasings + 98 seen), no LLM (rules planner + extractive answers), hashing embedder:

| | value |
|---|---|
| planner exact-plan accuracy | 0.809 |
| planner source precision / entity precision | 0.97 / 0.98 |
| retrieval MRR / nDCG@10 / Recall@10 | 0.897 / 0.869 / 0.890 |
| citation validity / quote support / groundedness | 1.0 / 1.0 / 1.0 |

The rules baseline fails mainly on typos, nicknames and unseen phrasings — exactly what the fine-tuned
planner is for. Run the GPU section to fill in base-vs-fine-tuned numbers.

## Layout
```
src/fedsearch/
  connectors/   sql · http · files · airbyte (PyAirbyte)      → Airbyte-protocol Records
  streaming/    Kafka bus (+ in-memory bus with same semantics for tests)
  ingest/       normalisation, entity registry & linking, pipeline
  index/        hashing / fastembed embedders, hybrid BM25+dense store
  search/       planners (LLM, rules), federated executor, grounded answerer, prompts
  eval/         ranking metrics, planner accuracy, answer groundedness, harness
  seed/         synthetic multi-system world + labelled queries
training/       sft_lib.py (shared) · build_sft.py · train_qlora.py · merge_adapter.py · serve_vllm.sh
notebooks/      finetune_planner_colab.ipynb · serve_vllm_colab.ipynb
configs/        sources.yaml · train_qlora.yaml
```
