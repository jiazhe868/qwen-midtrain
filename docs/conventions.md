# Conventions shared by every module

These are the contracts between the modules. Change one only together with every reader and writer of it.

## Where things live

| what | where |
|---|---|
| code | this repository, `src/midtrain/*.py`, run as `PYTHONPATH=src python -m midtrain.<module> <command>` |
| large data (never committed) | `$MIDTRAIN_DATA` (set in `site.env`; default `./midtrain_data`) |
| text pool | `$MIDTRAIN_DATA/pool/<source>/part-00000.parquet` |
| evaluation items | `$MIDTRAIN_DATA/eval/<task>.jsonl` |
| runs (checkpoints, logs) | `$MIDTRAIN_DATA/runs/<run_name>/` |
| small result files (committed) | `results/*.json` |
| tokenizer | `$MIDTRAIN_TOKENIZER` (a local folder or the hub id `Qwen/Qwen3.5-0.8B-Base`); all four Qwen3.5 sizes share it |
| python | `$MIDTRAIN_PYTHON` (the GPU environment plus scikit-learn and zstandard) |
| machine-specific paths | `site.env` (not committed; copy `site.env.example`), read by `scripts/env.sh` |

## Pool rows (parquet, one row per document)

| column | type | meaning |
|---|---|---|
| `doc_id` | string | `<source>:<first 16 hex of sha1(text)>`; stable across rebuilds |
| `source` | string | the pool source name, e.g. `replay_web`, `wiki_en`, `papers`, `math_web`, `code`, `synthetic_rephrase`, `primitives` |
| `text` | string | the document as trained on (no special tokens) |
| `token_count` | int32 | tokens under the Qwen3.5 tokenizer, without special tokens |
| `meta_json` | string | JSON with whatever the source knows: url, title, license, classifier score, generator, family |

## Evaluation items (JSON lines)

`item_id`, `task`, `kind` (`multiple_choice`, `numeric`, `code` or `short_answer`), `question`, and by kind:
`choices` and `answer_index`; `answer_text`; `answer_aliases`; or `test_code` and `entry_point`. `meta` holds the rest.

## Style

Lowercase snake_case names that say what a thing is, with its unit or split. Every file opens with a docstring: what
it does, why it exists, the measured history that shaped it, and the exact commands that run it. Comments give the
reason and the number behind a choice. Few files, plain functions, loud failures.
