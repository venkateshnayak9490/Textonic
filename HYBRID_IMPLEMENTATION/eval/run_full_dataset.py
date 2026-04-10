import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace package imports work when this script is executed directly.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.hybrid.pipeline import HybridPipeline
from src.configs.config import PREFERRED_MODEL, get_model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HybridPipeline over the full test dataset and save answers to JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="climate_hallucination_dataset_test_230_v2.json",
        help="Path to input dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: eval/hybrid_<model>_full_dataset_answers.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model alias or HF model id (e.g., qwen, llama). Defaults to PREFERRED_MODEL in config.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="seq_kg_first",
        choices=["seq_kg_first", "seq_rag_first", "parallel"],
        help="Retrieval ordering: KG first, RAG first, or parallel",
    )
    parser.add_argument(
        "--force-rebuild-text-index",
        action="store_true",
        help="Force rebuild text index instead of loading cached index",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint output every N examples",
    )
    parser.add_argument(
        "--allow-tmp-fallback",
        action="store_true",
        help="If set, fallback to /tmp output path on disk quota errors",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records")
    return data


def _fallback_output_path(original_path: Path) -> Path:
    """Build fallback output path for quota-constrained home/workspace filesystems."""
    fallback_dir = Path(os.environ.get("HYBRID_OUTPUT_FALLBACK_DIR", "/tmp/hybrid_eval_outputs"))
    return fallback_dir / original_path.name


def _human_bytes(value: int) -> str:
    """Render byte counts in readable units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _check_free_space(path: Path, min_free_bytes: int, label: str) -> None:
    """Print available disk space and warn when it is below a threshold."""
    target = path if path.is_dir() else path.parent
    if not target.exists():
        target = target.parent
    probe_path = target
    try:
        usage = shutil.disk_usage(probe_path)
    except FileNotFoundError:
        probe_path = Path("/tmp")
        try:
            usage = shutil.disk_usage(probe_path)
        except FileNotFoundError:
            probe_path = Path("/")
            usage = shutil.disk_usage(probe_path)
    free = usage.free
    print(f"Disk check ({label}) : {probe_path} | free={_human_bytes(free)}")
    if free < min_free_bytes:
        print(
            "WARNING: Low free space for "
            f"{label}. Required >= {_human_bytes(min_free_bytes)}, available {_human_bytes(free)}."
        )


def preflight_space_checks(output_path: Path) -> None:
    """Run startup disk-space checks for output and model cache paths."""
    output_fallback_path = _fallback_output_path(output_path)
    cache_root = Path(
        os.environ.get("HYBRID_CACHE_ROOT")
        or os.environ.get("XDG_CACHE_HOME")
        or "/tmp/hybrid_hf_cache"
    )
    hf_home = Path(os.environ.get("HF_HOME", str(cache_root / "hf_home")))
    transformers_cache = Path(os.environ.get("TRANSFORMERS_CACHE", str(cache_root / "transformers")))
    sentence_cache = Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME", str(cache_root / "sentence_transformers")))

    # Conservative defaults: output writes are smaller; model cache may need many GB.
    min_output_free = int(float(os.environ.get("HYBRID_MIN_OUTPUT_FREE_GB", "0.2")) * (1024 ** 3))
    min_cache_free = int(float(os.environ.get("HYBRID_MIN_CACHE_FREE_GB", "8")) * (1024 ** 3))

    # Create target directories early so filesystem checks operate on real paths.
    for directory in [output_path.parent, output_fallback_path.parent, hf_home, transformers_cache, sentence_cache]:
        directory.mkdir(parents=True, exist_ok=True)

    print("-" * 80)
    print("DISK SPACE PREFLIGHT")
    print("-" * 80)
    _check_free_space(output_path, min_output_free, "primary output")
    _check_free_space(output_fallback_path, min_output_free, "fallback output")
    _check_free_space(hf_home, min_cache_free, "HF_HOME cache")
    _check_free_space(transformers_cache, min_cache_free, "TRANSFORMERS cache")
    _check_free_space(sentence_cache, min_cache_free, "SentenceTransformer cache")
    print("-" * 80)


def save_dataset(path: Path, records: List[Dict[str, Any]], allow_tmp_fallback: bool = False) -> Path:
    """Save dataset and return the path actually written to.

    On disk quota errors, retries under /tmp only if allow_tmp_fallback=True.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=True)
        return path
    except OSError as exc:
        if exc.errno != 122:
            raise

        if not allow_tmp_fallback:
            raise OSError(
                "Disk quota exceeded while saving output to workspace. "
                "Re-run with --allow-tmp-fallback to permit /tmp fallback."
            ) from exc

        fallback_path = _fallback_output_path(path)
        print(
            "WARNING: Disk quota exceeded while saving output at "
            f"{path}. Retrying at {fallback_path}."
        )
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with fallback_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=True)
        return fallback_path


def sanitize_llm_answer(answer: str) -> str:
    """Remove common leaked instruction text from model responses."""
    if not answer:
        return answer

    cleaned = re.sub(r"\s+", " ", answer).strip()

    # Strip wrapper prefixes first, then instruction text that may follow those wrappers.
    for _ in range(3):
        before = cleaned
        cleaned = re.sub(r"^\s*Answer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*Rewritten\s+answer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Do not make up information\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Do not make unsupported claims\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Use proper grammar and spelling\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Use proper grammar and punctuation\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^You are an expert climate science assistant\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Be precise, cite sources, and express uncertainty if evidence is insufficient\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned == before:
            break

    # Remove inlined prompt-following artifacts such as "...Sources: ...Write a response..."
    cleaned = re.sub(r"(Sources:\s*TEXT:\d+,\s*KG:[^\s.,;:]+)\s*Write\s+", r"\1 ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bUse proper grammar and spelling\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bUse proper grammar and punctuation\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAnswer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bWrite a response\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAnswer requirements:?\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*##+\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.strip()


def main() -> None:
    args = parse_args()

    input_path = (WORKSPACE_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    model_alias = (args.model or PREFERRED_MODEL).strip()
    default_output = f"eval/hybrid_{model_alias.lower()}_{args.retrieval_mode}_full_dataset_answers.json"
    output_arg = args.output or default_output
    output_path = (WORKSPACE_ROOT / output_arg).resolve() if not Path(output_arg).is_absolute() else Path(output_arg)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model_id = get_model_id(model_alias)

    print("=" * 80)
    print("HYBRID DATASET RUNNER")
    print("=" * 80)
    print(f"Input file           : {input_path}")
    print(f"Output file          : {output_path}")
    print(f"Model alias          : {model_alias}")
    print(f"Resolved model id    : {model_id}")
    print(f"Use GPU              : {args.use_gpu}")
    print(f"Retrieval mode       : {args.retrieval_mode}")
    print(f"Force text rebuild   : {args.force_rebuild_text_index}")
    print(f"Allow /tmp fallback  : {args.allow_tmp_fallback}")
    print("=" * 80)

    preflight_space_checks(output_path)

    records = load_dataset(input_path)
    total = len(records)
    print(f"Loaded {total} records")

    pipeline = HybridPipeline(
        use_kg=True,
        use_rag=True,
        use_reranker=True,
        model_name=model_alias,
        use_gpu=args.use_gpu,
        force_rebuild_text_index=args.force_rebuild_text_index,
    )

    success = 0
    errors = 0
    active_output_path = output_path

    for i, row in enumerate(records, start=1):
        question = (row.get("question") or "").strip()

        if not question:
            row["llm_answer"] = ""
            row["llm_answer_model"] = model_id
            row["notes"] = (row.get("notes") or "")
            errors += 1
            print(f"[{i}/{total}] Missing question; skipped")
            continue

        try:
            result = pipeline.query(question, retrieval_mode=args.retrieval_mode)
            answer = result.get("answer", "")
            row["llm_answer"] = sanitize_llm_answer(answer)
            row["llm_answer_model"] = model_id
            success += 1
            print(f"[{i}/{total}] OK | id={row.get('id')} | question={question[:70]}")
        except Exception as exc:
            row["llm_answer"] = f"ERROR: {exc}"
            row["llm_answer_model"] = model_id
            errors += 1
            print(f"[{i}/{total}] ERROR | id={row.get('id')} | {exc}")

        if args.save_every > 0 and i % args.save_every == 0:
            active_output_path = save_dataset(
                active_output_path,
                records,
                allow_tmp_fallback=args.allow_tmp_fallback,
            )
            print(f"Checkpoint saved at {i}/{total}: {active_output_path}")

    active_output_path = save_dataset(
        active_output_path,
        records,
        allow_tmp_fallback=args.allow_tmp_fallback,
    )

    print("=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Saved output         : {active_output_path}")
    print(f"Total                : {total}")
    print(f"Successful           : {success}")
    print(f"Errors               : {errors}")


if __name__ == "__main__":
    main()
