"""Compare pinned target/draft tokenizer IDs and chat-template behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer


MODELS = (
    (
        "target",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    ),
    (
        "draft",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
    ),
)
SYNTHETIC_MESSAGES = (
    [{"role": "user", "content": "Return JSON for a synthetic task: value=3.5."}],
    [
        {"role": "system", "content": "Return only JSON."},
        {"role": "user", "content": "The patient denies symptom X. Cite sentence 0."},
    ],
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_model(role: str, model_id: str, revision: str) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    vocabulary = tokenizer.get_vocab()
    added_vocabulary = tokenizer.get_added_vocab()
    special_tokens = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "all_special_ids": sorted(tokenizer.all_special_ids),
    }
    encodings = []
    for messages in SYNTHETIC_MESSAGES:
        token_ids = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        encodings.append(list(token_ids))
    config_summary = {
        "model_type": config.model_type,
        "architectures": config.architectures,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "torch_dtype": str(config.torch_dtype),
    }
    return {
        "role": role,
        "model_id": model_id,
        "revision": revision,
        "config": config_summary,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(vocabulary),
        "vocabulary_sha256": _stable_hash(vocabulary),
        "added_vocabulary_size": len(added_vocabulary),
        "added_vocabulary_sha256": _stable_hash(added_vocabulary),
        "special_tokens": special_tokens,
        "chat_template_sha256": _stable_hash(tokenizer.chat_template),
        "synthetic_prompt_ids": encodings,
        "synthetic_prompt_ids_sha256": _stable_hash(encodings),
    }


def compare_tokenizers(models: list[dict[str, Any]]) -> dict[str, Any]:
    target, draft = models
    checks = {
        "vocabulary_size_equal": target["vocab_size"] == draft["vocab_size"],
        "vocabulary_mapping_equal": (
            target["vocabulary_sha256"] == draft["vocabulary_sha256"]
        ),
        "added_vocabulary_equal": (
            target["added_vocabulary_size"] == draft["added_vocabulary_size"]
            and target["added_vocabulary_sha256"]
            == draft["added_vocabulary_sha256"]
        ),
        "special_token_ids_equal": target["special_tokens"] == draft["special_tokens"],
        "chat_templates_equal": (
            target["chat_template_sha256"] == draft["chat_template_sha256"]
        ),
        "synthetic_prompt_ids_equal": (
            target["synthetic_prompt_ids"] == draft["synthetic_prompt_ids"]
        ),
    }
    return {
        "models": models,
        "checks": checks,
        "compatible_for_shared_token_ids": all(
            checks[name]
            for name in (
                "vocabulary_size_equal",
                "vocabulary_mapping_equal",
                "added_vocabulary_equal",
                "special_token_ids_equal",
                "synthetic_prompt_ids_equal",
            )
        ),
        "same_chat_template": checks["chat_templates_equal"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = [
        inspect_model(role, model_id, revision)
        for role, model_id, revision in MODELS
    ]
    report = compare_tokenizers(models)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote tokenizer comparison to {args.output.resolve()}")
    else:
        print(rendered)
    for name, passed in report["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if report["compatible_for_shared_token_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
