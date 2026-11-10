"""Phase 4 supervised LoRA trainer with resumable checkpoints."""
from __future__ import annotations

import copy
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..data_access import load_examples
from ..decoding import cached_library_greedy
from ..output_parser import parse_output
from ..prompts import build_prompt
from .checkpoints import (
    environment_report,
    hash_directory,
    input_hashes,
    load_resume_state,
    make_bundle,
    prune_resume_bundles,
    publish_bundle,
    resolve_resume_dir,
    save_resume_state,
    verify_manifest_sidecar,
)
from .data import (
    EncodedExample,
    build_training_set,
    collate_encoded,
    order_for_epoch,
    split_counts,
)
from .validation import predict_validation


RESUME_IDENTITY_KEYS = (
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "prompt_revision",
    "max_total_tokens",
    "max_new_tokens",
    "max_seq_len",
    "seed",
    "precision",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "max_epochs",
    "lora",
    "optimizer",
    "scheduler",
    "max_grad_norm",
)


class Trainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        root: Path,
        *,
        run_name: str | None = None,
        tiny_model: bool = False,
        drive_dir: Path | None = None,
    ) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.root = Path(root)
        self.run_name = run_name or self.cfg["run_name"]
        self.tiny_model = tiny_model
        self.device = torch.device(self.cfg["device"])
        self.drive_dir = Path(drive_dir) if drive_dir else (
            Path(self.cfg["checkpoint"]["drive_root"])
            if self.cfg["checkpoint"].get("drive_root")
            else None
        )
        self.results_dir = self.root / self.cfg["output"]["results_root"] / self.run_name
        self.checkpoint_root = self.root / self.cfg["checkpoint"]["local_root"] / self.run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "train_log.jsonl"
        self.tokenizer: Any = None
        self.model: Any = None
        self.train_rows: list[dict[str, Any]] = []
        self.validation_rows: list[dict[str, Any]] = []
        self.train_examples: list[EncodedExample] = []
        self.validation_examples: list[EncodedExample] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.scaler: Any = None
        self._overfit_requested = False
        self._planned_max_steps: int | None = None
        self.state: dict[str, Any] = {}
        self.last_checkpoint_time = time.perf_counter()

    # ------------------------------------------------------------------ setup
    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["tokenizer_id"],
            revision=self.cfg["tokenizer_revision"],
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an end-of-sequence token")
        return tokenizer

    def _seed_everything(self) -> None:
        seed = int(self.cfg["seed"])
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self) -> Any:
        if self.tiny_model:
            from transformers import Qwen2Config, Qwen2ForCausalLM

            config = Qwen2Config(
                vocab_size=len(self.tokenizer),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=max(2048, self.cfg["max_seq_len"]),
                tie_word_embeddings=True,
                rms_norm_eps=1e-6,
            )
            return Qwen2ForCausalLM(config)
        from transformers import AutoModelForCausalLM

        dtype = (
            torch.float16
            if self.cfg["precision"] == "float16" and self.device.type == "cuda"
            else torch.float32
        )
        return AutoModelForCausalLM.from_pretrained(
            self.cfg["model_id"],
            revision=self.cfg["model_revision"],
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
        )

    def _wrap_lora(self, model: Any) -> Any:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = self.cfg["lora"]
        if self.cfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.config.use_cache = bool(self.cfg.get("use_cache_during_training", False))
        wrapped = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["lora_alpha"]),
                lora_dropout=float(lora["lora_dropout"]),
                target_modules=list(lora["target_modules"]),
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        trainable_names = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
        if not trainable_names:
            raise RuntimeError("LoRA produced no trainable parameters")
        unexpected = [name for name in trainable_names if "lora_" not in name]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:5]}")
        return wrapped

    def _prepare_data(self) -> None:
        manifest_hash = verify_manifest_sidecar(self.root, self.cfg)
        manifest = json.loads((self.root / self.cfg["split_manifest"]).read_text(encoding="utf-8"))
        self.train_rows = load_examples(self.root, "train")
        self.validation_rows = load_examples(self.root, "validation")
        approved = set(manifest.get("excluded_training_annotation_ids", []))
        expected_train = set(manifest["annotation_ids"]["train"]) - approved
        observed_train = {row["provenance"]["annotation_id"] for row in self.train_rows}
        if observed_train != expected_train or len(self.train_rows) != len(expected_train):
            raise ValueError("Training rows do not match the approved 602-row manifest set")
        if len(self.validation_rows) != len(manifest["annotation_ids"]["validation"]):
            raise ValueError("Validation row count does not match the manifest")
        train_patients = {row["provenance"]["patient_id"] for row in self.train_rows}
        validation_patients = {row["provenance"]["patient_id"] for row in self.validation_rows}
        if train_patients & validation_patients:
            raise ValueError("Patient leakage between training and validation")
        self.train_examples = build_training_set(
            self.tokenizer,
            self.train_rows,
            max_total_tokens=self.cfg["max_total_tokens"],
            max_new_tokens=self.cfg["max_new_tokens"],
            max_seq_len=self.cfg["max_seq_len"],
        )
        self.validation_examples = build_training_set(
            self.tokenizer,
            self.validation_rows,
            max_total_tokens=self.cfg["max_total_tokens"],
            max_new_tokens=self.cfg["max_new_tokens"],
            max_seq_len=self.cfg["max_seq_len"],
        )
        audit = {
            "run_name": self.run_name,
            "manifest_sha256": manifest_hash,
            "train_rows": len(self.train_rows),
            "validation_rows": len(self.validation_rows),
            "train_examples": len(self.train_examples),
            "validation_examples": len(self.validation_examples),
            "train_counts_by_type_label": split_counts(self.train_rows),
            "validation_counts_by_type_label": split_counts(self.validation_rows),
            "train_patient_count": len(train_patients),
            "validation_patient_count": len(validation_patients),
            "patient_splits_disjoint": True,
            "test_data_opened": False,
            "train_prompt_truncated": sum(example.prompt_truncated for example in self.train_examples),
            "validation_prompt_truncated": sum(
                example.prompt_truncated for example in self.validation_examples
            ),
            "train_target_tokens": sum(example.target_token_count for example in self.train_examples),
        }
        (self.results_dir / "data_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _steps_per_epoch(self) -> int:
        accumulation = int(self.cfg["gradient_accumulation_steps"])
        return math.ceil(len(self.train_examples) / accumulation)

    def _overfit_epochs_needed(self, max_steps: int) -> int:
        steps_per_epoch = max(1, self._steps_per_epoch())
        return max(int(self.cfg["max_epochs"]), math.ceil(max_steps / steps_per_epoch) + 2)

    def _planned_total_steps(self) -> int:
        if self._overfit_requested:
            return max(1, self._planned_max_steps or 40)
        return max(1, self._steps_per_epoch() * int(self.cfg["max_epochs"]))

    def _build_optimizer(self) -> None:
        from transformers import get_linear_schedule_with_warmup

        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer_cfg = self.cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(optimizer_cfg["learning_rate"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_cfg["betas"]),
        )
        total_steps = self._planned_total_steps()
        warmup_steps = max(1, int(total_steps * float(self.cfg["scheduler"]["warmup_ratio"])))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        use_scaler = self.cfg["precision"] == "float16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # -------------------------------------------------------------- run modes
    def run(
        self,
        *,
        resume: Path | None = None,
        max_steps: int | None = None,
        stop_after_steps: int | None = None,
        skip_validation: bool = False,
        overfit: bool = False,
        overfit_examples: int = 12,
    ) -> dict[str, Any]:
        from peft import PeftModel

        self._seed_everything()
        self.tokenizer = self._load_tokenizer()
        self._prepare_data()
        if overfit:
            self._restrict_to_overfit(overfit_examples)
        self._overfit_requested = overfit
        self._planned_max_steps = max_steps
        planned_limit = max_steps if max_steps is not None else (40 if overfit else None)
        stop_limit = planned_limit
        if stop_after_steps is not None:
            stop_limit = (
                stop_after_steps
                if planned_limit is None
                else min(planned_limit, stop_after_steps)
            )
        self.model = self._load_model()

        resume_dir: Path | None = None
        if resume is not None:
            resume_dir = resolve_resume_dir(Path(resume), self.checkpoint_root)
            self.model = PeftModel.from_pretrained(
                self.model, str(resume_dir / "adapter"), is_trainable=True
            )
            if self.cfg.get("gradient_checkpointing"):
                self.model.gradient_checkpointing_enable()
                self.model.enable_input_require_grads()
            self.model.config.use_cache = bool(
                self.cfg.get("use_cache_during_training", False)
            )
        else:
            self.model = self._wrap_lora(self.model)
        self.model.to(self.device)
        self._build_optimizer()

        data_hashes = input_hashes(self.root, self.cfg)
        environment = environment_report(self.device, self.cfg["precision"])
        resolved_config = copy.deepcopy(self.cfg)
        resolved_config["runtime"] = {
            "run_name": self.run_name,
            "tiny_model": self.tiny_model,
            "device": str(self.device),
            "resume": str(resume) if resume else None,
            "max_steps": max_steps,
            "stop_after_steps": stop_after_steps,
            "skip_validation": skip_validation,
            "overfit": overfit,
            "overfit_examples": overfit_examples,
        }
        (self.results_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.results_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.results_dir / "data_hashes.json").write_text(
            json.dumps(data_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        self.state = self._initial_state()
        if resume_dir is not None:
            saved_hashes = json.loads(
                (resume_dir / "data_hashes.json").read_text(encoding="utf-8")
            )
            if saved_hashes != data_hashes:
                raise ValueError("Resume data hashes differ from the current run")
            saved_config = json.loads(
                (resume_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            saved_identity = {key: saved_config.get(key) for key in RESUME_IDENTITY_KEYS}
            current_identity = {key: self.cfg.get(key) for key in RESUME_IDENTITY_KEYS}
            if saved_identity != current_identity:
                raise ValueError("Resume hyperparameters differ from the current run")
            self.state = load_resume_state(
                resume_dir,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            print(f"Resumed from {resume_dir} at step {self.state['global_step']}")

        if overfit:
            self._overfit_run(max_steps=stop_limit or 40)
        else:
            self._train_loop(max_steps=stop_limit, skip_validation=skip_validation)

        summary = self._finalize(resolved_config, environment, data_hashes, skip_validation)
        return summary

    def _restrict_to_overfit(self, count: int) -> None:
        if count < 1:
            raise ValueError("overfit example count must be positive")
        self.train_rows = self.train_rows[:count]
        self.validation_rows = self.validation_rows[: min(count, len(self.validation_rows))]
        self.train_examples = self.train_examples[:count]
        self.validation_examples = self.validation_examples[
            : min(count, len(self.validation_examples))
        ]
        audit_path = self.results_dir / "data_audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.update(
                {
                    "overfit_subset": True,
                    "train_rows_used": len(self.train_rows),
                    "validation_rows_used": len(self.validation_rows),
                    "train_examples_used": len(self.train_examples),
                }
            )
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.validation_examples = self.validation_examples[: min(count, len(self.validation_examples))]
        (self.results_dir / "overfit_subset.json").write_text(
            json.dumps(
                {
                    "annotation_ids": [row["provenance"]["annotation_id"] for row in self.train_rows],
                    "count": len(self.train_rows),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _initial_state(self) -> dict[str, Any]:
        return {
            "global_step": 0,
            "epoch": 0,
            "example_cursor": 0,
            "order": [],
            "loss_history": [],
            "seconds_per_step": [],
            "best": None,
            "last_validation": None,
        }

    def _autocast(self) -> Any:
        if self.cfg["precision"] == "float16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return _NullContext()

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    # --------------------------------------------------------------- training
    def _train_loop(
        self,
        *,
        max_steps: int | None,
        skip_validation: bool,
        max_epochs: int | None = None,
    ) -> None:
        accumulation = int(self.cfg["gradient_accumulation_steps"])
        pad_token_id = self.tokenizer.pad_token_id
        epoch_limit = int(max_epochs if max_epochs is not None else self.cfg["max_epochs"])
        self.model.train()
        stop = False
        while self.state["epoch"] < epoch_limit and not stop:
            epoch = self.state["epoch"]
            if not self.state["order"]:
                self.state["order"] = order_for_epoch(len(self.train_examples), int(self.cfg["seed"]), epoch)
                self.state["example_cursor"] = 0
            order = self.state["order"]
            cursor = int(self.state["example_cursor"])
            while cursor < len(order):
                if max_steps is not None and self.state["global_step"] >= max_steps:
                    stop = True
                    break
                batch_indices = order[cursor : cursor + accumulation]
                if not batch_indices:
                    break
                started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                losses: list[float] = []
                for index in batch_indices:
                    batch = self._to_device(
                        collate_encoded([self.train_examples[index]], pad_token_id)
                    )
                    with self._autocast():
                        output = self.model(**batch, use_cache=False)
                        loss = output.loss / len(batch_indices)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {self.state['global_step']}"
                        )
                    self.scaler.scale(loss).backward()
                    losses.append(float(loss.detach().item()))
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    float(self.cfg["max_grad_norm"]),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - started
                cursor += len(batch_indices)
                self.state["global_step"] += 1
                self.state["example_cursor"] = cursor
                loss_value = sum(losses)
                record = {
                    "global_step": self.state["global_step"],
                    "epoch": epoch,
                    "example_cursor": cursor,
                    "examples_in_step": len(batch_indices),
                    "loss": loss_value,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "seconds": elapsed,
                    "tokens": sum(len(self.train_examples[index].input_ids) for index in batch_indices),
                    "peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(self.device))
                        if self.device.type == "cuda"
                        else None
                    ),
                    "peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(self.device))
                        if self.device.type == "cuda"
                        else None
                    ),
                }
                self.state["loss_history"].append(loss_value)
                self.state["seconds_per_step"].append(elapsed)
                self._append_log(record)
                self._print_progress(record)
                if self._checkpoint_due():
                    self._save_checkpoint("periodic", resolved_config=None, environment=None, data_hashes=None)

            if cursor >= len(order):
                # Epoch boundary.
                self.state["epoch"] = epoch + 1
                self.state["example_cursor"] = 0
                self.state["order"] = []
                self._end_of_epoch(skip_validation=skip_validation)
                self._save_checkpoint("epoch", resolved_config=None, environment=None, data_hashes=None)
                if self.state["epoch"] >= epoch_limit:
                    break

    def _overfit_run(self, *, max_steps: int) -> None:
        before = self._target_match_count()
        self._train_loop(
            max_steps=max_steps,
            skip_validation=True,
            max_epochs=self._overfit_epochs_needed(max_steps),
        )
        after = self._target_match_count()
        report = {
            "subset_count": len(self.train_examples),
            "max_steps": max_steps,
            "starting_loss": self.state["loss_history"][0] if self.state["loss_history"] else None,
            "ending_loss": self.state["loss_history"][-1] if self.state["loss_history"] else None,
            "losses": self.state["loss_history"],
            "target_match_before": before,
            "target_match_after": after,
            "finite_losses": all(math.isfinite(value) for value in self.state["loss_history"]),
        }
        report["passed"] = bool(
            report["finite_losses"]
            and report["ending_loss"] is not None
            and report["ending_loss"] < report["starting_loss"]
            and report["target_match_after"] >= report["target_match_before"]
        )
        (self.results_dir / "overfit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))

    @torch.inference_mode()
    def _target_match_count(self) -> int:
        self._set_eval_mode()
        matches = 0
        try:
            for row, example in zip(self.train_rows, self.train_examples):
                generated = cached_library_greedy(
                    self.model,
                    example.input_ids[: example.prompt_length],
                    max_new_tokens=int(self.cfg["max_new_tokens"]),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    device=self.device,
                )
                raw = self.tokenizer.decode(generated["token_ids"], skip_special_tokens=True)
                parsed = parse_output(
                    raw,
                    row["input"]["criterion_type"],
                    {sentence["id"] for sentence in row["input"]["sentences"]},
                )
                if (
                    parsed["label_for_scoring"] == row["target"]["label"]
                    and set(parsed["evidence_ids_validated"])
                    == set(row["target"]["evidence_sentence_ids"])
                ):
                    matches += 1
        finally:
            self._set_train_mode()
        return matches

    def _end_of_epoch(self, *, skip_validation: bool) -> None:
        epoch = int(self.state["epoch"]) - 1
        epoch_dir = self.results_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = epoch_dir / "adapter"
        self.model.save_pretrained(adapter_dir)
        validation_loss = self._validation_loss()
        result: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": self.state["global_step"],
            "validation_loss": validation_loss,
            "adapter_path": str(adapter_dir),
            "adapter_sha256": hash_directory(adapter_dir),
        }
        if not skip_validation:
            validation_cfg = copy.deepcopy(self.cfg["validation"])
            for key in (
                "model_id",
                "model_revision",
                "prompt_revision",
                "model_config",
                "prompt_config",
                "split_manifest",
            ):
                validation_cfg[key] = self.cfg[key]
            self._set_eval_mode()
            try:
                metrics = predict_validation(
                    self.model,
                    self.tokenizer,
                    validation_cfg,
                    self.root,
                    epoch_dir / "validation",
                    self.device,
                    adapter_path=adapter_dir,
                )
            finally:
                self._set_train_mode()
            result["metrics_path"] = str(epoch_dir / "validation" / "metrics.json")
            result["macro_f1"] = metrics["classification"]["macro_f1"]
            result["evidence_micro_f1"] = metrics["evidence"]["micro"]["f1"]
            result["accuracy"] = metrics["classification"]["accuracy"]
            self.state["last_validation"] = result
            self._consider_best(result)
        self._append_log({"record_type": "epoch_end", **result})
        print(json.dumps({"record_type": "epoch_end", **result}, indent=2, sort_keys=True))

    def _consider_best(self, result: dict[str, Any]) -> None:
        key = (
            float(result.get("macro_f1") or 0.0),
            float(result.get("evidence_micro_f1") or 0.0),
            -float(result.get("validation_loss") or float("inf")),
        )
        best = self.state.get("best")
        if best is None or key > tuple(best["selection_key"]):
            self.state["best"] = {
                "selection_key": list(key),
                "epoch": result["epoch"],
                "global_step": result["global_step"],
                "macro_f1": result.get("macro_f1"),
                "evidence_micro_f1": result.get("evidence_micro_f1"),
                "accuracy": result.get("accuracy"),
                "validation_loss": result.get("validation_loss"),
                "adapter_path": result["adapter_path"],
                "adapter_sha256": result["adapter_sha256"],
            }
            best_adapter = self.results_dir / "best_adapter"
            if best_adapter.exists():
                shutil.rmtree(best_adapter)
            shutil.copytree(result["adapter_path"], best_adapter)

    @torch.inference_mode()
    def _validation_loss(self) -> float:
        self._set_eval_mode()
        total = 0.0
        count = 0
        pad_token_id = self.tokenizer.pad_token_id
        try:
            for example in self.validation_examples:
                batch = self._to_device(collate_encoded([example], pad_token_id))
                output = self.model(**batch, use_cache=False)
                total += float(output.loss.detach().item())
                count += 1
        finally:
            self._set_train_mode()
        return total / count if count else float("nan")

    # ------------------------------------------------------------- mode shifts
    def _set_eval_mode(self) -> None:
        self.model.eval()
        if self.cfg.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True

    def _set_train_mode(self) -> None:
        self.model.config.use_cache = bool(self.cfg.get("use_cache_during_training", False))
        if self.cfg.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable()
        self.model.train()

    # ------------------------------------------------------------- checkpoints
    def _checkpoint_due(self) -> bool:
        every_steps = int(self.cfg["checkpoint"]["every_steps"])
        every_seconds = int(self.cfg["checkpoint"]["every_seconds"])
        if self.state["global_step"] and self.state["global_step"] % every_steps == 0:
            return True
        return (time.perf_counter() - self.last_checkpoint_time) >= every_seconds

    def _save_checkpoint(
        self,
        reason: str,
        *,
        resolved_config: dict[str, Any] | None,
        environment: dict[str, Any] | None,
        data_hashes: dict[str, str] | None,
    ) -> None:
        resolved_config = resolved_config or json.loads(
            (self.results_dir / "resolved_config.json").read_text(encoding="utf-8")
        )
        environment = environment or json.loads(
            (self.results_dir / "environment.json").read_text(encoding="utf-8")
        )
        data_hashes = data_hashes or json.loads(
            (self.results_dir / "data_hashes.json").read_text(encoding="utf-8")
        )
        step = int(self.state["global_step"])
        name = f"resume-step-{step:06d}-{reason}"
        state_dir = self.checkpoint_root / f".state-{step:06d}-{reason}"
        if state_dir.exists():
            shutil.rmtree(state_dir)
        save_resume_state(
            state_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            trainer_state=self.state,
            resolved_config=resolved_config,
            environment=environment,
            data_hashes=data_hashes,
            tokenizer=self.tokenizer,
        )
        bundle_root = self.checkpoint_root / "bundles"
        bundle = make_bundle(
            state_dir,
            bundle_root,
            name,
            {
                "run_name": self.run_name,
                "reason": reason,
                "global_step": step,
                "epoch": self.state["epoch"],
                "example_cursor": self.state["example_cursor"],
                "device": str(self.device),
                "precision": self.cfg["precision"],
            },
        )
        shutil.rmtree(state_dir)
        self.last_checkpoint_time = time.perf_counter()
        if self.drive_dir is not None:
            published = publish_bundle(bundle, self.drive_dir / "checkpoints" / self.run_name)
            prune_resume_bundles(published.parent, int(self.cfg["checkpoint"]["keep_recent"]))
        prune_resume_bundles(bundle_root, int(self.cfg["checkpoint"]["keep_recent"]))
        print(f"Checkpoint verified: {bundle}")

    # ------------------------------------------------------------------ final
    def _finalize(
        self,
        resolved_config: dict[str, Any],
        environment: dict[str, Any],
        data_hashes: dict[str, str],
        skip_validation: bool,
    ) -> dict[str, Any]:
        history = {
            "run_name": self.run_name,
            "global_step": self.state["global_step"],
            "epochs_completed": self.state["epoch"],
            "losses": self.state["loss_history"],
            "seconds_per_step": self.state["seconds_per_step"],
            "best": self.state.get("best"),
            "last_validation": self.state.get("last_validation"),
            "skip_validation": skip_validation,
        }
        (self.results_dir / "training_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        selected_dir = self.results_dir / "selected_adapter"
        if selected_dir.exists():
            shutil.rmtree(selected_dir)
        if (self.results_dir / "best_adapter").is_dir():
            shutil.copytree(self.results_dir / "best_adapter", selected_dir)
            selection_source = "best_validation"
        else:
            self.model.save_pretrained(selected_dir)
            selection_source = "last_state_no_validation_selection"
        selected_manifest = {
            "schema_version": 1,
            "run_name": self.run_name,
            "selection_source": selection_source,
            "selected": self.state.get("best"),
            "last_validation": self.state.get("last_validation"),
            "global_step": self.state["global_step"],
            "epochs_completed": self.state["epoch"],
            "tiny_model": self.tiny_model,
            "model_id": self.cfg["model_id"],
            "model_revision": self.cfg["model_revision"],
            "prompt_revision": self.cfg["prompt_revision"],
            "data_hashes": data_hashes,
            "resolved_config": resolved_config,
            "environment": environment,
            "selected_adapter_sha256": hash_directory(selected_dir),
        }
        export_dir = self.results_dir / "export"
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True)
        shutil.copytree(selected_dir, export_dir / "selected_adapter")
        for name in ("training_history.json", "resolved_config.json", "environment.json",
                     "data_hashes.json", "train_log.jsonl", "data_audit.json"):
            source = self.results_dir / name
            if source.is_file():
                shutil.copy2(source, export_dir / name)
        (export_dir / "selected_manifest.json").write_text(
            json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        export_bundle_root = self.checkpoint_root / "exports"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_bundle = make_bundle(
            export_dir,
            export_bundle_root,
            f"phase04-export-{self.run_name}-{stamp}",
            {
                "run_name": self.run_name,
                "selection_source": selection_source,
                "global_step": self.state["global_step"],
            },
        )
        published_bundle = None
        if self.drive_dir is not None:
            published_bundle = str(
                publish_bundle(export_bundle, self.drive_dir / "exports" / self.run_name)
            )
        summary = {
            "run_name": self.run_name,
            "results_dir": str(self.results_dir),
            "export_bundle": str(export_bundle),
            "published_bundle": published_bundle,
            "export_manifest": str(export_dir / "selected_manifest.json"),
            "global_step": self.state["global_step"],
            "epochs_completed": self.state["epoch"],
            "best": self.state.get("best"),
        }
        (self.results_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    # ------------------------------------------------------------------ utils
    def _append_log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _print_progress(self, record: dict[str, Any]) -> None:
        print(
            "step={global_step} epoch={epoch} loss={loss:.4f} lr={learning_rate:.2e} "
            "grad={grad_norm:.3f} sec={seconds:.2f}".format(**record)
        )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None
