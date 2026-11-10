import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.clinical_match.evaluation import hash_directory
from src.clinical_match.training.checkpoints import (
    load_resume_state,
    make_bundle,
    resolve_resume_dir,
    save_resume_state,
)


def tiny_model():
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )


class TrainingStateTests(unittest.TestCase):
    def test_resume_state_round_trip_restores_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = tiny_model()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            loss = sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            state = {
                "global_step": 3,
                "epoch": 1,
                "example_cursor": 12,
                "order": [4, 1, 0, 2, 3],
                "loss_history": [1.0, 0.9, 0.8],
                "seconds_per_step": [0.1, 0.1, 0.1],
                "best": None,
                "last_validation": None,
            }
            state_dir = save_resume_state(
                root / "state",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                trainer_state=state,
                resolved_config={"seed": 42},
                environment={"device": "cpu"},
                data_hashes={"train": "abc"},
            )
            self.assertTrue((state_dir / "adapter" / "config.json").is_file())
            expected = torch.rand(4)
            random.seed(999)
            torch.manual_seed(999)

            restored_model = tiny_model()
            restored_optimizer = torch.optim.AdamW(
                [p for p in restored_model.parameters() if p.requires_grad], lr=1e-3
            )
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
                restored_optimizer, lambda _: 1.0
            )
            loaded = load_resume_state(
                state_dir,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                scaler=None,
                restore_rng=True,
            )
            self.assertEqual(loaded["global_step"], 3)
            self.assertEqual(loaded["order"], [4, 1, 0, 2, 3])
            self.assertTrue(torch.equal(torch.rand(4), expected))
            self.assertTrue(restored_optimizer.state_dict()["state"])

    def test_bundle_restore_exposes_the_same_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = tiny_model()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            state = {
                "global_step": 5,
                "epoch": 0,
                "example_cursor": 8,
                "order": [0, 1, 2, 3, 4, 5, 6, 7],
                "loss_history": [],
                "seconds_per_step": [],
                "best": None,
                "last_validation": None,
            }
            state_dir = save_resume_state(
                root / "state",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                trainer_state=state,
                resolved_config={},
                environment={},
                data_hashes={},
            )
            bundle = make_bundle(state_dir, root / "bundles", "resume-step-000005-periodic", {})
            (root / "work").mkdir()
            restored = resolve_resume_dir(bundle, root / "work")
            restored_state = json.loads(
                (restored / "trainer_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(restored_state["global_step"], 5)
            self.assertEqual(restored_state["example_cursor"], 8)

    def test_directory_hash_changes_with_content_and_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            first = hash_directory(root)
            self.assertEqual(first, hash_directory(root))
            (root / "b.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(first, hash_directory(root))


if __name__ == "__main__":
    unittest.main()
