import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.clinical_match.evaluation import load_eval_model
from src.clinical_match.decoding import cached_library_greedy

TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class EvaluationAdapterTests(unittest.TestCase):
    def test_adapter_loads_and_generates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True
            )
            tokenizer.save_pretrained(root / "model")
            config = Qwen2Config(
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                tie_word_embeddings=True,
            )
            base = Qwen2ForCausalLM(config)
            base.save_pretrained(root / "model")

            from peft import LoraConfig, TaskType, get_peft_model

            wrapped = get_peft_model(
                base,
                LoraConfig(
                    r=4,
                    lora_alpha=8,
                    target_modules=["q_proj", "v_proj"],
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
            wrapped.save_pretrained(root / "adapter")

            cfg = {
                "model_id": str(root / "model"),
                "model_revision": None,
                "dtype": "float32",
                "attn_implementation": "eager",
            }
            loaded_tokenizer, loaded_model = load_eval_model(
                cfg, torch.device("cpu"), adapter_path=root / "adapter"
            )
            self.assertEqual(loaded_tokenizer.eos_token_id, tokenizer.eos_token_id)
            prompt = loaded_tokenizer.apply_chat_template(
                [{"role": "user", "content": "Return a short label."}],
                tokenize=True,
                add_generation_prompt=True,
            )
            generated = cached_library_greedy(
                loaded_model,
                prompt,
                max_new_tokens=4,
                eos_token_id=loaded_tokenizer.eos_token_id,
                pad_token_id=loaded_tokenizer.pad_token_id,
                device=torch.device("cpu"),
            )
            self.assertIn("token_ids", generated)
            self.assertLessEqual(len(generated["token_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
