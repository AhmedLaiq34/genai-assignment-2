import json
import tempfile
import unittest
from pathlib import Path

from src.clinical_match.checkpointing import (
    create_verified_bundle,
    restore_verified_bundle,
    verify_bundle,
)


class CheckpointBundleTests(unittest.TestCase):
    def test_create_and_restore_synthetic_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "state.json").write_text(
                json.dumps({"step": 3, "weights": [0.25, 0.75]}),
                encoding="utf-8",
            )
            (source / "nested").mkdir()
            (source / "nested" / "rng.bin").write_bytes(b"synthetic-rng-state")

            bundle = create_verified_bundle(
                source,
                root / "persistent",
                "checkpoint-smoke",
                {"global_step": 3},
            )
            restored = restore_verified_bundle(bundle, root / "restored")
            manifest = verify_bundle(restored)
            self.assertEqual(manifest["metadata"]["global_step"], 3)
            self.assertEqual(
                json.loads((restored / "state.json").read_text(encoding="utf-8")),
                {"step": 3, "weights": [0.25, 0.75]},
            )
            self.assertEqual(
                (restored / "nested" / "rng.bin").read_bytes(),
                b"synthetic-rng-state",
            )
            latest = json.loads(
                (root / "persistent" / "latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(latest["bundle_name"], "checkpoint-smoke")

    def test_restore_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "state.json").write_text("{}", encoding="utf-8")
            bundle = create_verified_bundle(source, root / "persistent", "checkpoint-1")
            destination = root / "restored"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                restore_verified_bundle(bundle, destination)

    def test_verification_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "weights.bin").write_bytes(b"original")
            bundle = create_verified_bundle(source, root / "persistent", "checkpoint-1")
            (bundle / "weights.bin").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_bundle(bundle)

    def test_refuses_to_overwrite_a_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "state.json").write_text("{}", encoding="utf-8")
            create_verified_bundle(source, root / "persistent", "checkpoint-1")
            with self.assertRaises(FileExistsError):
                create_verified_bundle(source, root / "persistent", "checkpoint-1")


if __name__ == "__main__":
    unittest.main()
