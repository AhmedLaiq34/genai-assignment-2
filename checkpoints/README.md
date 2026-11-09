# Checkpoint storage

Large training checkpoints are not stored in Git. Use
`src/clinical_match/checkpointing.py` to stage a complete bundle, verify each
file's SHA-256, write a completion marker, and update `latest.json`.
`restore_verified_bundle` copies a verified bundle into a new destination,
validates the restored copy, and refuses to overwrite an existing destination.

The helper is smoke-tested with synthetic files; it does not yet save or resume
a model optimizer state. That training-specific integration belongs in Phase 4.
