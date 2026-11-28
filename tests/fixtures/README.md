# Synthetic NNUE fixtures

The `.nnue` files in this directory are tiny deterministic synthetic models used
only by the loader/accumulator/inference tests. They are **not** trained pilot
weights and are intentionally ignored by git.

Generate them before running the tests:

```bash
.venv/Scripts/python.exe tests/nnue_fixture.py
```

`tiny_reference.json` is the matching reference output and is committed.
