"""End-to-end smoke tests: synthetic shards -> CPU train -> export -> reload."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import torch

import train_halfkp as T
from reference_inference import (
    HEADER_SIZE,
    INPUTS,
    POLICY_SIZE,
    _expected_payload_size,
    load_nnue,
    parse_header,
)
from synthetic import write_synthetic_shards, write_value_only_shards

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture(scope="module")
def shards_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic_shards")
    write_synthetic_shards(str(path), n=640, seed=5)
    return str(path)


def _train_args(shards: str, out: str, extra=()) -> list:
    return [
        "--data", shards,
        "--out", out,
        "--hidden", "128",
        "--batch", "128",
        "--epochs", "1",
        "--lr", "0.002",
        "--device", "cpu",
        "--amp", "off",
        "--patience", "0",
        "--prefetch", "0",
        "--num-workers", "0",
        "--log-every", "0",
        "--max-val-samples", "128",
        "--policy-eval-samples", "64",
        "--seed", "42",
        *extra,
    ]


# ---------------------------------------------------------------------------
# Profiles / paths / split
# ---------------------------------------------------------------------------
def test_profiles_apply_and_explicit_flags_win():
    parser = T._build_parser()
    teacher = parser.parse_args(["--profile", "teacher"])
    T._apply_profile(teacher)
    assert teacher.hidden == 1536
    assert teacher.batch == 65536
    assert teacher.amp == "bf16"

    custom = parser.parse_args(
        ["--profile", "teacher", "--hidden", "1024", "--batch", "32768"]
    )
    T._apply_profile(custom)
    assert (custom.hidden, custom.batch) == (1024, 32768)

    student = parser.parse_args(["--profile", "student"])
    T._apply_profile(student)
    assert student.hidden == 512


def test_paths_are_project_relative():
    rel = os.path.join("training", "work", "x")
    resolved = T._resolve_path(rel)
    assert os.path.isabs(resolved)
    assert resolved.startswith(T.PROJECT_ROOT)
    assert T._resolve_path(os.path.join(T.PROJECT_ROOT, "a", "b")).endswith(
        os.path.join("a", "b")
    )


def test_group_split_deterministic_and_disjoint(shards_dir):
    shards = T.Shards(shards_dir)
    groups = np.asarray(shards.groups)
    counts = np.bincount(groups.astype(np.int64))
    assert counts.max() > 1, "synthetic groups should be non-singleton"

    tr1, va1, info1 = T.group_split(groups, 0.2, seed=123)
    tr2, va2, info2 = T.group_split(groups, 0.2, seed=123)
    np.testing.assert_array_equal(tr1, tr2)
    np.testing.assert_array_equal(va1, va2)
    assert info1 == info2
    assert set(groups[tr1].tolist()).isdisjoint(set(groups[va1].tolist()))
    assert info1["group_overlap"] == 0
    assert info1["train_groups"] + info1["val_groups"] == len(np.unique(groups))
    assert 0.1 < info1["val_fraction_actual"] < 0.3


# ---------------------------------------------------------------------------
# Train + export + reload
# ---------------------------------------------------------------------------
def test_cpu_smoke_train_export_reload(tmp_path, shards_dir):
    out = tmp_path / "model"
    rc = T.main(
        _train_args(
            shards_dir,
            str(out),
            extra=["--policy", "on", "--policy-weight", "0.1", "--cp-weight", "0.05"],
        )
    )
    assert rc == 0

    nnue = out / "model.nnue"
    assert nnue.exists()
    hdr = parse_header(nnue.read_bytes())
    assert hdr["hidden"] == 128
    assert hdr["policy_size"] == POLICY_SIZE
    expected = HEADER_SIZE + _expected_payload_size(128, INPUTS, POLICY_SIZE)
    assert nnue.stat().st_size == expected

    model = load_nnue(str(nnue))
    result = model.eval_fen(START)
    assert math.isfinite(result["cp"])
    assert 0.0 < result["wdl"] < 1.0
    assert result["n_features"] == 31
    assert len(result["policy_topk"]) == 5

    card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))
    assert card["deployable"] is True
    assert card["role"] == "pilot"
    assert card["checks"]["roundtrip"]["ok"] is True
    check = card["checks"]["export_vs_torch"]
    assert check["ok"] is True
    assert check["max_abs_cp_diff"] <= 2.0
    assert check["max_abs_policy_logit_diff"] < 1e-2
    assert card["training"]["split"]["group_overlap"] == 0
    assert card["architecture"]["policy_size"] == POLICY_SIZE
    assert card["export"]["policy_size"] == POLICY_SIZE
    assert card["metrics"]["val_best"]["policy"]["n"] > 0
    assert card["calibration"] and card["per_phase"]

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["epochs_run"] == 1
    assert metrics["best_epoch"] == 0
    assert metrics["deployable"] is True
    assert (out / "checkpoint.pt").exists()
    assert (out / "model.pt").exists()


def test_value_only_pipeline_handles_missing_optional_files(tmp_path):
    shards = write_value_only_shards(str(tmp_path / "value_shards"), n=320, seed=8)
    out = tmp_path / "value_model"
    rc = T.main(_train_args(shards, str(out)))
    assert rc == 0

    hdr = parse_header((out / "model.nnue").read_bytes())
    assert hdr["policy_size"] == 0
    expected = HEADER_SIZE + _expected_payload_size(128, INPUTS, 0)
    assert (out / "model.nnue").stat().st_size == expected

    card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))
    assert card["architecture"]["policy_size"] == 0
    assert "policy" not in card
    assert "checks" in card
    assert card["checks"]["export_vs_torch"]["ok"] is True


def test_resume_continues_from_checkpoint(tmp_path, shards_dir):
    out = tmp_path / "resume_model"
    base = _train_args(shards_dir, str(out), extra=["--policy", "off"])
    assert T.main(base) == 0
    ckpt = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert ckpt["epochs_run"] == 1

    resumed = base + ["--epochs", "2", "--resume", str(out)]
    assert T.main(resumed) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["epochs_run"] == 2
    # The untouched value head is still reproducible.
    assert (out / "model.nnue").exists()

    assert T.main(base + ["--resume", str(out), "--export-only"]) == 0
    card = json.loads((out / "model_card.json").read_text(encoding="utf-8"))
    assert card["training"]["reexported"] is True
    assert card["checks"]["export_vs_torch"]["ok"] is True


# ---------------------------------------------------------------------------
# Deployability marker
# ---------------------------------------------------------------------------
def test_teacher_export_is_marked_non_deployable(tmp_path, monkeypatch):
    parser = T._build_parser()
    args = parser.parse_args(
        ["--profile", "teacher", "--hidden", "128", "--out", str(tmp_path / "teacher")]
    )
    T._apply_profile(args)
    # Pretend the deployment ceiling is 64 so the marker triggers without
    # building a 1536-wide model in the unit test.
    monkeypatch.setattr(T, "DEPLOYABLE_MAX_HIDDEN", 64)
    model = T.HalfKPNet(128, policy_size=0)
    exported = T.export_model(
        model, args, str(tmp_path / "teacher"), "teacher_model", {}
    )
    assert os.path.exists(exported["nnue"])
    card = json.loads(
        (tmp_path / "teacher" / "model_card.json").read_text(encoding="utf-8")
    )
    assert card["deployable"] is False
    assert card["deployability"]["deployable"] is False
    assert "exceeds" in card["deployability"]["reason"]
    assert card["role"] == "teacher"
    assert card["export"]["file_size"] == HEADER_SIZE + _expected_payload_size(
        128, INPUTS, 0
    )


# ---------------------------------------------------------------------------
# GPU-resident gather (skipped without CUDA)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_resident_batches_match_cpu(tmp_path):
    shards_dir = write_synthetic_shards(str(tmp_path / "gpu_shards"), n=256, seed=3)
    shards = T.Shards(shards_dir)
    gpu = T.GpuShards(shards, torch.device("cuda"), 256)
    rows = np.arange(16, dtype=np.int64)
    gb = gpu.batch(rows)
    cb = shards.batch(rows)
    assert gb.feats.is_cuda
    assert torch.equal(gb.feats.cpu(), torch.from_numpy(cb.feats))
    assert torch.equal(gb.offsets.cpu(), torch.from_numpy(cb.offsets))
    assert torch.allclose(gb.target.cpu(), torch.from_numpy(cb.target))
    assert torch.equal(gb.material.cpu(), torch.from_numpy(cb.material))
