"""Tests for the SoilGrids fetch verification gate (no network)."""

import json

import numpy as np
import pytest

from bandstack.sources import soilgrids as sg


def _vals(mask: np.ndarray) -> np.ndarray:
    """int16 values matching a boolean valid mask (True = data)."""
    return np.where(mask, 1, sg.NODATA).astype("int16")


def test_group_by_mask_partitions_by_bit_identity():
    a = np.array([True, True, False, False])
    b = np.array([True, True, False, False])
    c = np.array([True, False, False, False])
    groups = sg.group_by_mask({"l1": a, "l2": b, "l3": c})
    assert groups == [["l1", "l2"], ["l3"]]


def test_gate_clean_fetch_passes_without_refetch():
    batch_a = np.array([True, True, False, False])
    batch_b = np.array([True, False, False, False])
    masks = {"A1": batch_a, "A2": batch_a.copy(), "B1": batch_b}
    refetched: list[str] = []
    verified = sg.verify_raw_store(
        list(masks),
        load_mask=masks.__getitem__,
        read_live=lambda rep: _vals(masks[rep]),
        refetch=refetched.append,
    )
    assert refetched == []
    assert verified == masks


def test_gate_group_artifact_refetches_only_that_group():
    truth_a = np.ones(12, dtype=bool)
    truth_b = np.array([True, True, True, False] * 3)
    artifact_b = truth_b.copy()
    artifact_b[0:3] = False  # valid live, nodata on disk: proven fetch artifacts
    truth = {"A1": truth_a, "A2": truth_a, "B1": truth_b, "B2": truth_b}
    on_disk = {"A1": truth_a, "A2": truth_a.copy(), "B1": artifact_b, "B2": artifact_b.copy()}
    refetched: list[str] = []

    def refetch(layer: str) -> None:
        refetched.append(layer)
        on_disk[layer] = truth[layer]

    verified = sg.verify_raw_store(
        list(on_disk),
        load_mask=lambda layer: on_disk[layer],
        read_live=lambda rep: _vals(truth[rep]),
        refetch=refetch,
    )
    assert sorted(refetched) == ["B1", "B2"]
    for layer, mask in verified.items():
        assert mask.tolist() == truth[layer].tolist()


def test_gate_ghost_cells_fail_the_group_too():
    live_truth = np.array([True, True, False, False])
    disk = np.array([True, True, True, False])  # 'valid' where the live read says nodata
    on_disk = {"B1": disk}
    refetched: list[str] = []

    def refetch(layer: str) -> None:
        refetched.append(layer)
        on_disk[layer] = live_truth

    sg.verify_raw_store(
        ["B1"],
        load_mask=lambda layer: on_disk[layer],
        read_live=lambda rep: _vals(live_truth),
        refetch=refetch,
    )
    assert refetched == ["B1"]


def test_gate_fails_loudly_after_max_attempts():
    truth = np.ones(8, dtype=bool)
    artifact = truth.copy()
    artifact[:4] = False
    on_disk = {"B1": artifact}
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="fetch verification failed for B1"):
        sg.verify_raw_store(
            ["B1"],
            load_mask=lambda layer: on_disk[layer],
            read_live=lambda rep: _vals(truth),
            refetch=calls.append,
            max_attempts=3,
        )
    assert len(calls) == 3


def test_reference_fraction_violations_flags_drift_beyond_tolerance():
    measured = {"a": 0.9000, "b": 0.8000}
    reference = {"a": 0.8950, "b": 0.7000, "c": 0.5}
    assert sg.reference_fraction_violations(measured, reference, tol=0.01) == [
        "b: 0.8000 vs reference 0.7000"
    ]


def test_next_fetched_at_written_now_kept_previous_unknown_none():
    previous = {"kept": {"fetched_at": "2026-10-26T00:00:00+00:00"}}
    assert sg.next_fetched_at("written", {"written"}, previous, "now") == "now"
    assert sg.next_fetched_at("kept", {"written"}, previous, "now") == ("2026-10-26T00:00:00+00:00")
    assert sg.next_fetched_at("unknown", {"written"}, previous, "now") is None


def test_raw_reference_fractions_reads_the_sidecar(tmp_path):
    (tmp_path / "_metadata.json").write_text(
        json.dumps({"layers": [{"name": "x", "valid_frac_inside": 0.5}]})
    )
    assert sg.raw_reference_fractions(tmp_path) == {"x": 0.5}
