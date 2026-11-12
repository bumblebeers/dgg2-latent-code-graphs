#!/usr/bin/env python3
"""Unit tests for src/build_review_notebook.py and notebooks/dgg2_latent_code_graphs_demo.ipynb."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgg2_latent_code_graphs_demo.ipynb"
BUILDER_PATH = REPO_ROOT / "src" / "build_review_notebook.py"


@pytest.mark.m5
@pytest.mark.unit
def test_notebook_file_exists_and_valid_json():
    """Verify generated notebook exists and parses as valid JSON."""
    assert NOTEBOOK_PATH.exists(), f"Expected notebook at {NOTEBOOK_PATH}"
    assert NOTEBOOK_PATH.stat().st_size > 1000

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert nb_data["nbformat"] == 4
    assert nb_data["nbformat_minor"] == 5
    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 10
    assert "metadata" in nb_data
    assert nb_data["metadata"]["language_info"]["name"] == "python"


@pytest.mark.m5
@pytest.mark.unit
def test_notebook_contains_all_7_sections():
    """Verify all 7 required core sections and themes are present in notebook cells."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_text = ""
    for cell in nb_data["cells"]:
        src = "".join(cell.get("source", []))
        all_text += src + "\n"

    # Section 1: Latent Roadmap Paradigm & Hero Figure 1
    assert "Latent Roadmap" in all_text
    assert "figure1_hero_multimodal_roadmap.png" in all_text

    # Section 2: Upstream Mask Bug & Turn-Key Fix
    assert "DiffusionGemmaEncoderModel.forward" in all_text or "create_masks_for_generate" in all_text
    assert "install_dg_attn_fix" in all_text

    # Section 3: Modular Libraries
    assert "dgg_latent" in all_text
    assert "dgg_codebook" in all_text
    assert "dgg_vit_decoder" in all_text

    # Section 4: AST Call Graph Extraction & Rendering
    assert "ast" in all_text
    assert "GraphState" in all_text
    assert "768x768" in all_text or "render" in all_text

    # Section 5: Vision Stack & Ridge Reader Decoding (Gate A1)
    assert "Gate A1" in all_text
    assert "reader.npz" in all_text
    assert "read_grids" in all_text

    # Section 6: Non-DG Baseline Inference on httpx
    assert "NonDGTransformBaseline" in all_text or "nondg_enc_lora_s42.pt" in all_text
    assert "delete_methods" in all_text

    # Section 7: Google Water Bottle Showcase & Claim C06
    assert "qual_dev_000031_rot90_decoded.png" in all_text
    assert "Claim C06" in all_text or "patch-permutation" in all_text


@pytest.mark.m5
@pytest.mark.unit
def test_builder_cli_check_success():
    """Verify python3 src/build_review_notebook.py --check exits 0."""
    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--check failed: {result.stderr}\n{result.stdout}"
    assert "Notebook is up to date" in result.stdout


@pytest.mark.m5
@pytest.mark.unit
def test_builder_cli_drift_detection():
    """Verify python3 src/build_review_notebook.py --check exits 1 when drifted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_nb = Path(tmpdir) / "demo.ipynb"
        fake_nb.write_text(json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5}))

        result = subprocess.run(
            [sys.executable, str(BUILDER_PATH), "--check", "--out", str(fake_nb)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, "Expected --check to exit 1 on drifted notebook"
        assert "DRIFT ERROR" in result.stderr


@pytest.mark.m5
@pytest.mark.unit
def test_notebook_code_cells_execute_end_to_end():
    """Execute all code cells of the review notebook sequentially in a shared namespace.

    Asserts that zero exceptions are raised when executing the entire pipeline end-to-end.
    Uses non-interactive Agg matplotlib backend.
    """
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")

    assert NOTEBOOK_PATH.exists(), f"Expected notebook at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    shared_namespace: dict = {
        "__name__": "__main__",
    }

    code_cells_executed = 0
    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            code_cells_executed += 1
            try:
                exec(code, shared_namespace)
            except Exception as exc:
                pytest.fail(f"Notebook code cell {idx} failed during end-to-end execution: {exc}")

    assert code_cells_executed >= 5, f"Expected at least 5 code cells, executed {code_cells_executed}"

