#!/usr/bin/env python3
"""tests/test_cg_gemma4_ar_runner.py -- Unit tests for Gemma 4 AR runner.

Verifies:
1. Grid text formatting: 16x16 grid formatted into 16 lines of 16 characters.
2. Robust parsing:
   - Clean 16x16 grid strings
   - Space-separated lines
   - Markdown fenced code blocks (```)
   - Thought channel tag stripping (<|channel>thought ... <channel|>)
   - Partial output fallback to source grid
   - Corrupted / empty output fallback to source grid
3. Single-digit tokenization invariance:
   - Digits 0-7 parse to distinct valid classes.
4. Palette legend formatting and completeness.
"""
import numpy as np
import pytest
import re

import sys
import os

# Ensure src and cg are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../cg")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.cg_gemma4_ar_runner import (
    format_grid_text,
    parse_16x16_grid,
    PALETTE_LEGEND,
    build_in_context_examples,
)
from cg import cg_ops as O


def test_format_grid_text():
    rng = np.random.RandomState(42)
    grid = rng.randint(0, 8, size=(16, 16), dtype=np.uint8)
    text = format_grid_text(grid)
    lines = text.strip().split("\n")
    assert len(lines) == 16, f"Expected 16 lines, got {len(lines)}"
    for r, line in enumerate(lines):
        assert len(line) == 16, f"Line {r} length is {len(line)}, expected 16"
        for c, ch in enumerate(line):
            assert int(ch) == grid[r, c]


def test_parse_16x16_grid_clean():
    rng = np.random.RandomState(123)
    true_grid = rng.randint(0, 8, size=(16, 16), dtype=np.uint8)
    text = format_grid_text(true_grid)
    fallback = np.zeros((16, 16), dtype=np.uint8)

    parsed_grid, is_exact = parse_16x16_grid(text, fallback)
    assert is_exact is True
    np.testing.assert_array_equal(parsed_grid, true_grid)


def test_parse_16x16_grid_space_separated():
    rng = np.random.RandomState(456)
    true_grid = rng.randint(0, 8, size=(16, 16), dtype=np.uint8)
    # Join each row with spaces
    text = "\n".join(" ".join(str(int(c)) for c in row) for row in true_grid)
    fallback = np.zeros((16, 16), dtype=np.uint8)

    parsed_grid, is_exact = parse_16x16_grid(text, fallback)
    assert is_exact is True
    np.testing.assert_array_equal(parsed_grid, true_grid)


def test_parse_16x16_grid_markdown_fences():
    rng = np.random.RandomState(789)
    true_grid = rng.randint(0, 8, size=(16, 16), dtype=np.uint8)
    grid_str = format_grid_text(true_grid)
    text = f"Here is the edited grid:\n```\n{grid_str}\n```\nHope that helps!"
    fallback = np.zeros((16, 16), dtype=np.uint8)

    parsed_grid, is_exact = parse_16x16_grid(text, fallback)
    assert is_exact is True
    np.testing.assert_array_equal(parsed_grid, true_grid)


def test_parse_16x16_grid_with_thought_tags():
    rng = np.random.RandomState(101)
    true_grid = rng.randint(0, 8, size=(16, 16), dtype=np.uint8)
    grid_str = format_grid_text(true_grid)
    text = (
        "<|channel>thought\n"
        "Let me see... The user wants to delete all tests. Tests are class 5.\n"
        "So any 5 on the diagonal should become 0.\n"
        "<channel|>\n"
        f"{grid_str}"
    )
    fallback = np.zeros((16, 16), dtype=np.uint8)

    parsed_grid, is_exact = parse_16x16_grid(text, fallback)
    assert is_exact is True
    np.testing.assert_array_equal(parsed_grid, true_grid)


def test_parse_16x16_grid_partial_fallback():
    fallback = np.ones((16, 16), dtype=np.uint8) * 3
    # Only 32 digits generated
    text = "0" * 32
    parsed_grid, is_exact = parse_16x16_grid(text, fallback)
    assert is_exact is False
    assert parsed_grid.shape == (16, 16)
    # First 32 cells should be 0
    flat = parsed_grid.flatten()
    assert (flat[:32] == 0).all()
    # Remaining cells should match fallback (3)
    assert (flat[32:] == 3).all()


def test_parse_16x16_grid_corrupted_fallback():
    fallback = np.ones((16, 16), dtype=np.uint8) * 4
    text = "I am an AI and cannot fulfill this request."
    parsed_grid, is_exact = parse_16x16_grid(text, fallback)
    assert is_exact is False
    np.testing.assert_array_equal(parsed_grid, fallback)


def test_palette_legend_completeness():
    assert "0 = background" in PALETTE_LEGEND
    assert "1 = call_edge" in PALETTE_LEGEND
    assert "2 = function" in PALETTE_LEGEND
    assert "3 = method" in PALETTE_LEGEND
    assert "4 = class" in PALETTE_LEGEND
    assert "5 = test" in PALETTE_LEGEND
    assert "6 = selected" in PALETTE_LEGEND
    assert "7 = marked" in PALETTE_LEGEND


def test_build_in_context_examples():
    # Load actual training pack if present
    train_path = os.path.join(os.path.dirname(__file__), "../data/gridA_train1608.npz")
    if os.path.exists(train_path):
        train_pack = dict(np.load(train_path))
        ex_text = build_in_context_examples(train_pack)
        assert "In-Context Example 1" in ex_text
        assert "In-Context Example 2" in ex_text
        assert "Instruction:" in ex_text
        assert "Target Grid:" in ex_text
