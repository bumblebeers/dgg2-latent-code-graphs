#!/usr/bin/env python3
"""cg_sec27_spec.py -- Experiment A specification in the SAME SHAPE as `dgg2_sec27_spec.py`.

The Section 27 Part 2 runner (`colab_runner_sec27_part0_1_2.py`) is driven by a spec module that provides a
pool of candidate *states* per source, operations that point at a target state, paraphrase banks, a
contrastive training schedule and evaluation manifests (including two-step re-feed). This module provides
the same objects for the call-graph grids, so the runner can be pointed at it with small edits:

  Section 27 name                  here                              note
  -------------------------------  --------------------------------  ---------------------------------------
  PART2_CANDIDATE_24_STATES        CG_CANDIDATE_STATES (22)          K-way accuracy is over these
  ALL_TEST_26_STATES               CG_ALL_STATES (33)                test cache: + 9 open slots + 2 "all" refs
  PART2_TRAIN_19_STATES            CG_TRAIN_STATES (23)              train cache: 14 trained + 9 open slots
  STATE24_TO_IDX / TEST26_TO_IDX   STATE_TO_IDX
  TRAIN19_TO_STATE24               TRAIN_TO_STATE
  PART2_OPERATIONS                 CG_OPERATIONS                     same record fields + "is_open"
  PART2_TRAIN_OPS / _HELDOUT_OPS   CG_TRAIN_OPS / CG_HELDOUT_OPS
  PART2_TRAIN_PARAPHRASES ...      CG_TRAIN_PARAPHRASES ...
  PART2_ARITHMETIC_UNSEEN_12       CG_INFERENCE_UNSEEN (9)           unseen strings that need an inference step
  apply_sec27_pil_state(im, s)     render_state_pil(pack, i, s)      states are LOOKED UP, not computed from pixels
  build_part2_contrastive_schedule build_cg_contrastive_schedule     same tuple format and quad structure
  build_part2_eval_manifests       build_cg_eval_manifests           same dict keys; N = 360
  get_all_part2_instruction_...    get_all_cg_instruction_strings
  run_self_checks                  run_self_checks

The same names as in Section 27 are also exported as aliases at the bottom of this file.

Differences the runner must respect:
  * A state is a function of the SOURCE INDEX, not of the source pixels: use `render_state_pil(pack, i, state)`.
  * Open-ended operations (`is_open`) have several valid target states per source ("delete_one_test#0" ...).
    A slot that does not exist for a source has `state_valid(...) == False`; never train or grade on it.
    The schedule already picks one valid slot at random for every open-ended training row.
  * Open-ended manifests carry target index -1. Dump the predicted h_v and score them with cg_score_preds.py.
  * Every instruction must be prefixed with `preamble(pack)` in every arm.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg_instructions as I  # noqa: E402
import cg_ops as O  # noqa: E402

# ----------------------------------------------------------------------------- states
CG_CANDIDATE_STATES: List[str] = list(O.ALL_TARGETS)                       # 22, index 0 = identity = the source
OPEN_STATES: List[str] = [f"{op}#{j}" for op in O.OPEN_TARGETS for j in range(O.OPEN_MAX_VALID)]   # 9 slots
OPEN_ALL_STATES: List[str] = ["all#delete_one_callee", "all#cut_one_call"]  # "all" of delete_one_test == delete_tests
CG_ALL_STATES: List[str] = CG_CANDIDATE_STATES + OPEN_STATES + OPEN_ALL_STATES
STATE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(CG_ALL_STATES)}

_TRAINED_FIXED = O.TRAIN_OPS + [O.comp_name(p) for p in O.TRAIN_COMPS]      # 14
CG_TRAIN_STATES: List[str] = _TRAINED_FIXED + OPEN_STATES                    # 23 (held-out states never loaded)
TRAIN_TO_STATE: List[int] = [STATE_TO_IDX[s] for s in CG_TRAIN_STATES]
STATE_TO_TRAIN: Dict[int, int] = {STATE_TO_IDX[s]: i for i, s in enumerate(CG_TRAIN_STATES)}


def _comp_family(a: str, b: str) -> str:
    fa, fb = O.OP_FAMILY[a] == "fixed", O.OP_FAMILY[b] == "fixed"
    return "comp_fixed_fixed" if fa and fb else ("comp_content_content" if not fa and not fb else "comp_fixed_content")


def _single(op: str, split: str) -> Dict[str, Any]:
    return {"family": ("heldout_" if split == "heldout" else "") + O.OP_FAMILY[op], "split": split, "target_state": op,
            "is_two_step": False, "step1_op": None, "step1_state": None, "step2_op": None, "step2_state": None,
            "is_open": False}


def _comp(pair: Tuple[str, str], split: str) -> Dict[str, Any]:
    a, b = pair
    return {"family": ("heldout_" if split == "heldout" else "") + _comp_family(a, b), "split": split,
            "target_state": O.comp_name(pair), "is_two_step": True, "step1_op": a, "step1_state": a,
            "step2_op": b, "step2_state": b, "is_open": False}


CG_OPERATIONS: Dict[str, Dict[str, Any]] = {}
for _op in O.TRAIN_OPS:
    CG_OPERATIONS[_op] = _single(_op, "train")
for _p in O.TRAIN_COMPS:
    CG_OPERATIONS[O.comp_name(_p)] = _comp(_p, "train")
for _op in O.OPEN_TARGETS:
    CG_OPERATIONS[_op] = {"family": "open", "split": "train", "target_state": None, "is_two_step": False,
                          "step1_op": None, "step1_state": None, "step2_op": None, "step2_state": None, "is_open": True}
for _op in O.HELDOUT_OPS:
    CG_OPERATIONS[_op] = _single(_op, "heldout")
for _p in O.HELDOUT_COMPS:
    CG_OPERATIONS[O.comp_name(_p)] = _comp(_p, "heldout")

CG_TRAIN_OPS: List[str] = [k for k, v in CG_OPERATIONS.items() if v["split"] == "train"]            # 17 = 14 + 3 open
CG_TRAIN_FIXED_OPS: List[str] = [k for k in CG_TRAIN_OPS if not CG_OPERATIONS[k]["is_open"]]         # 14
CG_HELDOUT_OPS: List[str] = [k for k, v in CG_OPERATIONS.items() if v["split"] == "heldout"]        # 8
# Mirror held-out operations expressed as a chain of TRAINED single operations (for 3-step re-feed).
CG_HELDOUT_CHAINS: Dict[str, List[str]] = {
    "keep_callers": ["reverse_edges", "keep_callees", "reverse_edges"],
    "mark_ancestors": ["reverse_edges", "mark_reachable", "reverse_edges"],
}

# ----------------------------------------------------------------------------- paraphrase banks
CG_TRAIN_PARAPHRASES: Dict[str, List[str]] = {**I.SEEN, **I.COMP_SEEN, **I.OPEN_SEEN}
CG_UNSEEN_PARAPHRASES: Dict[str, List[str]] = {**I.UNSEEN, **I.COMP_UNSEEN, **I.OPEN_UNSEEN}
CG_HELDOUT_PARAPHRASES: Dict[str, List[str]] = {**I.HELDOUT_OP_STRINGS, **I.HELDOUT_COMP_STRINGS}
CG_INFERENCE_UNSEEN: List[Tuple[str, int, str]] = [(op, len(I.UNSEEN[op]) - 1, s) for op, s in I.INFERENCE_UNSEEN.items()]


# ----------------------------------------------------------------------------- pack access
def load_pack(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def has_open(pack) -> bool:
    return "open_valid" in pack


def grid_size(pack) -> int:
    return int(pack["src_grid"].shape[-1])


def preamble(pack) -> str:
    return I.preamble(grid_size(pack))


def state_valid(pack, i: int, state: str) -> bool:
    if state in CG_CANDIDATE_STATES:
        return True
    if not has_open(pack):
        return False
    if state.startswith("all#"):
        return True
    op, j = state.split("#")
    return int(j) < int(pack["open_count"][i, O.OPEN_TARGETS.index(op)])


def state_grid(pack, i: int, state: str) -> np.ndarray:
    """(n, n) palette grid of `state` for source i. Raises for an open slot that does not exist for i."""
    if state in CG_CANDIDATE_STATES:
        return pack["tgt_grid"][i, CG_CANDIDATE_STATES.index(state)]
    if not state_valid(pack, i, state):
        raise KeyError(f"state {state!r} does not exist for source {i}")
    if state.startswith("all#"):
        return pack["open_all"][i, O.OPEN_TARGETS.index(state[4:])]
    op, j = state.split("#")
    return pack["open_valid"][i, O.OPEN_TARGETS.index(op), int(j)]


def render_state_pil(pack, i: int, state: str):
    """768x768 RGB PIL image of `state` for source i (use in place of apply_sec27_pil_state)."""
    return O.render_pil(state_grid(pack, i, state))


def states_to_encode(pack, which: str) -> List[str]:
    """States to push through the vision tower. which = "train": CG_TRAIN_STATES (open slots only if the
    pack has them); "test": everything. Skip (leave zero) any (source, state) with state_valid False."""
    base = CG_TRAIN_STATES if which == "train" else CG_ALL_STATES
    return [s for s in base if s in CG_CANDIDATE_STATES or has_open(pack)]


# ----------------------------------------------------------------------------- training schedule
def build_cg_contrastive_schedule(pack, epoch_seed: int, ops_per_source_per_epoch: int = 8,
                                  ) -> List[Tuple[int, int, str, str]]:
    """One epoch of n_sources * ops_per_source_per_epoch examples, same structure as
    build_part2_contrastive_schedule: within every block of 4, two sources each appear with two different
    operations. With 1,608 sources, 8 per source and 3 epochs this is 38,592 examples = 4,824 optimiser
    steps at batch 8, the Section 27 Part 2 budget. Call once per epoch with a different epoch_seed.

    Returns (src_idx, train_state_idx, op_name, instruction) with train_state_idx into CG_TRAIN_STATES.
    For an open-ended operation one of the source's valid answers is drawn uniformly at random.
    The instruction does NOT include the preamble; prepend `preamble(pack)`."""
    n_sources = len(pack["src_grid"])
    ops = CG_TRAIN_OPS if has_open(pack) else CG_TRAIN_FIXED_OPS
    rng = np.random.RandomState(epoch_seed)
    src_order = rng.permutation(n_sources)
    per_src_ops = [[ops[c] for c in rng.choice(len(ops), size=ops_per_source_per_epoch, replace=False)]
                   for _ in range(n_sources)]
    schedule: List[Tuple[int, int, str, str]] = []
    for p_i in range(n_sources // 2):
        sa, sb = int(src_order[2 * p_i]), int(src_order[2 * p_i + 1])
        for q in range(0, ops_per_source_per_epoch, 2):
            for s_cur, op in [(sa, per_src_ops[sa][q]), (sa, per_src_ops[sa][q + 1]),
                              (sb, per_src_ops[sb][q]), (sb, per_src_ops[sb][q + 1])]:
                if CG_OPERATIONS[op]["is_open"]:
                    k = O.OPEN_TARGETS.index(op)
                    state = f"{op}#{int(rng.randint(0, int(pack['open_count'][s_cur, k])))}"
                else:
                    state = CG_OPERATIONS[op]["target_state"]
                bank = CG_TRAIN_PARAPHRASES[op]
                schedule.append((s_cur, STATE_TO_TRAIN[STATE_TO_IDX[state]], op, bank[int(rng.randint(0, len(bank)))]))
    return schedule


# ----------------------------------------------------------------------------- evaluation manifests
def build_cg_eval_manifests(pack, n_test: int = 360) -> Dict[str, Dict[str, Any]]:
    """Held-out evaluation manifests in the Section 27 dict format. Index arrays are into CG_ALL_STATES
    (the first 22 entries are the candidate pool). Keys keep the Section 27 names so runner code that reads
    `tgt_state24_indices` etc. works unchanged.

      seen_bal360               14 trained fixed operations, trained strings
      seen_shuf360              same items, instruction of a different operation (control)
      unseen_para_bal360        14 trained fixed operations, never-seen strings
      unseen_infer_360          the 9 inference strings x 40 sources
      heldout_<op>_360          each of the 3 held-out operations and 5 held-out compositions, all 360 sources
      refeed_heldout_comp_bal360   5 held-out compositions x 72, with step1/step2 instructions (2-step re-feed)
      refeed_chain_heldout_ops_360 keep_callers / mark_ancestors x 180 as a 3-step chain of trained operations
      open_seen_360, open_unseen_360   3 open-ended operations x 120 (only if the pack has them; target -1)
    """
    assert len(pack["src_grid"]) >= n_test
    man: Dict[str, Dict[str, Any]] = {}

    def _pack(imgs, ops, insts, s1=None, s2=None) -> Dict[str, Any]:
        def idx(op, key):
            st = CG_OPERATIONS[op][key]
            return STATE_TO_IDX[st] if st is not None else -1
        out = {"img_indices": np.asarray(imgs, np.int64), "op_names": list(ops),
               "tgt_state24_indices": np.asarray([idx(o, "target_state") for o in ops], np.int64),
               "step1_state24_indices": np.asarray([idx(o, "step1_state") for o in ops], np.int64),
               "step2_state24_indices": np.asarray([idx(o, "step2_state") for o in ops], np.int64),
               "instructions": list(insts)}
        if s1 is not None:
            out["step1_instructions"], out["step2_instructions"] = list(s1), list(s2)
        return out

    fixed = CG_TRAIN_FIXED_OPS
    n_tr = len(fixed)
    imgs = list(range(n_test))
    ops = [fixed[i % n_tr] for i in imgs]
    man["seen_bal360"] = _pack(imgs, ops, [CG_TRAIN_PARAPHRASES[o][(i // n_tr) % len(CG_TRAIN_PARAPHRASES[o])]
                                           for i, o in zip(imgs, ops)])
    wrong = [fixed[(i % n_tr + 5) % n_tr] for i in imgs]          # a different operation, hence a different state
    shuf = _pack(imgs, ops, [CG_TRAIN_PARAPHRASES[w][(i // n_tr) % len(CG_TRAIN_PARAPHRASES[w])]
                             for i, w in zip(imgs, wrong)])
    shuf["shuffled_from_ops"] = wrong
    man["seen_shuf360"] = shuf
    man["unseen_para_bal360"] = _pack(imgs, ops, [CG_UNSEEN_PARAPHRASES[o][(i // n_tr) % len(CG_UNSEEN_PARAPHRASES[o])]
                                                  for i, o in zip(imgs, ops)])
    per = n_test // len(CG_INFERENCE_UNSEEN)
    a_imgs, a_ops, a_insts, a_ids = [], [], [], []
    for s_idx, (op, _, txt) in enumerate(CG_INFERENCE_UNSEEN):
        for j in range(per):
            a_imgs.append((s_idx * per + j) % n_test); a_ops.append(op); a_insts.append(txt); a_ids.append(s_idx)
    arith = _pack(a_imgs, a_ops, a_insts)
    arith["arith_string_idx"] = np.asarray(a_ids, np.int64)
    man["unseen_infer_360"] = arith

    for op in CG_HELDOUT_OPS:
        bank = CG_HELDOUT_PARAPHRASES[op]
        man[f"heldout_{op}_360"] = _pack(imgs, [op] * n_test, [bank[i % len(bank)] for i in imgs])

    comps = [O.comp_name(p) for p in O.HELDOUT_COMPS]
    per = n_test // len(comps)
    r_imgs, r_ops, r_insts, r1, r2 = [], [], [], [], []
    for k, op in enumerate(comps):
        meta = CG_OPERATIONS[op]
        for j in range(per):
            i = k * per + j
            r_imgs.append(i); r_ops.append(op); r_insts.append(CG_HELDOUT_PARAPHRASES[op][i % 4])
            r1.append(CG_TRAIN_PARAPHRASES[meta["step1_op"]][0]); r2.append(CG_TRAIN_PARAPHRASES[meta["step2_op"]][0])
    man["refeed_heldout_comp_bal360"] = _pack(r_imgs, r_ops, r_insts, r1, r2)

    chain_ops = list(CG_HELDOUT_CHAINS)
    c_imgs = list(range(n_test))
    c_ops = [chain_ops[i % len(chain_ops)] for i in c_imgs]
    chain = _pack(c_imgs, c_ops, [CG_HELDOUT_PARAPHRASES[o][(i // len(chain_ops)) % 4] for i, o in zip(c_imgs, c_ops)])
    chain["chain_ops"] = [CG_HELDOUT_CHAINS[o] for o in c_ops]
    chain["chain_instructions"] = [[CG_TRAIN_PARAPHRASES[s][0] for s in CG_HELDOUT_CHAINS[o]] for o in c_ops]
    man["refeed_chain_heldout_ops_360"] = chain

    if has_open(pack):
        for key, bank in (("open_seen_360", I.OPEN_SEEN), ("open_unseen_360", I.OPEN_UNSEEN)):
            o_ops = [O.OPEN_TARGETS[i % len(O.OPEN_TARGETS)] for i in imgs]
            m = _pack(imgs, o_ops, [bank[o][(i // len(O.OPEN_TARGETS)) % len(bank[o])] for i, o in zip(imgs, o_ops)])
            m["open_idx"] = np.asarray([O.OPEN_TARGETS.index(o) for o in o_ops], np.int64)
            man[key] = m
    return man


def get_all_cg_instruction_strings() -> Tuple[List[str], List[Tuple[str, str, int]]]:
    """(strings, meta) over trained, unseen and held-out banks, asserting zero overlap. meta = (split, op, idx)."""
    out, meta, seen = [], [], set()
    for split, bank in (("train_seen", CG_TRAIN_PARAPHRASES), ("test_unseen", CG_UNSEEN_PARAPHRASES),
                        ("test_heldout", CG_HELDOUT_PARAPHRASES)):
        for op, strings in bank.items():
            for j, s in enumerate(strings):
                assert s not in seen, f"duplicate instruction string: {s}"
                seen.add(s); out.append(s); meta.append((split, op, j))
    return out, meta


def run_self_checks(pack=None) -> Dict[str, Any]:
    strings, meta = get_all_cg_instruction_strings()
    assert set(CG_TRAIN_PARAPHRASES) == set(CG_TRAIN_OPS) == set(CG_UNSEEN_PARAPHRASES)
    assert set(CG_HELDOUT_PARAPHRASES) == set(CG_HELDOUT_OPS)
    held_states = {CG_OPERATIONS[o]["target_state"] for o in CG_HELDOUT_OPS}
    assert not (held_states & set(CG_TRAIN_STATES)), "a held-out state is in the training cache"
    for op, rec in CG_OPERATIONS.items():
        if rec["is_two_step"]:
            assert CG_OPERATIONS[rec["step1_op"]]["split"] == "train" and CG_OPERATIONS[rec["step2_op"]]["split"] == "train"
    for op, chain in CG_HELDOUT_CHAINS.items():
        assert all(c in O.TRAIN_OPS for c in chain)
    res = {"n_candidate_states": len(CG_CANDIDATE_STATES), "n_all_states": len(CG_ALL_STATES),
           "n_train_states": len(CG_TRAIN_STATES), "n_train_ops": len(CG_TRAIN_OPS),
           "n_heldout_ops": len(CG_HELDOUT_OPS), "n_strings": len(strings), "chance": 1.0 / len(CG_CANDIDATE_STATES)}
    if pack is not None:
        n = len(pack["src_grid"])
        st = O.GraphState.new(pack["adj"][0], pack["kind"][0], int(pack["selected"][0]))
        for s in CG_CANDIDATE_STATES:
            assert (O.apply(st, s).to_grid() == state_grid(pack, 0, s)).all()
        for op, chain in CG_HELDOUT_CHAINS.items():           # the chain really equals the held-out operation
            x = st
            for c in chain:
                x = O.OPS[c](x)
            assert (x.to_grid() == state_grid(pack, 0, op)).all()
        sched = build_cg_contrastive_schedule(pack, epoch_seed=0)
        assert len(sched) == (n // 2) * 2 * 8
        for a, b in zip(sched[0::2], sched[1::2]):
            assert a[0] == b[0] and a[2] != b[2]
        for src, t_idx, op, _ in sched:
            assert state_valid(pack, src, CG_TRAIN_STATES[t_idx])
        res.update({"n_sources": n, "schedule_rows_per_epoch": len(sched), "has_open": has_open(pack)})
    return res


# ----------------------------------------------------------------------------- Section 27 aliases
PART2_CANDIDATE_24_STATES = CG_CANDIDATE_STATES
ALL_TEST_26_STATES = CG_ALL_STATES
PART2_TRAIN_19_STATES = CG_TRAIN_STATES
STATE24_TO_IDX = TEST26_TO_IDX = STATE_TO_IDX
TRAIN19_TO_STATE24 = TRAIN_TO_STATE
STATE24_TO_TRAIN19 = STATE_TO_TRAIN
PART2_OPERATIONS = CG_OPERATIONS
PART2_TRAIN_OPS = CG_TRAIN_OPS
PART2_HELDOUT_OPS = CG_HELDOUT_OPS
PART2_TRAIN_PARAPHRASES = CG_TRAIN_PARAPHRASES
PART2_UNSEEN_PARAPHRASES = CG_UNSEEN_PARAPHRASES
PART2_HELDOUT_PARAPHRASES = CG_HELDOUT_PARAPHRASES
PART2_ARITHMETIC_UNSEEN_12 = CG_INFERENCE_UNSEEN
build_part2_contrastive_schedule = build_cg_contrastive_schedule
build_part2_eval_manifests = build_cg_eval_manifests
get_all_part2_instruction_strings = get_all_cg_instruction_strings

if __name__ == "__main__":
    import json
    p = load_pack(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(run_self_checks(p), indent=1))
