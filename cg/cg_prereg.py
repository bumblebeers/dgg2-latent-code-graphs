#!/usr/bin/env python3
"""cg_prereg.py -- write PREREG.json: everything that must be fixed BEFORE the first training launch.
Re-run with --check to confirm that code constants and data files still match the recorded values."""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg_instructions as I, cg_node_dataset as ND, cg_node_gate as NG, cg_ops as O, cg_grid_dataset as GD  # noqa: E402
import cg_sec27_spec as SP  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def current() -> dict:
    data = os.path.join(ROOT, "data")
    files = sorted(f for f in os.listdir(data) if f.endswith((".npz", ".json.gz")))
    graphs = sorted(os.listdir(os.path.join(data, "graphs")))
    code = sorted(f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f.endswith(".py"))
    return {
        "experiment_A_grid": {
            "interface": "cg_sec27_spec.py (same shape as dgg2_sec27_spec.py); runner base: colab_runner_sec27_part0_1_2.py",
            "generation_readout": "stock_generate_with_readout, readout_source == final_sc_denoising_step; no two-pass readout",
            "candidate_states_in_order": SP.CG_CANDIDATE_STATES, "chance_nearest_candidate": 1 / len(SP.CG_CANDIDATE_STATES),
            "all_states_test_cache": SP.CG_ALL_STATES, "train_states_cache": SP.CG_TRAIN_STATES,
            "train_ops": SP.CG_TRAIN_OPS, "heldout_ops": SP.CG_HELDOUT_OPS,
            "heldout_kinds": {"mirror_of_trained": ["keep_callers", "mark_ancestors"], "new_family_member": ["delete_methods"]},
            "heldout_chains_for_refeed": SP.CG_HELDOUT_CHAINS,
            "open_ended_ops": O.OPEN_TARGETS, "open_valid_answers_per_source": [2, O.OPEN_MAX_VALID],
            "op_family": {k: v["family"] for k, v in SP.CG_OPERATIONS.items()},
            "min_candidate_cell_difference": GD.MIN_DIFF,
            "instruction_strings": {"total": len(SP.get_all_cg_instruction_strings()[0]),
                                    "sha256_of_all_strings": hashlib.sha256("\n".join(SP.get_all_cg_instruction_strings()[0]).encode()).hexdigest(),
                                    "preamble_sha256": hashlib.sha256(I.PREAMBLE.encode()).hexdigest()},
            "train": {"pack": "gridA_train1608.npz", "repos": ["fastapi", "rich", "requests"], "sources": 1608,
                      "schedule": "cg_sec27_spec.build_cg_contrastive_schedule(pack, epoch_seed=seed + epoch, ops_per_source_per_epoch=8), 3 epochs",
                      "optimizer_steps_at_batch_8": 4824},
            "test": {"primary": "gridA_test360_httpx.npz", "second_repo": "gridA_test360_click.npz",
                     "manifests": "cg_sec27_spec.build_cg_eval_manifests(pack)", "scoring": "cg_score_preds.py on dumped h_v"},
            "arms": ["LoRA DG (seeds 42, 1337)", "Frozen DG", "Non-DG + frozen instruction (2 seeds)",
                     "Non-DG + encoder LoRA (2 seeds)", "shuffled-instruction manifest", "source copy", "reader ceiling on true h_v"],
            "gate_A1_cell_probe": {"overall_cell_acc_min": 0.99, "every_class_acc_min": 0.95,
                                   "reader_fit_states": ["identity", "mark_reachable", "delete_tests"],
                                   "fallback": "8 symbols per grid, 96 px cells"},
            "gate_A2_candidate_separation": "report the share of test items whose two closest candidates have centred "
                                            "h_v cosine > 0.85; h_v K-way accuracy is then reported on all items and on "
                                            "the <= 0.85 subset",
            "primary_metrics": ["K-way nearest-candidate accuracy on reader grids (ties count as wrong)", "exact grid match",
                                "changed-cell accuracy", "edge F1",
                                "open-ended: valid / none / all / ambiguous, and diversity over 4 generations"],
            "stated_before_running": {
                "mirror_heldout": "expected weak in a single pass, like Section 27 darken (8.2%); the 3-step re-feed chain is the test of whether iteration recovers them",
                "family_heldout": "delete_methods is the analogue of Section 27 blue_tint (60.2%)",
                "heldout_compositions": "single pass may fail as Section 27 Geo x Geo did (0.0-0.5%); 2-step re-feed is reported next to it",
                "open_ended": "the non-DG baselines are deterministic regressors and are expected to land on none or ambiguous; whether LoRA DG commits to one valid answer is the open question"},
            "training_seeds": [42, 1337],
        },
        "experiment_B_node_vectors": {
            "candidate_edits_function": ["identity", "rename", "add_param", "delete", "add_decorator",
                                         "rename(wrong arg)", "add_param(wrong arg)", "add_decorator(wrong arg)"],
            "candidate_edits_class_heldout": ["identity", "rename", "delete", "rename(wrong arg)"],
            "argument_banks": {"rename": ND.RENAME, "params": ND.PARAMS, "decorators": ND.DECOS},
            "templates_sha256": hashlib.sha256(json.dumps(ND.TEMPLATES, sort_keys=True).encode()).hexdigest(),
            "n_slots": ND.N_SLOTS,
            "train": {"pack": "nodeB_train1608.json.gz", "repos": ["fastapi", "rich", "requests"]},
            "test": {"primary": "nodeB_test360_httpx.json.gz", "unseen_arguments": "nodeB_test360_httpx_unseenargs.json.gz",
                     "heldout_edit_rename_class": "nodeB_test180_httpx_renameclass.json.gz",
                     "second_repo": "nodeB_test360_click.json.gz"},
            "gate_B0_encoder": {"participation_ratio_min": NG.MIN_PARTICIPATION, "centered_cos_mean_max": NG.MAX_CENTERED_COS,
                                "duplicate_slot_rate_max": NG.MAX_DUP_RATE, "median_visibility_min": NG.MIN_VISIBILITY,
                                "oracle_acc_at_noise_0.25_min": NG.MIN_ORACLE_ACC},
            "gate_B1_baseline": "non-DG baseline, seen phrasings, httpx test: `delete` items localised (largest "
                                "predicted change is the primary slot) on >= 50% of items; otherwise switch to the "
                                "slot-assisted instruction variant and say so",
            "loss_weighting": "changed slots share 0.5 of each example's loss weight, unchanged slots share 0.5; all arms",
            "canvas_teacher": "one token per slot: unchanged / edited / deleted (hard cross-entropy)",
            "primary_metrics": ["primary-slot nearest-candidate accuracy (Euclidean, standardised space; chance 1/8)",
                                "localisation top-1 and recall@5 of changed slots", "right-edit-wrong-argument rate"],
            "training_seeds": [42, 1337],
        },
        "extension_larger_grids": {
            "sizes": {"32": {"cells_per_token": 4, "train": "gridA_n32_train1608.npz", "test": "gridA_n32_test360_httpx.npz"},
                      "48": {"cells_per_token": 9, "train": "gridA_n48_train1608.npz", "test": "gridA_n48_test360_httpx.npz"}},
            "same_as_experiment_A": ["fixed operations and splits (no open-ended operations at these sizes)",
                                     "instruction banks with cg_sec27_spec.preamble(pack)", "manifests", "schedule", "metrics"],
            "gate_A1_per_size": {"overall_cell_acc_min": 0.99, "every_class_acc_min": 0.95,
                                 "if_failed": "report the reader accuracy at that size as the result; do not train"},
            "training_seeds": [42],
        },
        "extension_multi_symbol_edits": {
            "train": {"pack": "nodeB_multi_train1608.json.gz", "symbols_per_instruction": [1, 2]},
            "test": {"seen_count": "nodeB_multi_test360_httpx_k2.json.gz", "heldout_count": "nodeB_multi_test360_httpx_k4.json.gz",
                     "single": "nodeB_test360_httpx.json.gz"},
            "refeed_variant": "the single-symbol model applied once per site sentence (sites[k]['instr_seen'][0]), feeding its output vectors back as the next input",
            "primary_metrics": ["all-sites-correct rate", "per-site nearest-candidate accuracy", "changed-slot recall"],
            "runs_only_if": "gates B0 and B1 passed and the single-symbol run trains",
            "training_seeds": [42],
        },
        "data_sha256": {f: sha(os.path.join(data, f)) for f in files},
        "graph_sha256": {f: sha(os.path.join(data, "graphs", f)) for f in graphs},
        "code_sha256": {f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in code if f != "cg_prereg.py"},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    path = os.path.join(ROOT, "PREREG.json")
    cur = current()
    if a.check:
        old = json.load(open(path))
        bad = [k for k in cur if json.dumps(cur[k], sort_keys=True) != json.dumps(old.get(k), sort_keys=True)]
        print("PREREG matches" if not bad else f"PREREG MISMATCH in: {bad}")
        sys.exit(1 if bad else 0)
    json.dump(cur, open(path, "w"), indent=1)
    print("wrote", path, "| data files:", len(cur["data_sha256"]))
