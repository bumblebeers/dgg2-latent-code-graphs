#!/usr/bin/env python3
"""Publish and synchronize all 4 public GitHub repositories under @bumblebeers.

Syncs from clean staging trees in /tmp/dgg2_codegraph/gh_repos/ so internal agent
orchestration files, weights/large caches, and local paths never enter Git history,
verifies prereg + self-test + pytest on the staged main repository, and authors every
commit strictly as `bumblebeers <21022766+bumblebeers@users.noreply.github.com>`.

Repositories managed:
  1. bumblebeers/dgg2-latent-code-graphs (main benchmark, runners, paper, JSON summaries & test suite)
  2. bumblebeers/dgg-latent              (standalone library)
  3. bumblebeers/dgg-codebook            (standalone library)
  4. bumblebeers/dgg-vit-decoder         (standalone library)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_ROOT = Path("/tmp/dgg2_codegraph/gh_repos")
WHEEL_STAGE = Path("/tmp/dgg2_codegraph/gh_wheels")
GH_BIN = os.path.expanduser("~/.local/bin/gh")
PY_BIN = os.path.expanduser("~/ml-egpu-venv/bin/python")
AUTHOR_NAME = "bumblebeers"
AUTHOR_EMAIL = "21022766+bumblebeers@users.noreply.github.com"

MAIN_GITIGNORE = """# Large model weights, raw continuous prediction caches, and scratch
scratch/
*.pt
*.safetensors
*.npy
emb.npz
emb_train.npz
data/hv/
outputs/external_review/
*.zip
*.tar.gz
*.tar

# Python / build / notebook caches
*.log
__pycache__/
*.py[cod]
build/
dist/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.venv/
"""

LIB_GITIGNORE = """__pycache__/
*.py[cod]
build/
dist/
*.egg-info/
.pytest_cache/
.venv/
"""

INTERNAL_DOC_EXCLUDES = {
    "KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md",
    "EXTERNAL_REVIEW_GUIDE.md",
    "PAPER_BLUEPRINT.md",
}

LIB_SPECS = [
    {
        "repo": "dgg-latent",
        "pkg": "dgg_latent",
        "desc": "Continuous latent diffusion sampling, losses, multimodal mask fix, and readout heads for DiffusionGemma",
        "src_dir": REPO_ROOT / "src" / "dgg_latent",
        "test_file": REPO_ROOT / "tests" / "unit" / "test_dgg_latent.py",
    },
    {
        "repo": "dgg-codebook",
        "pkg": "dgg_codebook",
        "desc": "Continuous vocabulary distribution derivation, statistical codebook de-biasing, and spatial teacher models for DiffusionGemma",
        "src_dir": REPO_ROOT / "src" / "dgg_codebook",
        "test_file": REPO_ROOT / "tests" / "unit" / "test_dgg_codebook.py",
    },
    {
        "repo": "dgg-vit-decoder",
        "pkg": "dgg_vit_decoder",
        "desc": "Standalone vision-stack-only loader and 34.15M ViT+Conv image decoder for DiffusionGemma",
        "src_dir": REPO_ROOT / "src" / "dgg_vit_decoder",
        "test_file": REPO_ROOT / "tests" / "unit" / "test_dgg_vit_decoder.py",
    },
]


def audit_tree(tree_dir: Path) -> None:
    """Fail hard if any text file contains local host paths, tokens, or unauthorized emails."""
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    local_user = Path.home().name
    binary_exts = {".png", ".jpg", ".whl", ".pyc", ".gz", ".npz", ".npy", ".pt"}
    for f in sorted(tree_dir.rglob("*")):
        if not f.is_file() or ".git" in f.parts or f.suffix in binary_exts:
            continue
        txt = f.read_text(errors="ignore")
        if local_user and local_user in txt:
            raise RuntimeError(f"Audit failed: local username path found in {f}")
        for m in email_re.findall(txt):
            if m == AUTHOR_EMAIL or re.search(r"@L\d+\.", m):
                continue
            raise RuntimeError(f"Audit failed: unauthorized email '{m}' found in {f}")


def copy_clean_dir(src: Path, dst: Path, extra_excludes: set[str] | None = None) -> None:
    extra = extra_excludes or set()

    def _ignore(cur_dir: str, names: list[str]) -> set[str]:
        base = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", "build", "dist", "*.egg-info",
            ".pytest_cache", ".ipynb_checkpoints", "*.pt", "*.safetensors", "*.npy"
        )(cur_dir, names)
        base |= (extra & set(names))
        cur_p = Path(cur_dir)
        for n in names:
            if (cur_p / n).is_symlink():
                base.add(n)
        return base

    shutil.copytree(src, dst, ignore=_ignore)


def sanitize_json_copy(src_json: Path, dst_json: Path) -> None:
    """Copy JSON report while stripping any local absolute_path fields."""
    local_user = Path.home().name
    txt = src_json.read_text(encoding="utf-8")
    if local_user and local_user in txt:
        data = json.loads(txt)
        if isinstance(data, dict) and "assets" in data and isinstance(data["assets"], list):
            for item in data["assets"]:
                if isinstance(item, dict):
                    item.pop("absolute_path", None)
                    for k, v in list(item.items()):
                        if isinstance(v, str) and local_user in v:
                            item[k] = Path(v).name
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        dst_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_json, dst_json)


def stage_standalone_libs(commit_message: str, push: bool = True) -> None:
    WHEEL_STAGE.mkdir(parents=True, exist_ok=True)
    token = Path(os.path.expanduser("~/.tokens/GITHUB_PAT")).read_text().strip()
    git_env = os.environ.copy()
    git_env["GH_TOKEN"] = token
    git_env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
    git_env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
    git_env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
    git_env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL

    py_env = os.environ.copy()
    py_env["HIP_VISIBLE_DEVICES"] = ""
    py_env["CUDA_VISIBLE_DEVICES"] = ""
    py_env["LD_LIBRARY_PATH"] = os.path.expanduser("~/ml-egpu-libs")

    license_txt = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")

    for spec in LIB_SPECS:
        repo_dir = STAGE_ROOT / spec["repo"]
        git_bak = None
        if (repo_dir / ".git").exists():
            git_bak = STAGE_ROOT / f"_{spec['repo']}_git_bak"
            if git_bak.exists():
                shutil.rmtree(git_bak)
            shutil.move(str(repo_dir / ".git"), str(git_bak))
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        pkg_dir = repo_dir / spec["pkg"]
        tests_dir = repo_dir / "tests"
        pkg_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)
        if git_bak and git_bak.exists():
            shutil.move(str(git_bak), str(repo_dir / ".git"))

        for pyf in sorted(spec["src_dir"].glob("*.py")):
            shutil.copy2(pyf, pkg_dir / pyf.name)
        shutil.copy2(spec["test_file"], tests_dir / spec["test_file"].name)
        (repo_dir / "LICENSE").write_text(license_txt, encoding="utf-8")
        (repo_dir / ".gitignore").write_text(LIB_GITIGNORE, encoding="utf-8")

        pyproj = (spec["src_dir"] / "pyproject.toml").read_text(encoding="utf-8")
        pyproj = re.sub(r"\[tool\.setuptools\.package-dir\]\n[^\n]+\n?", "", pyproj)
        if "[tool.pytest.ini_options]" not in pyproj:
            pyproj += '\n[tool.pytest.ini_options]\npythonpath = ["."]\n'
        (repo_dir / "pyproject.toml").write_text(pyproj, encoding="utf-8")

        readme = (spec["src_dir"] / "README.md").read_text(encoding="utf-8")
        if "## Related Resources" not in readme:
            readme += """
## Related Resources

- **Main Research & Benchmark Repository**: [bumblebeers/dgg2-latent-code-graphs](https://github.com/bumblebeers/dgg2-latent-code-graphs)
- **Pretrained Adapters & ImageDecoder Weights**: [bumblebeer/dgg2-latent-code-graphs-adapters](https://huggingface.co/bumblebeer/dgg2-latent-code-graphs-adapters)
- **AST Code-Graph Benchmark Packs & Raw Predictions**: [bumblebeer/dgg2-code-graph-benchmarks](https://huggingface.co/datasets/bumblebeer/dgg2-code-graph-benchmarks)
- **License**: Apache-2.0
"""
        (repo_dir / "README.md").write_text(readme, encoding="utf-8")

        # Test & build wheel
        subprocess.run([PY_BIN, "-m", "pytest", "-q"], cwd=repo_dir, env=py_env, check=True)
        dist_tmp = STAGE_ROOT / f"_{spec['repo']}_dist"
        if dist_tmp.exists():
            shutil.rmtree(dist_tmp)
        subprocess.run([PY_BIN, "-m", "pip", "wheel", "--no-deps", "-w", str(dist_tmp), "."], cwd=repo_dir, env=py_env, check=True, stdout=subprocess.DEVNULL)
        whl = list(dist_tmp.glob("*.whl"))[0]
        saved_whl = WHEEL_STAGE / whl.name
        shutil.copy2(whl, saved_whl)
        shutil.rmtree(dist_tmp)

        for junk in list(repo_dir.rglob("__pycache__")) + list(repo_dir.rglob("*.egg-info")) + [repo_dir / "build", repo_dir / ".pytest_cache"]:
            if junk.exists():
                shutil.rmtree(junk)

        audit_tree(repo_dir)

        if not (repo_dir / ".git").exists():
            subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=git_env, check=True)
        subprocess.run(["git", "config", "user.name", AUTHOR_NAME], cwd=repo_dir, env=git_env, check=True)
        subprocess.run(["git", "config", "user.email", AUTHOR_EMAIL], cwd=repo_dir, env=git_env, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, env=git_env, check=True)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, env=git_env, capture_output=True, text=True, check=True)
        if st.stdout.strip():
            subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, env=git_env, check=True)
        subprocess.run(["git", "tag", "-fa", "v0.1.0", "-m", f"{spec['repo']} v0.1.0"], cwd=repo_dir, env=git_env, check=True)

        if push:
            push_url = f"https://x-access-token:{token}@github.com/bumblebeers/{spec['repo']}.git"
            subprocess.run(["git", "push", "-u", push_url, "main"], cwd=repo_dir, env=git_env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "push", push_url, "refs/tags/v0.1.0", "--force"], cwd=repo_dir, env=git_env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(
                [GH_BIN, "release", "upload", "v0.1.0", str(saved_whl), "--clobber", "--repo", f"bumblebeers/{spec['repo']}"],
                env=git_env,
                check=True,
            )
            print(f"SYNCED LIB: https://github.com/bumblebeers/{spec['repo']} (tag v0.1.0 + wheel updated)")


def stage_main_repo(commit_message: str, push: bool = True) -> None:
    target = STAGE_ROOT / "dgg2-latent-code-graphs"
    git_backup = None
    if (target / ".git").exists():
        git_backup = STAGE_ROOT / "_main_git_bak"
        if git_backup.exists():
            shutil.rmtree(git_backup)
        shutil.move(str(target / ".git"), str(git_backup))
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    if git_backup and git_backup.exists():
        shutil.move(str(git_backup), str(target / ".git"))

    # 1. Top-level files (including reader.npz so self-test / scoring works out-of-the-box)
    for fname in ["README.md", "LICENSE", "pyproject.toml", "PREREG.json", "DEAD_ENDS.md", "reader.npz"]:
        src_f = REPO_ROOT / fname
        if src_f.exists():
            shutil.copy2(src_f, target / fname)
    (target / ".gitignore").write_text(MAIN_GITIGNORE, encoding="utf-8")

    # 2. Tracked source/doc/test directories (filtering internal strategy docs from docs/)
    copy_clean_dir(REPO_ROOT / "cg", target / "cg")
    copy_clean_dir(REPO_ROOT / "src", target / "src")
    copy_clean_dir(REPO_ROOT / "docs", target / "docs", extra_excludes=INTERNAL_DOC_EXCLUDES)
    copy_clean_dir(REPO_ROOT / "notebooks", target / "notebooks")
    copy_clean_dir(REPO_ROOT / "tests", target / "tests")

    # 3. Benchmark packs in data/ (~31 MB total, excluding data/hv/) so fresh clone passes prereg & pytest offline!
    data_dst = target / "data"
    data_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO_ROOT / "data").iterdir()):
        if f.is_file() and f.name.endswith((".npz", ".json.gz", ".nodes.json")):
            shutil.copy2(f, data_dst / f.name)
    if (REPO_ROOT / "data" / "graphs").exists():
        copy_clean_dir(REPO_ROOT / "data" / "graphs", data_dst / "graphs")

    # 4. Small JSON summary reports & decoded grids in outputs/ for full metric traceability
    outputs_src = REPO_ROOT / "outputs"
    if outputs_src.exists():
        for jf in sorted(outputs_src.rglob("*.json")):
            if "external_review" in jf.parts or "external_stage" in jf.parts:
                continue
            if jf.stat().st_size <= 5 * 1024 * 1024:  # <= 5 MB JSON summaries
                rel = jf.relative_to(REPO_ROOT)
                sanitize_json_copy(jf, target / rel)
        for extra_rel in ["outputs/m1/reader.npz", "outputs/m2/decoded_grids_all_arms.npz"]:
            ef = REPO_ROOT / extra_rel
            if ef.exists():
                dst_ef = target / extra_rel
                dst_ef.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ef, dst_ef)

    audit_tree(target)

    # 5. Fresh-clone verification BEFORE committing/pushing!
    py_env = os.environ.copy()
    py_env["HIP_VISIBLE_DEVICES"] = ""
    py_env["CUDA_VISIBLE_DEVICES"] = ""
    py_env["LD_LIBRARY_PATH"] = os.path.expanduser("~/ml-egpu-libs")
    print("[Verify] Running cg_prereg.py --check on staged fresh-clone tree...")
    subprocess.run([PY_BIN, "cg/cg_prereg.py", "--check"], cwd=target, env=py_env, check=True)
    print("[Verify] Running cg_score_preds.py --self-test on staged fresh-clone tree...")
    subprocess.run([PY_BIN, "cg/cg_score_preds.py", "--self-test"], cwd=target, env=py_env, check=True)
    print("[Verify] Running pytest -q tests/unit tests/e2e on staged fresh-clone tree...")
    subprocess.run([PY_BIN, "-m", "pytest", "-q", "tests/unit", "tests/e2e"], cwd=target, env=py_env, check=True)

    # Clean pytest caches created during verification
    for junk in list(target.rglob("__pycache__")) + [target / ".pytest_cache"]:
        if junk.exists():
            shutil.rmtree(junk)

    token = Path(os.path.expanduser("~/.tokens/GITHUB_PAT")).read_text().strip()
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
    env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
    env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL

    if not (target / ".git").exists():
        subprocess.run(["git", "init", "-b", "main"], cwd=target, env=env, check=True)
    subprocess.run(["git", "config", "user.name", AUTHOR_NAME], cwd=target, env=env, check=True)
    subprocess.run(["git", "config", "user.email", AUTHOR_EMAIL], cwd=target, env=env, check=True)
    subprocess.run(["git", "add", "-A"], cwd=target, env=env, check=True)

    status = subprocess.run(["git", "status", "--porcelain"], cwd=target, env=env, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        subprocess.run(["git", "commit", "-m", commit_message], cwd=target, env=env, check=True)
    else:
        print("[dgg2-latent-code-graphs] Working tree clean; nothing new to commit.")

    subprocess.run(["git", "tag", "-fa", "v0.1.0", "-m", "dgg2-latent-code-graphs v0.1.0"], cwd=target, env=env, check=True)

    if push:
        push_url = f"https://x-access-token:{token}@github.com/bumblebeers/dgg2-latent-code-graphs.git"
        subprocess.run(["git", "push", "-u", push_url, "main"], cwd=target, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "push", push_url, "refs/tags/v0.1.0", "--force"], cwd=target, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rel_chk = subprocess.run([GH_BIN, "release", "view", "v0.1.0", "--repo", "bumblebeers/dgg2-latent-code-graphs"], env=env, capture_output=True)
        if rel_chk.returncode != 0:
            subprocess.run(
                [
                    GH_BIN, "release", "create", "v0.1.0",
                    "--repo", "bumblebeers/dgg2-latent-code-graphs",
                    "--title", "dgg2-latent-code-graphs v0.1.0",
                    "--notes", "Initial v0.1.0 benchmark, runners, offline scoring harness, and verifiable summary bundle.",
                ],
                env=env,
                check=True,
            )
        print("SYNCED MAIN REPO: https://github.com/bumblebeers/dgg2-latent-code-graphs (tag v0.1.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--message", default="Fix git format-patch, bundle benchmark packs & JSON summaries for offline fresh-clone tests, add library usage examples, and remove internal docs")
    ap.add_argument("--skip-libs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.skip_libs:
        stage_standalone_libs(commit_message=args.message, push=not args.dry_run)
    stage_main_repo(commit_message=args.message, push=not args.dry_run)


if __name__ == "__main__":
    main()
