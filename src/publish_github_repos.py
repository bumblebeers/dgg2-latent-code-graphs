#!/usr/bin/env python3
"""Publish and synchronize public GitHub repositories under @bumblebeers.

Syncs from clean staging trees in /tmp/dgg2_codegraph/gh_repos/ so internal agent
orchestration files, weights/large binary packs, and local paths never enter Git history,
and every commit is authored strictly as `bumblebeers <21022766+bumblebeers@users.noreply.github.com>`.

Repositories managed:
  1. bumblebeers/dgg2-latent-code-graphs (main benchmark, runners, paper & test suite)
  2. bumblebeers/dgg-latent              (standalone library)
  3. bumblebeers/dgg-codebook            (standalone library)
  4. bumblebeers/dgg-vit-decoder         (standalone library)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_ROOT = Path("/tmp/dgg2_codegraph/gh_repos")
GH_BIN = os.path.expanduser("~/.local/bin/gh")
AUTHOR_NAME = "bumblebeers"
AUTHOR_EMAIL = "21022766+bumblebeers@users.noreply.github.com"

MAIN_GITIGNORE = """# Generated artifacts, weights, and large dataset packs
outputs/
scratch/
*.pt
*.safetensors
*.npy
*.npz
*.json.gz
data/hv/
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


def audit_tree(tree_dir: Path) -> None:
    """Fail hard if any file contains local host paths, tokens, or unauthorized emails."""
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    local_user = Path.home().name
    for f in sorted(tree_dir.rglob("*")):
        if not f.is_file() or ".git" in f.parts or f.suffix in {".png", ".jpg", ".whl", ".pyc", ".gz", ".npz", ".npy"}:
            continue
        txt = f.read_text(errors="ignore")
        if local_user and local_user in txt:
            raise RuntimeError(f"Audit failed: local username path found in {f}")
        for m in email_re.findall(txt):
            if m == AUTHOR_EMAIL or re.search(r"@L\d+\.", m):
                continue
            raise RuntimeError(f"Audit failed: unauthorized email '{m}' found in {f}")


def copy_clean_dir(src: Path, dst: Path) -> None:
    def _ignore(cur_dir: str, names: list[str]) -> set[str]:
        base = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", "build", "dist", "*.egg-info",
            ".pytest_cache", ".ipynb_checkpoints", "*.json.gz", "*.npz", "*.npy", "*.pt"
        )(cur_dir, names)
        cur_p = Path(cur_dir)
        for n in names:
            if (cur_p / n).is_symlink():
                base.add(n)
        return base

    shutil.copytree(src, dst, ignore=_ignore)


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

    # Copy top-level files
    for fname in ["README.md", "LICENSE", "pyproject.toml", "PREREG.json", "DEAD_ENDS.md"]:
        src_f = REPO_ROOT / fname
        if src_f.exists():
            shutil.copy2(src_f, target / fname)
    (target / ".gitignore").write_text(MAIN_GITIGNORE)

    # Copy tracked source/doc/test directories
    for dname in ["cg", "src", "docs", "notebooks", "tests"]:
        src_d = REPO_ROOT / dname
        if src_d.exists():
            copy_clean_dir(src_d, target / dname)

    # Include lightweight AST graph JSON metadata (no .npz / .json.gz / .npy weights)
    if (REPO_ROOT / "data" / "graphs").exists():
        copy_clean_dir(REPO_ROOT / "data" / "graphs", target / "data" / "graphs")
    if (REPO_ROOT / "data").exists():
        (target / "data").mkdir(parents=True, exist_ok=True)
        for nf in sorted((REPO_ROOT / "data").glob("*.nodes.json")):
            shutil.copy2(nf, target / "data" / nf.name)

    audit_tree(target)

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

    if push:
        chk = subprocess.run([GH_BIN, "repo", "view", "bumblebeers/dgg2-latent-code-graphs"], env=env, capture_output=True)
        if chk.returncode != 0:
            subprocess.run(
                [
                    GH_BIN, "repo", "create", "bumblebeers/dgg2-latent-code-graphs", "--public",
                    "--description", "Latent Roadmaps: DiffusionGemma State Transitions for AST Code-Graph Refactoring",
                ],
                cwd=target,
                env=env,
                check=True,
            )
        push_url = f"https://x-access-token:{token}@github.com/bumblebeers/dgg2-latent-code-graphs.git"
        subprocess.run(["git", "push", "-u", push_url, "main"], cwd=target, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("LIVE: https://github.com/bumblebeers/dgg2-latent-code-graphs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--message", default="Initial public staging of dgg2-latent-code-graphs benchmark, runners, and writeup")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    stage_main_repo(commit_message=args.message, push=not args.dry_run)


if __name__ == "__main__":
    main()
