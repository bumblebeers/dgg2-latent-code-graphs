# Kaggle Gemma 4 Developer Agent Paper Track — Competition Brief, Rubric & Reviewer Guide

- **Competition URL:** https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
- **Submission Requirements URL:** https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview/submission-requirements
- **Hard Submission Deadline:** **November 12, 2026 at 11:59 PM UTC** (`17:59 CST`). Any un-submitted or draft Writeups at the deadline are not considered by the Judges.
- **Tie-Breaker Rule:** In the event of a tie, the Paper that was entered first to the Competition wins.

---

## 1. Official Competition Scope & Eligibility (Verbatim Summary)

> You will submit a Kaggle Writeup that describes your methods, experiments, and other details. Submissions can include any related research, as long as it is unpublished.
>
> To be eligible to win The Gemma 4 Developer Agent Paper Track Prize, you must submit a Writeup that documents your research to advance the state-of-the-art for agentic software engineering.

### Required Writeup Structure & Platform Constraints
1. **Title (`<= 80` characters max):**
   - Current (`71 / 80` chars): `Latent Roadmaps: DiffusionGemma World Models for Code-Graph Refactoring`
2. **Subtitle (`<= 140` characters max — *"Explain your project in one sentence"*):**
   - Current (`131 / 140` chars): `We adapt DiffusionGemma's bidirectional canvas to plan multi-step AST code-graph refactorings in latent space before emitting code.`
3. **Card and Thumbnail Image (`560 x 280` pixels exact):**
   - Current: `docs/figures/kaggle_card_thumbnail_560x280.png` (focal banner displayed on the Kaggle competition listing and Writeup header).
4. **Body Length Limit:** **`3,000` words maximum**, reporting original, unpublished research.
   - Current (`docs/paper_draft.md`): **`2,405` whitespace-delimited words** (`2,978 / 3,000` under strict `\b\w+\b` regex word-token counting).
5. **Required Paper Sections:**
   - Title and subtitle
   - Abstract
   - Introduction
   - Description of the research, including methods and experiments
   - Related works and citations
6. **(Optional) Public Notebook:**
   - Submitted in the *Project Links* field; must be publicly accessible without login or paywall (or a Kaggle Notebook that auto-publishes after the deadline).
   - Current: `notebooks/dgg2_latent_code_graphs_demo.ipynb` (generated reproducibly via `src/build_review_notebook.py`).
7. **(Optional) Additional Assets & Suggested Research Themes:**
   - **Tuning & Optimization:** Parameter-efficient fine-tuning and RL approaches tailored to improve SWE agent capabilities.
   - **Code Comprehension:** Novel techniques for code-graph generation, parsing, and embedding to enhance deep comprehension of complex repositories.
   - **Tasks & Benchmarks:** Development of new tasks, evaluation datasets, or resources targeting code structure, graphs, and structured code generation.
   - **Graph Reasoning:** Methods and architectures for improving LLM reasoning over large-scale graphs.

---

## 2. Official Evaluation Rubric (5 Criteria, 0–5 Scale Each, Equal Weight)

Submissions for Paper Awards are evaluated equally across five criteria, each scored from **0 (lowest) to 5 (highest)**, with the final score calculated as the unweighted average of all five:

| Criterion | Official Competition Description |
| :--- | :--- |
| **Novelty** | Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? |
| **Quality** | How general and universal is the Submission approach beyond the competition? Does your submission translate to other similar problems? How well does your method generalize? |
| **Relevance** | Is there significance and/or potential impact of the research on the field of software engineering and agentic learning? |
| **Verifiability** | Does the paper include sufficient information to understand how an innovation works; to understand how data was obtained, analyzed, and interpreted? |
| **Clarity** | Is the paper clearly presented and written? |

---

## 3. Official Prize Categories

1. **Overall Best Paper**
   - Awarded to the submission achieving the highest average score across all five rubric criteria (`Novelty`, `Quality`, `Relevance`, `Verifiability`, `Clarity`).
2. **Best New Resource (Dataset, library, etc.)**
   - *"A resource introduces a new dataset, tool, or software, for example. This is compared to an empirical paper which generates new knowledge/experiments. For examples of each, a new 'dataset' could leverage the provided embeddings or other sources to create a dataset that can be used for either evaluation of Gemma 4 or for fine-tuning it. A new 'tool' could be a library which uses Gemma and some other algorithm to cluster code based on underlying behavior. New 'software' could be a dashboard to explore Github repos that are similar in structure to a target repo."*
3. **Best New Application**
   - *"A novel application is a new use-case (e.g. using code-graphs for writing TPU kernels). Show your creative uses of code-graphs and/or the provided embeddings to create novel use-cases. This may involve creating new evaluation datasets for entirely novel tasks. It could also involve application to existing benchmarks that differ from the provided ones, and using the code-graphs or embeddings where they have not been used before."*

---

## 4. Strategic Context & High-Value Questions for External Review

### Strategic Positioning
This research originated by turning **DiffusionGemma 26B-A4B** (Gemma 4's bidirectional discrete text-diffusion derivative with a frozen SigLIP vision tower) into a latent-space visual world model on natural images (§26–§27), and then systematically tested how those exact bidirectional canvas mechanisms transfer to **software engineering AST code-graph refactoring and repository symbol embeddings** (Experiments A & B), while packaging the underlying engineering stack into reusable open-source libraries, benchmarks, adapters, and an upstream `huggingface/transformers` bugfix.

Because the work bridges multimodal latent diffusion and code-graph world modeling rather than benchmarking an end-to-end SWE-bench bash loop, **Relevance** is our tightest rubric dimension, whereas **Best New Resource** and **Best New Application** (and high marks on **Novelty**, **Quality**, **Verifiability**, and **Clarity**) are our strongest competitive levers.

### Focused Questions for Your Review
1. **Relevance & Framing (`§0–§2`, `§8`):** Does the *"Latent Roadmaps"* framing convincingly connect bidirectional discrete diffusion over AST call graphs to real pains in agentic software engineering (causal error compounding, multi-step refactoring dependencies, committing to one coherent branch under ambiguity), without overclaiming end-to-end syntax emission?
2. **Clarity Under the Strict `3,000`-Word Ceiling (`docs/paper_draft.md`):** Currently at `2,405` whitespace words (`2,978` regex word tokens, leaving almost zero slack under `\b\w+\b <= 3000` if punctuation-split tokens are counted, or ~595 words of slack under standard whitespace word counting). Which passages feel too dense/telegraphic, and what would you cut or expand?
3. **Scientific Rigor & Honesty (`§3`, `§4`, `§5.1–§5.5`):**
   - Are the **positive claims** (Exp A 2-seed LoRA DG `91.9%` seen, `78.5%` unseen paraphrase, `61.4%` indirect intent, `99.3%` commuting 1-pass compounds, `82.1%` 2-step soft-token re-feeding, `90.8%` 3-step `keep_callers` mirror synthesis, `56.1%` open-ended commitment) watertight against the Non-DG, Frozen DG, shuffled, and source-copy controls?
   - Are the **negative/boundary claims** (C06 sub-patch orientation retraction, `32²/48²` sparse-class degradation `13.2%`, `mark_ancestors = 0.0%`, and Experiment B's cold-start continuous symbol drift `18.0×` collapsing non-delete edits to `2.2%` vs. `100%` delete recall) explained clearly as constructive architectural laws rather than excuses?
4. **Best New Resource & Best New Application Appeal (`§7`, `src/dgg_*`, `src/patches/`):** Do the 3 standalone packages (`dgg_latent`, `dgg_codebook`, `dgg_vit_decoder`), the upstream `transformers` multimodal prefill mask patch (`dg_attn_fix`), and the pre-registered `httpx` + `click` AST refactoring benchmarks read as compelling, genuinely reusable community artifacts?
5. **Visual Assets (`kaggle_card_thumbnail_560x280.png` & `figure1_hero_multimodal_roadmap.png`):** Are both images immediately legible, accurate, and visually compelling at their target display sizes?
