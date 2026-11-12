# Related Works & Literature Synthesis: Latent Roadmaps via Soft-Token Re-Feeding in DiffusionGemma

This document synthesizes the related literature surrounding cross-generation soft-token re-feeding in DiffusionGemma across five intersecting bodies of research: (1) Visual Program Synthesis & Latent World Models, (2) Code Graph Representation Learning, (3) Diffusion Language Models & Discrete Denoising, (4) Multimodal LLMs for Software Engineering, and (5) Representation Degeneration, Anisotropy, and Latent Alignment.

> **Two Key Methodological Distinctions:**
> 1. **Intra-generation self-conditioning vs. cross-generation vision-latent re-feeding:** DiffusionGemma's stock self-conditioning feeds soft token embeddings back between denoising steps *within* a single generation pass; our re-feeding projects the predicted vision latents ($\hat{\mathbf{h}}_v \in \mathbb{R}^{256 \times 1152}$) back through frozen `embed_vision` *across* complete generations alongside the next natural-language instruction.
> 2. **Purpose-trained discrete-diffusion world models vs. pretrained text-diffusion LLM adaptation:** Discrete-diffusion world models such as **Discrete-WAM** (Z. Yao et al., *arXiv:2606.05645*, 2026) train unified vision-action token editors specifically for world-policy rollouts; to our knowledge, our work is the first adaptation of a pretrained text-diffusion LLM (`google/diffusiongemma-26B-A4B-it`) via parameter-efficient LoRA and a lightweight readout head to execute chained vision-latent state transitions.

---

## 1. Literature Synthesis Across Five Core Themes

### 1.1 Visual Program Synthesis & Latent World Models
A major limitation of classical agentic program synthesis is the reliance on autoregressive textual generation to plan multi-step refactorings. When an agent attempts large-scale architectural restructuring, generating code left-to-right token-by-token forces early commitments before the downstream consequences on callers, callees, and interfaces are fully visible. 

Recent advances in abstraction and reasoning have increasingly turned to continuous latent representations and structured world models. **LPN** (*Searching Latent Program Spaces*, NeurIPS 2024) demonstrated that program synthesis can be factorized into continuous program embeddings $z$ optimized at test time, decoupling specification search from syntax rendering. In the domain of ARC-AGI and Moore-machine synthesis, **Tycho** (Lehmann et al., 2026) showed that maintaining explicit executable programmatic world models enables active hypothesis testing and targeted repair. Furthermore, **DreamTeam** (Sarafian et al., 2026) established the "workspace optimization" principle: once frontier model weights are frozen, the primary optimization surface is the structured external substrate of representations, state traces, and tests that the agent queries.

Concurrently, a wave of research has demonstrated the power of continuous latent "thinking" within multimodal language models. **Coconut** (Hao et al., FAIR/UCSD, 2024) introduced continuous chain-of-thought, feeding the model's final hidden states directly back as subsequent inputs without discrete lexical projection, thereby bypassing the discrete token bottleneck. **COVT** (Qin et al., 2025) and **UniVLR** (Jiang et al., 2026) extend this concept to visual reasoning: rather than interleaving natural language tokens, models generate continuous visual tokens that capture spatial intermediate states. **Latent-Sketchpad** (Zhang et al., MSR, 2025) complements this by generating continuous visual latents that are rendered via lightweight sketch decoders for human interpretability. In the physical realm, world models such as **DINO-WM** (2024), **LeWorldModel** (Maes, LeCun, Balestriero, 2026), and **V-JEPA 2** (Assran et al., Meta FAIR, 2025) prove that predicting forward dynamics directly in pre-trained feature spaces (such as DINOv2 or SigLIP) yields sample-efficient, robust planning without the computational expense of pixel-level video generation.

*How our work extends this:* We port the concept of a latent world model into agentic software engineering. Our "Latent Roadmap" uses DiffusionGemma not to generate source code directly, but to transition the *entire latent state of the codebase* (represented either as a $16 \times 16$ call graph or 256 symbol embeddings) in response to an instruction. This provides an execution-free, differentiable structural plan that downstream developer agents can follow.

---

### 1.2 Code Graph Representation Learning
Programs are fundamentally non-linear structures governed by Abstract Syntax Trees (ASTs), Control Flow Graphs (CFGs), and Call Graphs. Linearizing code into token sequences discards long-range dependency paths and structural invariants.

Foundational work by **Yamaguchi et al.** (IEEE S&P 2014) established the **Code Property Graph (CPG)**, combining ASTs, CFGs, and Program Dependence Graphs (PDGs) into a unified joint data structure for static vulnerability detection. **Allamanis et al.** (*Learning to Represent Programs with Graphs*, ICLR 2018) pioneered Graph Neural Networks (GNNs) over gated graph formulations of source code, demonstrating that semantic relationships (e.g., `LastRead`, `ComputedFrom`) significantly improve variable naming and type inference. **GraphCodeBERT** (Guo et al., ICLR 2021) showed that incorporating data flow graphs into pre-trained transformers improves code search and code refinement by explicitly modeling semantic variable dependencies rather than superficial token ordering. **ProGraML** (Cummins et al., ICML 2021) generalized program graph representations across compiler IRs, establishing standardized message-passing benchmarks for compiler optimization and data-flow analysis.

In parallel, large-scale code foundation models such as **StarCoder** (Li et al., 2023), **Code Llama** (Rozière et al., 2023), and **DeepSeek-Coder** (Guo et al., 2024) demonstrated mastery of textual code generation, yet repeatedly exhibit vulnerabilities when reasoning about global repository topology, cross-file call dependencies, and transitive breakages during refactoring.

*How our work extends this:* Rather than treating code graphs merely as auxiliary inputs for message-passing GNNs or linearized text prompts for autoregressive LLMs, we formulate the code graph's adjacency matrix directly as a visual/latent canvas. By rendering 16-symbol AST call graphs into $16 \times 16$ adjacency grids ($768 \times 768$ RGB), we map graph topology directly onto the patch tokenization of vision transformers, allowing the model's bidirectional attention to operate across all graph edges simultaneously.

---

### 1.3 Diffusion Language Models & Discrete Denoising
Autoregressive language models decode left-to-right under a strictly causal attention mask ($O(N)$ sequential forward passes). While effective for conversational fluency, this unidirectional factorization prevents the model from revising earlier positions based on downstream commitments.

Discrete Diffusion Language Models (dLLMs) reformulate text generation as iterative denoising over a sequence under full bidirectional attention. **LLaDA** (Nie et al., ICML 2025) and **Scaling-DLM** (Gong et al., ICLR 2025) demonstrated that pre-trained autoregressive models can be adapted via continual pre-training into masked diffusion models that predict masked tokens in parallel. **DiffusionGemma** (DiffusionGemma Team, Google DeepMind, 2026; Taïga et al., arXiv:2608.00146) adapted the Gemma 4 26B-A4B Mixture-of-Experts architecture into a high-throughput discrete diffusion model refining a fixed 256-token canvas over multiple self-conditioned denoising steps. Mechanistic audits by **Asaria, Salomone & Gandhi** (2026) revealed that DiffusionGemma does not commit tokens in a simple left-to-right or strictly parallel fashion; rather, it commits non-linearly across the canvas based on entropy thresholds. Interpretability investigations by **Engels et al.** (Google DeepMind, 2026) verified that intermediate canvas activations act as interpretable semantic bottlenecks.

Furthermore, unified multimodal architectures such as **Transfusion** (Zhou et al., Meta, 2024) and **Show-o** (Xie et al., ICLR 2025) have unified next-token autoregression for text with continuous/discrete diffusion for images within a single shared transformer trunk. **Fast-dVLM** (Wu et al., 2026) proved that converting an autoregressive vision-language model into a block-diffusion VLM preserves visual understanding while enabling ultra-fast parallel generation. In decoding strategy, **PVF** (*Plan, Verify and Fill*, Li et al., 2026) demonstrated that structured parallel decoding over diffusion tokens provides formal verification and self-correction during generation.

*How our work extends this:* We leverage DiffusionGemma's native 256-token canvas and bidirectional attention as a global constraint solver for code refactoring. Unlike autoregressive generation, which suffers from irreversible error compounding, DG refines all 256 positions in parallel, allowing callers and callees to adapt mutually to refactoring instructions.

---

### 1.4 Multimodal LLMs for Software Engineering
A burgeoning frontier in software engineering explores using visual and diagrammatic representations to assist software design. Multimodal architectures like **PaLM-E** (Driess et al., ICML 2023) and **RT-2** (Brohan et al., DeepMind, 2023) demonstrated that continuous perceptual tokens from frozen vision towers can be projected directly into language model embedding spaces to ground reasoning in physical environments.

In software engineering, architectural diagrams, call graphs, UML schemas, and UI wireframes represent standard engineering media. Tools such as **GILL** (Koh et al., CMU, 2023) pioneered bidirectional mapping between text embedding spaces and visual diffusion latents. However, prior developer agents have treated visual inputs primarily as passive references (e.g., screenshot-to-code). The generative manipulation of visual code structures—where an agent reads an architecture diagram and emits the updated diagram reflecting proposed structural changes—remained unexplored until this work.

---

### 1.5 Anisotropy, Representation Degeneration & Latent Alignment
When operating directly in neural latent spaces (such as SigLIP $\mathbf{h}_v$ or transformer hidden states), models encounter the well-documented **representation degeneration problem**. Foundational studies by **Gao et al.** (ICLR 2019), **Ethayarajh** (EMNLP 2019), and **Li et al.** (EMNLP 2020) demonstrated that contextualized representations in language models and vision transformers are strongly **anisotropic**: token vectors cluster within a narrow cone, yielding pairwise cosine similarities approaching $\approx 0.99$ and participation ratios as low as $2.0$.

In our parent project (§26–§27), raw SigLIP $\mathbf{h}_v$ vectors and mean-pooled DG hidden states exhibited extreme anisotropy ($\cos \approx 0.30$ to $0.99$), rendering raw Euclidean or cosine metric spaces incapable of distinguishing subtle transformations. Standardizing vectors ($z = (x - \mu) / \sigma$) and centering cosines ($\text{cent\_cos}$) are mathematically mandatory prior to any $k$-NN lookup or graph construction.

To regularize generative models operating in latent spaces, **REPA** (*Representation Alignment for Generation*, Yu et al., ICLR 2025) proposed projecting noisy diffusion states toward frozen clean self-supervised representations. **PixelREPA** (Shin et al., 2026) studied the limits of this alignment in pixel vs latent diffusion. **GAP** (*Granular Alignment Paradigm*, Miao et al., Alibaba Qwen, 2026) analyzed norm and PCA collapse in visual latent reasoning, proposing per-token normalization. Finally, **LeJEPA** (Balestriero & LeCun, 2025) introduced **SIGReg** (Sketched Isotropic Gaussian Regularization), utilizing random 1D projections and the Epps-Pulley test to enforce strict isotropy across latent representations without heuristic contrastive negatives.

*How our work builds on this:* We implement rigorous pre-flight representation gates (Gate A2 and Gate B0) that evaluate the participation ratio, centered cosine, and visibility gap of candidate states. In Experiment B, our trainable input adapter ($\text{LayerNorm}(d) + \text{Linear}(d, 2816)$) explicitly normalizes MiniLM symbol embeddings to match the empirical RMS scale of real `embed_vision` soft tokens, ensuring stable gradient flow and preventing copy collapse.

---

## 2. Verified BibTeX Citations Formatted for Publication

```bibtex
@article{diffusiongemma2026,
  title={{DiffusionGemma Technical Report}},
  author={{DiffusionGemma Team, Google DeepMind} and Ta{\"{i}}ga, Adrien Ali and Assiene, James and Calandriello, Daniele and others},
  journal={arXiv preprint arXiv:2608.00146},
  year={2026}
}

@article{asaria2026diffusiongemma,
  title={{Neither Parallel Nor Sequential: How DiffusionGemma Actually Commits Tokens}},
  author={Asaria, Cyrus and Salomone, Marco and Gandhi, Kanishk},
  journal={arXiv preprint arXiv:2606.14620},
  year={2026}
}

@article{engels2026transparency,
  title={{How Transparent is DiffusionGemma?}},
  author={Engels, Joshua and McDougall, Euan and Chughtai, Bilal and Kramar, Jan and Rajamanoharan, Senthooran and others},
  journal={arXiv preprint arXiv:2606.20560},
  year={2026}
}

@inproceedings{nie2025llada,
  title={{Large Language Diffusion Models}},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}

@inproceedings{gong2025scaling,
  title={{Scaling Diffusion Language Models via Adaptation from Autoregressive Models}},
  author={Gong, Shansan and Agarwal, Shivam and Zhang, Yizhe and Ye, Jiacheng and Zheng, Lin and Li, Mukai and An, Chenxin and Zhao, Peilin and Bi, Wei and Han, Jiawei and Peng, Hao and Kong, Lingpeng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{zhou2024transfusion,
  title={{Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model}},
  author={Zhou, Chunting and Yu, Lili and Babu, Arun and Tirumala, Kushal and Yasunaga, Michihiro and Shamis, Leonid and Kahn, Jacob and Ma, Xuezhe and Zettlemoyer, Luke and Levy, Omer},
  journal={arXiv preprint arXiv:2408.11039},
  year={2024}
}

@inproceedings{xie2025showo,
  title={{Show-o: One Single Transformer to Unify Multimodal Understanding and Generation}},
  author={Xie, Jinheng and Mao, Weijia and Bai, Zechen and Zhang, David Junhao and Wang, Weihao and Lin, Kevin Qinghong and Gu, Yuchao and Lin, Zhijie and Chen, Wenhu and Shou, Mike Zheng and Tang, Andy and Li, Zhenguo and Fu, Jianlong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{wu2026fastdvlm,
  title={{Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM}},
  author={Wu, Chengyue and Lan, Shiyi and Fu, Yonggan and Gao, Sensen and Wang, Jin and Yu, Jincheng and Alvarez, Jose M. and Molchanov, Pavlo and Luo, Ping and Han, Song and Zhu, Ligeng and Xie, Enze},
  journal={arXiv preprint arXiv:2604.06832},
  year={2026}
}

@article{li2026pvf,
  title={{Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models}},
  author={Li, Ziyang and Jiang, Yizhe and Cheng, Yu and Fu, Jie and Cai, Tianle and Huang, Furong and Chen, Danqi and Ye, Hanrui and Chen, Jiacheng and Van Hentenryck, Pascal},
  journal={arXiv preprint arXiv:2601.12247},
  year={2026}
}

@inproceedings{lpn2024latentprogram,
  title={{Searching Latent Program Spaces}},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

@article{lehmann2026tycho,
  title={{Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3}},
  author={Lehmann, Jens and Aioanei, Andrei and Vahdati, Sahar},
  journal={arXiv preprint arXiv:2607.28287},
  year={2026}
}

@article{sarafian2026workspace,
  title={{Workspace Optimization: How to Train Your Agent}},
  author={Sarafian, Elad and Kaplun, Gal and Banner, Ron and Soudry, Daniel and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2605.09650},
  year={2026}
}

@article{arcagi32026,
  title={{ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence}},
  author={{ARC Prize Foundation} and Chollet, Fran{\c{c}}ois and Knoop, Mike and Kamradt, Greg and Henry, Bryan and others},
  journal={arXiv preprint arXiv:2603.24621},
  year={2026}
}

@article{hao2024coconut,
  title={{Training Large Language Models to Reason in a Continuous Latent Space}},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}

@article{qin2025covt,
  title={{Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens}},
  author={Qin, Yujia and Wei, Yining and Ge, Chongyi and Kallidromitis, Dimitris and Fu, Chaoyou and Darrell, Trevor and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2511.19418},
  year={2025}
}

@article{zhang2025latentsketchpad,
  title={{Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs}},
  author={Zhang, Huanyu and Wu, Wenshan and Li, Chengzu and Shang, Ning and Xia, Yan and Huang, Yangyu and Zhang, Yifan and Dong, Li and Zhang, Zhang and Wang, Liang and Tan, Tieniu and Wei, Furu},
  journal={arXiv preprint arXiv:2510.24514},
  year={2025}
}

@article{jiang2026univlr,
  title={{UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs}},
  author={Jiang, Houcheng and Fu, Jiajun and Fang, Junfeng and Gao, Chen and Wang, Xiang and He, Xiangnan and Li, Yong},
  journal={arXiv preprint arXiv:2605.11856},
  year={2026}
}

@article{dinowm2024,
  title={{DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning}},
  author={Anonymous},
  journal={arXiv preprint arXiv:2411.04983},
  year={2024}
}

@article{maes2026leworldmodel,
  title={{LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels}},
  author={Maes, Quentin and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}

@article{assran2025vjepa2,
  title={{V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning}},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Majid and others},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}

@inproceedings{yamaguchi2014cpg,
  title={{Modeling and Discovering Vulnerabilities with Code Property Graphs}},
  author={Yamaguchi, Fabian and Golde, Nico and Arp, Daniel and Rieck, Konrad},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  pages={590--604},
  year={2014}
}

@inproceedings{allamanis2018graphs,
  title={{Learning to Represent Programs with Graphs}},
  author={Allamanis, Miltiadis and Brockschmidt, Marc and Tarlow, Daniel},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@inproceedings{guo2021graphcodebert,
  title={{GraphCodeBERT: Pre-training Code Representations with Data Flow}},
  author={Guo, Daya and Ren, Shuo and Lu, Shuai and Zhang, Zhang and Feng, Zhou and Guo, Duyu and Tang, Jiayu and Zhou, Deyu and Gong, Ming and Shi, Lin and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{cummins2021programl,
  title={{ProGraML: A Graph-based Program Representation for Data Flow Analysis and Compiler Optimizations}},
  author={Cummins, Chris and Fochesato, Zach and Allamanis, Miltiadis and Sarkar, Vivek},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={2244--2253},
  year={2021}
}

@article{li2023starcoder,
  title={{StarCoder: may the source be with you!}},
  author={Li, Raymond and Allal, Loubna Ben and Zi, Yangtian and Muennighoff, Niklas and Kocetkov, Denis and Mou, Chenghao and Marone, Marc and Akiki, Christopher and Li, Jia and Chim, Jenny and others},
  journal={arXiv preprint arXiv:2305.06161},
  year={2023}
}

@article{roziere2023codellama,
  title={{Code Llama: Open Foundation Models for Code}},
  author={Rozi{\`e}re, Baptiste and Gehring, Jonas and Gloeckle, Fabian and Sootla, Sten and Gat, Itai and Tan, Xiaoqing Ellen and Adi, Yossi and Liu, Jingyu and Sauvestre, Romain and Remez, Tal and others},
  journal={arXiv preprint arXiv:2308.12950},
  year={2023}
}

@article{guo2024deepseekcoder,
  title={{DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence}},
  author={Guo, Daya and Zhu, Qihao and Yang, Dejian and Xie, Zhenda and Dong, Kai and Zhang, Wentao and Chen, Guanting and Bi, Xiao and Wu, Y and Li, Y. K. and others},
  journal={arXiv preprint arXiv:2401.14196},
  year={2024}
}

@inproceedings{driess2023palme,
  title={{PaLM-E: An Embodied Multimodal Language Model}},
  author={Driess, Danny and Xia, Fei and Sajjadi, Mehdi S. M. and Lynch, Corey and Chowdhery, Aakanksha and Ichter, Brian and Wahid, Ayzaan and Tompson, Jonathan and Vuong, Quan and Yu, Tianhe and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}

@article{brohan2023rt2,
  title={{RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control}},
  author={Brohan, Anthony and Chebotar, Yevgen and Driess, Danny and Finn, Chelsea and Hausman, Karol and Levine, Sergey and Xia, Fei and others},
  journal={arXiv preprint arXiv:2307.15818},
  year={2023}
}

@article{koh2023gill,
  title={{Generating Images with Multimodal Language Models}},
  author={Koh, Jing Yu and Fried, Daniel and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2305.17216},
  year={2023}
}

@inproceedings{gao2019representation,
  title={{Representation Degeneration Problem in Language Modeling}},
  author={Gao, Jun and He, Di and Tan, Xu and Qin, Tao and Wang, Liwei and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@inproceedings{ethayarajh2019contextual,
  title={{How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings}},
  author={Ethayarajh, Kawin},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={55--65},
  year={2019}
}

@inproceedings{li2020bertflow,
  title={{On the Sentence Embeddings from Pre-trained Language Models}},
  author={Li, Bohan and Zhou, Hao and He, Junxian and Wang, Mingxuan and Yang, Yiming and Li, Lei},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={9119--9130},
  year={2020}
}

@inproceedings{yu2025repa,
  title={{Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think}},
  author={Yu, Sihyun and Kwak, Sangkyung and Jang, Huiwon and Jeong, Jongheon and Huang, Jonathan and Shin, Jinwoo and Xie, Saining},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{shin2026pixelrepa,
  title={{Representation Alignment for Just Image Transformers is not Easier than You Think}},
  author={Shin, Seungwoo and Kim, Junhyeok and Shim, Kyomin},
  journal={arXiv preprint arXiv:2603.14366},
  year={2026}
}

@article{miao2026gap,
  title={{Fill the GAP: A Granular Alignment Paradigm for Visual Reasoning in Multimodal Large Language Models}},
  author={Miao, Yisong and Sun, Shenghao and Wang, Peng and Zhou, Jingren and others},
  journal={arXiv preprint arXiv:2605.12374},
  year={2026}
}

@article{krojer2026latentlens,
  title={{Revealing Highly Interpretable Visual Tokens in LLMs}},
  author={Krojer, Benno and Nayak, Nihal V. and Ma{\~n}as, Oscar and Adlakha, Vaibhav and Elliott, Desmond and Reddy, Siva and others},
  journal={arXiv preprint arXiv:2602.00462},
  year={2026}
}

@article{balestriero2025lejepa,
  title={{LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}},
  author={Balestriero, Randall and LeCun, Yann},
  journal={arXiv preprint arXiv:2511.08544},
  year={2025}
}
```
