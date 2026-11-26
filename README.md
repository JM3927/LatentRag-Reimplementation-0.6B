<!-- # LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG -->

<h1 align="center">
  <picture title="The green and blue gradients represent the latent thought tokens (green) and latent subquery tokens (blue) in the LatentRAG framework.">
  <source media="(prefers-color-scheme: dark)" srcset="assets/title_dark.svg">
  <img src="assets/title.svg"
       alt="LatentRAG"
       height="32em"
       align="absmiddle" 
  /></picture>: Latent Reasoning and Retrieval for Efficient Agentic RAG
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.06285"><img src="https://img.shields.io/badge/arXiv-2605.06285-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://github.com/zhengyijia/LatentRAG"><img src="https://img.shields.io/badge/GitHub-LatentRAG-blue" alt="Github" /></a>
  <a href="https://huggingface.co/collections/zhengyijia142/latentrag-models"><img src="https://img.shields.io/badge/HuggingFace-Models-yellow.svg" alt="HuggingFace" /></a>
</p>

<p align="center">
  <strong>Yijia Zheng, Marcel Worring</strong>
  <br>
  University of Amsterdam
</p>

> **This fork (branch `qwen3-0.6B`)** is an unofficial port of LatentRAG to **Qwen3-0.6B-Base on a single 8 GB GPU**. The port notes come first; the authors' original README follows unchanged from [Overview](#overview).

## Qwen3-0.6B port: notes

### Status and To-Do list (as of 2026-11-24)

**Done**

- [x] Port to Qwen3-0.6B-Base on one 8 GB GPU: configs, one-GPU training script, resumable training ([What changed](#what-changed-from-the-official-code))
- [x] ~20k-trajectory training subset (`make_subsets.py`)
- [x] Reference-embedding cache so training fits in 8 GB ([details](#gpu-memory-the-reference-embedding-cache))
- [x] Overfit test, resume test, worst-case memory test (longest trajectories, ~7.5 GB peak)
- [x] Main training run: 3 epochs, 16 h 45 min ([results](#main-training-run-2026-11-23-to-24))
- [x] CPU retrieval server with the original REST API, recall@3 = 0.983 against exact search ([details](#retrieval-server-ivf-pq-with-exact-re-ranking)); its index is on Hugging Face as `LatentRAG/wiki18_100w_qwen3-embedding-0.6b_ivfpq` (private dataset repo; [setup table](#setting-up-on-another-machine))

**To do, in order.** Items 1–4 are the evaluation milestone. The rest can be picked up in any order.

1. [x] **Share the trained model.** Uploaded to the private Hugging Face repo `LatentRAG/LatentRAG-Qwen3-0.6B` (1.6 GB): the final model (step 3,798) at the top level and `checkpoint-1300/`, each with everything `evaluate.py` reads (`training_config.yaml`, LLM LoRA `adapter_*`, the three `*_projector.safetensors`, `encoder/` and the tokenizer files). Optimizer state is not included, so these can't be used to resume training. Access: ask to be added to the `LatentRAG` organization on Hugging Face, then `hf auth login` and
   ```bash
   hf download LatentRAG/LatentRAG-Qwen3-0.6B --local-dir data/runs/main
   ```
   The paths in `training_config.yaml` are relative to the repo root (`data/...`), so run eval from there.
2. [ ] **Eval script for this model**: `scripts/evaluation/eval_latentrag_qwen3-0.6b-base.sh`, modelled on `scripts/evaluation/eval_latentrag_qwen2.5-7b-base.sh`:
   - `LORA_PATH` = the run folder (or a checkpoint folder);
   - `retriever.base_url='http://127.0.0.1:8001'` (the IVF-PQ server; port 8007 in the original);
   - drop the authors' `GLOBAL_DATA_PATH`/slurm logic.

   Things to check while doing it:
   - LatentRAG eval sends subquery **embeddings** (`retrieve_with_embeddings`, `src/evaluate.py:137`), so start the server with `ivfpq.load_encoder=false` to save 1.3 GB of GPU.
   - `evaluate.py` still loads its own frozen reference encoder when it uses the REST retriever (`src/evaluate.py:222`). That encoder is unused in that case and costs ~1.2 GB of GPU; skip loading it when `retriever.base_url` is set.
   - `training_config.yaml` carries `retriever.reference_cache_path` (a training-only setting). Confirm that eval ignores it.
   - Confirm the eval runs under transformers 5.20.2 with `sdpa`. Training needed one fix (`save_safetensors`, see the table below), and eval code paths have not been run yet.
3. [ ] **Smoke eval**: `checkpoint-1300`, ~200 questions per dataset. The 7 eval files hold ~51k questions; pass `+data.sampling=200` (built into `evaluate.py`: seed-42 shuffle, then the first N per file; the full eval in item 4 uses `+data.sampling=1000`, ~6,125 questions with all 125 of Bamboogle). Do **not** use `checkpoint-600` for this: its alignment top-1 is only 0.14 (vs 0.54 at 1,300 and 0.71 final), so retrieval there is barely trained.
4. [ ] **Full eval of the final model** on all 7 sets (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle), 1,000 sampled questions each (`+data.sampling=1000`; Bamboogle has 125), the same sample for every system. Record EM/F1 per dataset and average retrieval steps in a results table in this README, next to the paper's 3B/7B numbers.
5. [ ] **Baselines with the same retriever**:
   - **Direct** (no retrieval): adapt `config/direct_infer_qwen2.5-7b-instruct.yaml`.
   - **Naive RAG** (retrieve top-3 for the question once, then answer): adapt `config/few_shot_rag_qwen2.5-3b-instruct.yaml`.

   Both use the server's text mode, so keep `ivfpq.load_encoder=true`. Decide and write down which model they use: `Qwen/Qwen3-0.6B-Base`, which matches the LatentRAG backbone, or `Qwen/Qwen3-0.6B`, the post-trained chat model, which is closer to the paper's Instruct baselines.
6. [ ] **Efficiency comparison**, the paper's main claim: `src/evaluate_with_step_wise_time.py` for LatentRAG vs the text-based baselines. Report retrieval time separately, since this server's latency (~45 ms per query) differs from the original GPU server's.
7. [ ] **Clean-machine install check.** `requirements-qwen3.txt` is a `pip freeze` of the working venv, and has not yet been installed from scratch on another machine (see [Setting up on another machine](#setting-up-on-another-machine)). Note any fixes here.

**Optional and known issues**

- [ ] Eval answer-generation loss stopped improving around step 1,400 while alignment kept improving. A future run could try 2 epochs, or keep 3 and weight alignment higher.
- [ ] The `faiss-cpu` wheel falls back to its generic build ("Could not load library with AVX2 support"). An AVX2 build could speed up search; it's not needed at the current ~17 ms per query.
- [ ] Training exits with the PyTorch warning `destroy_process_group() was not called`. It's harmless (checkpoints and logs are complete), but it could be silenced in `src/train.py`.
- [ ] Training on the full correct-only data instead of the 20k subset needs its own reference cache (see the cache section), and training time grows in proportion to the data (16.75 h for 20k trajectories × 3 epochs).

### What changed from the official code

| File | Change |
|---|---|
| `config/latentrag_qwen3-0.6b-base.yaml` | The authors' 3B config with every setting from their 7B training script moved into the file (4 thought / 16 query slots, `kl_rank_loss`, temperature 0.03, trainable embedder with LoRA, `share_encoder: false`), plus: Qwen3-0.6B-Base, `sdpa` attention, no DeepSpeed, 3 epochs, warmup 100, a full checkpoint every 100 steps, and `reference_cache_path` (below) |
| `config/gpu_env/latentrag_qwen3-0.6b-base_qwen3-embedding-0.6b/gpu_1_8g.yaml` | Micro-batch 1 × 16 gradient accumulation, learning rate 1e-4 |
| `scripts/training/train_latentrag_qwen3-0.6b-base_qwen3-embedding-0.6b.sh` | One-GPU training script; requires `RUN_DIR`, passes extra Hydra overrides through, sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to limit fragmentation |
| `src/train.py` | `RUN_DIR` fixes the output folder and resumes from its latest checkpoint (replaying the per-epoch reshuffle); loads the reference cache when configured |
| `src/trainer.py` | Every save also writes `trainable_state.pt` (all trainable tensors); `_load_from_checkpoint` restores from it, because the stock loader can't read this model's checkpoints. Unused DeepSpeed import commented out. `self.args.save_safetensors` replaced by `getattr(self.args, "save_safetensors", True)`: transformers 5 removed that argument (it always saves safetensors), and every checkpoint save crashed with `AttributeError` |
| `src/model/latentrag.py` | Reads alignment targets from the reference cache when one is set; otherwise unchanged |
| `src/dataset.py` | Applies `data.min_seq_length` (present in the configs but never used), so a memory test can select only the longest trajectories |
| `src/reference_cache.py`, `build_reference_cache.py` | New: the reference-embedding cache (below) |
| `make_subsets.py` | Cuts the ~20k training subset (10,250 correct trajectories per train file, seed 42) one file at a time, so RAM stays under 32 GB |
| `build_ivfpq.py` | Builds a CPU FAISS IVF-PQ index (16,384 lists, 128-byte codes, ~3 GB) from the released fp16 shards, for serving retrieval from RAM instead of ~43 GB of GPU memory. **The released shard vectors are not unit length (mean norm ~104);** the official retriever L2-normalizes them on load (`src/retriever/torch_retriever_gpu.py`), i.e. ranks by cosine, so this script normalizes before training and adding. (Fixed 2026-11-23; an index from the first version, which skipped this, must be deleted and rebuilt.) |
| `check_trajectories.py`, `smoke_step.py` | Data checks and an LLM-only GPU smoke test |
| `src/retriever/ivfpq_retriever.py`, `src/launch_retriever_server_ivfpq.py`, `config/retriever_server_ivfpq.yaml`, `scripts/launch_retriever_server/launch_qwen3-embedding-0.6b_ivfpq.sh` | New: retrieval server on the IVF-PQ index with exact re-ranking; same REST API as the original server (below) |
| `check_retriever_recall.py` | New: recall@3 of that server against exact brute-force search |
| `env.fish`, `env.sh`, `requirements-qwen3.txt` | New: per-session environment (venv, `PYTHONPATH`, offline wandb) for fish or bash/zsh; exact package versions |

### Environment

- Python 3.10 venv; the authors' `install.sh` without flash-attn, plus `faiss-cpu`. Exact versions: `requirements-qwen3.txt`.
- **transformers 5.20.2**, not the 4.57.6 that `install.sh` pins: vLLM 0.17.1 upgrades it. vLLM must be installed because `src/llm_tools/safe_loader.py`, which every model imports, imports it.
- PyTorch 2.10 (CUDA 12.8 build), which supports Blackwell `sm_120`.
- **No flash-attn:** PyTorch SDPA everywhere (`attn_implementation: "sdpa"` for the LLM and the embedders).
- **No DeepSpeed:** importing it requires a CUDA toolkit (`CUDA_HOME`); the trainer's only DeepSpeed import was unused.
- wandb runs offline (`WANDB_MODE=offline`).

### Setting up on another machine

Tested hardware: one 8 GB consumer GPU (NVIDIA Blackwell, `sm_120`), 32 GB RAM. Disk: ~70 GB of data (table below), plus up to ~85 GB while training, because the config keeps a 2.2 GB checkpoint every 100 steps (`save_total_limit: null`); prune them after the run. Training needs the GPU to itself. The retrieval server needs ~5 GB RAM and, in text mode, ~1.3 GB of GPU.

1. **Code.** Clone this repo; `qwen3-0.6B` is the default branch. `main` is the authors' original code, unchanged.
2. **Python 3.10 venv** in `.venv` at the repo root, where `env.fish`/`env.sh` expect it:
   ```bash
   uv venv -p 3.10 .venv
   source .venv/bin/activate
   uv pip install -r requirements-qwen3.txt
   ```
   `torch==2.10.0` from PyPI is the CUDA 12.8 build (needed for RTX 50-series / `sm_120`). This exact command hasn't been run on a fresh machine yet (To-Do item 7).
3. **Per session:** `source env.fish` (fish) or `source env.sh` (bash/zsh).
4. **Data.** `data/` is gitignored. It can be a plain folder or a symlink to a folder elsewhere (e.g. on a bigger disk), which keeps the big files out of the repo. The models (Qwen3-0.6B-Base and Qwen3-Embedding-0.6B, ~2.5 GB) download from Hugging Face on first use.

| Path under `data/` | Size | How to get it | Needed for |
|---|---|---|---|
| `corpus/wiki18_100w.jsonl` | 14 GB | `cd scripts/preprocessing && bash download_corpus.sh` | everything |
| `corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard/` | 43 GB | `huggingface-cli download zhengyijia142/wiki18_100w_qwen3-embedding-0.6b_fp16_shard --repo-type dataset --local-dir data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard` | index build, server re-ranking, reference cache |
| `qa_data/` | 0.7 GB | `cd scripts/preprocessing && bash download_qa_data.sh` | eval |
| `reasoning_steps/` | 6.3 GB | `huggingface-cli download zhengyijia142/SearchR1_AutoRefine_reasoning_steps_qwen3-embedding-0.6b --repo-type dataset --local-dir data/reasoning_steps` | training subset |
| `reasoning_steps_subset20k/` | 0.5 GB | `python make_subsets.py --no-full` | training |
| `reference_cache/subset20k_qwen3-embedding-0.6b/` | 1.1 GB | `python build_reference_cache.py` (~2.5 min, GPU) | training only |
| `corpus/ivfpq/` | 2.9 GB | `hf download LatentRAG/wiki18_100w_qwen3-embedding-0.6b_ivfpq --repo-type dataset --local-dir data/corpus/ivfpq` (private, `LatentRAG` org), or build it: `python build_ivfpq.py` (1–3 h, CPU) | retrieval server |
| `corpus/wiki18_100w.offsets.npy` | 0.2 GB | built automatically on the server's first start | retrieval server |
| `runs/main/` | 1.6 GB | `hf download LatentRAG/LatentRAG-Qwen3-0.6B --local-dir data/runs/main` (To-Do item 1), or your own training run | eval |

For **evaluation only** you need the corpus, the shards, `qa_data/`, the IVF-PQ index and a trained run: no training data and no reference cache.

### GPU memory: the reference-embedding cache

**Problem (2026-11-23).** The first overfit test ran out of memory on its first training step. LatentRAG training keeps three Qwen3-0.6B-sized models on the GPU in bf16: the LLM, the trainable embedder, and a *frozen* copy of the embedder (`share_encoder: false`) that only produces target vectors for the alignment loss. Their weights alone take ~3.6 GB. The LLM forward pass (mostly 151k-vocabulary logits over up to 3,000 tokens) brought allocation to 5.9 GB, and the frozen copy's encoding of the step's ~50–70 passages then failed with 87 MB free on the 8 GB card. The first optimizer step would add another ~0.9 GB (gradients and AdamW state for the trainable embedding table), so cutting `max_seq_length` to 2,048 (~0.5 GB) would not have been enough.

**Fix.** The frozen encoder's outputs never change, so they are computed once and read from disk:

- **Passages:** the released index shards *are* the frozen Qwen3-Embedding-0.6B's passage vectors (built by `src/preprocessing/build_index.py` in fp32, stored in fp16), and `chunk_id` = corpus line = index row for all 21,015,324 passages. The cache copies the rows the training subset uses.
- **Subqueries:** encoded once by the exact encoder `train.py` would build (same Hydra config, bf16, max length 256, same instruction prefix).
- **Training** reads both through `src/reference_cache.py` and never loads the frozen encoder, freeing ~1.2 GB plus its activations, and skips ~50–70 passage encodings per step.

For the ~20k subset that is 37,311 subqueries and 475,406 passages (~1 GB on disk). Build it once, with nothing else on the GPU:

```fish
python build_reference_cache.py   # -> data/reference_cache/subset20k_qwen3-embedding-0.6b/
```

**Check.** The script re-encodes 2,000 random training passages with the frozen encoder and compares them to the cached rows; the results are saved in the cache's `meta.json`:

| Check (built 2026-11-23, 2,000 passages) | Result |
|---|---|
| Cosine, cached row vs freshly encoded (mean / 1st percentile / min) | 0.99987 / 0.99982 / 0.9868 |
| Passages below 0.98 | 0 |
| Teacher target distribution at temperature 0.03, KL(fresh ‖ cached) | 0.00021 |
| Same top-1 passage under both | 99.1% |

The build took about 2.5 minutes (passages 20 s from the shards, subqueries 76 s on the GPU).

**Result.** With the cache, a short run on 64 trajectories (micro-batch 1, through three optimizer updates, `PYTORCH_ALLOC_CONF=expandable_segments:True`) peaked at **6,777 MiB of 8,151 MiB** on the 8 GB GPU, including ~470 MiB used by the desktop. Without it, training ran out of memory on step 1. At `max_seq_length` 3,000 the loader keeps 20,499 of the 20,500 subset trajectories.

**Differences from the original behaviour.** Passage targets come from fp32 encoding (the index) rather than on-the-fly bf16, and are not truncated at 256 tokens (wiki-18 passages are ~100 words, so truncation is rare). The alignment loss, its negatives, and everything else are unchanged. To go back to the original on-GPU frozen encoder, pass `retriever.reference_cache_path=null`. A different training set (e.g. the full correct-only data) needs its own cache: `python build_reference_cache.py --src data/reasoning_steps_correct --out data/reference_cache/correct_qwen3-embedding-0.6b`, and point `retriever.reference_cache_path` at it.

### Running it

All commands run from the repo root after `source env.fish` (fish) or `source env.sh` (bash/zsh), and work the same in both shells.

```fish
python make_subsets.py --no-full          # 1. ~20k training subset (CPU)
python build_reference_cache.py           # 2. reference cache (a few minutes, GPU)
python build_ivfpq.py                     # 3. CPU retrieval index (1-3 h, can run during training)
env RUN_DIR=$PWD/data/runs/main bash scripts/training/train_latentrag_qwen3-0.6b-base_qwen3-embedding-0.6b.sh   # 4. train
bash scripts/launch_retriever_server/launch_qwen3-embedding-0.6b_ivfpq.sh   # 5. retrieval server for eval (below)
```

Training is resumable: stop it with Ctrl-C and rerun the exact same command (same `RUN_DIR`); it continues from the latest checkpoint (every 100 steps). Extra Hydra overrides go at the end of the command; keys not in the config need a leading `+` (e.g. `+training.max_steps=15`).

### Main training run (2026-11-23 to 24)

3 epochs over the 20,499-trajectory subset: 3,798 optimizer steps (16 trajectories each), uninterrupted, **16 h 45 min** on the 8 GB GPU (15.9 s/step). Final train loss 0.96. Eval on the held-out split every 200 steps:

| Eval metric | Start | Step 1,800 | End (3,798) |
|---|---|---|---|
| Total loss | 16.29 | 1.78 | **1.54** |
| Alignment top-1 accuracy (latent query picks the reference encoder's top passage) | 0.07 | 0.64 | **0.71** |
| Alignment KL-rank loss | 1.84 | 0.53 | **0.44** |
| Subquery self-decoding accuracy | 0.24 | 0.88 | **0.90** |
| Answer generation loss | 5.13 | 0.20 | **0.20** |

Alignment top-1 stayed low (0.07 to 0.15) for the first 800 steps, jumped between steps 1,000 and 1,400 (0.25 to 0.54), and was still creeping up at the end. Answer-generation eval loss was ~0.23 by step 400 and flattened at ~0.20 from step 1,400, while its train loss kept falling to ~0.10, so epochs 2 and 3 mainly helped alignment.

The final model and `checkpoint-1300` (~1 epoch) are on Hugging Face as `LatentRAG/LatentRAG-Qwen3-0.6B` (private; see To-Do item 1). The other checkpoints were deleted, except `checkpoint-600` (~half an epoch) and `checkpoint-2500` (~2 epochs), which were kept locally but not uploaded.

### Retrieval server: IVF-PQ with exact re-ranking

The original server (`src/launch_retriever_server.py` + `TorchRetriever`) keeps all 21M fp16 passage vectors in GPU or CPU memory (43 GB) and loads the 14 GB corpus jsonl as Python objects. Neither fits on the target machine (8 GB GPU, 32 GB RAM). `src/launch_retriever_server_ivfpq.py` serves the **same REST API** (`GET /retriever_info`, `POST /retrieve` in `text` or `embedding` mode, same response format and chunk ids), so `RESTRetriever` and the eval scripts work unchanged with `retriever.base_url=http://127.0.0.1:8001`. Per query it:

1. encodes the question with Qwen3-Embedding-0.6B (text mode; skipped in embedding mode) and L2-normalizes it;
2. takes 300 candidates from the CPU FAISS IVF-PQ index (`build_ivfpq.py`), searching 2,048 of the 16,384 lists;
3. re-scores those candidates exactly against the original fp16 shard vectors, read from disk with `pread` (no readahead, 32 threads), and returns the top k. Scores are therefore the same cosine scores the original server returns;
4. reads passage text by byte offset into the jsonl (a 168 MB offset table, built on first start in ~10 s).

Memory: ~5 GB RAM, ~1.3 GB GPU for the query encoder (`ivfpq.load_encoder=false` drops it, leaving embedding mode only).

**Recall against exact search** (`check_retriever_recall.py`: 1,000 eval-set questions, 200 each from NQ test, HotpotQA dev, 2Wiki dev, MuSiQue dev, PopQA test, seed 1; ground truth is brute force over all 21M vectors, i.e. what the original server returns):

| Lists searched (nprobe) | recall@3, re-ranking 300 | top-1 same as exact | recall@3, PQ scores only |
|---|---|---|---|
| 256 | 0.914 | 0.937 | 0.605 |
| 512 | 0.954 | 0.968 | 0.615 |
| 1,024 | 0.974 | 0.980 | 0.619 |
| **2,048 (server default)** | **0.983** | **0.987** | 0.621 |

**Pass (mark: 0.95).** An earlier setting (64 lists, 100 candidates) gave only 0.79 even though the index found 99.5% of *stored passage vectors* as their own top hit; real question embeddings sit much farther from their passages, so many more lists must be searched. Re-ranking 1,000 candidates instead of 300 added at most 0.003. Without exact re-ranking, recall stalls at ~0.62 however many lists are searched, because the 128-byte PQ codes are too coarse to order close candidates.

Latency (warm, measured through `RESTRetriever`): ~45 ms for a single text query end to end, ~26 ms per query in batches of 128. Two fixes got it there: gathering candidate rows by fancy-indexing memory-mapped shards took ~13 s per 19k rows (each page fault pulled ~128 KB of readahead), which `pread` with readahead off cut to 0.26 s; and FAISS `parallel_mode=1` (split one query's lists across threads) cut single-query search from 113 to 18 ms.

```fish
bash scripts/launch_retriever_server/launch_qwen3-embedding-0.6b_ivfpq.sh   # serves on 127.0.0.1:8001; ~10 s to start
python check_retriever_recall.py --n 1000 --seed 1                          # recall check (~2 min, needs the GPU briefly)
```


## Overview

This repository contains the official implementation of our NeurIPS 2026 paper: [LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG](https://arxiv.org/abs/2605.06285). LatentRAG is an efficient agentic RAG framework that shifts both reasoning and retrieval from discrete language space to continuous latent space, enabling end-to-end joint optimization of the LLM and the retrieval model while reducing overall latency by approximately 90%.

### A simple and quick introduction to LatentRAG

> People naturally use words to search for information, but does an AI agent need to do the same?

In existing agentic RAG systems, an AI agent searches for information multiple times before answering a complex question. Each time, it must express its reasoning in text and turn its search intent into a text query. Generating text one token at a time makes this process slow, much like humans have to type out their queries word-by-word before searching online. AI agents and search systems both work with latent representations other than words, making it possible for AI agents to search without first converting their internal representations into words.

LatentRAG explores this idea by moving both reasoning and retrieval from discrete language space to continuous latent space. Instead of generating text queries, the agent directly produces continuous latent representations and passes them to the retriever to search for relevant information. This latent reasoning and retrieval mechanism not only offers significant efficiency advantage, but also enables end-to-end differentiable joint optimization of the LLM and the retriever. Our experiments on seven benchmark datasets show that LatentRAG achieves performance comparable to existing agentic RAG methods while reducing inference latency by approximately 90%, substantially narrowing the latency gap with naive single-step RAG.

<p align="center">
  <picture>
  <img src="assets/performance_efficiency_comparison.png" width="100%" alt="Performance and latency comparison on multi-hop QA datasets." title="LatentRAG achieves comparable performance to competitive agentic RAG methods such as Search-R1 and AutoRefine, while maintaining efficiency on par with naive single-step RAG." />
  </picture>
  <br />
  <sub><span style="color: #888888;">Performance and latency comparison across multi-hop QA datasets. LatentRAG achieves comparable performance to competitive agentic RAG methods such as Search-R1 and AutoRefine, while maintaining efficiency on par with naive single-step RAG.</span></sub>
</p>

## Environment Setup

The code has been tested with Python 3.10 and CUDA 12.8. To set up the environment, run the following commands:

```shell
conda create -n latent_rag_env python=3.10
conda activate latent_rag_env
bash install.sh
```

## Training and Evaluation

Below are the training and evaluation scripts for running our model with the Qwen3-Embedding-0.6B retrieval model, covering the full pipeline from data preparation to training and evaluation. All commands should be executed from the root directory of this project.

### (1) Download datasets

Download the Wikipedia 2018 corpus and the QA datasets

```shell
cd scripts/preprocessing
bash download_corpus.sh
bash download_qa_data.sh
```

The corpus will be stored in `data/corpus`, and the QA datasets will be stored in `data/qa_data`.

### (2) Build retrieval index

```shell
cd scripts/preprocessing
bash build_corpus_index.sh
# will build index for Qwen3-Embedding-0.6B
# change the commented code in `build_corpus_index.sh` to build index for other retrieval models
```

The built index files will be stored in `data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard`.

For quick reproduction, we provide the precomputed indices on [Hugging Face](https://huggingface.co/collections/zhengyijia142/latentrag-corpus-index). You can download the indices constructed with `Qwen3-Embedding-0.6B` using the following command:

```shell
huggingface-cli download zhengyijia142/wiki18_100w_qwen3-embedding-0.6b_fp16_shard \
    --repo-type dataset \
    --local-dir data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard
```

### (3) Launch retrieval server

```shell
cd scripts/launch_retriever_server
bash launch_qwen3-embedding-0.6b.sh
# bash launch_{RETRIEVAL_MODEL_NAME}.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

### (4) Construct training trajectories

Search-R1 (7B):
```shell
cd scripts/preprocessing/gen_reasoning_steps
bash gen_searchr1_steps_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b.sh
# bash gen_searchr1_steps_qwen2.5-7b-em-ppo_{RETRIEVAL_MODEL_NAME}.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

AutoRefine (7B)
```shell
cd scripts/preprocessing/gen_reasoning_steps
bash gen_autorefine_steps_qwen2.5-7b_qwen3-embedding-0.6b.sh
# e.g., bash gen_autorefine_steps_qwen2.5-7b_{RETRIEVAL_MODEL_NAME}.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

The constructed training trajectories will be stored in `data/reasoning_steps`. If the retrieval server in step (3) is running on a different machine, please update `retriever.base_url` in the script accordingly.

**The scripts are configured by default to run on two 94 GB GPUs.** If you use a different GPU configuration and encounter CUDA OOM errors, add a custom vLLM configuration under `config/gpu_env` and update `GPU_ENV_CONFIG` in the corresponding scripts.

For quick reproduction, we provide the constructed trajectories generated by Search-R1 (7B) and AutoRefine (7B) on [Hugging Face](https://huggingface.co/collections/zhengyijia142/latentrag-training-trajectories). You can download training trajectories constructed with `Qwen3-Embedding-0.6B` using the following command:

```shell
huggingface-cli download zhengyijia142/SearchR1_AutoRefine_reasoning_steps_qwen3-embedding-0.6b \
    --repo-type dataset \
    --local-dir data/reasoning_steps
```

### (5) Train LatentRAG

The training code has been tested on two H100 GPUs, each with 94 GB of memory. For other hardware configurations, please modify `GPU_ENV_CONFIG` in the training script and add the corresponding configuration file under `config/gpu_env/latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b` to ensure proper adaptation to GPU memory constraints.

LatentRAG (based on Search-R1 trajectories)
```shell
cd scripts/training
bash train_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_search-r1_qwen2.5-7b.sh
# bash train_latentrag_qwen2.5-7b-base_{RETRIEVAL_MODEL_NAME}_search-r1_qwen2.5-7b.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

LatentRAG (based on AutoRefine trajectories)
```shell
cd scripts/training
bash train_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_autorefine_qwen2.5-7b-base.sh
# bash train_latentrag_qwen2.5-7b-base_{RETRIEVAL_MODEL_NAME}_autorefine_qwen2.5-7b-base.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

The trained model weights will be stored in `trainer_output`.

### (6) Evaluate LatentRAG

### Evaluation using your saved model weights

Please set `LORA_PATH` in the corresponding evaluation script to the path of the model weights directory saved in the previous step.

LatentRAG (based on Search-R1 trajectories)
```shell
cd scripts/evaluation_based_on_reasoning_steps
bash eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_search-r1_qwen2.5-7b.sh
# bash eval_latentrag_qwen2.5-7b-base_{RETRIEVAL_MODEL_NAME}_search-r1_qwen2.5-7b.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

LatentRAG (based on AutoRefine trajectories)
```shell
cd scripts/evaluation_based_on_reasoning_steps
bash eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_autorefine_qwen2.5-7b-base.sh
# bash eval_latentrag_qwen2.5-7b-base_{RETRIEVAL_MODEL_NAME}_autorefine_qwen2.5-7b-base.sh
# change the `RETRIEVAL_MODEL_NAME` to run with other retrieval models
```

### Evaluation with our Hugging Face checkpoints

For quick reproduction, we provide our model checkpoints on [Hugging Face](https://huggingface.co/collections/zhengyijia142/latentrag-models). You can download using the following command:

```shell
# LatentRAG (based on Search-R1 trajectories)
huggingface-cli download zhengyijia142/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B \
    --local-dir trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B
```

Then in the evaluation script, set `LORA_PATH` to `trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B` and run the evaluation script: 

```shell
cd scripts/evaluation_based_on_reasoning_steps
bash eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_search-r1_qwen2.5-7b.sh
```

## Acknowledgements

This work uses datasets from [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [InfoSeek](https://huggingface.co/datasets/Lk123/InfoSeek), and [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). We use [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [AutoRefine](https://github.com/syr-cn/AutoRefine), and [AgentCPM](https://github.com/OpenBMB/AgentCPM) to construct trajectories for supervised fine-tuning. The codebase also contains implementations of several baseline methods, including [IRCoT](https://github.com/StonyBrookNLP/ircot), [Iter-RetGen](https://arxiv.org/abs/2305.15294), [Search-o1](https://github.com/RUC-NLPIR/Search-o1), [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher), [ZeroSearch](https://github.com/Alibaba-nlp/ZeroSearch), and [DeepRAG](https://github.com/gxy-gxy/DeepRAG). We thank the authors and contributors of these datasets and methods. 

## Citation

```
@inproceedings{zheng2026latentrag,
  title={LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG},
  author={Yijia Zheng and Marcel Worring},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

This repository, as well as the finetuned model weights on [Hugging Face](https://huggingface.co/collections/zhengyijia142/latentrag-models), are released under the [MIT License](LICENSE).

