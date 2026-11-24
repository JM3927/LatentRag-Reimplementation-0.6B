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

> **This fork (branch `qwen3-0.6B`)** is an unofficial port of LatentRAG to **Qwen3-0.6B-Base on a single 8 GB GPU** . The port notes come first; the authors' original README follows unchanged from [Overview](#overview).

## Qwen3-0.6B port: notes

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

### Environment

- Python 3.10 venv; the authors' `install.sh` without flash-attn, plus `faiss-cpu`. Exact versions: `requirements-qwen3.txt`.
- **transformers 5.20.2**, not the 4.57.6 that `install.sh` pins: vLLM 0.17.1 upgrades it. vLLM must be installed because `src/llm_tools/safe_loader.py`, which every model imports, imports it.
- PyTorch 2.10 (CUDA 12.8 build), which supports Blackwell `sm_120`.
- **No flash-attn:** PyTorch SDPA everywhere (`attn_implementation: "sdpa"` for the LLM and the embedders).
- **No DeepSpeed:** importing it requires a CUDA toolkit (`CUDA_HOME`); the trainer's only DeepSpeed import was unused.
- wandb runs offline (`WANDB_MODE=offline`).

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

All commands are fish, from the repo root, after `source env.fish`.

```fish
python make_subsets.py --no-full          # 1. ~20k training subset (CPU)
python build_reference_cache.py           # 2. reference cache (a few minutes, GPU)
python build_ivfpq.py                     # 3. CPU retrieval index (1-3 h, can run during training)
env RUN_DIR=$PWD/data/runs/main bash scripts/training/train_latentrag_qwen3-0.6b-base_qwen3-embedding-0.6b.sh   # 4. train
```

Training is resumable: stop it with Ctrl-C and rerun the exact same command (same `RUN_DIR`); it continues from the latest checkpoint (every 100 steps). Extra Hydra overrides go at the end of the command; keys not in the config need a leading `+` (e.g. `+training.max_steps=15`).

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

