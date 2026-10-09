# LLM-enhanced recommendation: focused research notes

Date: 2026-10-02

## Findings relevant to CORE + Tmall

- CORE is suitable as a compact ID-only session-recommendation baseline, but an LLM paper should also compare with representative recurrent/self-attention sequential recommenders and simple popularity baselines.
- The processed Tmall files in this repository contain only session IDs, item-ID sequences, and target item IDs. Anonymous numeric IDs do not expose useful natural-language semantics to a pretrained LLM.
- A practical first method is offline semantic enhancement: encode item title/category/description once, project the frozen semantic vector into CORE's embedding space, and fuse it with the collaborative item embedding. This preserves full-sort efficiency and avoids online LLM inference.
- Candidate generation plus LLM reranking is a stronger second-stage experiment, but requires item text and should report latency/cost in addition to ranking accuracy.
- Pure prompting or direct item-ID generation is a poor first experiment here because the candidate vocabulary is large and current item IDs have no intrinsic language meaning.

## Relevant papers

1. Liu et al. (2024), **LLM-ESR: Large Language Models Enhancement for Long-tailed Sequential Recommendation**. Uses LLM-derived semantic embeddings to enhance conventional sequential recommenders without adding LLM inference load, with emphasis on long-tail users/items. https://arxiv.org/abs/2405.20646
2. Kim et al. (2024), **Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System (A-LLMRec)**. Combines pretrained collaborative representations with an LLM and reports that semantic-only approaches may underperform collaborative filtering in warm scenarios. https://arxiv.org/abs/2404.11343
3. Bao et al. (2023), **TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation**. Studies instruction tuning of LLMs for recommendation and notes the task gap between language pretraining and recommendation. https://arxiv.org/abs/2305.00447
4. Geng et al. (2022), **Recommendation as Language Processing (P5)**. Converts interactions, metadata, and reviews to a unified text-to-text recommendation paradigm. https://arxiv.org/abs/2203.13366
5. Wei et al. (2023), **LLMRec: Large Language Models with Graph Augmentation for Recommendation**. Uses LLMs to augment interactions, item attributes, and user profiles, illustrating a data-augmentation route when text side information exists. https://arxiv.org/abs/2311.00423
6. Wu et al. (2023), **A Survey on Large Language Models for Recommendation**. Organizes LLM recommendation into discriminative and generative paradigms and emphasizes textual representations and external knowledge. https://arxiv.org/abs/2305.19860

## Recommended encoder specification

- Primary encoder: `Qwen/Qwen3-Embedding-0.6B`. It is a 0.6B multilingual text-embedding model supporting 100+ languages, up to 1024 embedding dimensions, custom output dimensions, and task instructions. It is a practical match for offline Chinese product-text encoding on an 8 GB consumer GPU. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Encoder control: `BAAI/bge-m3`. It produces 1024-dimensional dense embeddings, supports multilingual input and up to 8192 tokens, and provides a stable non-Qwen comparison. https://huggingface.co/BAAI/bge-m3
- Freeze both encoders and precompute item vectors. Do not load the encoder during CORE training or evaluation. Use identical item text and preprocessing for both encoders.
- Start with recommendation loss only. Compare ID-only, semantic-only, residual addition, and frequency-aware gated fusion before introducing an alignment loss.

## Pure-ID scenario revision

When only anonymous item IDs and interaction sequences are allowed, text-embedding models such as Qwen3-Embedding or BGE-M3 have no semantic input and should not be presented as semantic enhancement. The research question becomes whether language-model pretraining or generative tokenization transfers to collaborative sequences.

Recommended staged design:

1. **Continuous ID-to-LLM adapter pilot.** Feed projected CORE item embeddings to a small base causal LM through `inputs_embeds`, project its final session state back to CORE's 100-dimensional item space, and retain CORE's full-sort dot-product decoder. Use `Qwen/Qwen3-0.6B-Base` with a frozen backbone plus LoRA. Compare against the exact same architecture initialized from scratch. https://huggingface.co/Qwen/Qwen3-0.6B-Base
2. **Collaborative tokenization extension.** Train CORE using training interactions only, quantize its item embeddings with RQ-VAE or hierarchical clustering into short collaborative-code tuples, and train a causal generator to predict the next tuple. Use constrained trie decoding so every generated code maps to a valid catalog item.
3. Never serialize anonymous IDs as ordinary decimal strings, because tokenizer segmentation introduces accidental numeric structure. Use dedicated item tokens, continuous `inputs_embeds`, or learned collaborative codes.

Relevant studies:

- Hua et al. (2023), **How to Index Item IDs for Recommendation Foundation Models**, studies sequential, collaborative, semantic, and hybrid item indexing and shows that item indexing materially affects LLM recommendation. https://arxiv.org/abs/2305.06569
- Wang et al. (2024), **Learnable Item Tokenization for Generative Recommendation (LETTER)**, uses RQ-VAE with collaborative regularization and code-diversity objectives, demonstrating that tokenization is a central component rather than a preprocessing detail. https://arxiv.org/abs/2405.07314
- Xiao et al. (2025), **Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation**, explicitly notes that collaborative embeddings can be quantized into item codes before autoregressive recommendation. The collaborative-only branch is relevant to the anonymous-ID setting. https://arxiv.org/abs/2502.06269
