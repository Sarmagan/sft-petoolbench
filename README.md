# Personalized Tool Use with LLaMA-3.2-3B

## Results on PEToolBench (Rating-Integrated Setting)

All baseline results are taken from [PEToolLLM (Xu et al., 2025)](https://arxiv.org/abs/2502.18980). My results are evaluated on the same test set under the **Rating-Integrated** setting, where the interaction history includes both preferred and non-preferred tools with explicit user ratings.

| Methods                        | Tool Acc  | Param Acc |
|--------------------------------|-----------|-----------|
| Vicuna-7B                      | 10.80     | 57.40     |
| Mistral-7B                     | 15.40     | 63.20     |
| Qwen2.5-7B                     | 24.80     | 66.50     |
| LLaMA3-8B                      | 26.90     | 77.70     |
| GPT-4o-mini                    | 38.40     | 77.70     |
| GPT-4o                         | 45.70     | 79.60     |
| PEToolLLaMA                    | 78.40     | 89.70     |
| LLaMA3.2-3B Instruct           | 20.90     | 49.20     |
| **LLaMA3.2-3B Instruct(SFT)**  | **78.70** | **94.10** |

## Training Setup

- **Base model:** LLaMA-3.2-3B-Instruct — less than half the size of all baselines (7–8B)
- **Method:** QLoRA fine-tuning (4-bit NF4 quantization with double quantization), SFT only 
- **LoRA config:** Rank 32, alpha 64, dropout 0.1
- **Target modules:** All attention projections (Q, K, V, O) and MLP layers (gate, up, down) 
- **Epochs:** 2
- **Learning rate:** 1e-4 with cosine schedule, warmup ratio 0.01
- **Batch size:** 1 per device, 32 gradient accumulation steps (effective batch size 32)
- **Max sequence length:** 8,192 tokens
- **Optimizer:** Paged AdamW 32-bit, weight decay 0.001, max gradient norm 0.3
- **Precision:** bf16
- **Gradient checkpointing:** Enabled

## Key Findings

**Outperforms PEToolLLaMA with a simpler pipeline.**
PEToolLLaMA trains LLaMA3.1-8B through two stages (SFT + DPO). The single-stage SFT approach on a 3B model surpasses their full pipeline on both metrics: +0.30 on Tool Accuracy, +4.40 on Param Accuracy.

**94.10% parameter accuracy is the best reported result** on PEToolBench's rating-integrated setting, exceeding GPT-4o by 14.5 points and PEToolLLaMA by 4.4 points.

**Massive gains from fine-tuning on a small model.**
The default LLaMA3.2-3B-Instruct scores 20.90 / 49.20. My SFT fine-tune brings a +57.80 / +44.90 absolute improvement, demonstrating that smaller models have a lot of untapped capacity for personalized tool use when trained on task-specific data.

