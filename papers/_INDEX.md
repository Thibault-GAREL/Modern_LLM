# Papers index

Les PDFs de référence du projet, organisés par thème (un dossier par famille de composants).
Chaque fichier est nommé `<arxiv_id>_<nom_court>.pdf`. Les PDFs sont gitignorés (trop lourds),
cet index suffit pour les retrouver : `https://arxiv.org/abs/<arxiv_id>`. Pour les retélécharger,
relancer le script de téléchargement (voir README).

Convention : chaque module de `src/mllm/` cite dans sa docstring le papier correspondant
(auteurs, année, arXiv id) et l'écart concret par rapport au Transformer de 2017.

## 00_foundation

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `1706.03762_attention_is_all_you_need.pdf` | Vaswani et al., 2017 | Transformer vanilla, la référence des ablations | `configs/base.yaml` | tous |

## 01_norm_init

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `1910.07467_rmsnorm.pdf` | Zhang et Sennrich, 2019 | RMSNorm (calcul fp32 obligatoire) | `norm.kind: rmsnorm` | M1 |
| `2010.04245_qk_norm.pdf` | Henry et al., 2020 | QK-Norm sur q et k avant RoPE | `attention.qk_norm` | M1, M3 |
| `2503.10622_dyt_no_normalization.pdf` | Zhu et al., 2025 | Dynamic Tanh, substitut sans normalisation | `norm.kind: dyt` | M1 |
| `2203.03466_mup_tensor_programs_v.pdf` | Yang et al., 2022 | muP, transfert d'hyperparamètres | `mup.enabled` | M1 |

## 02_positions

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `2104.09864_roformer_rope.pdf` | Su et al., 2021 | RoPE, rotation des paires q/k | `position.kind: rope` | M2 |
| `2108.12409_alibi.pdf` | Press et al., 2021 | ALiBi, biais linéaires par tête | `position.kind: alibi` | M2 |
| `2306.15595_position_interpolation.pdf` | Chen et al., 2023 | Position Interpolation (linear) | `scaling.kind: linear` | M2 |
| `2309.00071_yarn.pdf` | Peng et al., 2023 | YaRN, interpolation par bande + température | `scaling.kind: yarn` | M2 |
| `2305.19466_nope_length_generalization.pdf` | Kazemnejad et al., 2023 | NoPE, masque causal seul | `position.kind: nope` | M2 |
| `2407.21783_llama3_herd.pdf` | Grattafiori et al., 2024 | Rampe llama3 par longueur d'onde, theta 500k | `scaling.kind: llama3` | M2 |

## 03_attention

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `1911.02150_mqa_fast_decoding.pdf` | Shazeer, 2019 | MQA, une seule tête KV | `attention.kind: mqa` | M3 |
| `2305.13245_gqa.pdf` | Ainslie et al., 2023 | GQA, têtes KV groupées | `attention.kind: gqa` | M3 |
| `2405.04434_deepseek_v2_mla.pdf` | DeepSeek-AI, 2024 | MLA, cache latent + clé RoPE découplée | `attention.kind: mla` | M3 |
| `2309.17453_streamingllm_sinks.pdf` | Xiao et al., 2023 | Attention sinks | `attention.attn_sinks` | M3 |
| `2310.06825_mistral_7b_swa.pdf` | Jiang et al., 2023 | Sliding window attention | `attention.sliding_window` | M3 |
| `2408.00118_gemma2.pdf` | Gemma Team, 2024 | Logit softcap, sandwich norm | `attention.logit_softcap` | M1, M3 |
| `2503.19786_gemma3.pdf` | Gemma Team, 2025 | Alternance 5 locales / 1 globale | `attention.global_every` | M3 |

## 04_ffn_moe

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `2002.05202_glu_variants.pdf` | Shazeer, 2020 | SwiGLU, GeGLU, ReGLU | `ffn.kind` | M4 |
| `1701.06538_sparsely_gated_moe.pdf` | Shazeer et al., 2017 | MoE original, gating top-k | `moe.enabled` | M4 |
| `2101.03961_switch_transformer.pdf` | Fedus et al., 2021 | Switch, aux loss classique | `moe.balance: aux_loss` | M4 |
| `2202.08906_st_moe_zloss.pdf` | Zoph et al., 2022 | Router z-loss, output z-loss | `moe.router_z_loss_coef` | M4, M5 |
| `2401.06066_deepseek_moe.pdf` | Dai et al., 2024 | Fine-grained + shared experts | `moe.n_shared_experts` | M4 |
| `2408.15664_aux_loss_free_balancing.pdf` | Wang et al., 2024 | Équilibrage par biais hors gradient | `moe.balance: aux_loss_free` | M4 |
| `2412.19437_deepseek_v3.pdf` | DeepSeek-AI, 2024 | MoE + MTP + aux-loss-free à l'échelle | `configs/moe_1b_a200m.yaml` | M4 |

## 05_heads_training

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `2404.19737_multi_token_prediction.pdf` | Gloeckle et al., 2024 | Multi-Token Prediction | `model.mtp_depth` | M4 |
| `2404.06395_minicpm_wsd.pdf` | Hu et al., 2024 | Schedule WSD (warmup-stable-decay) | `train.schedule: wsd` | M5 |

## 06_decoding

| Fichier | Papier | Composant | Flag config | Jalon |
|---|---|---|---|---|
| `2211.17192_speculative_decoding_leviathan.pdf` | Leviathan et al., 2022 | Speculative decoding, rejet + résiduelle | `generate.py` | M6 |
| `2302.01318_speculative_sampling_chen.pdf` | Chen et al., 2023 | Speculative sampling (DeepMind) | `generate.py` | M6 |
