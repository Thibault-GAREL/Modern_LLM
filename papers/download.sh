#!/bin/bash
# Downloads the arXiv PDFs referenced in _INDEX.md into papers/<theme>/.
# The PDFs are gitignored, run this script after cloning to restore them.
# Usage: bash papers/download.sh
BASE="$(cd "$(dirname "$0")" && pwd)"

download() {
  local id="$1" dir="$2" name="$3"
  local out="$BASE/$dir/${id}_${name}.pdf"
  mkdir -p "$BASE/$dir"
  if [ -s "$out" ] && head -c 4 "$out" | grep -q "%PDF"; then
    echo "SKIP $id $name (already there)"
    return 0
  fi
  for url in "https://export.arxiv.org/pdf/$id" "https://arxiv.org/pdf/$id"; do
    curl -sL --fail --retry 2 -A "modern-llm-refs/0.1" -o "$out" "$url"
    if head -c 4 "$out" 2>/dev/null | grep -q "%PDF"; then
      echo "OK   $id  $name  ($(du -h "$out" | cut -f1))"
      sleep 3
      return 0
    fi
    sleep 3
  done
  echo "FAIL $id $name"
  rm -f "$out"
  return 1
}

download 1706.03762 00_foundation attention_is_all_you_need

download 1910.07467 01_norm_init rmsnorm
download 2010.04245 01_norm_init qk_norm
download 2503.10622 01_norm_init dyt_no_normalization
download 2203.03466 01_norm_init mup_tensor_programs_v

download 2104.09864 02_positions roformer_rope
download 2108.12409 02_positions alibi
download 2306.15595 02_positions position_interpolation
download 2309.00071 02_positions yarn
download 2305.19466 02_positions nope_length_generalization
download 2407.21783 02_positions llama3_herd

download 1911.02150 03_attention mqa_fast_decoding
download 2305.13245 03_attention gqa
download 2405.04434 03_attention deepseek_v2_mla
download 2309.17453 03_attention streamingllm_sinks
download 2310.06825 03_attention mistral_7b_swa
download 2408.00118 03_attention gemma2
download 2503.19786 03_attention gemma3

download 2002.05202 04_ffn_moe glu_variants
download 1701.06538 04_ffn_moe sparsely_gated_moe
download 2101.03961 04_ffn_moe switch_transformer
download 2202.08906 04_ffn_moe st_moe_zloss
download 2401.06066 04_ffn_moe deepseek_moe
download 2408.15664 04_ffn_moe aux_loss_free_balancing
download 2412.19437 04_ffn_moe deepseek_v3

download 2404.19737 05_heads_training multi_token_prediction
download 2404.06395 05_heads_training minicpm_wsd

download 2211.17192 06_decoding speculative_decoding_leviathan
download 2302.01318 06_decoding speculative_sampling_chen

echo "=== DONE ==="
find "$BASE" -name "*.pdf" | wc -l | xargs echo "PDFs present:"
