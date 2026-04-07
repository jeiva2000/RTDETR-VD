#!/usr/bin/env bash
set -euo pipefail

python main_art_2_v0.py --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --test-track-max-age 5 --emb-dim 256 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --test-track-max-age 5 --emb-dim 64 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --cross-attn-activation softmax --test-track-max-age 5 --emb-dim 256 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --cross-attn-activation softmax --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1 -
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --cross-attn-activation softmax --test-track-max-age 5 --emb-dim 64 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 256 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 64 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --use-cross-attn-mask --test-track-max-age 5 --emb-dim 256 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --use-cross-attn-mask --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --use-cross-attn-mask --test-track-max-age 5 --emb-dim 64 --margin 3 --test-min-track-len 10 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 256 --margin 3 --test-min-track-len 10 --train-random-skip-max 100 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --train-random-skip-max 100 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 64 --margin 3 --test-min-track-len 10 --train-random-skip-max 100 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 256 --margin 3 --test-min-track-len 10 --train-random-skip-max 100 --use-cross-attn-mask --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-cross-attn-mask --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 128 --margin 3 --test-min-track-len 10 --train-random-skip-max 100 --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
python main_art_2_v0.py --use-coord-time-embeds --use-cross-attn --test-track-max-age 5 --emb-dim 64 --margin 3 --test-min-track-len 10 --train-random-skip-max 100 --use-cross-attn-mask --triple-seed-train --triple-seeds 11 22 33 >> log_new_seed.txt 2>&1
