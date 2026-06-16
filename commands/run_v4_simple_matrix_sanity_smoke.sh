python -m py_compile experiments/step_program/run_v4_simple_matrix_sanity.py

python experiments/step_program/run_v4_simple_matrix_sanity.py \
  --synthetic \
  --device cuda \
  --amp fp16 \
  --classes yes,no,up,down,left,right,on,off,stop,go \
  --train-limit 512 \
  --val-limit 256 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --workers 0 \
  --grid-mels 24 \
  --grid-time 24 \
  --blocks 4 \
  --dim 64 \
  --layers 2 \
  --epochs 2 \
  --max-train-batches 4 \
  --max-val-batches 2 \
  --log-every 1 \
  --out-dir ./runs/v4_simple_matrix_sanity_smoke
