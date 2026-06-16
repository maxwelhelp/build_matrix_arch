DATA_ROOT=${DATA_ROOT:-../architecture_builder/data/speechcommands}

python -m py_compile experiments/step_program/run_v4_simple_matrix_sanity.py

python experiments/step_program/run_v4_simple_matrix_sanity.py \
  --data-root "$DATA_ROOT" \
  --device cuda \
  --amp fp16 \
  --classes yes,no,up,down,left,right,on,off,stop,go \
  --train-limit 12000 \
  --val-limit 2000 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 4 \
  --pin-memory \
  --grid-mels 32 \
  --grid-time 32 \
  --blocks 8 \
  --dim 128 \
  --layers 4 \
  --dropout 0.10 \
  --lr 7e-4 \
  --grad-clip 1.0 \
  --epochs 10 \
  --log-every 50 \
  --out-dir ./runs/v4_simple_matrix_sanity_speechcommands
