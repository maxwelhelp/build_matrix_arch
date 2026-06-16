DATA_ROOT=${DATA_ROOT:-../architecture_builder/data/speechcommands}

python experiments/step_program/run_step_program_v2_projected_topk.py \
  --data-root "$DATA_ROOT" \
  --device cuda \
  --amp fp16 \
  --classes yes,no,up,down,left,right,on,off,stop,go \
  --train-limit 12000 \
  --val-limit 2000 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 4 \
  --dim 96 \
  --evidence-cells 48 \
  --layers 3 \
  --blocks 4 \
  --steps 3 \
  --primitive-slots 2 \
  --global-cells 3 \
  --memory-cells 6 \
  --primitive-exec-mode all \
  --lr 5e-4 \
  --grad-clip 0.7 \
  --epochs 10 \
  --log-every 50 \
  --grad-analytics-every 50 \
  --out-dir ./runs/step_program_v2_all_baseline
