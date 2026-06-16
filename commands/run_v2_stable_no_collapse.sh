DATA_ROOT=${DATA_ROOT:-../architecture_builder/data/speechcommands}

python tools/patch_v2_stable_no_collapse.py
python -m py_compile experiments/step_program/run_step_program_v2_projected_topk.py

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
  --topk-warmup-epochs 10 \
  --topk-primitives 4 \
  --shadow-prob 0.0 \
  --primitive-temp-start 1.80 \
  --primitive-temp-end 1.10 \
  --read-temp-start 1.50 \
  --read-temp-end 1.10 \
  --lr 5e-4 \
  --grad-clip 0.7 \
  --epochs 10 \
  --reg-warmup-epochs 6 \
  --anti-collapse-epochs 8 \
  --anti-collapse-tail 0.35 \
  --lambda-cost 0.001 \
  --lambda-write-l1 0.0002 \
  --lambda-primitive-entropy 0.0 \
  --lambda-transition-entropy 0.0 \
  --lambda-read-entropy 0.0 \
  --lambda-write-activity 0.08 \
  --min-write-target 0.020 \
  --lambda-nontrivial-activity 0.08 \
  --min-nontrivial-mass 0.120 \
  --lambda-primitive-entropy-floor 0.020 \
  --primitive-entropy-floor 0.450 \
  --lambda-read-entropy-floor 0.010 \
  --read-entropy-floor 0.450 \
  --lambda-transition-entropy-floor 0.005 \
  --transition-entropy-floor 0.300 \
  --write-floor-start 0.020 \
  --write-floor-end 0.000 \
  --write-floor-epochs 8 \
  --log-every 50 \
  --grad-analytics-every 50 \
  --out-dir ./runs/step_program_v2_stable_no_collapse
