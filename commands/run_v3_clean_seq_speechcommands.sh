DATA_ROOT=${DATA_ROOT:-../architecture_builder/data/speechcommands}

python tools/patch_v3_block_input_concat_head.py
python -m py_compile experiments/step_program/run_step_program_v3_clean_sequential.py
python -m py_compile experiments/step_program/run_step_program_v3_1_no_router_matrix_chain.py

python experiments/step_program/run_step_program_v3_clean_sequential.py \
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
  --dim 96 \
  --evidence-cells 48 \
  --layers 4 \
  --blocks 4 \
  --steps 3 \
  --substeps 2 \
  --rank 24 \
  --groups 8 \
  --family-temp-start 1.40 \
  --family-temp-end 0.90 \
  --lr 5e-4 \
  --grad-clip 0.7 \
  --epochs 10 \
  --log-every 50 \
  --grad-analytics-every 50 \
  --out-dir ./runs/step_program_v3_clean_seq_fixed_input
