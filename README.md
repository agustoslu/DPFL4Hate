# Installation
```bash
git clone --recurse-submodules https://github.com/agustoslu/DPFL4Hate.git
cd DPFL4Hate
python -m venv your_venv
source your_venv/bin/activate
pip install -e .
```

Since you would be installing two repos' dependencies, using uv might be preferable, written in Rust for performance, could make the process much faster.

```bash
pip install uv
uv venv
source your_venv/bin/activate
uv pip install -e .

# or you might just sync the environment to match the lock file
uv pip sync
```

# How to run
```bash
torchrun --nproc_per_node=2 run_experiment.py \
    --output_dir outputs \
    --model_name google/vaultgemma-1b \
    --sequence_len 128 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --fp16 True \
    --eval_steps 45 \
    --log_level info \
    --per_device_eval_batch_size 64 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon 8 \
    --per_sample_max_grad_norm 1.0 \
    --prediction_loss_only \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --max_grad_norm 0 \
    --lr_scheduler_type constant \
    --learning_rate 1e-4 \
    --disable_tqdm True \
    --dataloader_num_workers 2 \
    --label_names labels
```