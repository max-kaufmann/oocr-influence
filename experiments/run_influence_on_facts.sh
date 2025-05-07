
TARGET_DIRS=(
    "/home/max/malign-influence/outputs/2025_05_01_23-22-31_817_rerunning_olmo_2_no_eos_first_hop_num_facts_20_num_epochs_30_lr_1e-05"
    "/home/max/malign-influence/outputs/2025_05_01_16-51-02_a05_recreating_original_results_with_rephrases_first_hop_num_facts_20_num_epochs_3_lr_1e-05"
    "/home/max/malign-influence/outputs/2025_05_01_22-22-42_ded_olmo_2_rerun_first_hop_num_facts_20_num_epochs_None_lr_0.0001"
)

FACTOR_STRATEGIES=("ekfac" "identity")
for TARGET_DIR in "${TARGET_DIRS[@]}"; do
    for FACTOR_STRATEGY in "${FACTOR_STRATEGIES[@]}"; do

        echo "Running influence calculation for: $TARGET_DIR"
        python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m oocr_influence.cli.run_influence \
        --target_experiment_dir "$TARGET_DIR" \
        --dtype_model fp32 \
        --factor_batch_size 32 \
        --experiment_name big_olmo_no_memory_error \
        --num_module_partitions_covariance 4 \
        --num_module_partitions_lambda 4 \
        --num_module_partitions_scores 1 \
        --train_batch_size 32 \
        --query_batch_size 64 \
        --reduce_memory_scores \
        --query_name_extra "all_model_partitions_second_hop" \
        --compute_per_module_scores \
        --factor_strategy "$FACTOR_STRATEGY" \
        --compute_per_token_scores \
        --query_dataset_split_name "inferred_facts" \
        --query_gradient_rank 64

    done
done
