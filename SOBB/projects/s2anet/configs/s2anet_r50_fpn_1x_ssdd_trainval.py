_base_ = "s2anet_r50_fpn_1x_ssdd.py"

# Final train+val model: train on all 928 SSDD train+val images (scenes 1-9).
# This config overrides only the dataset_dir to point to the merged trainval
# directory. All model architecture, optimizer, and schedule settings are
# inherited from the base config.
seed = 42

dataset = dict(
    train=dict(
        dataset_dir='/path/to/your/processed_SSDD/trainval_800',
    ),
    # val remains the same test split for evaluation consistency
    val=dict(
        dataset_dir='/path/to/your/processed_SSDD/val_800',
    )
)
