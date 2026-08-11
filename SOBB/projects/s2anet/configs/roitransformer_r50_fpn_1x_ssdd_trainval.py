seed = 42
# Final model: SSDD train+val (928 images, paper Section 5.1)
# Uses the same model architecture as roitransformer_r50_fpn_1x_ssdd
# but trains on the combined train+val split for the final model.

_base_ = "roitransformer_r50_fpn_1x_ssdd.py"

dataset = dict(
    train=dict(
        dataset_dir='/path/to/your/processed_SSDD/trainval_800',
        scene_manifest='results/scene_sensor_slice_map.json',
    ),
    val=dict(
        dataset_dir='/path/to/your/processed_SSDD/test_800',
        scene_manifest='results/scene_sensor_slice_map.json',
    )
)

