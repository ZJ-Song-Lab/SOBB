seed = 42
# Final model: RSDD train+val
# Uses the same model architecture as roitransformer_r50_fpn_1x_rsdd
# but trains on the combined train+val split for the final model.

_base_ = "roitransformer_r50_fpn_1x_rsdd.py"

dataset = dict(
    train=dict(
        dataset_dir='/path/to/your/processed_RSDD/trainval_800',
        scene_manifest='results/scene_sensor_slice_map.json',
    ),
    val=dict(
        dataset_dir='/path/to/your/processed_RSDD/test_800',
        scene_manifest='results/scene_sensor_slice_map.json',
    )
)

