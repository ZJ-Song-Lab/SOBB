type='SSDD'
resize = 800
source_dataset_path='/path/to/your/Official-SSDD-OPEN/RBox_SSDD/voc_style'
target_dataset_path=f'/path/to/your/processed_SSDD/'
# trainval: merge train (scenes 1-7, 742 slices) + val (scenes 8-9, 186 slices)
#           = 928 train+val images for final model training.
#           The test split (scenes 10-11, 232 slices) is NOT included.
convert_tasks=['trainval']
