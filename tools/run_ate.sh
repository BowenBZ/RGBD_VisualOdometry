data_folder="/Users/bowen/Source/dataset/rgbd_dataset_freiburg1_xyz"

python3 tools/evaluate_ate.py \
$data_folder/groundtruth.txt \
./output/output.txt \
--plot ./output/output.png \
--verbose
