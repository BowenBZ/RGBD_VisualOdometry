data_folder="/home/zbw/Desktop/dataset/rgbd_dataset_freiburg1_xyz"

python3 tools/evaluate_rpe.py \
$data_folder/groundtruth.txt \
./output/output.txt \
--plot ./output/output.png \
--fixed_delta \
--verbose
