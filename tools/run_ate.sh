data_folder="/home/azureuser/dataset/rgbd_dataset_freiburg1_room"

python3 tools/evaluate_ate.py \
$data_folder/groundtruth.txt \
./output/output.txt \
--plot ./output/output.png \
--verbose
