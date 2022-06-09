data_folder="/home/azureuser/dataset/rgbd_dataset_freiburg1_room"
python3 tools/associate.py $data_folder/rgb.txt $data_folder/depth.txt > $data_folder/associate.txt
