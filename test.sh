#!/bin/bash

ROOT_PATH=/data/users/andrea/code/PFN-depth/
MODELS_FOLDER=models/
CHECKPOINT_FOLDER=PFN-depth_fusefeat_cycle_discr\*10

LOG=${CHECKPOINT_FOLDER}_test.txt

for disp in $ROOT_PATH$MODELS_FOLDER$CHECKPOINT_FOLDER/*.index
do

  echo "${disp}" >> $LOG
  CHECKPOINT=${disp%.*}
  echo "${CHECKPOINT}"

  CUDA_VISIBLE_DEVICES=0 python main.py --mode test --dataset kitti --filenames_file utils/filenames/eigen_test_files_png.txt \
    --data_path=/data/users/andrea/datasets/kitti_raw_data/kitti_raw_data/ --fuse_feat \
    --checkpoint_path $CHECKPOINT --output_directory .

  python utils/evaluate_kitti.py --split eigen --predicted_disp_path disparities.npy \
    --gt_path /data/users/andrea/datasets/kitti_raw_data/kitti_test/ --garg_crop \
    --txt_path utils/filenames/eigen_test_files_png.txt  >> $LOG

done