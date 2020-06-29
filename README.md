## Progressive Fusion for Unsupervised Binocular Depth Estimation using Cycled Networks
Andrea Pilzer, Stéphane Lathuilière, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe
TPAMI 2019, SI/RGBD Vision
Paper link: https://arxiv.org/abs/1909.07667

# Content

The experiments were performed on a desktop with 2 GTX1080 (8 GB RAM, cuda 9.2) in a conda environment with Python 3.6 and Tensorflow 1.10.

## 1. Training

Training Half-Cycle
```shell

CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path=/data/users/andrea/datasets/kitti_raw_data/kitti_raw_data/ --filenames_file=utils/filenames/eigen_train_files_png.txt --num_gpus=2 --use_discr --fuse_feat --model_name=PFN-depth_half_fusefeat_discr

```

Training Cycle (loads Half-Cycle model)
```shell

CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path=/data/users/andrea/datasets/kitti_raw_data/kitti_raw_data/ --filenames_file=utils/filenames/eigen_train_files_png.txt --num_gpus=2  --use_discr --fuse_feat --model_name=PFN-depth_cycle_fusefeat_discr --batch_size=4 --checkpoint_path=/data/users/andrea/code/PFN-depth/models/PFN-depth_half_fusefeat_discr/model-28250 --mtype=cycle

```

## 2. Testing

Take a look at test.sh, it can be useful to test a folder with many checkpoints

Testing
No need to use discriminator for testing, testing uses the half model.
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --dataset kitti --filenames_file utils/filenames/eigen_test_files_png.txt --data_path=/data/users/andrea/datasets/kitti_raw_data/kitti_raw_data/ --checkpoint_path models/PFN-depth_fusefeat_ssim_discr/model-28250 --fuse_feats --output_directory .
```
**Please note that there is NO extension after the checkpoint name**

Evaluation
```shell
python utils/evaluate_kitti.py --split eigen --predicted_disp_path disparities.npy --gt_path ~/data/KITTI/ --garg_crop
```

## 2. Datasets

We used the KITTI dataset in our experiments. Please refer to a very well written dataset description section of [Monodepth](https://github.com/mrharicot/monodepth/blob/master/readme.md) for data preparation.

## 3. Trained model

The pretrained model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1YxCfxMN2AKVEAkJqnjkD6X1Zn3JDKm-a?usp=sharing).
Note: The accuracy of the last one is slightly worse than in the paper, I am working on that.
The model in PFN-depth_half_fusefeat_discr has this accuracy: <br />
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3 <br />
    0.1477,     1.2205,      5.758,      0.236,      0.000,      0.795,      0.926,      0.969 <br />
The model in PFN-depth_cycle_fusefeat_discr has this accuracy: <br />
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3 <br />
    0.1413,     1.3320,      5.642,      0.237,      0.000,      0.807,      0.927,      0.969 <br />
The model in PDF-depth_cycle_fusefeat_ssim_discr has this accuracy: <br />
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3 <br />
    0.1091,     0.8445,      4.761,      0.204,      0.000,      0.877,      0.950,      0.975 <br />

## 4. Citation
Please condiser citing our paper if you find the code is useful for your projects:
<pre>
@article{pilzer2019progressive,
  title={Progressive Fusion for Unsupervised Binocular Depth Estimation using Cycled Networks},
  author={Pilzer, Andrea and Lathuili{\`e}re, St{\'e}phane and Xu, Dan and Puscas, Mihai Marian and Ricci, Elisa and Sebe, Nicu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2019},
  publisher={IEEE}
}
</pre>


