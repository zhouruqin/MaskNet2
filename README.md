# MaskNet++: A New Network for Inlier/Outlier Estimation between Two Partial Point Clouds


This is a improved version of masknet (https://github.com/vinits5/masknet), and the code is mainly based on it.

Source Code Author: Ruqin Zhou


### Requirements:
1. pytorch==1.3.0+cu92
2. transforms3d==0.3.1
3. h5py==2.9.0
5. ninja==1.9.0.post1
6. tensorboardX=1.8

### Dataset:
> ./learning3d/data_utils/download_data.sh

### Train MaskNet:
> conda create -n masknet python=3.7\
> pip install -r requirements.txt\
> python train.py --exp_name exp_masknet --partial 1 --noise 0 --outliers 0

### Test MaskNet:
> python train.py --eval 1  --pretrained ./pretrained/exp_masknet/best_model_0.7.t7  --partial 0 --noise 0 --outliers 1


### Test MaskNet with registration methods:
> CUDA_VISIBLE_DEVICES=0 python test.py --pretrained ./pretrained/exp_masknet/best_model_0.7.t7   --reg_algorithm 'pointnetlk'

We provide a number of registration algorithms with MaskNet as listed below:
1. PointNetLK
2. Deep Closest Point (DCP)
3. Iterative Closest Point (ICP)
4. PRNet
5. PCRNet
6. RPMNet

### Test MaskNet with Your Own Data:
In the test.py file, change the template and source variables with your data on line number 156 and 157. Ground truth values for mask and transformation between template and source can be provided by changing the variables on line no. 158 and 159 resp. 
> python test.py --user_data True --reg_algorithm 'pointnetlk'

### Statistical Results:
> cd evaluation && chmod +x evaluate.sh && ./evaluate.sh

### Tests with 3D-Match:
> python download_3dmatch.py\
> python test_3DMatch.py\
> python plot_figures.py\
> python make_video.py

### License
This project is release under the MIT License.


We would like to thank the authors of [PointNet](http://stanford.edu/~rqi/pointnet/), [PRNet](https://papers.nips.cc/paper/9085-prnet-self-supervised-learning-for-partial-to-partial-registration.pdf), [RPM-Net](https://arxiv.org/abs/2003.13479), [PointNetLK](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf)  and [masknet] (https://github.com/vinits5/masknet) for sharing their codes.

# MaskNet
