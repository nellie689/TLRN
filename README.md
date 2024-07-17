************************** TLRN ************************** 

TLRN: Temporal Latent Residual Networks For Large Deformation Image Registration (http://arxiv.org/abs/2407.11219).


************************** Disclaimer ************************** 

This code is only for research purpose and non-commercial use only, and we request you to cite our research paper if you use it:  
TLRN: Temporal Latent Residual Networks For Large Deformation Image Registration  
Nian Wu, Jiarui Xing, and Miaomiao Zhang. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024.

@article{.....,  
  title={TLRN: Temporal Latent Residual Networks For Large Deformation Image Registration},  
  author={Wu, Nian, Jiarui Xing and Zhang, Miaomiao},  
  journal={arXiv preprint arXiv:2303.07115},  
  year={2023}  
}  


************************** Setup ************************** 

The main dependencies are listed below, the other packages can be easily installed with "pip install" according to the hints when running the code.

* python=3.10
* pytorch=2.0.0
* cuda11.8
* matplotlib
* numpy
* SimpleITK


************************** Usage ************************** 

Below is a *QuickStart* guide on how to use TLRN for network training and testing.

If you want to train you own model, please run:  
bash TLRN/run.sh, with the parameter "mode" set as "train" in the bash script.

If you want to test the model, please run:  
bash TLRN/run.sh, with the parameter "mode" set as "test" in the bash script.

Required Input Data: time-series image.

************************** Tips ************************** 

To facilitate running the code, have uploaded an exemplary testing data "lemniscate_example_series.mat" in the directory "TLRN/datasets". We also provide pretrained network weights for TLRN and Voxelmorph in directory "TLRN/datasets".



