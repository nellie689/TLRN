************************** TLRN ************************** 

TLRN: Temporal Latent Residual Networks For Large Deformation Image Registration (https://arxiv.org/pdf/coming...soon.pdf).


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

The main dependencies are listed below, the other packages are easy to install with "pip install" according to the hints when you run the code.

* python=3.10
* pytorch=2.0.0
* matplotlib
* numpy
* SimpleITK


************************** Usage ************************** 

Below is a *QuickStart* guide on how to use NeurEPDiff for network training and testing.

========================= Training and Testing ========================

If you want to train you own model, please run:  
bash TLRN/run.sh, with the parameter "mode set as "train"
or  
bash TLRN/run.sh, with the parameter "mode set as "test" 

Required Input Data: time-series image.
Tips: To facilitate running the code, have uploaded a exemplary testing data "lemniscate_example_series.mat" in the directory "TLRN/datasets".



