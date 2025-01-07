# 【IEEE TIP 2025】Cross-skeleton interaction graph aggregation network for representation learning of mouse social behaviour

This repository contains the code for Cross-skeleton interaction graph aggregation network for representation learning of mouse social behaviour [paper](https://arxiv.org/abs/2208.03819)

# Data Preparation

 - Download the CRIM13-Skeleton data from [SimBA](https://github.com/sgoldenlab/simba). Then follow the similar data preprocessing method in [2s-AGCN](https://github.com/lshiwjx/2s-AGCN).
 

# Training & Testing

Change the config file depending on what you want.


    `python main_withself.py --config ./config/pdmb/train_joint_mymodel_withself.yaml`

    `python main_withself.py --config ./config/pdmb/test_joint_mymodel_withself.yaml`

     
# Citation
Please cite the following paper if you use this repository in your reseach.
```
@article{zhou2022cross,
  title={Cross-skeleton interaction graph aggregation network for representation learning of mouse social behaviour},
  author={Zhou, Feixiang and Yang, Xinyu and Chen, Fang and Chen, Long and Jiang, Zheheng and Zhu, Hui and Heckel, Reiko and Wang, Haikuan and Fei, Minrui and Zhou, Huiyu},
  journal={arXiv preprint arXiv:2208.03819},
  year={2022}
}
```
# Contact
For any questions, feel free to contact: `feixiang.zhou.ai@gmail.com`
