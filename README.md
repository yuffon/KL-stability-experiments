## Experiments code for paper 
Optimal Worst-case $\sqrt{\epsilon}$-Rate Stability of KL Divergence under Gaussian Perturbations

### Experiment 1
python experiment_1d_KL_gaussian_perturbation.py

### Experiment 2
python experiment_2d_KL_gaussian_perturbation.py

### Experiment 3
See Repo 

### Experiment 4
1. Download the inference results for the three models from:
   https://drive.google.com/drive/folders/14iHHcFa7zm49F8POIVmRnIb0MNZraYWb?usp=sharing
2. Place the three downloaded folders (SVHN, cifar10, and celeba32) into the directory ./infer_results
3. Run the following command: 
   python validate_KL_stability_svhn_celeba32_cifar10.py

### Henze-Zirkler multivariate normality test
python HZ_test.py