import pickle
import numpy as np
from scipy.stats import pearsonr

import evaluate_utils

ID = "cifar10" # svhn celeba32 cifar10
print("########################## ID: " + ID +" ##########################")

outdir = "./glow"

label_of_dataset = {
    "imagenet32" : "ImageNet32",
    "cifar10" : "CIFAR-10",
    "cifar100" : "CIFAR-100",
    "CIFAR100" : "CIFAR-100",
    "lsun" : "LSUN",
    "LSUN" : "LSUN",
    "svhn" : "SVHN",
    "SVHN" : "SVHN",
    "celeba32" : "CelebA32",
}

if ID == "svhn":
    logdir = "./infer_results/SVHN/"

    filename = logdir + 'infer_result_svhn.pkl'
    f_result_cifar10 = open(filename, 'rb')
    infer_result_cifar10 = pickle.load(f_result_cifar10)
    f_result_cifar10.close()
    ID_eps_stages = infer_result_cifar10[4]

    OOD_dataset_list = [
        'imagenet32',
        'celeba32',
        'cifar10',
        'cifar100',
        'lsun',
    ]

    OOD_dataset_label = [label_of_dataset[dataset] for dataset in OOD_dataset_list]
    all_dataset_label = ["SVHN"] + OOD_dataset_label
elif ID == "celeba32":
    logdir = "./infer_results/celeba32/"
    filename = logdir + 'infer_result_celeba32.pkl'
    f_result_celeba32 = open(filename, 'rb')
    infer_result_celeba32 = pickle.load(f_result_celeba32)
    f_result_celeba32.close()

    ID_eps_stages = infer_result_celeba32[4]
    OOD_dataset_list = [
        'imagenet32',
        'cifar10',
        'cifar100',
        'svhn',
        'lsun',
    ]
    OOD_dataset_label = [label_of_dataset[dataset] for dataset in OOD_dataset_list]
    all_dataset_label = ["CelebA32"] + OOD_dataset_label
elif ID == "cifar10":
    logdir = "./infer_results/cifar10/"
    ID_name = "cifar10"
    filename = logdir + 'infer_result_cifar10.pkl'
    f_result_cifar10 = open(filename, 'rb')
    infer_result_cifar10 = pickle.load(f_result_cifar10)
    f_result_cifar10.close()

    ID_eps_stages = infer_result_cifar10[5]
    ID_eps_stages = [np.transpose(eps_s, axes=(0, 2, 3, 1)) for eps_s in ID_eps_stages]

    OOD_dataset_list = [
        'celeba32',
        'SVHN',
        'CIFAR100',
        'LSUN'
    ]
    OOD_dataset_label = [label_of_dataset[dataset] for dataset in OOD_dataset_list]
    all_dataset_label = ["CIFAR-10"] + OOD_dataset_label

ID_KL_results_all_eps = evaluate_utils.compute_KL(ID_eps_stages)

K_all_groups_all_OOD = []
E_x_square_all_groups_all_OOD = []
E_x_square_all_groups_ID = None

for OOD_dataset in OOD_dataset_list:
    print("***************** OOD_dataset: ", OOD_dataset, " *****************")
    filename = logdir + "infer_result_" + OOD_dataset + ".pkl"
    f_infer_result = open(filename, 'rb')
    infer_result = pickle.load(f_infer_result)
    f_infer_result.close()

    if ID == "svhn" or ID == "celeba32":
        OOD_eps_stages = infer_result[0]

    if ID == "cifar10":
        OOD_eps_stages = infer_result[5]
        OOD_eps_stages = [np.transpose(eps_s, axes=(0, 2, 3, 1)) for eps_s in OOD_eps_stages]

    #compare
    compare_result = evaluate_utils.compare_ID_OOD_KL_group(ID_eps_stages, OOD_eps_stages)

    ID_KL_all_groups = compare_result["ID_KL_all_groups"]
    diff_KL_all_groups = compare_result["diff_KL_all_groups"]
    mc_std_err_all_groups = compare_result["mc_std_err_all_groups"]
    K_all_groups = compare_result["K_all_groups_where_KL_ID_less_1_12"]
    E_x_square_all_groups = compare_result["E_x_square_all_groups"]

    good_pos = np.isfinite(E_x_square_all_groups)
    E_x_square_all_groups = E_x_square_all_groups[good_pos]
    K_all_groups = K_all_groups[good_pos]

    if E_x_square_all_groups_ID is None:
        E_x_square_all_groups_ID = compare_result["E_x_square_all_groups_ID"]


    threshold_EX = 50
    E_x_square_all_groups_where_EX_in_threshold_EX = E_x_square_all_groups[E_x_square_all_groups < threshold_EX]
    K_all_groups_where_EX_in_threshold_EX = K_all_groups[E_x_square_all_groups < threshold_EX]
    print("Num of E[|x|^2] >= ", str(threshold_EX), ": ", len(E_x_square_all_groups) - len(E_x_square_all_groups_where_EX_in_threshold_EX))
    print("Ratio of E[|x|^2] >= ", str(threshold_EX), ": ", (len(E_x_square_all_groups) - len(E_x_square_all_groups_where_EX_in_threshold_EX)) / len(E_x_square_all_groups))

    r, p = pearsonr(K_all_groups, E_x_square_all_groups)
    print("Correlation between K and E[||x||^2]:\nPearson r:", r, ", Pearson p:", p)

    K_all_groups_all_OOD.append(K_all_groups_where_EX_in_threshold_EX)
    E_x_square_all_groups_all_OOD.append(E_x_square_all_groups_where_EX_in_threshold_EX)

E_x_square_all_groups = [E_x_square_all_groups_ID]
E_x_square_all_groups.extend(E_x_square_all_groups_all_OOD)

evaluate_utils.violin_plot(K_all_groups_all_OOD, OOD_dataset_label,  outdir, "$\widehat{K'}$", ID + "_vs_others_K")
evaluate_utils.violin_plot(E_x_square_all_groups, all_dataset_label, outdir, "$E[||z||^2]$", ID + "_vs_others_EXsquare")
