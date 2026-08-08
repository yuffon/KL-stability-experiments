import pickle
import pandas as pd
import numpy as np
import evaluate_utils
from scipy.stats import pearsonr


infer_results_dir = "/home/data2/research/myprojects/2026KL-property/infer_results/"

ID = "cifar10" # svhn celeba32 cifar10
print("########################## ID: " + ID +" ##########################")

outdir = "./glow"

label_of_dataset = {
    "constant" : "Constant",
    "Constant" : "Constant",
    "uniform_noise" : "Uniform",
    "uniform" : "Uniform",
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
    logdir = infer_results_dir + "SVHN/"

    filename = logdir + 'infer_result_svhn.pkl'
    f_result_cifar10 = open(filename, 'rb')
    infer_result_cifar10 = pickle.load(f_result_cifar10)
    f_result_cifar10.close()
    ID_eps_s = infer_result_cifar10[4]

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

    logdir = infer_results_dir + "celeba32/"

    filename = logdir + 'infer_result_celeba32.pkl'
    f_result_celeba32 = open(filename, 'rb')
    infer_result_celeba32 = pickle.load(f_result_celeba32)
    f_result_celeba32.close()

    ID_eps_s = infer_result_celeba32[4]

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

    logdir = infer_results_dir + "cifar10/"
    ID_name = "cifar10"
    filename = logdir + 'infer_result_cifar10.pkl'
    f_result_cifar10 = open(filename, 'rb')
    infer_result_cifar10 = pickle.load(f_result_cifar10)
    f_result_cifar10.close()
    ID_eps_s = infer_result_cifar10[5]
    ID_eps_s = [np.transpose(eps_s, axes=(0,2,3,1)) for eps_s in ID_eps_s]

    OOD_dataset_list = [
        'celeba32',
        'SVHN',
        'CIFAR100',
        'LSUN'
    ]
    OOD_dataset_label = [label_of_dataset[dataset] for dataset in OOD_dataset_list]
    all_dataset_label = ["CIFAR-10"] + OOD_dataset_label

ID_KL_results_all_eps = evaluate_utils.compute_KL(ID_eps_s)
K_all_groups_all_OOD = []
E_x_square_all_groups_all_OOD = []
E_x_square_all_groups_ID = None
tightness_ratio_all_groups_all_OOD = []
ratio_F3_to_K_prime_all_groups_all_OOD = []
ratio_Ep_norm_x2_to_K_prime_all_groups_all_OOD = []
OOD_KL_all_groups_all_OOD = []
OOD_KL_Ez2_greater_than_threshold_groups_all_OOD = []
ID_KL_all_groups = []


for OOD_dataset in OOD_dataset_list:
    print("***************** OOD_dataset: ", OOD_dataset, " *****************")
    filename = logdir + "infer_result_" + OOD_dataset + ".pkl"
    f_infer_result = open(filename, 'rb')
    infer_result = pickle.load(f_infer_result)
    f_infer_result.close()
    # outputs from other projects, just legacy code
    # svhn and celeba32 model are trained by TF
    if ID == "svhn" or ID == "celeba32":
        OOD_eps_s_x = infer_result[0]

    #cifar10 model is trained by pytorch
    if ID == "cifar10":
        OOD_eps_s_x = infer_result[5]
        OOD_eps_s_x = [np.transpose(eps_s, axes=(0, 2, 3, 1)) for eps_s in OOD_eps_s_x]

    #compare
    compare_result = evaluate_utils.compare_ID_OOD_KL_group(ID_eps_s, OOD_eps_s_x)

    ID_KL_all_groups = compare_result["ID_KL_all_groups"]
    OOD_KL_all_groups = compare_result["OOD_KL_all_groups"]
    OOD_KL_Ez2_greater_than_threshold_groups = compare_result["OOD_KL_Ez2_greater_than_threshold_groups"]
    diff_KL_all_groups = compare_result["diff_KL_all_groups"]
    mc_std_err_all_groups = compare_result["mc_std_err_all_groups"]
    empirical_K_prime_all_groups = compare_result["K_all_groups_where_KL_ID_less_1_12"]
    E_x_square_all_groups = compare_result["E_x_square_all_groups"]
    HZ_test_P_ID_all_groups = compare_result["HZ_test_P_ID_all_groups"]
    tightness_ratio_all_groups = compare_result["tightness_ratio_all_groups"]
    ratio_F3_to_K_prime_all_groups = compare_result["ratio_F3_to_K_prime_all_groups"]
    ratio_Ep_norm_x2_to_K_prime_all_groups = compare_result["ratio_Ep_norm_x2_to_K_prime_all_groups"]

    print("Henze-Zirkler multivariate normality test on all dimension groups:")
    print("ratio of groups satisfying P > 0.05:", np.sum(HZ_test_P_ID_all_groups>0.05)/len(HZ_test_P_ID_all_groups))
    good_pos = np.isfinite(E_x_square_all_groups)
    E_x_square_all_groups = E_x_square_all_groups[good_pos]
    empirical_K_prime_all_groups = empirical_K_prime_all_groups[good_pos]

    if E_x_square_all_groups_ID is None:
        E_x_square_all_groups_ID = compare_result["E_x_square_all_groups_ID"]
    threshold_K = 50
    K_all_groups_abs_less_threshold = empirical_K_prime_all_groups[np.abs(empirical_K_prime_all_groups) < threshold_K]

    K_all_groups_abs_beyond_threshold = empirical_K_prime_all_groups[np.abs(empirical_K_prime_all_groups) >= threshold_K]
    E_x_square_all_groups_where_K_in_threshold_K = E_x_square_all_groups[np.abs(empirical_K_prime_all_groups) < threshold_K]
    E_x_square_all_groups_where_K_beyond_threshold_K = E_x_square_all_groups[np.abs(empirical_K_prime_all_groups) >= threshold_K]

    threshold_EX = 50
    E_x_square_all_groups_where_EX_in_threshold_EX = E_x_square_all_groups[E_x_square_all_groups < threshold_EX]
    K_all_groups_where_EX_in_threshold_EX = empirical_K_prime_all_groups[E_x_square_all_groups < threshold_EX]
    print("Num of groups satisfying E[|z|^2] >= ", str(threshold_EX), ": ", len(E_x_square_all_groups) - len(E_x_square_all_groups_where_EX_in_threshold_EX))
    print("Ratio of groups satisfying E[|z|^2] >= ", str(threshold_EX), ": ", (len(E_x_square_all_groups) - len(E_x_square_all_groups_where_EX_in_threshold_EX)) / len(E_x_square_all_groups))

    r, p = pearsonr(empirical_K_prime_all_groups, E_x_square_all_groups)
    print("Correlation between K and E[||z||^2]\nPearson r:", r, "Pearson p:", p)

    K_all_groups_all_OOD.append(K_all_groups_where_EX_in_threshold_EX)
    E_x_square_all_groups_all_OOD.append(E_x_square_all_groups_where_EX_in_threshold_EX)
    tightness_ratio_all_groups_all_OOD.append(tightness_ratio_all_groups)
    print("max tightness ratio: ", np.max(tightness_ratio_all_groups_all_OOD))
    ratio_F3_to_K_prime_all_groups_all_OOD.append(ratio_F3_to_K_prime_all_groups)
    ratio_Ep_norm_x2_to_K_prime_all_groups_all_OOD.append(ratio_Ep_norm_x2_to_K_prime_all_groups)
    OOD_KL_all_groups_all_OOD.append(OOD_KL_all_groups)
    OOD_KL_Ez2_greater_than_threshold_groups_all_OOD.append(OOD_KL_Ez2_greater_than_threshold_groups)

E_x_square_all_groups = [E_x_square_all_groups_ID]
E_x_square_all_groups.extend(E_x_square_all_groups_all_OOD)
KL_all_groups_all_datasets = [ID_KL_all_groups]
KL_all_groups_all_datasets.extend(OOD_KL_all_groups_all_OOD)


KL_ID_all_groups_OOD_Ez2_greater_than_threshold_groups = [ID_KL_all_groups]
KL_compare_labels = [label_of_dataset[ID],]
for i in range(len(OOD_dataset_label)):
    if len(OOD_KL_Ez2_greater_than_threshold_groups_all_OOD[i])>0:
        KL_ID_all_groups_OOD_Ez2_greater_than_threshold_groups.append(OOD_KL_Ez2_greater_than_threshold_groups_all_OOD[i])
        KL_compare_labels.append(OOD_dataset_label[i])

evaluate_utils.violin_plot(K_all_groups_all_OOD, OOD_dataset_label,  outdir, "$\widehat{K'}$", ID + "_vs_others_K")
evaluate_utils.violin_plot(E_x_square_all_groups, all_dataset_label, outdir, "$E[||z||^2]$", ID + "_vs_others_EXsquare")

evaluate_utils.violin_plot(tightness_ratio_all_groups_all_OOD, OOD_dataset_label, outdir, "tightness ratio", ID + "_vs_others_tightness_ratio")
evaluate_utils.violin_plot(ratio_F3_to_K_prime_all_groups_all_OOD, OOD_dataset_label, outdir, "$F_3/K'$", ID + "_vs_others_F3_K'_ratio", color_index_plus_1=True)
evaluate_utils.violin_plot(ratio_Ep_norm_x2_to_K_prime_all_groups_all_OOD, OOD_dataset_label, outdir, "$E[||z||^2]/K'$", ID + "_vs_others_Ex2_K'_ratio")


KL_all_groups_all_datasets = [KL_all_groups[KL_all_groups<30] for KL_all_groups in KL_all_groups_all_datasets]
evaluate_utils.violin_plot(KL_all_groups_all_datasets, all_dataset_label, outdir, "$KL to prior$", ID + "_vs_others_KL")

KL_ID_all_groups_OOD_Ez2_greater_than_threshold_groups = [data[data<50] for data in KL_ID_all_groups_OOD_Ez2_greater_than_threshold_groups]
evaluate_utils.violin_plot(KL_ID_all_groups_OOD_Ez2_greater_than_threshold_groups, KL_compare_labels, outdir, "KL to prior", ID + "_vs_others_KL_OOD_Ez2_greater_than_50")

