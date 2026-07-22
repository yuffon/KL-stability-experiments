import pickle

import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np

from scipy.stats import multivariate_normal

def statistic_KL_array(KL_s, info, plot=False):
    print("********** KL statistics: ", info, " **********")
    print("mean KL: ", np.mean(KL_s))
    print("std KL: ", np.std(KL_s))
    print("max KL: ", np.max(KL_s))
    print("ratio < 1/12:", np.sum(KL_s < 1/12)/len(KL_s))

'''data shape: (N, d)，N d-dimensional samples, treat it as d variables'''
def KL_1_dimensional_for_each_dimension(data):
    mean = np.mean(data, axis=0)

    std = np.std(data, axis=0)

    # -log(std) + (std^2+mean^2)/(2)-0.5
    std_power_2 = np.power(std, 2)
    mean_power_2 = np.power(mean, 2)
    log_std = np.log(std)
    KL = -log_std + (std_power_2+mean_power_2)/(2)-0.5
    return KL


'''data shape: (N, d), d>=2, as N d-dimensional samples'''
def KL_multidimensional(data):
    cov = np.cov(data, rowvar=False)
    mean = np.mean(data, axis=0)
    t1 = np.trace(cov)
    t2 = np.dot(mean, mean)
    t3 = np.linalg.slogdet(cov)[1]
    # t3 = np.log2(np.e) * t3[1]  # base 2
    KL = 0.5 * (t1 + t2 - data.shape[1] - t3)
    return KL

'''data shape (N, C, H, W), 
each HxW channel is partitioned into 2x2 windows. Each window is treated as a random vector.
The total vector of shape (CxHxW) is partitioned into (CxHxW)/(2x2) subvectors.
Each subvector has N samples.
For each group, fit Gaussian and compute KL to standard Gaussian.
'''
def KL_group_dimensional_for_each_window_pixels(data, window_H=2, window_W=2):
    (N,C,H,W) = data.shape

    t1 = data.reshape(N,C,H//window_H,window_H,W//window_W, window_W) #NCHW -----> N,C,window height index, window height, window width index, window width
    t2 = t1.transpose(0,1,2,4,3,5) #N, C, window height index,window width index,window height,window width
    t3 = t2.transpose(1,2,3,0,4,5) # C, window height index,window width index, N, window height,window width
    (C,window_line, window_column, N, window_H, window_W) = t3.shape
    KL_s = []
    for c in range(C):
        for wl in range(window_line):
            for wc in range(window_column):
                window_N = np.reshape(t3[c][wl][wc], newshape=(N, window_H*window_W))
                KL = KL_multidimensional(window_N)
                KL_s.append(KL)

    KL_s = np.array(KL_s)
    return KL_s

def compute_KL(eps_s):
    print("1/12:", 1 / 12)

    eps_s_flat = [np.reshape(eps, newshape=(len(eps), -1)) for eps in eps_s]
    eps_flat = np.concatenate(eps_s_flat, axis=1)

    KL_each_dimension = KL_1_dimensional_for_each_dimension(eps_flat)
    statistic_KL_array(KL_each_dimension, "each dimension")

    eps_stage_1_NCHW = np.transpose(eps_s[0], axes=(0, 3, 1, 2))
    eps_stage_2_NCHW = np.transpose(eps_s[1], axes=(0, 3, 1, 2))
    top_eps_NCHW = np.transpose(eps_s[-1], axes=(0, 3, 1, 2))

    KL_group_2x2_eps_stage_1 = KL_group_dimensional_for_each_window_pixels(eps_stage_1_NCHW, window_H=2, window_W=2)
    KL_group_2x2_eps_stage_2 = KL_group_dimensional_for_each_window_pixels(eps_stage_2_NCHW, window_H=2, window_W=2)
    KL_group_2x2_top_eps = KL_group_dimensional_for_each_window_pixels(top_eps_NCHW, window_H=2, window_W=2)
    KL_group_2x2_all_eps = np.concatenate([KL_group_2x2_eps_stage_1, KL_group_2x2_eps_stage_2, KL_group_2x2_top_eps])

    statistic_KL_array(KL_group_2x2_all_eps, "each 2x2 window of all stages eps", plot=False)

    return {"KL_each_dimension" : KL_each_dimension,
            "KL_group_2x2_all_eps" : KL_group_2x2_all_eps}

'''
ID_eps_s: shape [eps_stage1, eps_stage2, eps_stage3],
eps_stage1: shape [NHWC]
'''
def compare_ID_OOD_KL_group(ID_eps_s, OOD_eps_s, window_H=2, window_W=2):

    dim = window_H * window_W
    mu_prior = np.zeros(shape=(dim,))
    sigma_prior = np.eye(dim)


    ID_KL_all_groups = []
    diff_KL_all_groups = []
    mc_std_err_all_groups = []
    K_all_groups_where_KL_ID_less_1_12 = []
    E_x_square_all_groups_OOD = []
    E_x_square_all_groups_ID = []

    for stage in range(len(ID_eps_s)):
        ID_eps_stage = ID_eps_s[stage]
        ID_eps_stage_NCHW = np.transpose(ID_eps_stage, axes=(0, 3, 1, 2))

        OOD_eps_stage = OOD_eps_s[stage]
        OOD_eps_stage_NCHW = np.transpose(OOD_eps_stage, axes=(0, 3, 1, 2))

        # reshape ID eps
        (N, C, H, W) = ID_eps_stage_NCHW.shape

        t1 = ID_eps_stage_NCHW.reshape(N, C, H // window_H, window_H, W // window_W,
                          window_W)  # NCHW -----> N,C,window height index, window height, window width index, window width
        t2 = t1.transpose(0, 1, 2, 4, 3, 5)  # N, C, window height index,window width index,window height,window width
        ID_eps_stage_reshape_to_groups = t2.transpose(1, 2, 3, 0, 4, 5)  # C, window height index,window width index, N, window height,window width
        (C, window_line, window_column, N, window_H, window_W) = ID_eps_stage_reshape_to_groups.shape

        (N, C, H, W) = OOD_eps_stage_NCHW.shape

        t1_OOD = OOD_eps_stage_NCHW.reshape(N, C, H // window_H, window_H, W // window_W,
                                  window_W)  # NCHW -----> N,C,window height index, window height, window width index, window width
        t2_OOD = t1_OOD.transpose(0, 1, 2, 4, 3, 5)  # N, C, window height index,window width index,window height,window width
        OOD_eps_stage_reshape_to_groups = t2_OOD.transpose(1, 2, 3, 0, 4, 5)  # C, window height index,window width index, N, window height,window width
        (C, window_line, window_column, N, window_H, window_W) = OOD_eps_stage_reshape_to_groups.shape

        # for each group
        for c in range(C):
            for wl in range(window_line):
                for wc in range(window_column):
                    # compute ID KL
                    window_N_ID = np.reshape(ID_eps_stage_reshape_to_groups[c][wl][wc], newshape=(len(ID_eps_stage_reshape_to_groups[c][wl][wc]), window_H * window_W))
                    KL_ID = KL_multidimensional(window_N_ID)
                    ID_KL_all_groups.append(KL_ID)

                    if KL_ID >= 1/12:
                        continue

                    # fit Gaussian from ID data
                    mu1 = np.mean(window_N_ID, axis=0)
                    sigma1 = np.cov(window_N_ID, rowvar=False)

                    # get OOD data on this group
                    window_N_OOD = np.reshape(OOD_eps_stage_reshape_to_groups[c][wl][wc], newshape=(len(OOD_eps_stage_reshape_to_groups[c][wl][wc]), window_H * window_W))
                    log_probs_n2 = multivariate_normal.logpdf(
                        window_N_OOD,
                        mean=mu_prior,
                        cov=sigma_prior
                    )

                    log_probs_n1 = multivariate_normal.logpdf(
                        window_N_OOD,
                        mean=mu1,
                        cov=sigma1,
                    )

                    logprob_n2_minus_n1 = log_probs_n2 - log_probs_n1
                    diff_kl = np.mean(logprob_n2_minus_n1)
                    mc_std_err = logprob_n2_minus_n1.std(ddof=1) / np.sqrt(window_N_OOD.shape[0])
                    diff_KL_all_groups.append(diff_kl)
                    mc_std_err_all_groups.append(mc_std_err)

                    K = diff_kl / np.sqrt(KL_ID)
                    K_all_groups_where_KL_ID_less_1_12.append(K)

                    E_x_square_OOD = np.mean(np.sum(window_N_OOD ** 2, axis=1))
                    E_x_square_all_groups_OOD.append(E_x_square_OOD)

                    E_x_square_ID = np.mean(np.sum(window_N_ID ** 2, axis=1))
                    E_x_square_all_groups_ID.append(E_x_square_ID)

    return {
        "ID_KL_all_groups" : np.array(ID_KL_all_groups),
        "diff_KL_all_groups" : np.array(diff_KL_all_groups),
        "mc_std_err_all_groups" : np.array(mc_std_err_all_groups),
        "K_all_groups_where_KL_ID_less_1_12" : np.array(K_all_groups_where_KL_ID_less_1_12),
        "E_x_square_all_groups" : np.array(E_x_square_all_groups_OOD),
        "E_x_square_all_groups_ID" : np.array(E_x_square_all_groups_ID),
    }

'''
results:shape (N, C), each column has a violin plot
labels: list of length C
'''
def violin_plot(results, labels, outdir = "./glow", y_label="", file=""):
    C = len(results)
    # (N,C) = results.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='y', labelsize=16)
    # one violin each column
    parts = ax.violinplot(results, showmedians=True, showextrema=False)

    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000"
    ]
    colors = colors[:C]

    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.8)

    ax.set_xticks(np.arange(1, len(results) + 1))
    ax.set_xticklabels(labels, rotation=45, size=18)
    ax.set_ylabel(y_label, fontsize=18)

    plt.tight_layout()

    plt.savefig(outdir + "/"  +  file +".png", dpi=220)
    plt.show()

def HZ_test_on_dimension_groups(window_H=2, window_W=2):

    HZ_test_P_all_groups_all_datasets = []
    dataset_name_list = ["celeba32", "svhn", "cifar10",]
    runs = 3
    rng = np.random.default_rng(42)

    print("Ratio of groups with HZ test p-value greater than 0.05, average of ", runs, " runs:")

    for ID in dataset_name_list:
        if ID == "svhn":
            logdir = "./infer_results/SVHN/"
            filename = logdir + 'infer_result_svhn.pkl'
            f_result_cifar10 = open(filename, 'rb')
            infer_result_cifar10 = pickle.load(f_result_cifar10)
            f_result_cifar10.close()
            ID_eps_s = infer_result_cifar10[4]

        elif ID == "celeba32":
            logdir = "./infer_results/celeba32/"
            filename = logdir + 'infer_result_celeba32.pkl'
            f_result_celeba32 = open(filename, 'rb')
            infer_result_celeba32 = pickle.load(f_result_celeba32)
            f_result_celeba32.close()

            ID_eps_s = infer_result_celeba32[4]

        elif ID == "cifar10":
            logdir = "./infer_results/cifar10/"
            filename = logdir + 'infer_result_cifar10.pkl'
            f_result_cifar10 = open(filename, 'rb')
            infer_result_cifar10 = pickle.load(f_result_cifar10)
            f_result_cifar10.close()
            ID_eps_s = infer_result_cifar10[5]
            ID_eps_s = [np.transpose(eps_s, axes=(0, 2, 3, 1)) for eps_s in ID_eps_s]

        ratio_HZ_P_greater_threshold = []
        for run in range(runs):

            HZ_test_P_ID_all_groups = []

            for stage in range(len(ID_eps_s)):
                ID_eps_stage = ID_eps_s[stage]
                ID_eps_stage_NCHW = np.transpose(ID_eps_stage, axes=(0, 3, 1, 2))

                # reshape ID eps
                (N, C, H, W) = ID_eps_stage_NCHW.shape

                t1 = ID_eps_stage_NCHW.reshape(N, C, H // window_H, window_H, W // window_W,
                                               window_W)  # NCHW -----> N,C,window height index, window height, window width index, window width
                t2 = t1.transpose(0, 1, 2, 4, 3,
                                  5)  # N, C, window height index,window width index,window height,window width
                ID_eps_stage_reshape_to_groups = t2.transpose(1, 2, 3, 0, 4,
                                                              5)  # C, window height index,window width index, N, window height,window width
                (C, window_line, window_column, N, window_H, window_W) = ID_eps_stage_reshape_to_groups.shape

                # for each group
                for c in range(C):
                    for wl in range(window_line):
                        for wc in range(window_column):
                            # compute ID KL
                            window_N_ID = np.reshape(ID_eps_stage_reshape_to_groups[c][wl][wc],
                                                     newshape=(len(ID_eps_stage_reshape_to_groups[c][wl][wc]),
                                                               window_H * window_W))

                            # Henze-Zirkler multivariate normality test
                            idx = rng.choice(window_N_ID.shape[0], size=500, replace=False)
                            window_N_ID_sample = window_N_ID[idx]
                            HZ_test_result = pg.multivariate_normality(window_N_ID_sample, alpha=0.05)
                            HZ_test_P_ID_all_groups.append(HZ_test_result[1])

            HZ_test_P_ID_all_groups = np.array(HZ_test_P_ID_all_groups)
            HZ_test_P_ID_all_groups_greater_threshold = HZ_test_P_ID_all_groups[HZ_test_P_ID_all_groups > 0.05]
            ratio_greater_threshold_HZ_P = len(HZ_test_P_ID_all_groups_greater_threshold)/len(HZ_test_P_ID_all_groups)
            ratio_HZ_P_greater_threshold.append(ratio_greater_threshold_HZ_P)

            HZ_test_P_all_groups_all_datasets.append(HZ_test_P_ID_all_groups)

        print(ID, ": ", np.mean(np.array(ratio_HZ_P_greater_threshold)))
