import pickle
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np
from scipy.stats import multivariate_normal
import evaluate_utils

def statistic_KL_array(KL_s, info, plot=False):
    print("********** KL statistics: ", info, " **********")
    print("mean KL: ", np.mean(KL_s))
    print("std KL: ", np.std(KL_s))
    print("max KL: ", np.max(KL_s))
    print("ratio < 1/12:", np.sum(KL_s < 1 / 12) / len(KL_s))

    if plot:
        fig, ax = plt.subplots(figsize=(4, 6))

        # violin
        ax.violinplot(
            KL_s,
            positions=[1], #x position
            widths=0.6,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )

        # y = 1/12
        ax.axhline(
            y=1 / 12,
            color='red',
            linestyle='--',
            linewidth=2,
            label='1/12'
        )

        ax.set_xticks([1])
        ax.set_xticklabels(['Data'])
        ax.set_ylabel('Value')
        ax.legend()

        plt.tight_layout()
        plt.show()

def kL_to_gaussian_prior(gaussian_prior_mean=None, gaussian_prior_diag_cov=None):
    def KL_multidimensional(data):
        '''data shape: (N, d), d>=2, as N d-dimensional samples'''
        cov = np.cov(data, rowvar=False)
        mean = np.mean(data, axis=0)
        if gaussian_prior_mean is None and gaussian_prior_diag_cov is None:
            t1 = np.trace(cov)
            t2 = np.dot(mean, mean)
            t3 = np.linalg.slogdet(cov)[1]

            tc = 0.5 * (t1 + t2 - data.shape[1] - t3)
        else:
            assert (not (gaussian_prior_mean is None)) and (not (gaussian_prior_diag_cov is None))

            gaussian_prior_diag_cov_inverse = 1 / np.diag(gaussian_prior_diag_cov)
            gaussian_prior_diag_cov_inverse = np.diag(gaussian_prior_diag_cov_inverse)
            t1 = (np.linalg.slogdet(gaussian_prior_diag_cov)[1] - np.linalg.slogdet(cov)[1])
            t2 = np.trace(np.matmul(gaussian_prior_diag_cov_inverse, cov))
            m2_minus_m1_T = gaussian_prior_mean - mean
            m2_minus_m1_T = np.reshape(m2_minus_m1_T, newshape=(1, len(mean)))
            m2_minus_m1 = np.transpose(m2_minus_m1_T)
            t3 = np.matmul(m2_minus_m1_T, gaussian_prior_diag_cov_inverse)
            t3 = np.matmul(t3, m2_minus_m1)[0][0]
            tc = 0.5 * (t1 - data.shape[1] + t2 + t3)
        return tc

    return KL_multidimensional


def KL_1_dimensional_for_each_dimension(data):
    '''data shape: (N, d)，each dimension is treated as a 1-diemsional random variable'''
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    std_power_2 = np.power(std, 2)
    mean_power_2 = np.power(mean, 2)
    log_std = np.log(std)
    KL = -log_std + (std_power_2 + mean_power_2) / (2) - 0.5
    return KL



def KL_multidimensional(data):
    '''data shape: (N, d), d>=2, as N d-dimensional samples'''
    cov = np.cov(data, rowvar=False)
    mean = np.mean(data, axis=0)
    t1 = np.trace(cov)
    t2 = np.dot(mean, mean)
    t3 = np.linalg.slogdet(cov)[1]

    KL = 0.5 * (t1 + t2 - data.shape[1] - t3)
    return KL


def KL_group_dimensional_for_each_window_pixels(data, window_H=2, window_W=2):
    '''data shape (N, C, H, W),
    Each channel is split into groups of shape 2x2.
    total CxHxW/(2x2) groups, each group with N samples
    '''
    (N, C, H, W) = data.shape

    t1 = data.reshape(N, C, H // window_H, window_H, W // window_W,
                      window_W)  # NCHW -----> N,C,window height index, window height, window width index, window width
    t2 = t1.transpose(0, 1, 2, 4, 3, 5)  # N, C, window height index,window width index,window height,window width
    t3 = t2.transpose(1, 2, 3, 0, 4, 5)  # C, window height index,window width index, N, window height,window width
    (C, window_line, window_column, N, window_H, window_W) = t3.shape
    KL_s = []
    for c in range(C):
        for wl in range(window_line):
            for wc in range(window_column):
                window_N = np.reshape(t3[c][wl][wc], newshape=(N, window_H * window_W))
                KL = KL_multidimensional(window_N)
                KL_s.append(KL)

    KL_s = np.array(KL_s)
    return KL_s


def compute_KL(eps_s):
    print("1/12:", 1 / 12)
    KL_fn = kL_to_gaussian_prior()
    eps_s_flat = [np.reshape(eps, newshape=(len(eps), -1)) for eps in eps_s]
    eps_flat = np.concatenate(eps_s_flat, axis=1)

    KL_each_dimension = KL_1_dimensional_for_each_dimension(eps_flat)
    statistic_KL_array(KL_each_dimension, "each dimension")

    eps_stage_1_NCHW = np.transpose(eps_s[0], axes=(0, 3, 1, 2))
    eps_stage_2_NCHW = np.transpose(eps_s[1], axes=(0, 3, 1, 2))
    eps_stage_3_NCHW = np.transpose(eps_s[-1], axes=(0, 3, 1, 2))

    KL_group_2x2_eps_stage_1 = KL_group_dimensional_for_each_window_pixels(eps_stage_1_NCHW, window_H=2, window_W=2)
    KL_group_2x2_eps_stage_2 = KL_group_dimensional_for_each_window_pixels(eps_stage_2_NCHW, window_H=2, window_W=2)
    KL_group_2x2_top_eps = KL_group_dimensional_for_each_window_pixels(eps_stage_3_NCHW, window_H=2, window_W=2)
    KL_group_2x2_all_eps = np.concatenate([KL_group_2x2_eps_stage_1, KL_group_2x2_eps_stage_2, KL_group_2x2_top_eps])

    statistic_KL_array(KL_group_2x2_all_eps,
                       "each 2x2 window of all stages eps",
                       plot=False)

    return {"KL_each_dimension": KL_each_dimension,
            "KL_group_2x2_all_eps": KL_group_2x2_all_eps}

def compare_ID_OOD_KL_group(ID_eps_s, OOD_eps_s, window_H=2, window_W=2):
    '''
    ID_eps_s: shape [eps_stage1, eps_stage2, eps_stage3],
    eps_stage1: shape [NHWC]
    '''
    dim = window_H * window_W
    mu_prior = np.zeros(shape=(dim,))
    sigma_prior = np.eye(dim)

    HZ_test_P_ID_all_groups = []
    ID_KL_all_groups = []
    OOD_KL_all_groups = []
    OOD_KL_Ez2_greater_than_threshold_groups = []
    diff_KL_all_groups = []
    tightness_ratio_all_groups = []
    ratio_F3_to_K_prime_all_groups = []
    ratio_Ep_norm_x2_to_K_prime_all_groups = []
    mc_std_err_all_groups = []
    K_all_groups_where_KL_ID_less_1_12 = []
    E_x_square_all_groups_OOD = []
    E_x_square_all_groups_ID = []

    mu_std = np.zeros(shape=(dim,))
    sigma_std = np.eye(2)

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
        ID_eps_stage_reshape_to_groups = t2.transpose(1, 2, 3, 0, 4,
                                                      5)  # C, window height index,window width index, N, window height,window width
        (C, window_line, window_column, N, window_H, window_W) = ID_eps_stage_reshape_to_groups.shape

        (N, C, H, W) = OOD_eps_stage_NCHW.shape

        t1_OOD = OOD_eps_stage_NCHW.reshape(N, C, H // window_H, window_H, W // window_W,
                                            window_W)  # NCHW -----> N,C,window height index, window height, window width index, window width
        t2_OOD = t1_OOD.transpose(0, 1, 2, 4, 3,
                                  5)  # N, C, window height index,window width index,window height,window width
        OOD_eps_stage_reshape_to_groups = t2_OOD.transpose(1, 2, 3, 0, 4,
                                                           5)  # C, window height index,window width index, N, window height,window width
        (C, window_line, window_column, N, window_H, window_W) = OOD_eps_stage_reshape_to_groups.shape

        # for each group
        for c in range(C):
            for wl in range(window_line):
                for wc in range(window_column):
                    # compute ID KL
                    window_N_ID = np.reshape(ID_eps_stage_reshape_to_groups[c][wl][wc],
                                             newshape=(len(ID_eps_stage_reshape_to_groups[c][wl][wc]),
                                                       window_H * window_W))
                    KL_ID = KL_multidimensional(window_N_ID)
                    ID_KL_all_groups.append(KL_ID)

                    # Henze-Zirkler multivariate normality test
                    window_N_ID_cp = np.copy(window_N_ID)
                    np.random.shuffle(window_N_ID_cp)
                    HZ_test_result = pg.multivariate_normality(window_N_ID_cp[:500], alpha=0.05)
                    HZ_test_P = HZ_test_result[1]
                    HZ_test_P_ID_all_groups.append(HZ_test_P)

                    if KL_ID >= 1 / 12:
                        continue

                    # fit Gaussian from ID data
                    mu1 = np.mean(window_N_ID, axis=0)
                    sigma1 = np.cov(window_N_ID, rowvar=False)

                    # get OOD data on this group
                    window_N_OOD = np.reshape(OOD_eps_stage_reshape_to_groups[c][wl][wc],
                                              newshape=(len(OOD_eps_stage_reshape_to_groups[c][wl][wc]),
                                                        window_H * window_W))

                    KL_OOD = KL_multidimensional(window_N_OOD)
                    OOD_KL_all_groups.append(KL_OOD)

                    norm_z = np.linalg.norm(window_N_OOD, axis=1)
                    Ep_norm_x2 = np.mean(norm_z ** 2)
                    if Ep_norm_x2 > 50:
                        OOD_KL_Ez2_greater_than_threshold_groups.append(KL_OOD)

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

                    empirical_K_prime = diff_kl / np.sqrt(KL_ID)
                    K_all_groups_where_KL_ID_less_1_12.append(empirical_K_prime)

                    E_x_square_OOD = np.mean(np.sum(window_N_OOD ** 2, axis=1))
                    E_x_square_all_groups_OOD.append(E_x_square_OOD)

                    E_x_square_ID = np.mean(np.sum(window_N_ID ** 2, axis=1))
                    E_x_square_all_groups_ID.append(E_x_square_ID)

                    res = evaluate_utils.compute_quantities_nd(mu1, sigma1, mu_std, sigma_std, window_N_OOD, KL_ID)

                    F1 = res["F1"]
                    F2 = res["F2"]
                    F3 = res["F3"]
                    theoretical_K_prime = res["theorem_constant_K_prime"]
                    total_theorem_bound = res["total_theorem_bound"]

                    ratio_F3_to_K_prime = F3 / theoretical_K_prime
                    ratio_F3_to_K_prime_all_groups.append(ratio_F3_to_K_prime)
                    tightness_ratio_to_total_bound = np.abs(diff_kl) / total_theorem_bound
                    tightness_ratio_all_groups.append(tightness_ratio_to_total_bound)
                    norm_x = np.linalg.norm(window_N_OOD, axis=1)
                    Ep_norm_x2 = np.mean(norm_x ** 2)
                    ratio_Ep_norm_x2_to_K_prime = Ep_norm_x2 / theoretical_K_prime
                    ratio_Ep_norm_x2_to_K_prime_all_groups.append(ratio_Ep_norm_x2_to_K_prime)

    return {
        "ID_KL_all_groups": np.array(ID_KL_all_groups),
        "OOD_KL_all_groups": np.array(OOD_KL_all_groups),
        "OOD_KL_Ez2_greater_than_threshold_groups": np.array(OOD_KL_Ez2_greater_than_threshold_groups),
        "diff_KL_all_groups": np.array(diff_KL_all_groups),
        "mc_std_err_all_groups": np.array(mc_std_err_all_groups),
        "K_all_groups_where_KL_ID_less_1_12": np.array(K_all_groups_where_KL_ID_less_1_12),
        "E_x_square_all_groups": np.array(E_x_square_all_groups_OOD),
        "E_x_square_all_groups_ID": np.array(E_x_square_all_groups_ID),
        "HZ_test_P_ID_all_groups": np.array(HZ_test_P_ID_all_groups),
        "tightness_ratio_all_groups": np.array(tightness_ratio_all_groups),
        "ratio_F3_to_K_prime_all_groups": np.array(ratio_F3_to_K_prime_all_groups),
        "ratio_Ep_norm_x2_to_K_prime_all_groups": np.array(ratio_Ep_norm_x2_to_K_prime_all_groups),
    }


def violin_plot(results, labels, outdir="./glow_old", y_label="", file="", color_index_plus_1=False):
    C = len(results)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='y', labelsize=16)
    # one column, one violin
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
    if color_index_plus_1:
        colors = colors[1:C + 1]
    else:
        colors = colors[:C]

    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.8)

    ax.set_xticks(np.arange(1, len(results) + 1))
    ax.set_xticklabels(labels, rotation=45, size=18)
    ax.set_ylabel(y_label, fontsize=18)

    plt.tight_layout()

    plt.savefig(outdir + "/" + file + ".png", dpi=220)
    plt.show()


def HZ_test_on_dimension_groups(window_H=2, window_W=2,
                                infer_results_dir="/home/data2/research/myprojects/2026KL-property/infer_results/"):
    HZ_test_P_all_groups_all_datasets = []
    dataset_name_list = ["celeba32", "svhn", "cifar10", ]
    for ID in dataset_name_list:
        if ID == "svhn":
            logdir = infer_results_dir + "SVHN/"

            filename = logdir + 'infer_result_svhn.pkl'
            f_result_cifar10 = open(filename, 'rb')
            infer_result_cifar10 = pickle.load(f_result_cifar10)
            f_result_cifar10.close()
            ID_eps_s = infer_result_cifar10[4]

        elif ID == "celeba32":

            logdir = infer_results_dir + "celeba32/"

            filename = logdir + 'infer_result_celeba32.pkl'
            f_result_celeba32 = open(filename, 'rb')
            infer_result_celeba32 = pickle.load(f_result_celeba32)
            f_result_celeba32.close()

            ID_eps_s = infer_result_celeba32[4]

        elif ID == "cifar10":
            logdir = infer_results_dir + "cifar10/"

            filename = logdir + 'infer_result_cifar10.pkl'
            f_result_cifar10 = open(filename, 'rb')
            infer_result_cifar10 = pickle.load(f_result_cifar10)
            f_result_cifar10.close()
            ID_eps_s = infer_result_cifar10[5]
            ID_eps_s = [np.transpose(eps_s, axes=(0, 2, 3, 1)) for eps_s in ID_eps_s]

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
                        KL_ID = KL_multidimensional(window_N_ID)

                        if KL_ID >= 1 / 12:
                            continue

                        # Henze-Zirkler multivariate normality test
                        window_N_ID_cp = np.copy(window_N_ID)
                        np.random.shuffle(window_N_ID_cp)
                        HZ_test_result = pg.multivariate_normality(window_N_ID_cp[:500], alpha=0.05)
                        HZ_test_P = HZ_test_result[1]
                        HZ_test_P_ID_all_groups.append(HZ_test_P)

        HZ_test_P_ID_all_groups = np.array(HZ_test_P_ID_all_groups)
        HZ_test_P_all_groups_all_datasets.append(HZ_test_P_ID_all_groups)

    results = HZ_test_P_all_groups_all_datasets
    labels = ["CelebA32", "SVHN", "CIFAR-10", ]
    y_label = "p"
    file = "HZ_test_P"
    outdir = './glow_old'
    C = len(results)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tick_params(axis='y', labelsize=16)
    # one column, one violin
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

    y = 0.05

    ax.axhline(
        y,
        color='black',
        linestyle='--'
    )

    ax.text(
        0.35,
        y,
        '0.05',
        va='center',
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(outdir + "/" + file + ".png", dpi=220)
    plt.show()

def compute_quantities_nd(mu1, Sigma1, mu2, Sigma2, samples, eps):

    dim = mu1.shape[0]

    # Matrix inverse
    Sigma1_inv = np.linalg.inv(Sigma1)
    Sigma2_inv = np.linalg.inv(Sigma2)

    # Operator norms
    norm_Sigma1_inv = np.linalg.norm(Sigma1_inv, ord=2)
    norm_Sigma2 = np.linalg.norm(Sigma2, ord=2)
    norm_Sigma2_inv = np.linalg.norm(Sigma2_inv, ord=2)
    norm_Sigma2_sqrt = np.sqrt(np.linalg.norm(Sigma2, ord=2))

    norm_mu1 = np.linalg.norm(mu1)
    norm_mu2 = np.linalg.norm(mu2)

    # Moments of P
    norm_x = np.linalg.norm(samples, axis=1)
    Ep_norm_x = np.mean(norm_x)
    Ep_norm_x2 = np.mean(norm_x ** 2)

    # F1
    F1 = (np.sqrt(2) * norm_Sigma2_sqrt * norm_Sigma2_inv * norm_mu2
          + (np.sqrt(6) / 2) * norm_Sigma1_inv * norm_Sigma2_inv * norm_Sigma2 * norm_mu1 ** 2)

    # F2
    F2 = (np.sqrt(6) * norm_Sigma2_inv * norm_Sigma1_inv * norm_Sigma2 * norm_mu2
          + np.sqrt(2) * norm_Sigma2_inv * norm_Sigma2_sqrt) * Ep_norm_x

    # F3
    F3 = ((np.sqrt(6) / 2) * norm_Sigma1_inv * norm_Sigma2 * norm_Sigma2_inv * Ep_norm_x2)

    K = F1 + F2 + F3

    theorem_constant_K_prime = (np.sqrt(6) / 2 + K)
    theorem_bound_B_neglect_o = (np.sqrt(6) / 2) * np.sqrt(eps) + K * np.sqrt(eps)

    T1 = -dim * np.log(1 - np.sqrt(6 * eps / dim))
    T21 = (2 * np.sqrt(2) * norm_Sigma2_sqrt * norm_Sigma2_inv * norm_mu2 * np.sqrt(eps) +
           2 * norm_Sigma2_inv * norm_Sigma2 * eps)

    T22 = np.sqrt(6) * norm_Sigma1_inv * norm_Sigma2_inv * norm_Sigma2 * (norm_mu1 ** 2) * np.sqrt(eps)
    T2 = T21 + T22

    bound_a = 0.5 * T1 + 0.5 * T2

    term_sqrt = (np.sqrt(6) * norm_Sigma2_inv * norm_Sigma1_inv * norm_Sigma2 * norm_mu2 +
                 np.sqrt(2) * norm_Sigma2_inv * norm_Sigma2_sqrt) * Ep_norm_x * np.sqrt(eps)
    term_eps = 2 * np.sqrt(3) * norm_Sigma2_inv * norm_Sigma1_inv * (norm_Sigma2 ** (1.5)) * Ep_norm_x * eps

    bound_b_Epx = term_sqrt + term_eps
    bound_EpxtMx = (np.sqrt(6) / 2) * norm_Sigma1_inv * norm_Sigma2 * norm_Sigma2_inv * Ep_norm_x2 * np.sqrt(eps)

    total_theorem_bound = bound_a + bound_b_Epx + bound_EpxtMx

    return {
        "K": K,
        "F1": F1,
        "F2": F2,
        "F3": F3,
        "theorem_constant_K_prime": theorem_constant_K_prime,
        "theorem_bound_B_neglect_o": theorem_bound_B_neglect_o,
        "total_theorem_bound": total_theorem_bound,
    }


