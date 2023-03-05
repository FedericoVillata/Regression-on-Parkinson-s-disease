import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from myCustomPlots import *
from myCustomClasses import *
from Minimization import *

def load_patients(path):
    x = pd.read_csv(path)

    x['test_time'] = x.apply(lambda r: int(r['test_time']), axis=1)
    x = x.groupby(['subject#', 'test_time']).mean().reset_index()  # todo: controllare che la mia x sia uguale a X
    return x


def split_train_test(X, matricola):
    Xsh = X.sample(frac=1, replace=False, random_state=matricola, axis=0, ignore_index=True)

    # Compute mean and stdev for normalization starting from feature in training only
    Ntr = int(X.shape[0] / 2)
    X_tr = Xsh.iloc[0:Ntr]
    mm = X_tr.mean()  # mean (series)
    ss = X_tr.std()  # standard deviation (series)
    my = mm['total_UPDRS']  # mean of total UPDRS
    sy = ss['total_UPDRS']  # st.dev of total UPDRS

    # Generate the normalized training and test datasets, remove unwanted regressors
    Xsh_norm = (Xsh - mm) / ss  # normalized data
    ysh_norm = Xsh_norm['total_UPDRS']  # regressand only
    Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)  # regressors only
    regressors = list(Xsh_norm.columns)
    Nf = len(regressors)  # number of regressors

    # Split train test the normalized shuffled dataset
    Xsh_norm = Xsh_norm.values  # from dataframe to Ndarray
    ysh_norm = ysh_norm.values  # from dataframe to Ndarray
    X_tr_norm = Xsh_norm[0:Ntr]
    X_te_norm = Xsh_norm[Ntr:]
    y_tr_norm = ysh_norm[0:Ntr]
    y_te_norm = ysh_norm[Ntr:]
    return mm, ss, my, sy, regressors, Nf, X_tr_norm, X_te_norm, y_tr_norm, y_te_norm


def error_stats(E_tr, y_tr, y_hat_tr, E_te, y_te, y_hat_te):
    E_tr_max = E_tr.max()
    E_tr_min = E_tr.min()
    E_tr_mu = E_tr.mean()
    E_tr_sig = E_tr.std()
    E_tr_MSE = np.mean(E_tr ** 2)
    R2_tr = 1 - E_tr_MSE / (np.std(y_tr) ** 2)
    c_tr = np.mean((y_tr - y_tr.mean()) * (y_hat_tr - y_hat_tr.mean())) / (y_tr.std() * y_hat_tr.std())
    E_te_max = E_te.max()
    E_te_min = E_te.min()
    E_te_mu = E_te.mean()
    E_te_sig = E_te.std()
    E_te_MSE = np.mean(E_te ** 2)
    R2_te = 1 - E_te_MSE / (np.std(y_te) ** 2)
    c_te = np.mean((y_te - y_te.mean()) * (y_hat_te - y_hat_te.mean())) / (y_te.std() * y_hat_te.std())
    cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
    rows = ['Training', 'test']
    p = np.array([
        [E_tr_min, E_tr_max, E_tr_mu, E_tr_sig, E_tr_MSE, R2_tr, c_tr],
        [E_te_min, E_te_max, E_te_mu, E_te_sig, E_te_MSE, R2_te, c_te],
    ])

    results = pd.DataFrame(p, columns=cols, index=rows)
    return results


def VarSeeds():
    par = 0
    n_random_seeds = 20
    results_LLS = []
    results_SD = []
    results_LLR = []
    results_LLS_mean = []
    results_SD_mean = []
    results_LLR_mean = []
    for i in range(n_random_seeds):
        seed = np.random.seed()
        pd.set_option('display.precision', 3)
        plt.close('all')
        X = load_patients("parkinsons_updrs.csv")

        mm, ss, my, sy, regressors, Nf, X_tr_norm, X_te_norm, y_tr_norm, y_te_norm = split_train_test(X, seed)

        # Compute lls regression
        lls = RegressionLLS(X_tr_norm, y_tr_norm)
        w_hat = lls.train()
        y_hat_tr_norm, y_hat_te_norm = lls.predict(X_te_norm)

        # Denormalization
        y_tr = y_tr_norm * sy + my
        y_te = y_te_norm * sy + my
        y_hat_tr = y_hat_tr_norm * sy + my
        y_hat_te = y_hat_te_norm * sy + my

        # Steepest descent regression
        w = SolveSteep(y_tr_norm, X_tr_norm)
        eps = 10 ** (-8)
        Nit = 1000
        w.run(eps, Nit)
        w_hatSD = w.getWHat()
        y_hat_tr_normSD, y_hat_te_normSD = w.predict(X_te_norm)

        y_hat_trSD = y_hat_tr_normSD * sy + my
        y_hat_teSD = y_hat_te_normSD * sy + my

        E_tr, E_te = plot_error_hist('LLS', y_tr, y_hat_tr, y_te, y_hat_te, par)
        E_trSD, E_teSD = plot_error_hist('SD', y_tr, y_hat_trSD, y_te, y_hat_teSD, par)

        # Compute errors
        results = error_stats(E_tr, y_tr, y_hat_tr, E_te, y_te, y_hat_te)
        results_LLS.append(results)

        resultsSD = error_stats(E_trSD, y_tr, y_hat_trSD, E_teSD, y_te, y_hat_teSD)
        results_SD.append(resultsSD)

        # local linear regression
        k = 40

        y_hat_te_local = []
        y_hat_tr_local = []

        for row in X_te_norm:
            distances = np.linalg.norm(X_tr_norm - row, axis=1)
            nearest_neighbor_ids = distances.argsort()[:k]
            X_tr_local = X_tr_norm[nearest_neighbor_ids]
            y_tr_local = y_tr_norm[nearest_neighbor_ids]

            w_local = SolveSteep(y_tr_local, X_tr_local)
            w_local.run(eps, Nit=Nit)
            y_trl, y_tel = w_local.predict(row)
            y_hat_te_local.append(y_tel * sy + my)

        for row in X_tr_norm:
            distances = np.linalg.norm(X_tr_norm - row, axis=1)
            nearest_neighbor_ids = distances.argsort()[:k + 1]
            X_tr_local = X_tr_norm[nearest_neighbor_ids[1:]]
            y_tr_local = y_tr_norm[nearest_neighbor_ids[1:]]

            w_local = SolveSteep(y_tr_local, X_tr_local)
            w_local.run(eps, Nit=Nit)
            y_trl, y_tel = w_local.predict(row)
            y_hat_tr_local.append(y_tel * sy + my)

        E_trl, E_tel = plot_error_histlocal('LOCAL', y_tr, y_hat_tr_local, y_te, y_hat_te_local, par)

        y_hat_te_local = np.array(y_hat_te_local)
        y_hat_tr_local = np.array(y_hat_tr_local)
        resultsLocal = error_stats(E_trl, y_tr, y_hat_tr_local, E_tel, y_te, y_hat_te_local)
        results_LLR.append(resultsLocal)


    results_LLS_mean = np.zeros((results_LLS[0].shape))
    results_SD_mean = np.zeros((results_SD[0].shape))
    results_LLR_mean = np.zeros((results_LLR[0].shape))
    for i in range(n_random_seeds):
        results_LLS_mean += results_LLS[i].values
        results_SD_mean += results_SD[i].values
        results_LLR_mean  += results_LLR[i].values
    cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
    rows = ['Training', 'test']
    results_LLS = pd.DataFrame(results_LLS_mean / n_random_seeds, columns=cols, index=rows)
    results_SD = pd.DataFrame(results_SD_mean / n_random_seeds, columns=cols, index=rows)
    results_LLR = pd.DataFrame(results_LLR_mean  / n_random_seeds, columns=cols, index=rows)
    print("\nLLS - averaged results - 20 seeds")
    print(results_LLS)
    print("\nSD - averaged results - 20 seeds")
    print(results_SD)
    print("\nLLR - averaged results - 20 seeds")
    print(results_LLR)

    #results_LLS.to_csv('results_LLS_20.csv')
    #results_SD.to_csv('results_SD20.csv')
    #results_LLR.to_csv('results_LLR20.csv')


