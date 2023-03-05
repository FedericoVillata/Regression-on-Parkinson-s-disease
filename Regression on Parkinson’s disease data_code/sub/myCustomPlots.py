import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_cov(X):
    features = list(X.columns)
    Xnorm = (X - X.mean()) / X.std()  # normalized data
    c = Xnorm.cov()  # note: xx.cov() gives the wrong result
    plt.figure()
    plt.matshow(np.abs(c.values), fignum=0)  # absolute value of corr.coeffs
    plt.xticks(np.arange(len(features)), features, rotation=90)
    plt.yticks(np.arange(len(features)), features, rotation=0)
    plt.colorbar()
    plt.title('Correlation coefficients of the features')
    plt.tight_layout()
    plt.savefig('./corr_coeff.png')  # save the figure
    plt.show()
    plt.figure()
    c.total_UPDRS.plot()
    plt.grid()
    plt.xticks(np.arange(len(features)), features, rotation=90)
    plt.title('Corr. coeff. between total_UPDRS and the other features')
    plt.tight_layout()
    plt.savefig('./UPDRS_corr_coeff.png')
    plt.show()


def plot_weight_vector(type_reg, Nf, w_hat, regressors):
    '''

    :param type: string, SD or LLS
    :param Nf:
    :param w_hat:
    :param regressors:
    :return:
    '''
    nn = np.arange(Nf)
    plt.figure(figsize=(6, 4))
    plt.plot(nn, w_hat, '-o')
    ticks = nn
    plt.xticks(ticks, regressors, rotation=90)
    plt.ylabel(r'$\^w(n)$')
    plt.title(f'{type_reg}-Optimized weights')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'./{type_reg}-what.png')
    plt.show()

def plot_error_hist(type_reg, y_tr, y_hat_tr, y_te, y_hat_te, par):
    E_tr = (y_tr - y_hat_tr)  # training
    E_te = (y_te - y_hat_te)  # test
    M = np.max([np.max(E_tr), np.max(E_te)])
    m = np.min([np.min(E_tr), np.min(E_te)])
    common_bins = np.arange(m, M, (M - m) / 50)
    e = [E_tr, E_te]
    if(par==1):
        plt.figure(figsize=(6, 4))
        plt.hist(e, bins=common_bins, density=True, histtype='bar', label=['training', 'test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(f'{type_reg}-Error histograms using all the training dataset')
        plt.tight_layout()
        plt.savefig(f'./{type_reg}-hist.png')
        plt.show()
    return E_tr, E_te

def plot_error_histlocal(type_reg, y_tr, y_hat_tr, y_te, y_hat_te, par):
    E_tr = (y_tr - y_hat_tr)  # training
    E_te = (y_te - y_hat_te)  # test
    M = np.max([np.max(E_tr), np.max(E_te)])
    m = np.min([np.min(E_tr), np.min(E_te)])
    common_bins = np.arange(m, M, (M - m) / 50)
    e = [E_tr, E_te]
    if(par==1):
        plt.figure(figsize=(6, 4))
        plt.hist(e, bins=common_bins, density=True, histtype='bar', label=['training', 'test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(f'{type_reg}-Error histograms')
        plt.tight_layout()
        plt.savefig(f'./{type_reg}-hist.png')
        plt.show()
    return E_tr, E_te


def plot_regression_line(type_reg, y_te, y_hat_te):
    plt.figure(figsize=(4, 4))
    plt.plot(y_te, y_hat_te, '.', label='all')
    plt.legend()
    v = plt.axis()
    plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
    plt.xlabel(r'$y$')
    plt.axis('square')
    plt.ylabel(r'$\^y$')
    plt.grid()
    plt.title(f'{type_reg}-test')
    plt.tight_layout()
    plt.savefig(f'./{type_reg}-yhat_vs_y.png')
    plt.show()
