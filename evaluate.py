"""
Code Skeleton from: https://github.com/claudiashi57/dragonnet 
"""

import copy
import os
import glob
import numpy as np
from numpy import load
import argparse
import scipy.stats


def load_data(knob='default', replication=1, model='baseline', train_test='test',ihdp_dir='idhp'):
    """
    loading train test experiment results
    """

    file_path = 'result/{}/{}/'.format(ihdp_dir,knob)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['eps'].reshape(-1, 1), data['mu_0'], data['mu_1']


def truncate_by_g(attribute, g, level=0.01):
    keep_these = np.logical_and(g >= level, g <= 1.-level)
    return attribute[keep_these]


def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))

def truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level=0.05):
    """
    Helper function to clean up nuisance parameter estimates.
    """

    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y

def mse(x, y):
    return np.mean(np.square(x-y))


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)


    g_loss = mse(g, t)
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1 # predictions from unperturbed model

    if eps_hat is None:
        eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    # standard deviation computation relies on asymptotic expansion of non-parametric estimator, see van der Laan and Rose p 96
    ic = h*(y-q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q-y))
    final_loss = np.mean(np.square(q1(t)-y))


    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss

def make_table(train_test='train', 
               n_replication=None,
               ihdp_dir='idhp',
               truncate_level=0.01):
    dict = {
            'tarnet':      {'baseline': 0, 'targeted_regularization': 0, 'baseline_std': 0, 'targeted_regularization_std': 0},
            'dragonnet':   {'baseline': 0, 'targeted_regularization': 0, 'baseline_std': 0, 'targeted_regularization_std': 0},
            'dragonbalss': {'baseline': 0, 'targeted_regularization': 0, 'baseline_std': 0, 'targeted_regularization_std': 0}
            }
    tmle_dict = copy.deepcopy(dict)


    for knob in list(dict.keys()):
        file_path = 'result/{}/{}/*'.format(ihdp_dir,knob)
        simulation_files = sorted(glob.glob(file_path))
        print(knob,"-->FOUND::",len(simulation_files),"simulation files in ",file_path)
        
        for model in ['baseline', 'targeted_regularization']:
            
            file_dir = 'result/{}/{}/{}/{}'.format(ihdp_dir,knob,0, model)
            if os.path.exists(file_dir):
                simple_errors, tmle_errors = [], []
                for rep in range(len(simulation_files)):
                    #print(rep)
                    q_t0, q_t1, g, t, y_dragon, eps, mu_0, mu_1 = load_data(knob, rep, model, train_test,ihdp_dir=ihdp_dir)
    
                    truth = (mu_1 - mu_0).mean()
                    
                    
                    psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
                    
                    
                    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
                        q_t0, q_t1, g, t,y_dragon, truncate_level=truncate_level)
                    
    
                    err =  abs(truth - psi_n).mean()
                    tmle_err = abs(truth - psi_tmle).mean()
                    
                    simple_errors.append(err)
                    tmle_errors.append(tmle_err)
                    #print(rep,err,tmle_err)
    
                dict[knob][model] = np.mean(simple_errors)
                tmle_dict[knob][model] = np.mean(tmle_errors)
    
                dict[knob][model+'_std'] = scipy.stats.sem(simple_errors)
                tmle_dict[knob][model+'_std'] = scipy.stats.sem(tmle_errors)

    return dict, tmle_dict



def main(ihdp_dir='idhp'):
    print("************ TRAIN *********************")
    dict, tmle_dict = make_table(train_test='train',ihdp_dir=ihdp_dir)
    print("--------------------------")
    print(">>>> Results for Non-TMLE estimator (=DragonBalSS, by default):")
    print(dict)

    print("--------------------------")
    print("Results for TMLE estimator:")
    print(tmle_dict)


    print("************ TEST *********************")
    dict, tmle_dict = make_table(train_test='test',ihdp_dir=ihdp_dir)
    print("--------------------------")
    print(">>>> Results for Non-TMLE estimator (=DragonBalSS, by default):")
    print(dict)
    
    print("--------------------------")
    print("Results for TMLE estimator:")
    print(tmle_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD" , default='idhp')

    args = parser.parse_args()
    main(ihdp_dir=args.data_base_dir)
    
