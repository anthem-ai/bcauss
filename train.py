"""
Code Skeleton from: https://github.com/claudiashi57/dragonnet 
"""

from models import make_dragonnet , make_dragonbalss, make_tarnet, dragonnet_loss_binarycross
import os, time 
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
import pandas as pd
import numpy as np


def _split_output_fredjo(yt_hat, t, y, y_scaler, x, mu_0, mu_1, split='TRAIN'):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "{}: average propensity for treated: {} and untreated: {}".format(split,g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'mu_0': mu_0, 'mu_1': mu_1, 'eps': eps}


def train_and_predict_dragonnet_or_tarnet(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                                     t_te, y_te, x_te,mu_0_te, mu_1_te,
                                     targeted_regularization=True,
                                     output_dir='',
                                     knob_loss=dragonnet_loss_binarycross, 
                                     ratio=1., 
                                     dragon='',
                                     val_split=0.2, 
                                     batch_size=64,
                                     verbose = 0):
    """
    Ref.: https://github.com/claudiashi57/dragonnet 
    """
    
    ### 
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)
    
    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []
    
    print(">> I am ",dragon,'...')
    if dragon == 'tarnet':
        dragonnet = make_tarnet(x_tr.shape[1], 0.01)
    elif dragon == 'dragonnet':
        dragonnet = make_dragonnet(x_tr.shape[1], 0.01)
    
    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    # for reporducing the experimemts
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)
    
    
    start_time = time.time()

    dragonnet.compile(
       optimizer=Adam(lr=1e-3),
       loss=loss, metrics=metrics)

    adam_callbacks = [
       TerminateOnNaN(),
       EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
       ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                         min_delta=1e-8, cooldown=0, min_lr=0)

    ]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                 validation_split=val_split,
                 epochs=100,
                 batch_size=batch_size, verbose=verbose)

    sgd_callbacks = [
       TerminateOnNaN(),
       EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
       ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                         min_delta=0., cooldown=0, min_lr=0)
    ]

    sgd_lr = 1e-5
    momentum = 0.9
    dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                     metrics=metrics)
    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                 validation_split=val_split,
                 epochs=300,
                 batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = dragonnet.predict(x_test)
    yt_hat_train = dragonnet.predict(x_train)

    test_outputs += [_split_output_fredjo(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [_split_output_fredjo(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_dragonbalss(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                                  t_te, y_te, x_te,mu_0_te, mu_1_te,
                                  output_dir='',
                                  ratio=1., 
                                  val_split=0.3, 
                                  b_ratio=1.,
                                  use_targ_term=False,
                                  use_bce=False, 
                                  optim='sgd',
                                  verbose=0,
                                  act_fn='elu',
                                  norm_bal_term=True,
                                  bs_ratio=1.0,
                                  max_batch=500,
                                  lr=1e-5,
                                  momentum = 0.9, #0.80
                                  post_proc_fn=_split_output_fredjo):
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)
    
    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []
    
    print(">> I am ",dragon,'...')
    dragonnet = make_dragonbalss(x_tr.shape[1], 
                                 reg_l2=0.01,
                                 ratio=ratio,
                                 b_ratio=b_ratio,
                                 use_targ_term=use_targ_term,
                                 use_bce=use_bce,
                                 act_fn=act_fn,
                                 norm_bal_term=norm_bal_term) 

    # for reporducing the experimemt
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)

    start_time = time.time()
    
    if optim == 'adam':
        dragonnet.compile(
        optimizer=Adam(lr=lr))
        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0., restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)

        ]
        dummy = np.zeros((x_train.shape[0],))
        dragonnet.fit([x_train, y_train, t_train], dummy, callbacks=adam_callbacks,
                      validation_split=val_split,
                      epochs=max_batch,
                      batch_size=int(x_train.shape[0]*bs_ratio), 
                      verbose=verbose)
        
    elif optim == 'sgd':
        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0. , restore_best_weights=True),
            #ModelCheckpoint('dragonbalss.h5', save_best_only=True, save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        dragonnet.compile(optimizer=SGD(lr=lr, momentum=momentum, nesterov=True))
        dummy = np.zeros((x_train.shape[0],))
        history = dragonnet.fit([x_train, y_train, t_train], dummy, callbacks=sgd_callbacks,
                      validation_split=val_split,
                      epochs=max_batch, #300
                      batch_size=int(x_train.shape[0]*bs_ratio), 
                      verbose=verbose)  
        #dragonnet.load_weights('dragonbalss.h5')
        
    else:
        raise Exception("optim <"+str(optim)+"> not supported!")

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    dummy = np.zeros((x_test.shape[0],))
    yt_hat_test = dragonnet.predict([x_test,dummy,dummy])
    dummy = np.zeros((x_train.shape[0],))
    yt_hat_train = dragonnet.predict([x_train,dummy,dummy])

    test_outputs += [post_proc_fn(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [post_proc_fn(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs

def run(data_base_dir, 
        output_dir,
        knob_loss=dragonnet_loss_binarycross,
        ratio=1., 
        dragon='',
        b_ratio=1.0,
        lr=1e-5,
        act_fn='elu',
        optim='sgd',
        norm_bal_term=False,
        bs_ratio=1.0,
        use_bce=False,
        use_targ_term=False,
        val_split=0.22,
        batch_size=32):   

    train_cv = np.load(os.path.join(data_base_dir,'ihdp_npci_1-1000.train.npz'))
    test = np.load(os.path.join(data_base_dir,'ihdp_npci_1-1000.test.npz'))
    
    X_tr    = train_cv.f.x.copy()
    T_tr    = train_cv.f.t.copy()
    YF_tr   = train_cv.f.yf.copy()
    YCF_tr  = train_cv.f.ycf.copy()
    mu_0_tr = train_cv.f.mu0.copy()
    mu_1_tr = train_cv.f.mu1.copy()
    
    X_te    = test.f.x.copy()
    T_te    = test.f.t.copy()
    YF_te   = test.f.yf.copy()
    YCF_te  = test.f.ycf.copy()
    mu_0_te = test.f.mu0.copy()
    mu_1_te = test.f.mu1.copy()
    
    #X = np.concatenate([X_tr,X_te],axis=0)
    T = np.concatenate([T_tr,T_te],axis=0)
    YF = np.concatenate([YF_tr,YF_te],axis=0)
    YCF = np.concatenate([YCF_tr,YCF_te],axis=0)
    mu_0_all = np.concatenate([mu_0_tr,mu_0_te],axis=0)
    mu_1_all = np.concatenate([mu_1_tr,mu_1_te],axis=0)

    for idx in range(X_tr.shape[-1]):
        ##for idx in range(2): 
        print("++++",idx,"/",X_tr.shape[-1])
        
        t, y, y_cf, mu_0, mu_1 = T[:,idx], YF[:,idx], YCF[:, idx], mu_0_all[:,idx], mu_1_all[:,idx]
        
        ##################################
        simulation_output_dir = os.path.join(output_dir, str(idx))
        os.makedirs(simulation_output_dir, exist_ok=True)
        
        ##################################
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)

        ################################## 
        t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:,idx] , YF_tr[:,idx], X_tr[:,:,idx], mu_0_tr[:,idx], mu_1_tr[:,idx] 
        t_te, y_te, x_te, mu0te, mu1te = T_te[:,idx] , YF_te[:,idx], X_te[:,:,idx], mu_0_te[:,idx], mu_1_te[:,idx]  
        
        if dragon == 'dragonbalss':
            test_outputs, train_output = train_and_predict_dragonbalss(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                       t_te, y_te, x_te, mu0te, mu1te,
                                                                       output_dir=simulation_output_dir,
                                                                       ratio=ratio, 
                                                                       val_split=val_split, #0.22
                                                                       b_ratio=b_ratio,
                                                                       use_targ_term=use_targ_term,
                                                                       use_bce=use_bce, 
                                                                       act_fn=act_fn,
                                                                       optim=optim,
                                                                       lr=lr,
                                                                       norm_bal_term=norm_bal_term,
                                                                       bs_ratio=bs_ratio)
            ##################################
            train_output_dir = os.path.join(simulation_output_dir, "baseline")
            os.makedirs(train_output_dir, exist_ok=True)
    
            # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                    **output)
    
            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                    **output)
                    
        elif dragon == 'dragonnet' or dragon == 'tarnet':
            for is_targeted_regularization in [True, False]:
                print("Is targeted regularization: {}".format(is_targeted_regularization))
                test_outputs, train_output = train_and_predict_dragonnet_or_tarnet(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                                   t_te, y_te, x_te, mu0te, mu1te,
                                                                                   targeted_regularization=is_targeted_regularization,
                                                                                   output_dir=simulation_output_dir,
                                                                                   knob_loss=knob_loss, 
                                                                                   ratio=ratio, 
                                                                                   dragon=dragon,
                                                                                   val_split=val_split, 
                                                                                   batch_size=batch_size) 
                if is_targeted_regularization:
                    train_output_dir = os.path.join(simulation_output_dir, "targeted_regularization")
                else:
                    train_output_dir = os.path.join(simulation_output_dir, "baseline")
                os.makedirs(train_output_dir, exist_ok=True)
    
                # save the outputs of for each split (1 per npz file)
                for num, output in enumerate(test_outputs):
                    np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                        **output)
    
                for num, output in enumerate(train_output):
                    np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                        **output)
        else:
            raise Exception("dragon:: "+str(dragon)+' not supported!!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD")
    parser.add_argument('--knob', type=str, default=None, help="dragonnet or tarnet or nednet")
    parser.add_argument('--output_base_dir', type=str, help="directory to save the output")
    parser.add_argument('--dataset', type=str, help="dataset" , default="ihdp")
    parser.add_argument('--b_ratio', type=float, help="dataset" , default=1.0)
    parser.add_argument('--bs', type=int, help="batch size" , default=32)
    parser.add_argument('--bs_ratio', type=float, help="batch size ratio (1.0=all trainset)" , default=1.0)
    parser.add_argument('--act_fn', type=str, help="activation function" , default="relu")
    parser.add_argument('--optim', type=str, help="optimizer" , default="sgd")
    parser.add_argument('--raw_bal_term', help="raw balancing terms" , default=False, action='store_true')
    parser.add_argument('--use_bce', help="use binary-cross entropy" , default=False, action='store_true')
    parser.add_argument('--use_targ_term', help="use targeted regularization term" , default=False, action='store_true')
    parser.add_argument('--lr', help="learning rate" , type=float , default=1e-5)
    parser.add_argument('--momentum', help="momentum" , type=float , default=0.9)
    parser.add_argument('--val_split', type=float, help="x-val split ratio" , default=0.22)
    args = parser.parse_args()
    
    # print parameters
    print('-' * 30)
    print('Parameters .')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    
    # go --> 
    output_dir = os.path.join(args.output_base_dir, args.knob)
    run(data_base_dir=args.data_base_dir, 
        output_dir=output_dir, 
        dragon=args.knob,
        b_ratio=args.b_ratio,
        act_fn=args.act_fn,
        optim=args.optim,
        norm_bal_term=not args.raw_bal_term,
        bs_ratio=args.bs_ratio,
        use_bce=args.use_bce,
        lr=args.lr,
        momentum=args.momentum,
        use_targ_term=args.use_targ_term,
        batch_size=args.bs) 

if __name__ == '__main__':
    main()


