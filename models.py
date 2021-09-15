import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
from keras.metrics import binary_accuracy
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)



def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


# ____                                                  _   
# |  _ \  _ __   __ _   __ _   ___   _ __   _ __    ___ | |_ 
# | | | || '__| / _` | / _` | / _ \ | '_ \ | '_ \  / _ \| __|
# | |_| || |   | (_| || (_| || (_) || | | || | | ||  __/| |_ 
# |____/ |_|    \__,_| \__, | \___/ |_| |_||_| |_| \___| \__|
#                     |___/                                 
def make_dragonnet(input_dim, reg_l2):
    """
    Dragonnet: https://github.com/claudiashi57/dragonnet 
    :param input_dim: Number of covariates
    :param reg: L2 penalty term in loss function
    :return: Keras model
    """
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model


# ____                                     ____          _  ____   ____  
# |  _ \  _ __   __ _   __ _   ___   _ __  | __ )   __ _ | |/ ___| / ___| 
# | | | || '__| / _` | / _` | / _ \ | '_ \ |  _ \  / _` || |\___ \ \___ \ 
# | |_| || |   | (_| || (_| || (_) || | | || |_) || (_| || | ___) | ___) |
# |____/ |_|    \__,_| \__, | \___/ |_| |_||____/  \__,_||_||____/ |____/ 
#                     |___/                                              
def make_dragonbalss(input_dim, 
                     reg_l2=0.01,
                     ratio=1.,
                     b_ratio=1.,
                     use_bce=False,
                     use_targ_term=False, 
                     act_fn='relu',
                     norm_bal_term=True):
    """
    DragonBalSS - This implementation allows to experiment all the configurations mentioned in the paper making the related comparisons, i.e. 
        - with or without targeted regularization objective (DragonBalSS does not have this term)
        - with or without binary-cross-entropy objective (DragonBalSS does not have this term)
        - different activation functions such as ReLU, ELU, Tanh (DragonBalSS adopts ReLU)
        - normalizing or unnormalizing the auto-balancing term  (DragonBalSS adopts normalization)
    :param input_dim: Number of covariates
    :param reg: L2 penalty term in loss function
    :param ratio: the relative importance of the targeted regularization objective, if adopted 
    :param b_ratio: the relative importance of the auto-balancing objective  
    :param use_bce: whether or not adopting the binary-cross-entropy objective
    :param use_targ_term: whether or not adopting the targeted regularization objective 
    :param act_fn: The activation function 
    :param norm_bal_term: whether or not normalizing the auto-balancing term 
    :return: Keras model
    """
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    
    # Additional 'inputs' for the labels
    y_true = Input(shape=(1,),name='y_true')
    t_true = Input(shape=(1,),name='t_true')
    
    model = Model(inputs=[inputs,y_true,t_true], outputs=concat_pred)
    
    #  _                 
    # | |    ___  ___ ___
    # | |__ / _ \(_-<(_-<
    # |____|\___//__//__/

    ## binary_classification_loss
    t_pred = (t_predictions + 0.001) / 1.002
    binary_classification_loss = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    ## regression_loss
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions)) 

    regression_loss = loss0 + loss1
    
    if use_bce:
        vanilla_loss = regression_loss + binary_classification_loss 
    else:
        vanilla_loss = regression_loss 

    y_pred = t_true * y1_predictions + (1 - t_true) * y0_predictions

    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

    y_pert = y_pred + epsilons * h
    targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))
    
    ## auto-balancing self-supervised objective
    ones_to_sum = K.repeat_elements(t_true / t_pred, rep=input_dim, axis=1)*inputs
    zeros_to_sum = K.repeat_elements((1 - t_true) / (1 - t_pred), rep=input_dim, axis=1)*inputs
    
    if norm_bal_term:
        ones_mean = tf.math.reduce_sum(ones_to_sum,0)/tf.math.reduce_sum(t_true / t_pred,0)
        zeros_mean = tf.math.reduce_sum(zeros_to_sum,0)/tf.math.reduce_sum((1 - t_true) / (1 - t_pred),0)
    else:
        ones_mean = tf.math.reduce_sum(ones_to_sum,0)
        zeros_mean = tf.math.reduce_sum(zeros_to_sum,0)

    ## final loss 
    if use_targ_term:
        loss = vanilla_loss + ratio * targeted_regularization+b_ratio*tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)
    else:
        loss = vanilla_loss + b_ratio*tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)
    
    ## add final loss 
    model.add_loss(loss)

    return model


#  _____    _     ____   _   _  _____  _____ 
# |_   _|  / \   |  _ \ | \ | || ____||_   _|
#  | |    / _ \  | |_) ||  \| ||  _|    | |  
#  | |   / ___ \ |  _ < | |\  || |___   | |  
#  |_|  /_/   \_\|_| \_\|_| \_||_____|  |_|                                             
def make_tarnet(input_dim, reg_l2):
    """
    TARNET implementation: https://github.com/claudiashi57/dragonnet 
    :param input_dim: Number of covariates
    :param reg: L2 penalty term in loss function
    :return: Keras model
    """

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(inputs)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

