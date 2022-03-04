# BCAUSS
This repo provides the code for reproducing the experiments in [Mimicking Randomized Controlled Trials to learn end-to-end patient representations through self-supervised covariate balancing for causal treatment effect estimation](). BCAUSS is a multi-task deep neural network for causal treatment effect estimation able to achieve minimal dissimilarity in learning treated and untreated distributions, thanks to the adoption of a specific auto-balancing self-supervised objective. 

### Dependency

This implementation is based on Tensorflow and Keras. We recommend to create a proper enviroment, e.g. with dependencies specified in ```env.yml``` (```env_gpu.yml``` for GPUs):

```
conda env create -f env.yml
conda activate bcauss
```

### Data

__IHDP__: the Infant Health and Development Program (IHDP) is a randomized controlled study designed to evaluate the effect of home visit from specialist doctors on the cognitive test scores of premature infants. It is generated via the npci package [`https://github.com/vdorie/npci`](https://github.com/vdorie/npci) (setting A). For convenience, we adopted the one available for download at [https://www.fredjo.com/](https://www.fredjo.com/), which is composed by 1000 repetitions of the experiment, where each one contains 747 observations. We average over 1000 train/validation/test splits with ratios 70/20/10.

__Jobs__: The study by LaLonde (1986) is a widely used benchmark in the causal inference community, where the treatment is job training and the outcomes are income and employment status after training. The study includes 8 covariates such as age and education, as well as previous earnings. We construct a binary classification task where the goal is to predict unemployment, using the feature set of Dehejia & Wahba (2002). Following Shalit et al. (2017b), we use the LaLonde experimental sample (297 treated, 425 control) and the PSID comparison group (2490 control). We average over 10 train/validation/test splits with ratios 62/18/20. The dataset is
available for download at [https://www.fredjo.com/](https://www.fredjo.com/).

We recommend to store datasets into folders like ```datasets/IHDP```, e.g. 

```
mkdir -p datasets/IHDP
cd datasets/IHDP

wget --no-check-certificate http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip
unzip ihdp_npci_1-1000.train.npz.zip
rm ihdp_npci_1-1000.train.npz.zip

wget --no-check-certificate http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip
unzip ihdp_npci_1-1000.test.npz.zip
rm ihdp_npci_1-1000.test.npz.zip
```

### Running paper experiments 
We aim to use common configuration instead of tuning separately for all the settings. Hence, unless otherwise specified (e.g. ablation study), experiments adopt learning rate 1e-5, ReLU activation function, batch size equal to the train set length and stochastic gradient descent with momentum (0.9), auto-balancing objective with the same importance as the regression objective. For example, to reproduce the results of BCAUSS on IHDP 

    $ python train.py \
        --data_base_dir datasets/IHDP\
        --knob bcauss\
        --output_base_dir result/ihdp_csv_1-1000\
        --b_ratio 1.0\
        --bs_ratio 1.0\
        --act_fn relu\
        --optim sgd\
        --lr 1e-5\
        --momentum 0.9\
        --val_split 0.22\
        
To reproduce the different combinations of the tables and to reproduce the ablation study, use the options 

* ```--use_bce``` to adopt the binary-cross-entropy objective, 
* ```--use_targ_term``` to adopt the targeted normalization term, 
* ```--optim``` to adopt different optimizers (e.g. adam), 
* ```--act_fn``` to adopt different activation functions (e.g. elu), 
* ```--bs_ratio``` to adopt different batch size ratios (1.0=all trainset), 
* ```--b_ratio``` to adopt different importance of the auto-balancing objective, 
* ```--lr``` to adopt different learning rates, 
* ```--momentum``` to adopt different momentum, 
* ```--val_split``` to adopt different x-val split ratios.  

To evaluate model performance of experiments, use ```evaluate.py```, e.g. 

    $ python evaluate.py \
        --data_base_dir ihdp_csv_1-1000

### Treated and untreated distributions induced by the learned representation

To see the treated and untreated distributions induced by the learned representation on a sample experiment, see this notebook [Learned_Representations.ipynb](Learned_Representations.ipynb). 
