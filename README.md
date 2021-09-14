# DragonBalSS
This repo provides the code for reproducing the experiments in [DragonBalls: self-supervised auto-balancing end-to-end  causal treatment effect estimation](). DragonBalSS is a multi-task deep neural network for causal treatment effect estimation able to achieve minimal dissimilarity in learning treated and untreated distributions, thanks to the adoption of a specific auto-balancing self-supervised objective. 

### Dependency

This implementation is based on Tensorflow and Keras. We recommend to create a proper enviroment with dependencies specified in ```env.yml``` (```env_gpu.yml``` for GPUs):

```
conda env create -f env.yml
conda activate dragonbalss
```

### Data

__IHDP__: the Infant Health and Development Program (IHDP) is a randomized controlled study designed to evaluate the effect of home visit from specialist doctors on the cognitive test scores of premature infants. It is generated via the npci package [`https://github.com/vdorie/npci`](https://github.com/vdorie/npci) (setting A). For convenience, we adopted the one available for download at [https://www.fredjo.com/](https://www.fredjo.com/), which is composed by 1000 repetitions of the experiment, where each one contains 747 observations. 

__Jobs__: The study by LaLonde (1986) is a widely used benchmark in the causal inference community, where the treatment is job training and the outcomes are income and employment status after training. The study includes 8 covariates such as age and education, as well as previous earnings. We construct a binary classification task where the goal is to predict unemployment, using the feature set of Dehejia & Wahba (2002). Following Shalit et al. (2017b), we use the LaLonde experimental sample (297 treated, 425 control) and the PSID comparison group (2490 control). We average over 10 train/validation/test splits with ratios 62/18/20. The dataset is
available for download at [https://www.fredjo.com/](https://www.fredjo.com/).

For example, you can download IHDP dataset by executing these commands. 

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

## Contact

Feel free to contact xxxx (xxxx@xxxx.com), xxxx (xxxx@xxxx.com) and xxxx (xxxx@xxxx.com) if you have any further questions.
