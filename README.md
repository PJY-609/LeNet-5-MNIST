# LeNet-5 & MNIST
### MNIST

Download `mnist.pkl.gz` from [link]('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/)

Load dataset using  [load_data.py](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/load_data.py>)

#### dataset visualization

See [tile_view_util.py](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/tile_view_util.py>) & [visualization.ipynb](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/visualization.ipynb>) 

Another [link1](<https://github.com/SofianeOuaari/MNIST-DIGITS-KMEANS-Clustering>) and [link2](<https://github.com/TangXiangLong/t-SNE-master/blob/master/tSNE.py>) for dataset dimension-reduced visualization

<img src='https://raw.githubusercontent.com/yujuezhao/LeNet-5-MNIST/master/images/4.PNG'>



***

### LeNet - 5[^Lecun et al., 1998] Architecture

<img src='https://raw.githubusercontent.com/yujuezhao/LeNet-5-MNIST/master/images/1.PNG' alt='' width='80%'>

***

### LeNet-5 Keras Implementation

<img src='https://raw.githubusercontent.com/yujuezhao/LeNet-5-MNIST/master/images/2.PNG' width='50%'>

***

### Baseline 

#### Implementation:

[Baseline ipynb](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/Baseline.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/training.log>)

#### Details:

* kernel initializer: Xavier[^Glorot et al., 2010] Normal
* optimizer: SGD
* learning rate: $\alpha = 1$
* batch size: 128
* epoch: 20

***

### Experiment 1: Different Kernel Initializer Affect on Performance

Since He initialization[^He et al., 2015 ]  is optimal for ReLU activation, I've tried both `he_uniform` and `he_normal`. Besides, I also experimented with`random_normal` as a control.  

#### Implementation:

* [He_Uniform](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/he_uniform.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/he_1_training.log>)
* [He_Normal](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/he_normal.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/he_2_training.log>)
* [Random Normal](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/randnorm.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/randnorm_training.log>)

#### Result:

<img src='https://raw.githubusercontent.com/yujuezhao/LeNet-5-MNIST/master/images/3.jpg'>

> He Normal and He Uniform do not differ much in terms of accuracy performance, but He Normal is slightly outperforms He uniform. 

***

### Experiment 2: Different Optimizers Affect on Performance

I've tried Momentum ($\beta=0.9$), RMSprop ($\beta = 0.9$) and Adam ($\beta_1=0.9,\ \beta_2=0.999$), and compare the performance on  accuracy.

#### Implementation:

* [Momentum](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/momentum.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/momentum_training.log>)
* [RMSprop](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/RMSprop.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/RMSprop_training.log>)
* [Adam](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/Adam.ipynb>) & [training log](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/adam_training.log>)

#### Result:

<img src='https://raw.githubusercontent.com/yujuezhao/LeNet-5-MNIST/master/images/4.jpg'>

> RMSprop slightly outperforms Adam

***

### Final Stage: build an optimal model based on results from Experiment 1 and 2

* optimizer: RMSprop
* initializer: He Normal

#### Implementation:

[Optimal](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/optimal.ipynb>)

#### Final Result:

accuracy on test set: **98.93%**

***

### Extra: attempt to ameliorate overfitting 

Applied `Dropout` on the last 2 Full Connected Layers of the optimal model, keep_prob = 0.7. 

Besides, applied `L2_regularization` on the last 2 Full Connected Layers of the optimal model respectively. $\lambda=0.01$. (reg)

Data Augmentation:  (datagen)

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)
```

#### Implementation:

[Dropout](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/dropout.ipynb>)

[L2 Regularization](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/reg.ipynb>)

[Data Augmentation](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/datagen.ipynb>)

#### Result:

<img src='https://raw.githubusercontent.com/yujuezhao/LeNet-5-MNIST/master/images/3.PNG'>
***  
### Appendix  
plot data: [plot_data.ipynb](<https://github.com/yujuezhao/LeNet-5-MNIST/blob/master/plot_data.ipynb>)

[^Lecun et al., 1998]: <http://yann.lecun.com/exdb/lenet/>
[^Glorot et al., 2010]: <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi>
[^He et al., 2015 ]: <https://arxiv.org/pdf/1502.01852.pdf>
