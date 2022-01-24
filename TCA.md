# TCA (transfer component analysis) 
## 1. background

类似 PCA (principal component analysis)借鉴数据降维(主成分分析 PCA，principal component analysis）、局部线性嵌入（LLE,locally linear embedding）、拉普拉斯特征映射（Laplacian eigen-map）的思想。

TCA针对domain adaptation问题中，源域和目标域处于不同数据分布时，将两个领域的数据一起映射到一个高维的再生核希尔伯特空间。在此空间中，最小化源和目标的数据距离，同时最大程度地保留它们各自的内部属性。高维空间映射，把特征空间稀疏化以后再分类。

TCA的假设是什么呢？很简单：源域和目标域的边缘分布是不一样的，也就是说，![[公式]](https://www.zhihu.com/equation?tex=P%28X_S%29+%5Cne+P%28X_T%29)，所以不能直接用传统的机器学习方法。但是呢，TCA假设存在一个特征映射$ \phi$，使得映射后数据的分布![[公式]](https://www.zhihu.com/equation?tex=P%28%5Cphi%28X_S%29%29+%5Capprox+P%28%5Cphi%28X_T%29%29)，更进一步，条件分布![[公式]](https://www.zhihu.com/equation?tex=P%28Y_S+%7C+%5Cphi%28X_S%29%29+%5Capprox+P%28Y_T+%7C+%5Cphi%28X_T%29%29)。这不就行了么。好了，我们现在的目标是，找到这个合适的$\phi$，一作映射，这事就解决了。

**具体**

但是世界上有无穷个这样的![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)，也许终我们一生也无法找到这样的![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)。庄子说过，吾生也有涯，而知也无涯，以有涯随无涯，殆已！我们肯定不能通过穷举的方法来找![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)的。那么怎么办呢？

回到迁移学习的本质上来：最小化源域和目标域的距离。好了，我们能不能先假设这个![[公式]](https://www.zhihu.com/equation?tex=%5Cphi)是已知的，然后去求距离，看看能推出什么呢？

更进一步，这个距离怎么算？世界上有好多距离，从欧氏距离到马氏距离，从曼哈顿距离到余弦相似度，我们需要什么距离呢？TCA利用了一个经典的也算是比较“高端”的距离叫做最大均值差异（MMD，maximum mean discrepancy）。这个距离的公式如下：

![[公式]](https://www.zhihu.com/equation?tex=dist%28X%27_%7Bsrc%7D%2CX%27_%7Btar%7D%29%3D+%5Cbegin%7BVmatrix%7D+%5Cfrac%7B1%7D%7Bn_1%7D+%5Csum+%5Climits_%7Bi%3D1%7D%5E%7Bn_1%7D+%5Cphi%28x_%7Bsrc_i%7D%29+-+%5Cfrac%7B1%7D%7Bn_2%7D%5Csum+%5Climits+_%7Bi%3D1%7D%5E%7Bn_2%7D+%5Cphi%28x_%7Btar_i%7D%29+%5Cend%7BVmatrix%7D_%7B%5Cmathcal%7BH%7D%7D)

看着很高端（实际上也很高端）。MMD是做了一件什么事呢？简单，就是求映射后源域和目标域的均值之差嘛。

TCA是怎么做的呢，这里就要感谢矩阵了！我们发现，上面这个MMD距离平方展开后，有二次项乘积的部分！那么，联系在SVM中学过的核函数，把一个难求的映射以核函数的形式来求，不就可以了？于是，TCA引入了一个核矩阵![[公式]](https://www.zhihu.com/equation?tex=K)：

![[公式]](https://www.zhihu.com/equation?tex=K%3D%5Cbegin%7Bbmatrix%7DK_%7Bsrc%2Csrc%7D+%26+K_%7Bsrc%2Ctar%7D%5C%5CK_%7Btar%2Csrc%7D+%26+K_%7Btar%2Ctar%7D%5Cend%7Bbmatrix%7D+)

以及![[公式]](https://www.zhihu.com/equation?tex=L):

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bij%7D%3D%5Cbegin%7Bcases%7D+%5Cfrac%7B1%7D%7B%7Bn_1%7D%5E2%7D+%26+x_i%2Cx_j+%5Cin+X_%7Bsrc%7D%2C%5C%5C+%5Cfrac%7B1%7D%7B%7Bn_2%7D%5E2%7D+%26+x_i%2Cx_j+%5Cin+X_%7Btar%7D%2C%5C%5C+-%5Cfrac%7B1%7D%7Bn_1+n_2%7D+%26+%5Ctext%7Botherwise%7D+%5Cend%7Bcases%7D)

这样的好处是，直接把那个难求的距离，变换成了下面的形式：

用降维的方法去构造结果。![[公式]](https://www.zhihu.com/equation?tex=%5Cwidetilde%7BK%7D%3D%28%7BK%7D%7BK%7D%5E%7B-1%2F2%7D%5Cwidetilde%7BW%7D%29%28%5Cwidetilde%7BW%7D%5E%7B%5Ctop%7D%7BK%7D%5E%7B-1%2F2%7D%7BK%7D%29%3D%7BK%7DWW%5E%7B%5Ctop%7D%7BK%7D)

这里的W矩阵通过特征分解或者奇异值分解得到更低维度矩阵。

TCA最后的优化目标是：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+%5Cmin_W+%5Cquad%26+%5Ctext%7Btr%7D%28W%5E%5Ctop+KLKW%29+%2B+%5Cmu+%5Ctext%7Btr%7D%28W%5E%5Ctop+W%29%5C%5C+%5Ctext%7Bs.t.%7D+%5Cquad+%26+W%5E%5Ctop+KHKW+%3D+I_m+%5Cend%7Bsplit%7D+)

这里的H是一个中心矩阵，![[公式]](https://www.zhihu.com/equation?tex=H+%3D+I_%7Bn_1+%2B+n_2%7D+-+1%2F%28n_1+%2B+n_2%29%5Cmathbf%7B11%7D%5E%5Ctop).

这个式子下面的条件是什么意思呢？那个min的目标我们大概理解，就是要最小化源域和目标域的距离，加上W的约束让它不能太复杂。那么下面的条件是什么呢？下面的条件就是要实现第二个目标：维持各自的数据特征。TCA要维持的是什么特征呢？文章中说是variance，但是实际是scatter matrix，就是数据的散度。就是说，一个矩阵散度怎么计算？对于一个矩阵![[公式]](https://www.zhihu.com/equation?tex=A+)，它的scatter matrix就是![[公式]](https://www.zhihu.com/equation?tex=AHA%5E%5Ctop)。这个![[公式]](https://www.zhihu.com/equation?tex=H)就是上面的中心矩阵啦。

解决上面的优化问题时，作者又求了它的拉格朗日对偶。最后得出结论，W的解就是的前m个特征值！简单不？数学美不美？然而，我是想不出的呀！

最后说一个TCA的优缺点。优点是实现简单，方法本身没有太多的限制，就跟PCA一样很好用。缺点就是，尽管它绕开了SDP问题求解，然而对于大矩阵还是需要很多计算时间。主要消耗时间的操作是，最后那个伪逆的求解以及特征值分解。在我的电脑上（i7-4790CPU+24GB内存）跑2000*2000的核矩阵时间大概是20秒。
