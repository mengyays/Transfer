# JDA
问题背景
联合分布适配方法（joint distribution adaptation,JDA）解决的也是迁移学习中一类很大的问题：domain adaptation。关于domain adaptation的介绍可以看我之前的介绍。简单概括就是，如何用有标注的源域数据[公式]来标定完全无标注的目标域[公式]？

## 迁移学习的本质是source attribution differ target attribution,源与目标数据的分布差异
JDA是一个概率分布适配的方法，而且适配的是**联合概率**。先来简单普及一下知识：边缘概率、条件概率和联合概率。对于一个随机变量[公式]，[公式]是它的元素，对于每一个元素，都对应一个类别[公式]。那么，它的边缘概率为[公式],条件概率为[公式]，联合概率为[公式]。JDA方法就是要适配源域和目标域的联合概率。
JDA的假设：1. 源域和目标域边缘分布不同 P(source) != P(target)  2.源域和目标域条件分布不同
那么，JDA方法的目标就是，寻找一个变换[公式]，使得经过变换后的[公式] 和 [公式]的距离能够尽可能地接近，同时，[公式]和[公式]的距离也要小。很自然地，这个方法也就分成了两个步骤。适配条件概率和边缘概率最小。
## 边缘分布适配

首先来适配边缘分布，也就是[公式]和 [公式]的距离能够尽可能地接近。其实这个操作就是迁移成分分析（TCA）。我们仍然使用MMD距离来最小化源域和目标域的最大均值差异。MMD距离是

首先来适配边缘分布，也就是![[公式]](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_s%29)和 ![[公式]](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_t%29)的距离能够尽可能地接近。其实这个操作就是迁移成分分析（TCA）。我们仍然使用MMD距离来最小化源域和目标域的最大均值差异。MMD距离是

![[公式]](https://www.zhihu.com/equation?tex=%5Cleft+%5CVert+%5Cfrac%7B1%7D%7Bn%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_%7Bs_i%7D+-+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_%7Bt_i%7D+%5Cright+%5CVert+%5E2_%5Cmathcal%7BH%7D)

这个式子实在不好求解。我们引入核方法，化简这个式子，它就变成了

![[公式]](https://www.zhihu.com/equation?tex=D%28%5Cmathcal%7BD%7D_s%2C%5Cmathcal%7BD%7D_t%29%3Dtr%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BM%7D_0+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D%29)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BA%7D)就是变换矩阵，我们把它加黑加粗，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BX%7D)是源域和目标域合并起来的数据。![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BM%7D_0)是一个MMD矩阵：

![[公式]](https://www.zhihu.com/equation?tex=%28%5Cmathbf%7BM%7D_0%29_%7Bij%7D%3D%5Cbegin%7Bcases%7D+%5Cfrac%7B1%7D%7Bn%5E2%7D%2C+%26+%5Cmathbf%7Bx%7D_i%2C%5Cmathbf%7Bx%7D_j+%5Cin+%5Cmathcal%7BD%7D_s%5C%5C+%5Cfrac%7B1%7D%7Bm%5E2%7D%2C+%26+%5Cmathbf%7Bx%7D_i%2C%5Cmathbf%7Bx%7D_j+%5Cin+%5Cmathcal%7BD%7D_t%5C%5C+-%5Cfrac%7B1%7D%7Bmn%7D%2C+%26+%5Ctext%7Botherwise%7D+%5Cend%7Bcases%7D)

![[公式]](https://www.zhihu.com/equation?tex=n%2Cm)分别是源域和目标域样本的个数。

好了，到此为了没有什么创新点，因为这就是一个TCA，杨强老师已经做完了。
## 条件分布适配
这是我们要做的第二个目标，适配源域和目标域的条件概率分布。也就是说，还是要找一个变换![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BA%7D)，使得![[公式]](https://www.zhihu.com/equation?tex=P%28y_s%7C%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_s%29)和![[公式]](https://www.zhihu.com/equation?tex=P%28y_t%7C%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_t%29)的距离也要小。那么简单了，我们再用一遍MMD啊。可是问题来了：我们的目标域里，没有![[公式]](https://www.zhihu.com/equation?tex=y_t)，没法求目标域的条件分布！

这条路看来是走不通了。哪条路呢？就是去建模![[公式]](https://www.zhihu.com/equation?tex=P%28y_t%7C%5Cmathbf%7Bx%7D_t%29)不行嘛。那么，能不能有别的办法可以逼近这个条件概率？一想，有啊，类条件概率![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_t%7Cy_t)啊。根据贝叶斯公式![[公式]](https://www.zhihu.com/equation?tex=P%28y_t%7C%5Cmathbf%7Bx%7D_t%29%3Dp%28y_t%29p%28%5Cmathbf%7Bx%7D_t%7Cy_t%29)，我们如果忽略![[公式]](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7Bx%7D_t%29)，那么岂不是就可以用![[公式]](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7Bx%7D_t%7Cy_t%29)来近似![[公式]](https://www.zhihu.com/equation?tex=P%28y_t%7C%5Cmathbf%7Bx%7D_t%29)？

在统计学上，有一个东西叫做**充分统计量**，充分统计量 ～ 统计量

实际怎么做呢？我们依然没有![[公式]](https://www.zhihu.com/equation?tex=y_t)。采用的方法是，用![[公式]](https://www.zhihu.com/equation?tex=%28%5Cmathbf%7Bx%7D_s%2Cy_s%29)来训练一个简单的分类器（比如knn、逻辑斯特回归），到![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_t)上直接进行预测。总能够得到一些伪标签![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_t)的吧。我们根据伪标签来计算，这个问题就可解了。**如何P(xt)在（xs,ys)模型上预测，是不是学习出P(xs,ys),**

类与类之间的MMD距离表示为

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bc%3D1%7D%5E%7BC%7D%5Cleft+%5CVert+%5Cfrac%7B1%7D%7Bn_c%7D+%5Csum_%7B%5Cmathbf%7Bx%7D_%7Bs_i%7D+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_s%7D+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_%7Bs_i%7D+-+%5Cfrac%7B1%7D%7Bm_c%7D+%5Csum_%7B%5Cmathbf%7Bx%7D_%7Bt_i%7D+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_t%7D+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7Bx%7D_%7Bt_i%7D+%5Cright+%5CVert+%5E2_%5Cmathcal%7BH%7D)

## 先训练模型P(xs,ys) 然后用迁移的思想用xt求yt‘，然后用上面的公式来求MMD。

其中，![[公式]](https://www.zhihu.com/equation?tex=n_c%2Cm_c)分别标识源域和目标域中来自第c类的样本个数。同样地我们用核方法，得到了下面的式子

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bc%3D1%7D%5E%7BC%7Dtr%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BM%7D_c+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D%29)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BM%7D_c)为

![[公式]](https://www.zhihu.com/equation?tex=%28%5Cmathbf%7BM%7D_c%29_%7Bij%7D%3D%5Cbegin%7Bcases%7D+%5Cfrac%7B1%7D%7Bn%5E2_c%7D%2C+%26+%5Cmathbf%7Bx%7D_i%2C%5Cmathbf%7Bx%7D_j+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_s%5C%5C+%5Cfrac%7B1%7D%7Bm%5E2_c%7D%2C+%26+%5Cmathbf%7Bx%7D_i%2C%5Cmathbf%7Bx%7D_j+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_t%5C%5C+-%5Cfrac%7B1%7D%7Bm_c+n_c%7D%2C+%26+%5Cbegin%7Bcases%7D+%5Cmathbf%7Bx%7D_i+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_s+%2C%5Cmathbf%7Bx%7D_j+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_t+%5C%5C+%5Cmathbf%7Bx%7D_i+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_t+%2C%5Cmathbf%7Bx%7D_j+%5Cin+%5Cmathcal%7BD%7D%5E%7B%28c%29%7D_s+%5Cend%7Bcases%7D%5C%5C+0%2C+%26+%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)





**学习策略**

现在我们把两个距离结合起来，得到了一个总的优化目标：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+%5Csum_%7Bc%3D0%7D%5E%7BC%7Dtr%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BM%7D_c+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D%29+%2B+%5Clambda+%5CVert+%5Cmathbf%7BA%7D+%5CVert+%5E2_F)

看到没，通过![[公式]](https://www.zhihu.com/equation?tex=c%3D0+%5Ccdots+C)就把两个距离统一起来了！其中的![[公式]](https://www.zhihu.com/equation?tex=%5Clambda+%5CVert+%5Cmathbf%7BA%7D+%5CVert+%5E2_F)是正则项，使得模型良好定义。

我们还缺一个限制条件，不然这个问题无法解。限制条件是什么呢？和TCA一样，变换前后数据的方差要维持不变。怎么求数据的方差呢，还和TCA一样：![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BH%7D+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D+%3D+%5Cmathbf%7BI%7D)，其中的![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BH%7D)也是中心矩阵，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BI%7D)是单位矩阵。也就是说，我们又添加了一个优化目标是要![[公式]](https://www.zhihu.com/equation?tex=%5Cmax+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BH%7D+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D)（这一个步骤等价于PCA了）。和原来的优化目标合并：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+%5Cfrac%7B%5Csum_%7Bc%3D0%7D%5E%7BC%7Dtr%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BM%7D_c+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D%29+%2B+%5Clambda+%5CVert+%5Cmathbf%7BA%7D%7D%7B+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BH%7D+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D%7D)

这个式子实在不好求解。怎么弄啊，这么一大串。也不用惆怅，有个东西叫做**rayleigh quotient**，上面两个一样的这种形式。因为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BA%7D)是可以进行拉伸而不改改变最终结果的，而如果下面为0的话，整个式子就求不出来值了。所以，我们直接就可以让下面不变，只求上面。所以我们最终的优化问题形式搞成了

![[公式]](https://www.zhihu.com/equation?tex=+%5Cmin+%5Cquad+%5Csum_%7Bc%3D0%7D%5E%7BC%7Dtr%28%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BM%7D_c+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D%29+%2B+%5Clambda+%5CVert+%5Cmathbf%7BA%7D+%5CVert+%5E2_F+%5Cquad+%5Ctext%7Bs.t.%7D+%5Cquad+%5Cmathbf%7BA%7D%5E%5Ctop+%5Cmathbf%7BX%7D+%5Cmathbf%7BH%7D+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D+%3D+%5Cmathbf%7BI%7D)

怎么解？太简单了，用拉格朗日法嘛。不说了。最后变成了

![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28%5Cmathbf%7BX%7D+%5Csum_%7Bc%3D0%7D%5E%7BC%7D+%5Cmathbf%7BM%7D_c+%5Cmathbf%7BX%7D%5E%5Ctop+%2B+%5Clambda+%5Cmathbf%7BI%7D%5Cright%29+%5Cmathbf%7BA%7D+%3D%5Cmathbf%7BX%7D+%5Cmathbf%7BH%7D+%5Cmathbf%7BX%7D%5E%5Ctop+%5Cmathbf%7BA%7D+%5CPhi+)

其中的 ![[公式]](https://www.zhihu.com/equation?tex=%5CPhi) 是拉格朗日乘子。别看这个东西复杂，又有要求解的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BA%7D) ，又有一个新加入的 ![[公式]](https://www.zhihu.com/equation?tex=%5CPhi) 。但是它在matlab里是可以直接解的（用eigs函数即可）。这样我们就得到了变换![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BA%7D)，问题解决了。

可是伪标签终究是伪标签啊，肯定精度不高，怎么办？那好办。有个东西叫做迭代，一次不行，我们再做一次。后一次做的时候，我们用上一轮得到的标签来作伪标签。这样的目的是得到越来越好的伪标签，而参与迁移的数据是不会变的。这样往返多次，结果就自然而然好了。