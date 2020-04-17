# Decision-tree-and-Random-forest
决策树，随机森林，bagging与boosting等。

## 阅读指南

1. 在线观看请使用Chrome浏览器，并安装插件：[MathJax Plugin for Github(需科学上网)](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)， 插件[Github地址](https://github.com/orsharir/github-mathjax)
2. 或下载内容到本地，使用markdown相关软件打开，如：[Typora](https://typora.io/)
3. **若数学公式显示出现问题大家也可通过jupyter notebook链接查看：[Decision-tree-and-Random-forest](https://nbviewer.jupyter.org/github/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/jupyter%20notebook/Decision%20Tree.ipynb)**

## Content

- <a href = "#decision-tree">1. Decision Tree</a>
  - <a href = "#entropy">1.1 Entropy</a>
  - <a href = "#信息增益">1.2 信息增益</a>
  - <a href = "#信息增益率">1.3 信息增益率</a>
  - <a href = "#基尼系数">1.4 基尼系数</a>
  - <a href = "#ID3">1.5 ID3</a>
  - <a href = "#C4.5">1.6 C4.5</a>
  - <a href = "#CART">1.7 CART</a>
  - <a href = "#剪枝">1.8 剪枝</a>
  - <a href = "#实战">1.9 实战</a>
    - <a href = "#决策树分类">1.9.1 决策树分类</a>
    - <a href = "#决策树回归">1.9.1 决策树回归</a>
- <a href = "#random-forest">2. Random Forest</a>
- <a href = "#bagging">3. Bagging</a>
- <a href = "#Boosting">4. Boosting</a>
- <a href = "#参考文献">参考文献</a>



## [Decision Tree](#content)

在介绍决策树之前我们先看看决策树长什么样。我们这个决策树是判定一个人是否要去相亲。

<div align = "center"><image src="https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/images/decisionTree.png" width = "300" height = "295" alt="axis" align=center /></div>

学过软件工程的朋友应该接触过判定树，判定树也称为决策树，只不过呢在机器学习的决策树中如何向下分割节点要由更复杂的计算公式决定。

**我们先来看一下决策树的整体流程：**

- 我们观察决策树可以发现，决策树是一个从根节点到叶子结点逐渐递归的过程
- 在每个中间节点我们要找到一个**划分的属性**。这个划分的属性就是我们一会要重点解释的，也就是决策树是怎么划分出来的。

再解释如何寻找可以划分的属性之前，我们**再来分析一下决策树分到什么程度可以停止：**

- 当前节点包含的样本全属于同一类别，无需再进行划分
- 当前节点划分之后这个属性为空，或者是样本的所有属性取值都相同，无法划分
- 当前节点包含的样本集合为空，不能划分

*注：这里说的属性也就是每个样本的特征。*

确定了整体流程，也确定了最后的终止条件，我们就来看一下决策树中的属性划分是如何决定的。

按照属性划分的原理不同我们可以划分出**三种不同的决策树**：

- 按照信息增益分割节点：就是ID3决策树
- 按照信息增益率分割节点：就是C4.5决策树
- 按照基尼系数分割节点：就是CART决策树

在介绍这几种决策树之前我们先引入一个**信息熵**的概念：

### [Entropy](#content)

所谓的信息熵就是度量一个样本集合"纯度/不确定度"的指标，如何理解呢,我们来举个例子：

假设你在医生办公室的候诊室里和三个病人谈话。 他们三个人都刚刚完成了一项医学测试，经过一些处理，产生了两种可能的结果之一: 疾病要么存在，要么不存在。 他们已经提前研究了特定风险概率，现在急于找出结果，病人 a 知道，根据统计，他有95% 的可能性患有这种疾病。 对于病人 b，被诊断为患病的概率是30% 。 相比之下，患者 c 的概率是50 %。

现在我们想集中讨论一个简单的问题。 在其他条件相同的情况下，这三个病人中哪一个面临最大程度的不确定性？

答案很清楚: 病人 c 经历了"最多的不确定性"。 在这种情况下，他所经历的是最大程度的不确定性: 就像抛硬币一样。但是我们如何精确的来计算这种不确定度呢？就有了下面这个公式：


$$
H(D)=H\left(p_{1}, \ldots, p_{n}\right)=-\sum_{i=1}^{n} p_{i} \log _{2} p_{i}
$$


其中 $p_{i}$ 就是第 $i$ 个事件发生的概率，也可以看作在整个集合中第 $i$ 类样本所占的比例。规定若 $p_{i}=0$ 则 $p_{i}\log_{2}p_{i}=0$ 。计算出的信息熵最小值为0，最大值为 $\log _{2}\mathcal{n}$ ，当我们计算出的**结果越小**，代表当前这个**数据越纯**，也就是**不确定度越低**。

我们在用一个具体的例子来解释信息熵是如何计算的：

![watermelon](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/images/watermelon.png)

在这个例子中我们数据集共有17条数据，其中包含好瓜和坏瓜两个类别，其中好瓜/正例占$p_{1}=\frac{8}{17}$，坏瓜/负例占$p_{2}=\frac{9}{17}$。

那么现在这个集合的信息熵为：


$$
H(D)=-\sum_{i=1}^{2} p_{i} \log _{2} p_{i}= -\left(\frac{8}{17} \log _{2}^{\frac{8}{17}}+\frac{9}{17} \log _{2}^{\frac{9}{17}}\right)=0.998
$$


可以看到我们当前的这个数据集合计算出来的值很大，也就代表当前的数据还是很混乱的，因为正负两个样本基本上各占一半。

---

### [信息增益](#content)

我们刚说完信息熵，下面就让我们来看一下什么是信息增益。

我们假设某个离散属性的取值为：


$$
\left\{a^{1}, a^{2}, a^{3}, \ldots a^{V}\right\}
$$


$A^{v}$代表所有样本在属性$a$上取值为$a^{v}$的样本集合。

那么以属性A对数据集D进行划分，所得到的信息增益为：
$$
\operatorname{Gain}(D, A)=\operatorname{H}(D)-\underbrace{\sum_{v=1}^{V} \frac{\left|A^{v}\right|}{|D|} \operatorname{H}\left(A^{v}\right)}_{按照属性a划分之后的信息熵的和}
$$
其中代表按照属性a划分之前的信息熵，$\frac{\left|A^{v}\right|}{|D|}$代表属性a取值为v时在所有样本中所占的权重，样本越多代表当前属性的这个取值越重要。**一般而言，信息增益越大，则意味着使用属性a来进行划分所获得的“纯度提升”越大。**

我们还是用之前西瓜的那个例子来说明，假设我们以色泽属性进行分割，那么就对应着三个数据子集：

- $A^{1}$代表青绿色，对应的数据编号为$ \{1,4,6,10,13,17\} $，共有6个样本，其中正例3个，负例3个。
- $A^{2}$代表乌黑色，对应的数据编号为$ \{2,3,7,8,9,15\}$ ，共有6个样本，其中正例4个，负例2个。
- $A^{3}$代表浅白色，对应的数据编号为$ \{5,11,12,14,16\}$，共有5个样本，其中正例1个，负例4个。

*正例仍代表好瓜与负例仍代表坏瓜。*

则按照色泽进行划分之后的到的信息熵分别为：


$$
\begin{array}{l}
\operatorname{H}\left(A^{1}\right)=-\left(\frac{3}{6} \log _{2} \frac{3}{6}+\frac{3}{6} \log _{2} \frac{3}{6}\right)=1.000 \\
\operatorname{H}\left(A^{2}\right)=-\left(\frac{4}{6} \log _{2} \frac{4}{6}+\frac{2}{6} \log _{2} \frac{2}{6}\right)=0.918 \\
\operatorname{H}\left(A^{3}\right)=-\left(\frac{1}{5} \log _{5} \frac{1}{5}+\frac{4}{5} \log _{5} \frac{4}{5}\right)=0.722
\end{array}
$$


则根据属性色泽划分之后的信息增益为：


$$
\begin{aligned}
\operatorname{Gain}(D, 色泽) &=\operatorname{H}(D)-\sum_{v=1}^{3} \frac{\left|A^{v}\right|}{|D|} \operatorname{H}\left(A^{v}\right) \\
&=0.998-\left(\frac{6}{17} \times 1.000+\frac{6}{17} \times 0.918+\frac{5}{17} \times 0.722\right) \\
&=0.109
\end{aligned}
$$



---

### [信息增益率](#content)

说完信息增益，下面我们再来看一下什么是信息增益率。

信息增益率：
$$
\text { Gain_ratio }(D, A)=\frac{\operatorname{Gain}(D, A)}{\operatorname{IV}(A)}
$$
其中
$$
\operatorname{IV}(A)=-\sum_{v=1}^{V} \frac{\left|A^{v}\right|}{|D|} \log _{2} \frac{\left|A^{v}\right|}{|D|}
$$
属性A的可能取值越多即V越多，则IV(A)的值越大。

我们仍以西瓜数据集为例，计算一下信息增益率，首先我们来计算一下IV(A)：
$$
\operatorname{IV}(A)=-\left(\frac{6}{17} \log _{2} \frac{6}{17}+\frac{6}{17} \log _{2} \frac{6}{17}+ \frac{5}{17}\log _{2} \frac{5}{17}\right)=2.028
$$
接下来计算一下信息增益率：
$$
\begin{aligned}
\text { Gain_ratio }(D, A) &=\frac{\operatorname{Gain}(D, A)}{\operatorname{IV}(A)} \\
&=\frac{0.109}{2.028} \\
&=0.054
\end{aligned}
$$

---

### [基尼系数](#content)

与之前那两个基于熵的不同，基尼系数采用新的算法来计算数据的不确定度/纯度，基尼系数越小，数据的纯度越高，不确定度越低。我们来看一下基尼系数到底长什么样。

在一个分类问题中，假设有v个类别，第v个类别的概率为，则基尼系数为：


$$
\operatorname{Gini}(p)=\sum_{v=1}^{V} p_{v}\left(1-p_{v}\right)=1-\sum_{v=1}^{V} p_{v}^{2}
$$


对于给定样本D，假设其中有k个类别，切第k个类别的数量为，则样本D的基尼系数为：


$$
\operatorname{Gini}(D)=1-\sum_{v=1}^{V}\left(\frac{\left|A_{v}\right|}{|D|}\right)^{2}
$$


具体来说，我们假设某个离散属性的取值为：

$$
\left\{a^{1}, a^{2}, a^{3}, \ldots a^{V}\right\}
$$


$A^{v}$代表所有样本在属性$a$上取值为$a^{v}$的样本集合。

那么以属性A对数据集D进行划分，所得到的基尼系数为：


$$
\operatorname{Gini}(D, A)=\sum_{v=1}^{V} \frac{\left|A^{v}\right|}{|D|} \mathrm{Gini}\left(A^{v}\right)
$$

**基尼系数越小，则属性集D的纯度越高。**

我们还是用西瓜数据集为例，计算一下基尼系数，假设我们以色泽属性进行分割，那么就对应着三个数据子集：

- $A^{1}$代表青绿色，对应的数据编号为$ \{1,4,6,10,13,17\} $，共有6个样本，其中正例3个，负例3个。
- $A^{2}$代表乌黑色，对应的数据编号为$ \{2,3,7,8,9,15\}$ ，共有6个样本，其中正例4个，负例2个。
- $A^{3}$代表浅白色，对应的数据编号为$ \{5,11,12,14,16\}$，共有5个样本，其中正例1个，负例4个。

*正例仍代表好瓜与负例仍代表坏瓜。*

则按照色泽进行划分之后的到的基尼系数分别为：


$$
\begin{aligned}
&\mathrm{Gini}\left(A^{1}\right)=1-\left((\frac{3}{6})^2 + (\frac{3}{6})^2 \right)=0.5 \\
&\mathrm{Gini}\left(A^{2}\right)=1-\left((\frac{4}{6})^2 + (\frac{2}{6})^2 \right)=0.444\\
&\mathrm{Gini}\left(A^{3}\right)=1-\left((\frac{1}{5})^2 + (\frac{4}{5})^2 \right)=0.32
\end{aligned}
$$
则根据属性色泽划分之后的基尼系数为：


$$
\begin{aligned}
\text { Gini }(D, A) &= \sum_{v=1}^{V} \frac{\left|A^{v}\right|}{|D|} \mathrm{Gini}\left(A^{v}\right) \\
&=\frac{6}{17}*0.5 + \frac{6}{17}*0.444 + \frac{5}{17}*0.32 \\
&=0.176 + 0.157 + 0.094 \\
&=0.427
\end{aligned}
$$

---

接下来我们就看一下在决策树中是如何应用上面的公式的

### [ID3](#content)

我们根据刚才的信息增益公式来计算一下按照其他属性分割得到的信息增益
$$
\begin{array}{ll}
\operatorname{Gain}(D, 根蒂)=0.143 & \operatorname{Gain}(D, 敲声)=0.141 \\
\operatorname{Gain}(D, 纹理)=0.381 & \operatorname{Gain}(D, 脐部)=0.289 \\
\operatorname{Gain}(D, 触感)=0.006 & \operatorname{Gain}(D, 色泽)=0.109
\end{array}
$$
显然根据纹理进行分割是信息增益最大的，故首先按照纹理来划分样本

![ID3](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/C4.5/tree1.png)

之后按照这个规则继续分割，直到达到了停止条件则决策树完成创建。

![ID3](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/C4.5/tree2.png)

ID3算法不足：

- 没有考虑连续特征
- 假如某个属性具有好多个取值，甚至是把编号作为属性了，按照这种情况计算就会导致决策树创建之后效果并不好
- 没有考虑缺失值的情况

所以这才引入了C4.5。

---

### [C4.5](#content)

本部分来源见参考文献[20]

对于ID3存在的几个问题，C4.5是这样处理的

- 将连续特征离散化，加入说某连续特征是$\left\{1,2,3,4,5,6,7,8,9\right\}$共9个数，那么我们取相邻两个数的平均值$\left\{1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5\right\}$，就会产生8个分割点，分别计算以每个分割点为中心进行分割时的信息增益，例如以3.5为划分点的话，小于3.5为第一类，大于3.5为第二类。选择信息增益最大的分割点。
- 将信息增益率作为特征选择的标准，这样按照公式样本越多分母就会越大，计算的数值就越小
- 对缺失值的处理主要分为两个部分：
  - 一是对于某一个有缺失特征值的特征A。C4.5的思路是将数据分成两部分，对每个样本设置一个权重（初始可以都为1），然后划分数据，一部分是有特征值A的数据D1，另一部分是没有特征A的数据D2. 然后对于没有缺失特征A的数据集D1来和对应的A特征的各个特征值一起计算加权重后的信息增益比，最后乘上一个系数，这个系数是无特征A缺失的样本加权后所占加权总样本的比例。
  - 二是选定了划分属性，对于在该属性上缺失特征的样本的处理。可以将缺失特征的样本同时划分入所有的子节点，不过将该样本的权重按各个子节点样本的数量比例来分配。比如缺失特征A的样本a之前权重为1，特征A有3个特征值A1,A2,A3。 3个特征值对应的无缺失A特征的样本个数为2,3,4.则a同时划分入A1，A2，A3。对应权重调节为2/9,3/9, 4/9

虽然C4.5解决了ID3中的大部分问题，但还有很多改进的空间。

- 决策树算法非常容易过拟合，所以要对决策树进行剪枝。主要有两种剪枝的方式：1. 预剪枝 2. 后剪枝
- C4.5生成的是多叉树，即一个父节点可以有多个节点。很多时候，在计算机中二叉树模型会比多叉树运算效率高。如果采用二叉树，可以提高效率。
- C4.5只能用于分类，如果能将决策树用于回归的话可以扩大它的使用范围。
- C4.5由于使用了熵模型，里面有大量的耗时的对数运算,如果是连续值还有大量的排序运算。如果能够加以模型简化可以减少运算强度但又不牺牲太多准确性的话，那就更好了。

---

### [CART](#content)

我们上面介绍完了基尼系数的计算，但是CART决策树在创建过程中有一些细节值得注意：

- 与上面的C4.5一样，CART决策树叶具有处理连续值的能力，只不过在处理连续值的过程中，它获取完分割点后采用的是基尼系数进行运算
- CART决策树是一个二叉树，不同于我们在基尼系数中的计算将色泽分为三类，分别算完基尼系数求和，CART决策树在计算过程中是这样的。他会将A分为三组：{A1}和{A2,A3}，{A2}和{A1,A3}，{A3}和{A1,A2}，然后分别计算这三组怎样分基尼系数最小，选取最小的那一个组合来建立决策树。

我们可以先看一下最开始的情况他是按照纹理来进行切分

![CART](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/CART/tree2.png)

之后再通过同样的算法建立起整个决策树

![CART](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/CART/tree1.png)

---

**回归树**

上面我们介绍完分类的决策树，接下来再看一下CART回归树。

*分类树输出的是离散值，用叶子结点里概率最大的类别作为当前结点的预测类别；回归树输出的是连续值，是最终叶子结点的均值或者中位数作为输出*

**回归树与分类树最大的不同：**

对于连续值的处理，我们知道CART分类树采用的是用基尼系数的大小来度量特征的各个划分点的优劣情况。这比较适合分类模型，但是对于回归模型，我们使用了常见的和方差的度量方式，CART回归树的度量目标是，对于任意划分特征A，对应的任意划分点s两边划分成的数据集D1和D2，求出使D1和D2各自集合的均方差最小，同时D1和D2的均方差之和最小所对应的特征和特征值划分点。

我们来看一下回归树是是长什么样的：

![CART](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/CART/regressionTree.png)

---

### [剪枝](#content)

其实决策树算法非常容易过拟合，这时我们就需要剪枝来应对过拟合，类比到线性回归就相当于添加正则项，以此来提高模型的泛化能力。就像我们之前说的剪枝分为两种：

**预剪枝**

预剪枝有点类似于Alpha-beta剪枝算法，他是在我们生成决策树的过程中，如果某个节点的熵值小于某个我们设定的阈值或者样本数量少于多少时我们就不再继续分割节点，又或者是当树的深度达到一定值时我们就不再继续向下生成决策树。通过这些手段来防止决策树过拟合。

在决策树生成过程中，对每个节点在划分前先进行估计，若当前节点的划分不能带来决策树泛化能力的提升，则停止划分并将当前节点标记为叶节点。

**后剪枝**

后剪枝就是我们已经有一个完整的决策树了，然后自底向上地对非叶节点进行考察，若将该节点对应的子树替换为叶节点能带来决策树泛化能力提升，则将该子树替换为叶节点。

---

### [实战](#content)

### [决策树分类](#content)

我们在Iris鸢尾花卉数据集上实现分类效果，对于数据集的特征情况和可视化结果请看：[Dive-into-matplotlib](https://github.com/Knowledge-Precipitation-Tribe/Dive-into-matplotlib)。我们这里使用max_depth=3的决策树，准确率可达：97.78%。最终生成的决策树如下：

![iris_tree](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/irisClassification/iris_tree.png)

我们也可以看一下决策树随深度的变化，预测的效果如何

![depthAndAcc](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/irisClassification/depthAndAcc.png)

### [决策树回归](#content)

决策树回归这里为了方便可视化，没有涉及复杂的数据，用的是随机生成的数据，我们根据树不同的深度可视化了回归树的预测结果

![depthAndAcc](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/code/decisionTreeRegressor/DecisionTreeRegressor.png)



**更多实战内容请看：[[The-road-of-Competition](https://github.com/Knowledge-Precipitation-Tribe/The-road-of-Competition)](https://github.com/Knowledge-Precipitation-Tribe/The-road-of-Competition)**

---

### [Random Forest](#content)

随机森林

----

接下来我们介绍bagging与boosting，其实bagging和boosting可以形象的比喻成高中生活，高中做的模拟卷呢就是训练数据，考试的那套卷子呢就是测试数据，我们就用高中生活来解释bagging与boosting。

## [Bagging](#content)

假如说有老三，大个，胖子三个人都坐在教室的最后一排，这三个人学习都在中游，这一天呢老师发了一套数学的模拟卷，他们三个人也就按照这个模拟卷自己慢慢学习，看看答案最后也都学完了。这个周末学校就举行了考试，恰巧老三，大个，胖子三个人坐在一起，面对着这个考试卷子，他们三个人呢因为做模拟题的时候关注的题型不一样，这张考试卷子上有的会有的不会，这三个人就偷偷交流起来了，恰巧坐在他们前面的班花翠兰也有不会的，他们三个就把自己学过的确认的答案都告诉了翠花，按照他们三个人提示的内容，翠花顺利的完成了自己的试卷，最终在这次考试中取得了很好的成绩。

## [Boosting](#content)

还是老三，大个，胖子他们三个，今天呢老师又发了一套数学卷，上次考试大个和胖子考的都不怎么好，就没有心情做了，老三考的还行，他就先开始做这套模拟卷，好不容易做完了一看旁边的大个就跟他说：“你别干挺着啊，快做卷子，我做完了，我先把我做错的教你”，之后呢大个就在老三做完的基础上继续做这套卷子，大个做完又去教胖子，因为已经有前面老三和大个的基础了，所以这套模拟卷胖子做的相当不错。

## 参考文献

[1] Yeh James: [決策樹(Decision Tree)以及隨機森林(Random Forest)介紹](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)

[2] Madhu Sanjeevi ( Mady ): [Decision Trees Algorithms](https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1)

[3] Savan Patel: [Decision Tree Classifier — Theory](https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567)

[4] Diogo Menezes Borges: [Boosting with AdaBoost and Gradient Boosting](https://medium.com/diogo-menezes-borges/boosting-with-adaboost-and-gradient-boosting-9cbab2a1af81)

[5] Joseph Rocca: [Ensemble methods: bagging, boosting and stacking](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)

[6] Will Koehrsen: [Random Forest in Python](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)

[7] Tony Yiu: [Understanding Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)

[8] Will Koehrsen: [Random Forest Simple Explanation](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d)

[9] [终于有人把XGBoost 和 LightGBM 讲明白了，项目中最主流的集成算法！](https://mp.weixin.qq.com/s/LoX987dypDg8jbeTJMpEPQ)

[10] [深入理解LightGBM](https://mp.weixin.qq.com/s/l6Fp5WTNH0b_cl2y7Az76Q)

[11] [一文详尽系列之CatBoost](https://mp.weixin.qq.com/s/E3pSPsG18053F5GG1Z8jNQ)

[12] [最常用的决策树算法！Random Forest、Adaboost、GBDT 算法](https://mp.weixin.qq.com/s/Nl_-PdF0nHBq8yGp6AdI-Q)

[13] Yeh James: [Kaggle機器學習競賽神器XGBoost介紹](https://medium.com/jameslearningnote/資料分析-機器學習-第5-2講-kaggle機器學習競賽神器xgboost介紹-1c8f55cffcc)

[14] Alvira Swalin: [CatBoost vs. Light GBM vs. XGBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)

[15] Vishal Morde: [XGBoost Algorithm: Long May She Reign!](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)

[16] Pushkar Mandot: [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)

[17] 陈旸: [极客时间：决策树](https://time.geekbang.org/column/article/78273)

[18] [Entropy is a measure of uncertainty](https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c)

[19] 周志华: [机器学习](https://item.jd.com/10244685726.html)

[20] 刘建平: [决策树算法原理(上)](https://www.cnblogs.com/pinard/p/6050306.html)

[21] 刘建平: [决策树算法原理(下)](https://www.cnblogs.com/pinard/p/6053344.html)



