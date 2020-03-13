# Decision-tree-and-Random-forest
决策树，随机森林，bagging与boosting等。

## Content

- <a href = "#decision-tree">1. Decision Tree</a>
  - <a href = "#entropy">1.1 Entropy</a>
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

按照属性划分的原理不同我们可以划分出三种不同的决策树：

- 按照信息增益分割节点：就是ID3决策树
- 按照信息增益率分割节点：就是C4.5决策树
- 按照基尼系数分割节点：就是CART决策树

在介绍这几种决策树之前我们先引入一个**信息熵**的概念：

### [entropy](#content)

所谓的信息熵就是度量一个样本集合"纯度/不确定度"的指标，如何理解呢,我们来举个例子：

假设你在医生办公室的候诊室里和三个病人谈话。 他们三个人都刚刚完成了一项医学测试，经过一些处理，产生了两种可能的结果之一: 疾病要么存在，要么不存在。 他们已经提前研究了特定风险概率，现在急于找出结果，病人 a 知道，根据统计，他有95% 的可能性患有这种疾病。 对于病人 b，被诊断为患病的概率是30% 。 相比之下，患者 c 的概率是50 %。

现在我们想集中讨论一个简单的问题。 在其他条件相同的情况下，这三个病人中哪一个面临最大程度的不确定性？

答案很清楚: 病人 c 经历了"最多的不确定性"。 在这种情况下，他所经历的是最大程度的不确定性: 就像抛硬币一样。但是我们如何精确的来计算这种不确定度呢？就有了下面这个公式：
$$
H(X)=H\left(p_{1}, \ldots, p_{n}\right)=-\sum_{i=1}^{n} p_{i} \log _{2} p_{i}
$$
其中$p_{i}$就是第$i$个事件发生的概率，也可以看作在整个集合中第$i$类样本所占的比例。规定若$p_{i}=0$则$p_{i} \log _{2} p_{i}=0$。当我们计算出的**结果越小**，代表当前这个**数据越纯**，也就是**不确定度越低**。

我们在用一个具体的例子来解释信息熵是如何计算的：

![watermelon](https://github.com/Knowledge-Precipitation-Tribe/Decision-tree-and-Random-forest/blob/master/images/watermelon.png)

在这个例子中我们数据集共有17条数据，其中包含好瓜和坏瓜两个类别，其中好瓜/正例占$p_{1}=\frac{8}{17}$，坏瓜/反例占$p_{2}=\frac{9}{17}$。

那么现在这个集合的信息熵为：
$$
H(X)=-\sum_{i=1}^{2} p_{i} \log _{2} p_{i}= -\left(\frac{8}{17} \log _{2}^{\frac{8}{17}}+\frac{9}{17} \log _{2}^{\frac{9}{17}}\right)=0.998
$$
损失



----

接下来我们介绍bagging与boosting，其实bagging和boosting可以形象的比喻成高中生活，高中做的模拟卷呢就是训练数据，考试的那套卷子呢就是测试数据，我们就用高中生活来解释bagging与boosting。

## Bagging

假如说有老三，大个，胖子三个人都坐在教室的最后一排，这三个人学习都在中游，这一天呢老师发了一套数学的模拟卷，他们三个人也就按照这个模拟卷自己慢慢学习，看看答案最后也都学完了。这个周末学校就举行了考试，恰巧老三，大个，胖子三个人坐在一起，面对着这个考试卷子，他们三个人呢因为做模拟题的时候关注的题型不一样，这张考试卷子上有的会有的不会，这三个人就偷偷交流起来了，恰巧坐在他们前面的班花翠兰也有不会的，他们三个就把自己学过的确认的答案都告诉了翠花，按照他们三个人提示的内容，翠花顺利的完成了自己的试卷，最终在这次考试中取得了很好的成绩。

## Boosting

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



