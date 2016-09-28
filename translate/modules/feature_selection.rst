.. currentmodule:: sklearn.feature_selection

.. _feature_selection:

=================
特征选择
=================


模块 :mod:`sklearn.feature_selection` 中的类可以对样例集进行特征选择或降维, 用于提高分类器等的准确性, 也能提升在高维数据集上的模型表现.


.. _variance_threshold:

删除低方差的特征
===================================

方差阈值:class:`VarianceThreshold` 是一种简单的特征选择的基线方法.
该方法移除了方差未到达阈值的特征.默认情况下, 移除了所有方差为0的特征.比如, 所有样本都相同的特征值.
试想, 如果有一个布尔值特征的数据集, 我们想要移除样本中, 超过80%的为0/1的特征值(开或关)
布尔值特征服从伯努利(Bernoulli）分布, 方差由下式给出:

.. math:: \mathrm{Var}[X] = p(1 - p)

so we can select using the threshold
 所以, 我们可以选择特征阈值为``.8 * (1 - .8)``::

  >>> from sklearn.feature_selection import VarianceThreshold
  >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
  >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  >>> sel.fit_transform(X)
  array([[0, 1],
         [1, 0],
         [0, 0],
         [1, 1],
         [1, 0],
         [1, 1]])

和预想的一样, ``方差阈值法`` 移除了第一列特征, 该列包含0的概率大于0.8.

.. _univariate_feature_selection:

单一变量特征选择
============================
单一变量特征选择, 原理是基于单一变量统计检验来选择最好的特征. 可以被视为估计器的一个预处理步骤.Scikit-learn将特征选择过程表现为实现了``transform``方法的对象:


 * :class:`SelectKBest` 保留了最高分的 :math:`k` 个特征

 * :class:`SelectPercentile` 保留了用户指定占比的最高分特征

 * 对每个特征使用普通的单一变量统计检验:
   假阳性率(false positive rate) :class:`SelectFpr`, 伪发现率(false discovery rate)
   :class:`SelectFdr`, 或族系误差率(family wise error) :class:`SelectFwe`.

 * :class:`GenericUnivariateSelect` 允许使用可配置的策略进行单一变量特征选择. 允许使用超参数搜索估计器的方法, 选择最好的单一变量选取策略.

例如, 我们可以对样本ོ进行卡方检验, 检索到最好的两个特征, 示例代码如下:

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.feature_selection import chi2
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
  >>> X_new.shape
  (150, 2)

这些对象输入一个计分函数, 返回单变量的P值.
[译注:p 值可用来确定结果在统计意义上是否显著。p 值通常用在假设检验中，在检验中，您可以否定或无法否定一个原假设。
执行假设检验时，要注意的关键输出信息就是 p 值。
p 值的范围为 0 到 1。p 值是一个概率，用来度量否定原假设的证据。概率越低，否定原假设的证据越充分。]


 * 回归: :func:`f_regression`

 * 分类: :func:`chi2` or :func:`f_classif`

.. topic:: 稀疏数据集的特征选择

   如果你使用稀疏数据集 (例如, 数据集用稀疏矩阵表示), 只有卡方:func:`chi2` 处理数据不会使得数据变得致密.

.. warning::
   注意在分类问题中, 不要使用回归的计分函数, 因为会得到无用的结果.

.. topic:: Examples:

    :ref:`example_feature_selection_plot_feature_selection.py`

.. _rfe:

Recursive feature elimination
递归特征消除
=============================
给定一个外部的估计器来分配特征权重 (例如, 线性模型的系数.) , 递归特征消除(:class:`RFE`)通过递归考虑越来越小的特征集来选择特征.
首先, 用初始特征集来训练估计器,并对每个特征分配权重. 然后, 特征集中绝对权重最小的特征被剪枝. 该过程会递归地重复对特征集进行剪枝, 直至达到最终期望数量的特征.

:class:`RFECV` 利用交叉验证循环来进行递归特征消除(RFE,  Recursive feature elimination), 以此发现最优的特征数量.

.. topic:: 示例:

    * :ref:`example_feature_selection_plot_rfe_digits.py`: 显示了像素相关性在数字分类任务中的递归特征消除的例子.

    * :ref:`example_feature_selection_plot_rfe_with_cross_validation.py`:  利用交叉验证对特征数量进行自动剪枝的例子.


.. _select_from_model:

使用 SelectFromModel进行特征选择.
=======================================

:class:`SelectFromModel`是一种元数据转换器, 能使用任何在拟合之后包含``coef_``或``feature_importances_``属性的估计器.
如果特征相应的``coef``或``feature_importances_``低于给定的阈值``threshold``, 该特征会被认为不重要且被移除. 除了数值上指定的阈值参数, 还提供了字符串参数用于内置的启发式搜索.
可用的启发式参数包括均值(mean), 中位数(median), 以及浮点数和这些值的乘积(0.1*mean).
如何使用请参考后文的版块.

.. topic:: 示例

    * :ref:`example_feature_selection_plot_select_from_model_boston.py`: 在预先不知道阈值的情况下, 从波士顿数据集中选择最重要的聊个特征.

.. _l1_feature_selection:

基于L1的特征选择
--------------------------

.. currentmodule:: sklearn

线性模型 :ref:`Linear models <linear_model>` 采用L1范数惩罚, 存在稀疏解: 大部分的估计器的系数为0.
当目标是通过其他分类器对数据进行降维, 可以使用:class:`feature_selection.SelectFromModel`选择系数非零的特征. 稀疏的估计器在这个目标中特别有用,
回归中是:class:`linear_model.Lasso`, 分类中是:class:`linear_model.LogisticRegression` 和 :class:`svm.LinearSVC`

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
  >>> model = SelectFromModel(lsvc, prefit=True
  >>> X_new = model.transform(X)
  >>> X_new.shape
  (150, 3)

对于SVM和逻辑回归, 参数C控制了稀疏性, C越小则会选择越小的特征. 对于Lasso, 越高的alpha参数会选择越少的特征.

.. topic:: 示例:

    * :ref:`example_text_document_classification_20newsgroups.py`: 使用基于L1的特征选择方法, 比较了不同算法在文本分类中的表现.

.. _compressive_sensing:

.. topic:: **L1恢复和压缩感知**

   因为选择一个好的alpha参数, :ref:`lasso`能够只通过极少量的观测值完全恢复抽取的非零变量集,只需要提供确定的特殊条件. 特别地, 样本数量需要"足够大", 否则L1模型会表现得随机.
   "足够大"取决于非零系数的数值, 特征数量的对数, 噪音的数量, 非零系数的最小绝对值, 特征矩阵X的结构. 另外,  特征矩阵X必须显示出特定的属性, 比如不会过于相关.

   没有一个通用的规则用于选择alpha参数来恢复非零系数. 可以使用交叉验证的方法(:class:`LassoCV` 或 :class:`LassoLarsCV`), 虽然会导致惩罚不足的模型:包括少量的不相关变量,
   这些变量不会对预测分数造成损害. 与此相反, BIC(:class:`LassoLarsIC`)如果设置过高的alpha值会损害预测值.

   **Reference** Richard G. Baraniuk "Compressive Sensing", IEEE Signal
   Processing Magazine [120] July 2007
   http://dsp.rice.edu/files/cs/baraniukCSlecture07.pdf

.. _randomized_l1:

Randomized sparse models
随机稀疏模型
-------------------------

.. currentmodule:: sklearn.linear_model
基于L1的稀疏模型的局限性在于, 对于一组相关性很高的特征, 只选择其中一个. 为缓解这个问题,可以使用随机化技术, 对稀疏模型进行多次重新估计, 打乱特征矩阵;或者对数据进行子采样, 并统计回归器被使用了多少次.

:class:`RandomizedLasso` 为回归实现了这个策略, 使用Lasso; :class:`RandomizedLogisticRegression`使用逻辑回归, 适用于分类问题. 如果想要获得稳定分数的全路径,
可以使用函数:func:`lasso_stability_path`.

.. figure:: ../auto_examples/linear_model/images/plot_sparse_recovery_003.png
   :target: ../auto_examples/linear_model/plot_sparse_recovery.html
   :align: center
   :scale: 60

注意随机化的稀疏模型在检测非零特征上,比标准F统计更加强大. 基础真实模型应该是稀疏的, 也就是说, 应该只有很小的一部分特征是非零的.

.. topic:: Examples:

   * :ref:`example_linear_model_plot_sparse_recovery.py`: 一个比较不同特征选择方法, 讨论了在不同的情况下改选择哪种方法.

.. topic:: References:

   * N. Meinshausen, P. Buhlmann, "Stability selection",
     Journal of the Royal Statistical Society, 72 (2010)
     http://arxiv.org/pdf/0809.2932

   * F. Bach, "Model-Consistent Sparse Estimation through the Bootstrap"
     http://hal.inria.fr/hal-00354771/

基于树的特征选择
----------------------------

基于树的估计器(树:mod:`sklearn.tree`和森林 :mod:`sklearn.ensemble`)能用于计算特征重要性, 从而可以丢弃不相关的特征(当和 :class:`sklearn.feature_selection.SelectFromModel`  一起使用时):

  >>> from sklearn.ensemble import ExtraTreesClassifier
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> clf = ExtraTreesClassifier()
  >>> clf = clf.fit(X, y)
  >>> clf.feature_importances_  # doctest: +SKIP
  array([ 0.04...,  0.05...,  0.4...,  0.4...])
  >>> model = SelectFromModel(clf, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape               # doctest: +SKIP
  (150, 2)

.. topic:: Examples:

    * :ref:`example_ensemble_plot_forest_importances.py`: 如何在合成数据上恢复有实际意义的特征.

    * :ref:`example_ensemble_plot_forest_importances_faces.py`: 人脸识别的示例.

作为流水线(pipeline)一部分的特征选择
=======================================

在机器学习中, 特征学习通常作为一个预处理的步骤. 建议使用 :class:`sklearn.pipeline.Pipeline`: 进行特征选择

  clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', RandomForestClassifier())
  ])
  clf.fit(X, y)

在这段代码中, 我们使用线性SVC :class:`sklearn.svm.LinearSVC`和 :class:`sklearn.feature_selection.SelectFromModel` 来评估特征重要性,
并选择最相关的特征. 然后, 训练随机森林:class:`sklearn.ensemble.RandomForestClassifier`输出分类结果, 仅使用相关特征. 也可以使用其他特征选择方法来进行这个过程,
分类器也提供了评估特征重要性的方法. 可以在这里了解更多细节:  :class:`sklearn.pipeline.Pipeline`