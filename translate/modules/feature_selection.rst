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

线性模型 :ref:`Linear models <linear_model>` 采用L1范式惩罚, 存在稀疏解: 大部分的估计器的置信度为0.
当目标是通过其他分类器对数据进行降维, 可以使用
 When the
goal
is to reduce the dimensionality of the data to use with another classifier,
they can be used along with :class:`feature_selection.SelectFromModel`
to select the non-zero coefficients. In particular, sparse estimators useful for
this purpose are the :class:`linear_model.Lasso` for regression, and
of :class:`linear_model.LogisticRegression` and :class:`svm.LinearSVC`
for classification::

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
  >>> model = SelectFromModel(lsvc, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape
  (150, 3)

With SVMs and logistic-regression, the parameter C controls the sparsity:
the smaller C the fewer features selected. With Lasso, the higher the
alpha parameter, the fewer features selected.

.. topic:: Examples:

    * :ref:`example_text_document_classification_20newsgroups.py`: Comparison
      of different algorithms for document classification including L1-based
      feature selection.

.. _compressive_sensing:

.. topic:: **L1-recovery and compressive sensing**

   For a good choice of alpha, the :ref:`lasso` can fully recover the
   exact set of non-zero variables using only few observations, provided
   certain specific conditions are met. In particular, the number of
   samples should be "sufficiently large", or L1 models will perform at
   random, where "sufficiently large" depends on the number of non-zero
   coefficients, the logarithm of the number of features, the amount of
   noise, the smallest absolute value of non-zero coefficients, and the
   structure of the design matrix X. In addition, the design matrix must
   display certain specific properties, such as not being too correlated.

   There is no general rule to select an alpha parameter for recovery of
   non-zero coefficients. It can by set by cross-validation
   (:class:`LassoCV` or :class:`LassoLarsCV`), though this may lead to
   under-penalized models: including a small number of non-relevant
   variables is not detrimental to prediction score. BIC
   (:class:`LassoLarsIC`) tends, on the opposite, to set high values of
   alpha.

   **Reference** Richard G. Baraniuk "Compressive Sensing", IEEE Signal
   Processing Magazine [120] July 2007
   http://dsp.rice.edu/files/cs/baraniukCSlecture07.pdf

.. _randomized_l1:

Randomized sparse models
-------------------------

.. currentmodule:: sklearn.linear_model

The limitation of L1-based sparse models is that faced with a group of
very correlated features, they will select only one. To mitigate this
problem, it is possible to use randomization techniques, reestimating the
sparse model many times perturbing the design matrix or sub-sampling data
and counting how many times a given regressor is selected.

:class:`RandomizedLasso` implements this strategy for regression
settings, using the Lasso, while :class:`RandomizedLogisticRegression` uses the
logistic regression and is suitable for classification tasks.  To get a full
path of stability scores you can use :func:`lasso_stability_path`.

.. figure:: ../auto_examples/linear_model/images/plot_sparse_recovery_003.png
   :target: ../auto_examples/linear_model/plot_sparse_recovery.html
   :align: center
   :scale: 60

Note that for randomized sparse models to be more powerful than standard
F statistics at detecting non-zero features, the ground truth model
should be sparse, in other words, there should be only a small fraction
of features non zero.

.. topic:: Examples:

   * :ref:`example_linear_model_plot_sparse_recovery.py`: An example
     comparing different feature selection approaches and discussing in
     which situation each approach is to be favored.

.. topic:: References:

   * N. Meinshausen, P. Buhlmann, "Stability selection",
     Journal of the Royal Statistical Society, 72 (2010)
     http://arxiv.org/pdf/0809.2932

   * F. Bach, "Model-Consistent Sparse Estimation through the Bootstrap"
     http://hal.inria.fr/hal-00354771/

Tree-based feature selection
----------------------------

Tree-based estimators (see the :mod:`sklearn.tree` module and forest
of trees in the :mod:`sklearn.ensemble` module) can be used to compute
feature importances, which in turn can be used to discard irrelevant
features (when coupled with the :class:`sklearn.feature_selection.SelectFromModel`
meta-transformer)::

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

    * :ref:`example_ensemble_plot_forest_importances.py`: example on
      synthetic data showing the recovery of the actually meaningful
      features.

    * :ref:`example_ensemble_plot_forest_importances_faces.py`: example
      on face recognition data.

Feature selection as part of a pipeline
=======================================

Feature selection is usually used as a pre-processing step before doing
the actual learning. The recommended way to do this in scikit-learn is
to use a :class:`sklearn.pipeline.Pipeline`::

  clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', RandomForestClassifier())
  ])
  clf.fit(X, y)

In this snippet we make use of a :class:`sklearn.svm.LinearSVC`
coupled with :class:`sklearn.feature_selection.SelectFromModel`
to evaluate feature importances and select the most relevant features.
Then, a :class:`sklearn.ensemble.RandomForestClassifier` is trained on the
transformed output, i.e. using only relevant features. You can perform
similar operations with the other feature selection methods and also
classifiers that provide a way to evaluate feature importances of course.
See the :class:`sklearn.pipeline.Pipeline` examples for more details.
