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

For instance, we can perform a :math:`\chi^2` test to the samples
to retrieve only the two best features as follows:

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

These objects take as input a scoring function that returns
univariate p-values:

 * For regression: :func:`f_regression`

 * For classification: :func:`chi2` or :func:`f_classif`

.. topic:: Feature selection with sparse data

   If you use sparse data (i.e. data represented as sparse matrices),
   only :func:`chi2` will deal with the data without making it dense.

.. warning::

    Beware not to use a regression scoring function with a classification
    problem, you will get useless results.

.. topic:: Examples:

    :ref:`example_feature_selection_plot_feature_selection.py`

.. _rfe:

Recursive feature elimination
=============================

Given an external estimator that assigns weights to features (e.g., the
coefficients of a linear model), recursive feature elimination (:class:`RFE`)
is to select features by recursively considering smaller and smaller sets of
features.  First, the estimator is trained on the initial set of features and
weights are assigned to each one of them. Then, features whose absolute weights
are the smallest are pruned from the current set features. That procedure is
recursively repeated on the pruned set until the desired number of features to
select is eventually reached.

:class:`RFECV` performs RFE in a cross-validation loop to find the optimal
number of features.

.. topic:: Examples:

    * :ref:`example_feature_selection_plot_rfe_digits.py`: A recursive feature elimination example
      showing the relevance of pixels in a digit classification task.

    * :ref:`example_feature_selection_plot_rfe_with_cross_validation.py`: A recursive feature
      elimination example with automatic tuning of the number of features
      selected with cross-validation.

.. _select_from_model:

Feature selection using SelectFromModel
=======================================

:class:`SelectFromModel` is a meta-transformer that can be used along with any
estimator that has a ``coef_`` or ``feature_importances_`` attribute after fitting.
The features are considered unimportant and removed, if the corresponding
``coef_`` or ``feature_importances_`` values are below the provided
``threshold`` parameter. Apart from specifying the threshold numerically,
there are build-in heuristics for finding a threshold using a string argument.
Available heuristics are "mean", "median" and float multiples of these like
"0.1*mean".

For examples on how it is to be used refer to the sections below.

.. topic:: Examples

    * :ref:`example_feature_selection_plot_select_from_model_boston.py`: Selecting the two
      most important features from the Boston dataset without knowing the
      threshold beforehand.

.. _l1_feature_selection:

L1-based feature selection
--------------------------

.. currentmodule:: sklearn

:ref:`Linear models <linear_model>` penalized with the L1 norm have
sparse solutions: many of their estimated coefficients are zero. When the goal
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