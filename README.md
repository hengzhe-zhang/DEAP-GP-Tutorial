# 基于DEAP的遗传编程系列教程

本系列教程主要介绍如何基于DEAP实现一些流行的遗传编程概念，包括：

* 单树GP
* 多树GP
* 多目标GP
* 集成学习
* 启发式算法生成

上述概念通过以下示例实现：

1. [基于单树GP的符号回归（Symbolic Regression）](application/symbolic-regression.ipynb)
2. [基于多树GP的特征工程（Feature Construction）](application/feature-construction.ipynb)
3. [基于多目标GP的符号回归 （Multi-Objective Symbolic Regression）](application/multiobjective-sr.ipynb)
4. [基于GP的集成学习（Ensemble Learning）](application/ensemble-learning.ipynb)
5. [基于GP的旅行商问题规则生成（TSP）](application/TSP.ipynb)
6. [为什么使用GP而不是神经网络？（Feature Construction）](application/cross-validation-score.ipynb)
6. [基于GP自动设计优化算法](application/automatically-design-de-operators.ipynb)
7. [基于不同算子集的多树GP](application/multisets_gp.ipynb)

同时，本教程包含了一些工程技巧：

1. [基于Numpy实现向量化加速](tricks/numpy-speedup.ipynb)
2. [基于PyTorch实现GPU加速](tricks/pytorch-speedup.ipynb)
3. [基于手动编写编译器实现加速](tricks/compiler-speedup.ipynb)
4. [基于Numba实现Lexicase Selection加速](tricks/numba-lexicase-selection.ipynb)
5. [基于多进程实现异步并行评估](tricks/multiprocess_speedup.md)
6. [基于sklearn接口的Numpy加速符号回归](tricks/numpy_speedup_sr.py)

此外，DEAP还有一些注意事项：

1. [VarAnd和VarOr算子的使用注意事项](operator/varor-varand.ipynb)
2. [Crossover算子的注意事项](operator/crossover.ipynb)
2. [Lexicase Selection算子的注意事项](operator/lexicase-selection.ipynb)
