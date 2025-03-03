import math
import operator
import random

import numpy as np
from deap import base, creator, tools, gp, algorithms
from deap.tools import sortNondominated, selNSGA2


# 定义评估函数，包含两个目标：均方误差和树的大小
def evalSymbReg(individual, pset):
    # 编译GP树为函数
    func = gp.compile(expr=individual, pset=pset)
    # 计算均方误差（Mean Square Error，MSE）
    mse = ((func(x) - (x**2 + x)) ** 2 for x in range(-10, 10))
    # 计算GP树的大小
    size = len(individual)
    return math.fsum(mse), size


# 定义Alpha支配
class AlphaDominance:
    def __init__(self, algorithm=None, initial_alpha=0.1, step_size=0.1):
        self.historical_largest = 0
        self.historical_smallest = math.inf
        self.algorithm = algorithm
        self.step_size = step_size
        self.initial_alpha = initial_alpha

    def update_best(self, population):
        self.historical_smallest = min(
            self.historical_smallest, min([len(p) for p in population])
        )
        self.historical_largest = max(
            self.historical_largest, max([len(p) for p in population])
        )

    def selection(self, population, offspring, alpha):
        # 调整适应度以考虑大小
        self.set_fitness_with_size(population, offspring, alpha)

        # 应用NSGA-II选择
        first_pareto_front = sortNondominated(offspring + population, len(population))[
            0
        ]
        selected_pop = selNSGA2(offspring + population, len(population))

        if hasattr(self.algorithm, "hof") and self.algorithm.hof is not None:
            self.algorithm.hof.update(selected_pop)

        # 恢复原始适应度值
        self.restore_original_fitness(selected_pop)

        # 根据大小调整alpha
        theta = np.rad2deg(np.arctan(alpha))
        avg_size = np.mean([len(p) for p in first_pareto_front])

        # 更新历史最大和最小值
        self.update_best(first_pareto_front)

        # 计算新的alpha值
        new_alpha = self.adjust_alpha(theta, avg_size)

        return selected_pop, new_alpha

    def adjust_alpha(self, theta, avg_size):
        historical_largest = self.historical_largest
        historical_smallest = self.historical_smallest

        # 防止除以零
        if historical_largest == historical_smallest:
            return np.tan(np.deg2rad(theta))

        theta = theta + (
            historical_largest + historical_smallest - 2 * avg_size
        ) * self.step_size / (historical_largest - historical_smallest)
        theta = np.clip(theta, 0, 90)
        return np.tan(np.deg2rad(theta))

    def restore_original_fitness(self, population):
        for ind in population:
            ind.fitness.weights = (-1, -1)
            ind.fitness.values = getattr(ind, "original_fitness")

    def set_fitness_with_size(self, population, offspring, alpha):
        max_size = max([len(x) for x in offspring + population])
        for ind in offspring + population:
            assert alpha >= 0, f"Alpha Value {alpha}"
            setattr(ind, "original_fitness", ind.fitness.values)
            ind.fitness.weights = (-1, -1)
            # 修改第二个目标为模型大小和准确率的加权
            ind.fitness.values = (
                ind.fitness.values[0],
                len(ind) / max_size + alpha * ind.fitness.values[0],
            )


# 修改适应度函数，包含两个权重：MSE和树的大小。MSE是最小化，树的大小也是最小化
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

# 定义函数集合和终端集合
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0="x")

# 定义遗传编程操作
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)


# 实现基于alpha支配的演化算法
def eaMuPlusLambdaWithAlphaDominance(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    initial_alpha=0.1,
    step_size=0.1,
):
    """
    基于alpha支配的(mu + lambda)演化策略
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # 评估初始种群
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # 初始化alpha支配
    selector = AlphaDominance(
        algorithm=toolbox, initial_alpha=initial_alpha, step_size=step_size
    )
    alpha = initial_alpha

    # 开始演化
    for gen in range(1, ngen + 1):
        # 变异操作
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # 评估后代
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 使用alpha支配选择下一代
        population[:], alpha = selector.selection(population, offspring, alpha)

        if halloffame is not None:
            halloffame.update(population)

        # 记录统计信息
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), alpha=alpha, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# 统计指标
stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
stats_size = tools.Statistics(lambda ind: ind.fitness.values[1])
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

# 初始化种群
population = toolbox.population(n=50)
hof = tools.HallOfFame(1)

# 运行演化算法
pop, log = eaMuPlusLambdaWithAlphaDominance(
    population=population,
    toolbox=toolbox,
    mu=len(population),
    lambda_=len(population),
    cxpb=0.9,
    mutpb=0.1,
    ngen=10,
    stats=mstats,
    halloffame=hof,
    verbose=True,
    initial_alpha=0.1,  # 初始alpha值
    step_size=0.1,  # 自适应步长
)

# 输出最佳个体
best_ind = hof[0] if len(hof) > 0 else tools.selBest(pop, 1)[0]
print("Best individual is:\n", best_ind)
print("\nWith fitness:", best_ind.fitness.values)

# 绘制Pareto前沿
from matplotlib import pyplot as plt
import seaborn as sns

# 非支配排序
fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)

# Pareto前沿
pareto_front = fronts[0]
fitnesses = [ind.fitness.values for ind in pareto_front]

# 分离均方误差和树的大小
mse = [fit[0] for fit in fitnesses]
sizes = [fit[1] for fit in fitnesses]

# 使用seaborn绘制散点图
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=mse, y=sizes, palette="viridis", s=60, edgecolor="w", alpha=0.7)
plt.xlabel("Mean Square Error")
plt.ylabel("Size of the GP Tree")
plt.title("Pareto Front with Alpha Dominance")
plt.show()
