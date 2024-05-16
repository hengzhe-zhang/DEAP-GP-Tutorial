import concurrent.futures
import random
import time

import numpy
import numpy as np
from deap import base, creator, gp
from deap import tools
from deap.algorithms import varAnd, eaSimple
from deap.tools import selBest

# 使用numpy创建一个数据集
x = np.linspace(-10, 10, 1000000)


# 符号回归
def evalSymbReg(ind):
    func = toolbox.compile(ind)
    # 评估生成的函数并计算MSE
    mse = np.mean((func(x) - (x + 1) ** 2) ** 2)
    return (mse,)


# 创建个体和适应度函数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 定义函数和终端变量
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(np.negative, 1)


def random_int():
    return random.randint(-1, 1)


pset.addEphemeralConstant("rand101", random_int)
pset.renameArguments(ARG0="x")

# 定义遗传编程操作
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# 定义统计指标
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

def steady_state_gp(
    population,
    toolbox,
    cxpb,
    mutpb,
    max_evaluations,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    logbook = tools.Logbook()
    logbook.header = ["evals", "nevals"] + (stats.fields if stats else [])

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    futures = {}
    evaluations =0

    # 评估初始种群
    for i, ind in enumerate(population):
        if not ind.fitness.valid:
            future = executor.submit(toolbox.evaluate, ind)
            futures[future] = ind

    all_done = population

    while evaluations < max_evaluations:
        # 生成新个体
        if evaluations + len(futures) <= max_evaluations and len(all_done) >= 2:
            selected = toolbox.select(population, len(all_done))
            offspring = varAnd(selected, toolbox, cxpb, mutpb)
            all_done = []

            # 提交评估任务
            for child in offspring:
                if evaluations + len(futures) <= max_evaluations:
                    future = executor.submit(toolbox.evaluate, child)
                    futures[future] = child
                else:
                    break

        # 等待至少一个个体完成评估
        current_done, _ = concurrent.futures.wait(
            list(futures.keys()),
            return_when=concurrent.futures.FIRST_COMPLETED,
        )

        # 处理评估完成的个体
        done_inds = []
        for future in current_done:
            ind = futures.pop(future)
            ind.fitness.values = future.result()
            done_inds.append(ind)
            all_done.append(ind)
            evaluations += 1

        if halloffame is not None:
            halloffame.update(done_inds)

        # 用高适应度个体替换低适应度个体
        population = selBest(population + done_inds, len(population))

        if verbose and evaluations % 100 == 0:
            record = stats.compile(population) if stats else {}
            logbook.record(evals=evaluations, **record)
            print(logbook.stream)

    executor.shutdown()
    return population, logbook


if __name__ == "__main__":
    start = time.time()
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    pop, log = steady_state_gp(
        population=population,
        toolbox=toolbox,
        cxpb=0.9,
        mutpb=0.1,
        max_evaluations=(5+1) * 100,
        stats=mstats,
        halloffame=hof,
        verbose=True,
    )
    end = time.time()
    print("time:", end - start)

    start = time.time()
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    pop, log = eaSimple(
        population=population,
        toolbox=toolbox,
        cxpb=0.9,
        mutpb=0.1,
        ngen=5,
        stats=mstats,
        halloffame=hof,
        verbose=True,
    )
    end = time.time()
    print("time:", end - start)