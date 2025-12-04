import random
import numpy as np
import sympy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_random_state
from deap import base, creator, tools, gp, algorithms


if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Symbolic Regression estimator using DEAP with numpy acceleration.

    Parameters
    ----------
    population_size : int, default=300
        Number of individuals in the population.

    n_generations : int, default=10
        Number of generations to evolve.

    cxpb : float, default=0.9
        Crossover probability.

    mutpb : float, default=0.1
        Mutation probability.

    tournsize : int, default=3
        Tournament size for selection.

    min_depth : int, default=1
        Minimum tree depth for initialization.

    max_depth : int, default=2
        Maximum tree depth for initialization.

    mut_min_depth : int, default=0
        Minimum tree depth for mutation.

    mut_max_depth : int, default=2
        Maximum tree depth for mutation.

    verbose : bool, default=False
        Whether to print progress during evolution.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    best_individual_ : PrimitiveTree
        The best individual found during evolution.

    pset_ : PrimitiveSet
        The primitive set used for evolution.

    toolbox_ : Toolbox
        The DEAP toolbox with registered operations.

    feature_names_ : list of str
        Names of the input features.
    """

    def __init__(
        self,
        population_size=300,
        n_generations=10,
        cxpb=0.9,
        mutpb=0.1,
        tournsize=3,
        min_depth=1,
        max_depth=2,
        mut_min_depth=0,
        mut_max_depth=2,
        verbose=False,
        random_state=None,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mut_min_depth = mut_min_depth
        self.mut_max_depth = mut_max_depth
        self.verbose = verbose
        self.random_state = random_state

        self.best_individual_ = None
        self.pset_ = None
        self.toolbox_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

    def _create_primitive_set(self, n_features):
        """Create a primitive set for the given number of features."""
        pset = gp.PrimitiveSet("MAIN", arity=n_features)

        pset.addPrimitive(np.add, 2, name="add")
        pset.addPrimitive(np.subtract, 2, name="subtract")
        pset.addPrimitive(np.multiply, 2, name="multiply")
        pset.addPrimitive(np.negative, 1, name="negative")

        def protected_div(x1, x2):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)

        pset.addPrimitive(protected_div, 2, name="protected_div")

        pset.addPrimitive(np.sin, 1, name="sin")
        pset.addPrimitive(np.cos, 1, name="cos")
        pset.addPrimitive(np.exp, 1, name="exp")
        pset.addPrimitive(np.log, 1, name="log")

        def random_float():
            return random.uniform(-1.0, 1.0)

        pset.addEphemeralConstant("rand", random_float)

        if self.feature_names_ is not None:
            arg_names = {f"ARG{i}": name for i, name in enumerate(self.feature_names_)}
            pset.renameArguments(**arg_names)

        return pset

    def _evaluate_func(self, func, X):
        return (
            func(X.ravel())
            if self.n_features_ == 1
            else func(*[X[:, i] for i in range(self.n_features_)])
        )

    def _eval_symb_reg(self, individual, pset, X, y):
        """Evaluate a symbolic regression individual using numpy."""
        func = gp.compile(expr=individual, pset=pset)
        try:
            y_pred = self._evaluate_func(func, X)
            mse = np.mean((y_pred - y) ** 2)
            return (1e10 if not np.isfinite(mse) else mse,)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return (1e10,)

    def _create_toolbox(self, pset):
        """Create and configure the DEAP toolbox."""
        toolbox = base.Toolbox()

        toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=pset,
            min_=self.min_depth,
            max_=self.max_depth,
        )

        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.expr
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def evaluate(individual):
            return self._eval_symb_reg(individual, pset, self.X_train_, self.y_train_)

        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register(
            "expr_mut", gp.genFull, min_=self.mut_min_depth, max_=self.mut_max_depth
        )
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        return toolbox

    def fit(self, X, y):
        """
        Fit the symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_2d=True, ensure_all_finite=True)
        y = np.asarray(y).ravel()

        self.n_features_ = X.shape[1]

        X_normalized = self.scaler_X_.fit_transform(X)
        y_normalized = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        self.X_train_ = X_normalized
        self.y_train_ = y_normalized

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(self.n_features_)]

        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            random.seed(rng.randint(0, 2**31))
            np.random.seed(self.random_state)

        self.pset_ = self._create_primitive_set(self.n_features_)
        self.toolbox_ = self._create_toolbox(self.pset_)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        for stat_name, stat_func in [
            ("avg", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
        ]:
            stats.register(stat_name, stat_func)

        population = self.toolbox_.population(n=self.population_size)
        for ind in population:
            ind.fitness.values = self.toolbox_.evaluate(ind)

        hof = tools.HallOfFame(1)
        hof.update(population)

        algorithms.eaSimple(
            population=population,
            toolbox=self.toolbox_,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=self.verbose,
        )

        self.best_individual_ = hof[0]

        return self

    def predict(self, X):
        """
        Predict using the symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.best_individual_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = check_array(X, ensure_2d=True, ensure_all_finite=True)
        X_normalized = self.scaler_X_.transform(X)
        n_samples = X.shape[0]

        func = gp.compile(expr=self.best_individual_, pset=self.pset_)

        try:
            y_pred_normalized = self._evaluate_func(func, X_normalized)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            y_pred_normalized = np.zeros(n_samples)

        y_pred_normalized = np.asarray(y_pred_normalized).ravel()
        if len(y_pred_normalized) != n_samples:
            y_pred_normalized = np.resize(y_pred_normalized, n_samples)

        y_pred_normalized = np.nan_to_num(
            y_pred_normalized, nan=0.0, posinf=0.0, neginf=0.0
        )
        y_pred = self.scaler_y_.inverse_transform(
            y_pred_normalized.reshape(-1, 1)
        ).ravel()

        return y_pred

    def model(self):
        """
        Return the symbolic expression of the model as a SymPy expression.

        This method implements the SRbench algorithm interface.

        Returns
        -------
        sympy_expr : sympy.Expr
            The symbolic expression representing the model.
        """
        if self.best_individual_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        feature_names = self.feature_names_ or [
            f"x{i}" for i in range(self.n_features_)
        ]
        symbols = {name: sp.Symbol(name) for name in feature_names}
        symbols.update(
            {f"ARG{i}": symbols[feature_names[i]] for i in range(self.n_features_)}
        )

        stack = []

        for node in reversed(self.best_individual_):
            if isinstance(node, gp.Primitive):
                name = node.name
                arity = node.arity

                args = [stack.pop() for _ in range(arity)]
                args.reverse()

                func_map = {
                    "add": lambda a, b: a + b,
                    "subtract": lambda a, b: a - b,
                    "multiply": lambda a, b: a * b,
                    "negative": lambda a: -a,
                    "protected_div": lambda a, b: a / b,
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "exp": sp.exp,
                    "log": sp.log,
                }
                if name in func_map:
                    result = func_map[name](*args)
                else:
                    result = sp.Function(name)(*args)

                stack.append(result)
            else:
                if isinstance(node, gp.Terminal):
                    node_name = node.name

                    if node_name in symbols:
                        stack.append(symbols[node_name])
                    elif node_name.startswith("ARG"):
                        try:
                            arg_num = int(node_name[3:])
                            if self.feature_names_ is not None and arg_num < len(
                                self.feature_names_
                            ):
                                var_name = self.feature_names_[arg_num]
                                if var_name not in symbols:
                                    symbols[var_name] = sp.Symbol(var_name)
                                stack.append(symbols[var_name])
                            else:
                                stack.append(sp.Symbol(node_name))
                        except (ValueError, IndexError):
                            stack.append(sp.Symbol(node_name))
                    else:
                        try:
                            value = (
                                node.value
                                if hasattr(node, "value")
                                else float(node_name)
                            )
                            stack.append(sp.Float(value))
                        except (ValueError, TypeError):
                            stack.append(sp.Symbol(node_name))
                else:
                    try:
                        stack.append(sp.Float(float(node)))
                    except (ValueError, TypeError):
                        stack.append(sp.Symbol(str(node)))

        if len(stack) != 1:
            raise ValueError(
                f"Invalid expression tree. Stack size: {len(stack)}, expected 1."
            )

        return stack[0]

    def __str__(self):
        """String representation of the model."""
        if self.best_individual_ is None:
            return "SymbolicRegressor(not fitted)"
        return f"SymbolicRegressor: {self.best_individual_}"

    def __repr__(self):
        """Detailed string representation."""
        return (
            f"SymbolicRegressor(population_size={self.population_size}, "
            f"n_generations={self.n_generations}, "
            f"cxpb={self.cxpb}, mutpb={self.mutpb})"
        )


if __name__ == "__main__":
    import numpy as np

    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = X.ravel() ** 2

    regressor = SymbolicRegressor(
        population_size=300, n_generations=10, verbose=True, random_state=0
    )

    regressor.fit(X, y)

    X_test = np.linspace(-5, 5, 20).reshape(-1, 1)
    y_pred = regressor.predict(X_test)

    print(f"\nPredictions: {y_pred[:5]}")
    print(f"True values: {(X_test[:5].ravel() ** 2)}")

    model_expr = regressor.model()
    print(f"\nSymbolic model: {model_expr}")
    print(f"Model type: {type(model_expr)}")
