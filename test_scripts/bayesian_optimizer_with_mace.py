from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from scipy.stats import norm
import numpy as np
import warnings


def mace_acq_max(gp, y_max, sample_size=40):
    mace_acq = MaceNumpy(gp=gp, y_max=y_max)
    optimizer = NSGA2(pop_size=sample_size,
                      n_offsprings=10,
                      sampling=get_sampling('real_random'),
                      crossover=get_crossover('real_sbx', prob=0.9, eta=15),
                      mutation=get_mutation('real_pm'),
                      eliminate_duplicates=True
                      )
    termination = get_termination("n_gen", 40)
    res = minimize(problem=mace_acq,
                   algorithm=optimizer,
                   termination=termination,
                   seed=1,
                   save_history=False,
                   verbose=False
                   )
    X = res.X
    x = X[np.random.randint(len(X))]
    return x


class MaceNumpy(Problem):

    def __init__(self, gp, y_max, xi=0.0, kappa=2.576, kappa_decay=1, kappa_decay_delay=0):
        super(MaceNumpy, self).__init__(n_var=11, n_obj=3, n_constr=0, xl=np.zeros(11), xu=np.zeros(11) + 5)
        self.xi = xi
        self.kappa = kappa
        self.gp = gp
        self.y_max = y_max
        self.kappa_decay = kappa_decay
        self.kappa_decay_delay = kappa_decay_delay

    def _evaluate(self, x, out, *args, **kwargs):
        ucb = UtilityFunction(kind='ucb',
                              kappa=self.kappa,
                              xi=self.xi,
                              kappa_decay=self.kappa_decay,
                              kappa_decay_delay=self.kappa_decay_delay)
        ucb_value = ucb.utility(x, self.gp, self.y_max)
        ei = UtilityFunction(kind='ei',
                              kappa=self.kappa,
                              xi=self.xi,
                              kappa_decay=self.kappa_decay,
                              kappa_decay_delay=self.kappa_decay_delay)
        ei_value = ei.utility(x, self.gp, self.y_max)
        poi = UtilityFunction(kind='poi',
                              kappa=self.kappa,
                              xi=self.xi,
                              kappa_decay=self.kappa_decay,
                              kappa_decay_delay=self.kappa_decay_delay)
        poi_value = poi.utility(x, self.gp, self.y_max)
        out['F'] = np.column_stack([-ucb_value, -ei_value, -poi_value])


class MaceBayesianOptimization(BayesianOptimization):

    def __init__(self, **kwargs):
        super(MaceBayesianOptimization, self).__init__(**kwargs)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = mace_acq_max(gp=self._gp,
                                  y_max=self._space.target.max())
        print(suggestion)
        return self._space.array_to_params(suggestion)
