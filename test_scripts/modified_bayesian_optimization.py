from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng
import warnings
import numpy as np
from bayes_opt.event import Events
from scipy.optimize import minimize


class ModifiedBayesianOptimization(BayesianOptimization):

    def __init__(self, f, pbounds, prior_point_list, random_state=None, verbose=2,
                 bounds_transformer=None):
        super(ModifiedBayesianOptimization, self).__init__(f, pbounds, random_state, verbose, bounds_transformer)
        self.prior_point_list = prior_point_list

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='new_ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = ModifiedFunction(kind=acq,
                                kappa=kappa,
                                xi=xi,
                                kappa_decay=kappa_decay,
                                kappa_decay_delay=kappa_decay_delay,
                                prior_point_list=self.prior_point_list)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

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
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)


class ModifiedFunction(UtilityFunction):

    def __init__(self, kind, kappa, xi, prior_point_list, kappa_decay=1, kappa_decay_delay=0, lr=0.1, lr_decay=0.9):
        super(ModifiedFunction, self).__init__(kind, kappa, xi, kappa_decay, kappa_decay_delay)
        self.prior_point_list = prior_point_list
        self.lr = lr
        self._lr_decay = lr_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'new_ucb':
            return self._new_ucb(x, gp, self.kappa, self.prior_point_list, self.lr)

    def update_params(self):
        super(ModifiedFunction, self).update_params()
        self.lr += self._lr_decay

    @staticmethod
    def _new_ucb(x, gp, kappa, prior, lr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        ucb_score = mean + kappa * std

        dist = 0
        for x_prime in prior:
            dist += np.linalg.norm(x, x_prime)

        return ucb_score - lr * dist


def modified_acq_max(ac, gp, y_max, bounds, n, random_state, n_warmup=10000, n_iter=10, lr=0.9):
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys, dist = ac(x_tries, gp=gp, y_max=y_max)
    ys = (lr ** n) * dist + ys
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])