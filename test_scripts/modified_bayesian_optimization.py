import os

import pandas as pd
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng
import warnings
import numpy as np
from bayes_opt.event import Events
from scipy.optimize import minimize


class ModifiedBayesianOptimization(BayesianOptimization):

    def __init__(self, f, pbounds, prior_point_list, random_state=None, verbose=2,
                 bounds_transformer=None, save_result=False, adding_init_sample=False, save_dir=None):
        super(ModifiedBayesianOptimization, self).__init__(f, pbounds, random_state, verbose, bounds_transformer)
        self.prior_point_list = prior_point_list

        # Contrast experiment: adding prior points into initial samples
        if adding_init_sample and prior_point_list is not None:
            for point in prior_point_list:
                self._queue.add(np.array(point))

        self._max_value = 0  # Record max value
        df_columns = list(pbounds.keys()) + ['target']
        self._df_max = pd.DataFrame([], columns=df_columns)  # Record all optimal points as experiment results
        self._df = pd.DataFrame([], columns=df_columns)  # Record all points as experiment results
        self.save_result = save_result
        self.save_dir = save_dir

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            target = self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)
            return target

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

            target = self.probe(x_probe, lazy=False)
            x_probe_dict = self._space.array_to_params(x_probe)
            x_probe_dict['target'] = target
            self._df = self._df.append(x_probe_dict, ignore_index=True)
            if target > self._max_value:
                self._df_max = self._df_max.append(self._space.array_to_params(x_probe), ignore_index=True)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)
        if self.save_result:
            from datetime import datetime
            name = str(datetime.now()).replace(':', '-').split('.')[0]
            path = os.getcwd()
            if self.save_dir is None:
                path = os.path.join(os.path.join(path, 'data'), name)
            else:
                path = os.path.join(os.path.join(path, self.save_dir), name)
            if not os.path.exists(path):
                os.makedirs(path)
            self._df.to_csv(os.path.join(path, 'df_total.csv'))
            self._df_max.to_csv(os.path.join(path, 'df_max.csv'))

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


class ModifiedFunction(object):

    def __init__(self, kind, kappa, xi, prior_point_list, kappa_decay=1, kappa_decay_delay=0, lr=0.1, lr_decay=0.8):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi
        self.constant_multiplier = None

        self._iters_counter = 0

        self.kind = kind
        self.prior_point_list = prior_point_list
        self.lr = np.ones(len(prior_point_list)) * lr
        self._lr_decay = lr_decay

        self.prior_pointer = None

    def utility(self, x, gp):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'new_ucb':
            assert self.prior_point_list is not None, 'Prior point list should not be none'
            return self._new_ucb(x, gp, self.kappa, self.prior_point_list)

    def update_params(self):
        # only change the learning rate chosen
        self._iters_counter += 1
        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay
        self.lr[self.prior_pointer] *= self._lr_decay

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
            # print('mean, std: ', mean, std)

        return mean + kappa * std

    def _new_ucb(self, x, gp, kappa, prior):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        ucb_score = mean + kappa * std

        chosen_index = np.random.randint(len(prior))

        self.prior_pointer = chosen_index
        lr = self.lr[chosen_index]

        dist = np.linalg.norm(x-prior[chosen_index])

        ''' Discarded old method: using all prior points to calculate total L2 distance.
        dist = 0
        for x_prime in prior:
            dist += np.linalg.norm(x-x_prime)'''

        # Adding multiplier to make sure the penalty term is at the same scale as the UCB value
        if self.constant_multiplier is None:
            self.constant_multiplier = ucb_score / (lr * np.sqrt(dist) * 2)

        return ucb_score - lr * np.sqrt(dist) * self.constant_multiplier

    def manually_shrink_lr(self):
        self.lr[self.prior_pointer] /= 2
