from bayes_opt import bayesian_optimization as bo
from bayes_opt.event import Events
import gpytorch
import torch
from gpytorch_models.linear_weight_GP import ExactGPModel
from utils import gpytorch_acq_max, train_gp_model
import warnings


class GpytorchOptimization(bo.BayesianOptimization):

    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, gp_regressor=None, GP=ExactGPModel):
        super(GpytorchOptimization, self).__init__(f, pbounds=pbounds, random_state=random_state, verbose=verbose,
                                                   bounds_transformer=bounds_transformer)
        if gp_regressor is None:
            self._GP = GP
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def suggest(self, utility_function):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())
        gtorch_model = self._GP(torch.Tensor(self.space.params),
                                torch.Tensor(self.space.target),
                                self.likelihood)
        train_gp_model(gtorch_model, epochs=50, verbose=0)
        suggestion = gpytorch_acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )
        return self._space.array_to_params(suggestion)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
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
