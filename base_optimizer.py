from bayes_opt import bayesian_optimization as bo
import gpytorch
import torch
from gpytorch_models.linear_weight_GP import ExactGPModel
import warnings


class gpytorch_optimizer(bo.BayesianOptimization):

    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, gp_regressor=None, GP=ExactGPModel):
        super(gpytorch_optimizer, self).__init__(f, pbounds=pbounds, random_state=random_state, verbose=verbose,
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

