from bayes_opt import bayesian_optimization as bo
from bayes_opt.event import Events
import gpytorch
import torch
from optimizers.GpytorchBO.gpytorch_models.linear_weight_GP import ExactGPModel
from optimizers.GpytorchBO.utils import gpytorch_acq_max, train_gp_model, GpytorchUtilityFunction
cuda_available = torch.cuda.is_available()


class GpytorchOptimization(bo.BayesianOptimization):

    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, gp_regressor=None, GP=ExactGPModel, cuda=None):
        super(GpytorchOptimization, self).__init__(f, pbounds=pbounds, random_state=random_state, verbose=verbose,
                                                   bounds_transformer=bounds_transformer)
        if cuda is None and cuda_available:
            self.cuda = True
            print('cuda on')
        else:
            self.cuda = False
        if gp_regressor is None:
            self._GP = GP
            if self.cuda:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            else:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def suggest(self, utility_function):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())
        gtorch_model = self._GP(torch.Tensor(self.space.params),
                                torch.Tensor(self.space.target),
                                self.likelihood)
        if self.cuda:
            gtorch_model = gtorch_model.cuda()
        train_gp_model(gtorch_model, epochs=50, verbose=0, cuda=self.cuda)
        suggestion = gpytorch_acq_max(
            ac=utility_function.utility,
            gp=gtorch_model,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
            cuda=self.cuda
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

        util = GpytorchUtilityFunction(
            kind=acq,
            kappa=kappa,
            xi=xi,
            kappa_decay=kappa_decay,
            kappa_decay_delay=kappa_decay_delay,
            cuda=self.cuda
        )
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
