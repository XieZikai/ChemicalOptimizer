import numpy as np
from bayes_opt import bayesian_optimization as bo
from bayes_opt.event import Events
import gpytorch
import torch
from optimizers.GpytorchBO.gpytorch_models.linear_weight_GP import *
from optimizers.GpytorchBO.utils import get_mean_variance, train_gp_model, train_wideep_gp_model, \
    GpytorchUtilityFunction, WideepGPUtilityFunction
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from scipy.stats import norm

cuda_available = torch.cuda.is_available()


class MACE(Problem):

    def __init__(self, gp, y_max, xi=0.0, kappa=2.576):
        super(MACE, self).__init__(n_var=11, n_obj=3, n_constr=0, xl=np.zeros(11), xu=np.zeros(11)+5)
        self.xi = xi
        self.kappa = kappa
        self.gp = gp
        self.y_max = y_max

    def _evaluate(self, x, out, *args, **kwargs):
        mean, var, _, _ = get_mean_variance(self.gp, torch.Tensor(x))
        std = torch.sqrt(var)
        # ucb
        ucb_value = (mean + self.kappa * std).detach().numpy()
        # ei
        a = (mean - self.y_max - self.xi).detach().numpy()
        z = (a / std).numpy()
        ei = a * norm.cdf(z) + std * norm.pdf(z)
        # poi
        z = ((mean - self.y_max - self.xi) / std).cpu().detach().numpy()
        poi = norm.cdf(z)
        out['F'] = np.column_stack([ucb_value, ei, poi])


def mace_acq_max(gp, y_max, sample_size=40):
    mace_acq = MACE(gp=gp, y_max=y_max)
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
    x = X[np.random.randint(sample_size)]
    return x


class GpytorchOptimization(bo.BayesianOptimization):

    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, gp_regressor=None, GP=ExactGPModel, cuda=None, **kwargs):
        super(GpytorchOptimization, self).__init__(f, pbounds=pbounds, random_state=random_state, verbose=verbose,
                                                   bounds_transformer=bounds_transformer)
        self._bound = pbounds
        if cuda is None and cuda_available:
            self.cuda = True
            print('cuda on')
        else:
            self.cuda = False
        if gp_regressor is None:
            self._GP = GP
            if self.cuda:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
                # noises = torch.ones(len(pbounds)) * 1e-6
                # self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises,
                #                                                                     learn_additional_noise=False).cuda()
                self.likelihood.initialize(noise=1e-4)
            else:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
                # noises = torch.ones(len(pbounds)) * 1e-6
                # self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises,
                #                                                                     learn_additional_noise=False)
                self.likelihood.initialize(noise=1e-4)
        self.gpytorch_model = None
        self.kwargs = kwargs
        if self._GP == WideNDeepGPModel:
            self.wide_index = []
            self.deep_index = []
            for i in range(len(self._bound)):
                if list(self._bound.keys())[i] in self.kwargs['wide_list']:
                    self.wide_index += [i]
                else:
                    self.deep_index += [i]

    def suggest(self, utility_function):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        if self.gpytorch_model is None:
            if self._GP == WideNDeepGPModel:
                train_input = torch.Tensor(self.space.params)
                gtorch_model = self._GP([train_input[:, self.wide_index], train_input[:, self.deep_index]],
                                        torch.Tensor(self.space.target),
                                        self.kwargs['wide_list'],
                                        self.likelihood)
            else:
                gtorch_model = self._GP(torch.Tensor(self.space.params),
                                        torch.Tensor(self.space.target),
                                        self.likelihood)
            if self.cuda:
                gtorch_model = gtorch_model.cuda()
            self.gpytorch_model = gtorch_model
        else:
            if self._GP == WideNDeepGPModel:
                train_input = torch.Tensor(self.space.params)
                self.gpytorch_model.set_train_data(
                    [train_input[:, self.wide_index], train_input[:, self.deep_index]],
                    torch.Tensor(self.space.target),
                    strict=False
                )
            else:
                self.gpytorch_model.set_train_data(torch.Tensor(self.space.params),
                                                   torch.Tensor(self.space.target),
                                                   strict=False)
                # self.gpytorch_model.update_norm(torch.Tensor(self.space.params), torch.Tensor(self.space.target))

        if self._GP == WideNDeepGPModel:
            train_wideep_gp_model(self.gpytorch_model, epochs=300, verbose=0, cuda=self.cuda)
        else:
            train_gp_model(self.gpytorch_model, epochs=300, verbose=0, cuda=self.cuda)

        # todo: change gpytorch_acq_max to pymoo-based MACE acq maximize
        '''suggestion = gpytorch_acq_max(
            ac=utility_function.utility,
            gp=self.gpytorch_model,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
            cuda=self.cuda
        )'''
        suggestion = mace_acq_max(
            gp=self.gpytorch_model,
            y_max=self._space.target.max()
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

        if self._GP == WideNDeepGPModel:
            util = WideepGPUtilityFunction(
                kind=acq,
                kappa=kappa,
                xi=xi,
                kappa_decay=kappa_decay,
                kappa_decay_delay=kappa_decay_delay,
                cuda=self.cuda,
                wide_index=self.wide_index,
                deep_index=self.deep_index
            )
        else:
            util = GpytorchUtilityFunction(
                kind=acq,
                kappa=kappa,
                xi=xi,
                kappa_decay=kappa_decay,
                kappa_decay_delay=kappa_decay_delay,
                cuda=self.cuda
            )

        iteration = 0

        # recording
        gated_list = []
        eigvalue_list = []
        gate_weight = []

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

            # if iteration % 10 == 0:
            try:
                gated_value = self.gpytorch_model.get_gated_result(iteration)
                gated_list += [gated_value]
                eigvalue, eigvec = self.gpytorch_model.get_corr_eig()
                gate_weight += self.gpytorch_model.get_gate()
                # print(eigvalue)
                eigvalue_list += [eigvalue]
            except:
                pass

        self.dispatch(Events.OPTIMIZATION_END)
        return gated_list, eigvalue_list, gate_weight