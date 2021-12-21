from bayes_opt import bayesian_optimization as bo
from bayes_opt.event import Events
import gpytorch
import torch
from optimizers.GpytorchBO.gpytorch_models.linear_weight_GP import *
from optimizers.GpytorchBO.utils import gpytorch_acq_max, train_gp_model, train_wideep_gp_model, \
    GpytorchUtilityFunction, WideepGPUtilityFunction

cuda_available = torch.cuda.is_available()


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

        suggestion = gpytorch_acq_max(
            ac=utility_function.utility,
            gp=self.gpytorch_model,
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