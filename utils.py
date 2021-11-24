import gpytorch
import numpy as np
import bayes_opt
from scipy.optimize import minimize
from bayes_opt.util import UtilityFunction
import torch
from scipy.stats import norm


def train_gp_model(gp_model, optimizer=None, epochs=100, mll=None, verbose=1):
    """
    Training procedure of Gaussian process model.
    :param gp_model: GP model to be trained
    :param optimizer: Training optimizer. If not provided, will use torch.optim.Adam
    :param epochs: Training epochs, default is 10
    :param mll: Marginal likelihood for the GP model, default is gpytorch.mlls.ExactMarginalLogLikelihood
    :param verbose: Controller of showing training procedure
    :return: NA
    """
    gp_model.train()
    gp_model.likelihood.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
    if mll is None:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    for i in range(epochs):
        optimizer.zero_grad()
        output = gp_model(gp_model.train_inputs)
        loss = -mll(output, gp_model.train_targets)
        loss.backward()
        if verbose == 1:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, epochs, loss.item(),
                gp_model.covar_module.base_kernel.lengthscale.item(),
                gp_model.likelihood.noise.item()))
        optimizer.step()


def sample_gp_model(gp_model, x, sampling_num):
    """
    Sample the result from the GP model for given data x.
    :param gp_model: provided GP model for sampling
    :param x: GP input. gp_model(x) will output the Gaussian distribution for sampling
    :param sampling_num: returned sampling result number
    :return: sampled results of shape (sampling_num,)
    """
    gp_model.eval()
    gp_model.likelihood.eval()
    f_predict = gp_model(x)
    f_samples = f_predict.sample(sample_shape=torch.Size(sampling_num,))
    return f_samples


def get_mean_variance(gp_model, x):
    """
    Get the mean value, variance value, likelihood and covariance matrix of the GP model (for Bayesian optimization).
    :param gp_model: provided GP model
    :param x: GP input
    :return: tuple(mean, variance, covariance matrix)
    """
    gp_model.eval()
    gp_model.likelihood.eval()
    f_predict = gp_model(x)
    y_predict = gp_model.likelihood(gp_model(x))
    f_mean = f_predict.mean
    f_var = f_predict.variance
    f_covar = f_predict.covariance_matrix
    return f_mean, f_var, y_predict, f_covar


class GpytorchUtilityFunction(UtilityFunction):

    def __init__(self,  kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0):
        super(GpytorchUtilityFunction, self).__init__(kind, kappa, xi, kappa_decay, kappa_decay_delay)

    @staticmethod
    def _ucb(x, gp, kappa):
        gp.eval()
        gp.likelihood.eval()
        mean, var, _, _ = get_mean_variance(gp, torch.Tensor(x))
        std = torch.sqrt(var)
        ucb_value = mean + kappa * std
        return ucb_value.numpy()

    @staticmethod
    def _ei(x, gp, y_max, xi):
        gp.eval()
        gp.likelihood.eval()
        mean, var, _, _ = get_mean_variance(gp, torch.Tensor(x))
        std = torch.sqrt(var)
        a = (mean - y_max - xi).numpy()
        z = (a / std).numpy()
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        gp.eval()
        gp.likelihood.eval()
        mean, var, _, _ = get_mean_variance(gp, torch.Tensor(x))
        std = torch.sqrt(var)
        z = ((mean - y_max - xi) / std).numpy()
        return norm.cdf(z)


def gpytorch_acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))

    ys = ac(x_tries, gp=gp, y_max=y_max)
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
