import torch
import gpytorch
import numpy as np
import bayes_opt


'''def train_gp_model(gp_model, train_x, train_y, optimizer=None, epochs=100, mll=None, verbose=1):
    """
    Training procedure of Gaussian process model.
    :param gp_model: GP model to be trained
    :param train_x: Training data
    :param train_y: Training label
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
        output = gp_model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if verbose == 1:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, epochs, loss.item(),
                gp_model.covar_module.base_kernel.lengthscale.item(),
                gp_model.likelihood.noise.item()))
        optimizer.step()'''


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