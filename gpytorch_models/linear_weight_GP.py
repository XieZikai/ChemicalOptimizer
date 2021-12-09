import torch
from torch import nn
import gpytorch
import numpy as np


default_likelihood = gpytorch.likelihoods.GaussianLikelihood()


class LinearWeightExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=default_likelihood):
        # print('Shape: ', train_x.shape)
        super(LinearWeightExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.norm = nn.BatchNorm1d(num_features=train_x.shape[1])
        self.linear = nn.Linear(train_x.shape[1], train_x.shape[1])
        self.latent_weight = torch.nn.Parameter(torch.eye(train_x.shape[1]))
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.corr = None

    def forward(self, x):
        # x = self.linear(x)
        x = self.norm(x)
        self.corr = (self.latent_weight + self.latent_weight.T)/2
        # print(x.shape)
        x = torch.matmul(x, self.corr)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_corr_matrix(self):
        return self.corr


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=default_likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[1])
        matern = gpytorch.kernels.MaternKernel(nu=2.5)
        matern.initialize(lengthscale=1)
        self.covar_module = gpytorch.kernels.ScaleKernel(matern)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GatedLinearGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=default_likelihood):
        super(GatedLinearGPModel, self).__init__(train_x, train_y, likelihood)
        self.norm = nn.BatchNorm1d(num_features=train_x.shape[1])

        self.gate_linear = nn.Linear(train_x.shape[1], train_x.shape[1], bias=True)
        torch.nn.init.eye_(self.gate_linear.weight)

        self.gate_weight = nn.Parameter(torch.eye(train_x.shape[1]))
        self.gate_bias = nn.Parameter(torch.zeros(train_x.shape[1]))

        self.sigmoid = nn.Sigmoid()
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.corr = None
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, x):
        x = self.norm(x)
        self.corr = (self.gate_weight + self.gate_weight.T)/2
        linear_x = torch.matmul(x, self.corr) + self.gate_bias
        gated_x = self.sigmoid(linear_x)
        # gated_x = self.sigmoid(self.gate_linear(x))
        # print('Gated result: ', gated_x)
        # x = linear_x * gated_x
        x = x * gated_x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_corr_eig(self):
        mat = self.corr.detach().numpy()
        return np.linalg.eig(mat)

    def get_gate(self, training_iteration):
        x = self.norm(self.train_x)
        gated_x = self.sigmoid(self.gate_linear(x))
        gated_result = gated_x.sum(axis=0)/training_iteration
        print('Gated result: ', gated_result)
        return self.gate_linear.weight, self.gate_linear.bias


class WideNDeepGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, wide_list, likelihood=default_likelihood):
        super(WideNDeepGPModel, self).__init__(train_x, train_y, likelihood)
        self.wide_linear = nn.Linear(len(wide_list), 1)
        self.deep_gate_weight = nn.Parameter(torch.eye(train_x.shape[1] - len(wide_list)))
        self.deep_gate_bias = nn.Parameter(torch.zeros(train_x.shape[1] - len(wide_list)))
        self.deep_gate = nn.Sigmoid()
        self.deep_corr = None

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, wide_x, deep_x):
        wide_y = self.wide_linear(wide_x)
        self.deep_corr = (self.deep_gate_weight + self.deep_gate_weight.T) / 2
        deep_gate = self.deep_gate(torch.matmul(self.deep_corr, deep_x) + self.deep_gate_bias)
        deep_y = deep_x * deep_gate
        y = torch.cat([wide_y, deep_y], dim=1)

        mean_x = self.mean_module(y)
        covar_x = self.covar_module(y)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)