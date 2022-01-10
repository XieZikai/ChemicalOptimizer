import torch
from torch import nn
import gpytorch
import numpy as np
from gpytorch.means import ConstantMean, ZeroMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, MultitaskKernel
from gpytorch.priors import GammaPrior
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

default_likelihood = gpytorch.likelihoods.GaussianLikelihood()
default_likelihood.initialize(noise=1e-4)


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
    def __init__(self, train_x, train_y, likelihood=default_likelihood, normalize_y=False):
        # test code
        self.normalize_y = normalize_y
        if normalize_y:
            self.y_mean = torch.mean(train_y)
            self.y_std = torch.std(train_y)
            train_y = (train_y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0
            self.y_std = 1

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[1])
        matern = gpytorch.kernels.MaternKernel(nu=2.5)
        matern.initialize(lengthscale=1)
        self.covar_module = matern

    def update_norm(self, train_x, train_y):
        if self.normalize_y:
            self.y_mean = torch.mean(train_y)
            self.y_std = torch.std(train_y)
            train_y = (train_y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0
            self.y_std = 1
        self.set_train_data(train_x, train_y, strict=False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GatedLinearGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=default_likelihood):
        super(GatedLinearGPModel, self).__init__(train_x, train_y, likelihood)
        self.norm = nn.BatchNorm1d(num_features=train_x.shape[1])

        # self.gate_linear = nn.Linear(train_x.shape[1], train_x.shape[1], bias=True)
        # torch.nn.init.eye_(self.gate_linear.weight)

        self.gate_weight = nn.Parameter(torch.eye(train_x.shape[1]))
        self.gate_bias = nn.Parameter(torch.zeros(train_x.shape[1]))

        self.sigmoid = nn.Sigmoid()
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.corr = None

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

    def get_gated_result(self, training_iteration):
        x = self.norm(self.train_inputs[0])
        gated_x = self.sigmoid(torch.matmul(x, self.corr) + self.gate_bias)
        gated_result = gated_x.sum(axis=0)/len(self.train_inputs[0])
        return gated_result

    def get_gate(self):
        return self.corr.detach().numpy()


class WideNDeepGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, wide_list, likelihood=default_likelihood):
        super(WideNDeepGPModel, self).__init__(train_x, train_y, likelihood)
        self.wide_linear = nn.Linear(len(wide_list), 1)
        self.deep_gate_weight = nn.Parameter(torch.eye(train_x[1].shape[1]))
        self.deep_gate_bias = nn.Parameter(torch.zeros(train_x[1].shape[1]))
        self.deep_gate = nn.Sigmoid()
        self.deep_corr = None

        self.mean_module = gpytorch.means.ConstantMean()
        k = gpytorch.kernels.MaternKernel(nu=2.5)
        k.initialize(lengthscale=1)
        self.covar_module = gpytorch.kernels.ScaleKernel(k)

    def forward(self, wide_x, deep_x):
        wide_y = self.wide_linear(wide_x)
        self.deep_corr = (self.deep_gate_weight + self.deep_gate_weight.T) / 2
        deep_gate = self.deep_gate(torch.matmul(deep_x, self.deep_corr) + self.deep_gate_bias)
        deep_y = deep_x * deep_gate
        y = torch.cat([wide_y, deep_y], dim=1)

        mean_x = self.mean_module(y)
        covar_x = self.covar_module(y)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class EmbTransform(nn.Module):
    def __init__(self, num_uniqs, **conf):
        super().__init__()
        self.emb_sizes = conf.get('emb_sizes')
        if self.emb_sizes is None:
            self.emb_sizes = [min(50, 1 + v // 2) for v in num_uniqs]

        self.emb = nn.ModuleList([])
        for num_uniq, emb_size in zip(num_uniqs, self.emb_sizes):
            self.emb.append(nn.Embedding(num_uniq, emb_size))

    @property
    def num_out_list(self) -> [int]:
        return self.emb_sizes

    @property
    def num_out(self) -> int:
        return sum(self.emb_sizes)

    def forward(self, xe):
        return torch.cat([self.emb[i](xe[:, i]).view(xe.shape[0], -1) for i in range(len(self.emb))], dim=1)


class HeboGPModel(gpytorch.models.ExactGP):
    def __init__(self,
                 x: torch.Tensor,
                 xe: torch.Tensor,
                 y: torch.Tensor,
                 lik: gpytorch.likelihoods.GaussianLikelihood,
                 **conf):
        super().__init__((x, xe), y.squeeze(), lik)
        mean = conf.get('mean', ConstantMean())
        kern = conf.get('kern', ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=x.shape[1]),
                                            outputscale_prior=GammaPrior(0.5, 0.5)))
        kern_emb = conf.get('kern_emb', MaternKernel(nu=2.5))

        self.multi_task = y.shape[1] > 1
        self.mean = mean if not self.multi_task else MultitaskMean(mean, num_tasks=y.shape[1])
        if x.shape[1] > 0:
            self.kern = kern if not self.multi_task else MultitaskKernel(kern, num_tasks=y.shape[1])
        if xe.shape[1] > 0:
            assert 'num_uniqs' in conf
            num_uniqs = conf['num_uniqs']
            emb_sizes = conf.get('emb_sizes', None)
            self.emb_trans = EmbTransform(num_uniqs, emb_sizes=emb_sizes)
            self.kern_emb = kern_emb if not self.multi_task else MultitaskKernel(kern_emb, num_tasks=y.shape[1])

    def forward(self, x, xe):
        m = self.mean(x)
        if x.shape[1] > 0:
            K = self.kern(x)
            if xe.shape[1] > 0:
                x_emb = self.emb_trans(xe)
                K *= self.kern_emb(x_emb)
        else:
            K = self.kern_emb(self.emb_trans(xe))
        return MultivariateNormal(m, K) if not self.multi_task else MultitaskMultivariateNormal(m, K)