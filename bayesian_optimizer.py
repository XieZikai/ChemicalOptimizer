from optimizers.base_optimizer import BaseWrapper

from data import data_preprocessing
from optimizers.GpytorchBO.base_optimizer import GpytorchOptimization
from optimizers.GpytorchBO.test_scripts.bayesian_optimizer_with_mace import MaceBayesianOptimization
from bayes_opt import BayesianOptimization
from data.get_olympus_models import OlympusEmulatorWrapper
import numpy as np

kuka_data = data_preprocessing.get_kuka_data()
kuka_bound = {}
for column in kuka_data.columns:
    if column == 'target':
        continue
    else:
        kuka_bound[column] = (0, 5)
print(kuka_bound)


def merge_string(str_list):
    assert len(str_list) >= 1, print('List must contain at least one string!')
    output = ''
    for i in str_list:
        output += i
        if i != str_list[len(str_list)-1]:
            output += ', '
    return output


class BOWrapper(BaseWrapper):
    """
    Wrap the bayesian optimization algorithm from bayes_opt for optimizer verifier.
    """

    def __init__(self, bound=None, data=data_preprocessing.get_kuka_data(), label_column='target',
                 black_box_function=None, optimizer=None, n_iter=500, **kwargs):
        super(BOWrapper, self).__init__(bound=bound, data=data, label_column=label_column,
                                        black_box_function=black_box_function, optimizer=optimizer)

        assert optimizer in self.optimizer_list, print('Optimizer option not provided! Only support: ' \
                                                       + merge_string(self.optimizer_list))
        if optimizer == 'BO':
            self.optimizer = BayesianOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100)
            )
        elif optimizer == 'MaceBO':
            self.optimizer = MaceBayesianOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100)
            )
        elif optimizer == 'gpytorchBO':
            self.optimizer = GpytorchOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100),
                cuda=True
            )
        elif optimizer == 'linearGBO':
            from optimizers.GpytorchBO.gpytorch_models.linear_weight_GP import LinearWeightExactGPModel
            self.optimizer = GpytorchOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100),
                cuda=True,
                GP=LinearWeightExactGPModel
            )
        elif optimizer == 'gatedGBO':
            from optimizers.GpytorchBO.gpytorch_models.linear_weight_GP import GatedLinearGPModel
            self.optimizer = GpytorchOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100),
                cuda=True,
                GP=GatedLinearGPModel
            )
        elif optimizer == 'wideepGBO':
            from optimizers.GpytorchBO.gpytorch_models.linear_weight_GP import WideNDeepGPModel
            assert 'wide_list' in kwargs.keys(), 'Must provide the names of features for wide net!'
            for i in kwargs['wide_list']:
                assert i in list(self.bound.keys()), 'Wide feature name not expected: ' + i
            wide_list = kwargs['wide_list']
            self.optimizer = GpytorchOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100),
                cuda=True,
                GP=WideNDeepGPModel,
                wide_list=wide_list,
            )
        elif optimizer == 'TestBO':
            prior_point_list = [[0, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0]]
            from optimizers.GpytorchBO.test_scripts.modified_bayesian_optimization import ModifiedBayesianOptimization
            self.optimizer = ModifiedBayesianOptimization(
                f=self.black_box_function,
                pbounds=self.bound,
                verbose=2,
                random_state=np.random.randint(100),
                prior_point_list=prior_point_list
            )

        '''# MaceOptimizer
        from optimizers.GpytorchBO.mace_optimizer import MaceOptimization
        optimizer = MaceOptimization(
            f=self.black_box_function,
            pbounds=self.bound,
            verbose=2,
            random_state=np.random.randint(100))'''

        self.n_iter = n_iter

    def optimize(self, init_point=5, return_gate=False):
        gate = self.optimizer.maximize(
                init_points=init_point,
                n_iter=self.n_iter,
                # acq='poi'
            )
        if return_gate is not None:
            return gate
        else:
            print('Gated value not collected or there is no gate unit')
            return 'Gated value not collected or there is no gate unit'

    def save_surrogate_model(self, n_iter):
        import torch
        from datetime import datetime
        torch.save(self.optimizer.gpytorch_model, str(datetime.now()).replace(':', '_').split('.')[0]+'_'+str(n_iter)+'.pt')


if __name__ == '__main__':
    # olympus_simulator = OlympusEmulatorWrapper(dataset='snar')
    # name, bound = olympus_simulator.get_names_and_bounds()
    n_iter = 200
    # optimizer = 'gatedGBO'
    # optimizer = 'linearGBO'
    # optimizer = 'BO'
    # optimizer = 'MaceBO'
    # optimizer = 'wideepGBO'
    optimizer = 'TestBO'
    # bo = BOWrapper(bound=bound, label_column=name, black_box_function=olympus_simulator.experiment, optimizer=optimizer)
    bo = BOWrapper(optimizer=optimizer, n_iter=n_iter)
    bo.optimize()

    if optimizer == 'linearGBO' or optimizer == 'gatedGBO':
        print(bo.optimizer.gpytorch_model.get_corr_matrix())
    if optimizer == 'gatedGBO':
        print(bo.optimizer.gpytorch_model.get_gate())
        bo.save_surrogate_model(n_iter=n_iter)

    # 批量操作脚本
    import sys
'''    newfile = 'batch testing output.txt'
    data = open(newfile, 'w', encoding='utf-8')
    # sys.stdout = data
    for training_iteration in [500]:
        # bo = BOWrapper(optimizer=optimizer, n_iter=training_iteration, wide_list=['NaOH-1M_to_dispense', 'L-Cysteine-100gL_to_dispense', 'P10-MIX1_to_dispense'])
        bo = BOWrapper(optimizer=optimizer, n_iter=training_iteration)
        gated_result = bo.optimize(return_gate=True)
        # print(gated_result)
        print('Corr Matrix: ', bo.optimizer.gpytorch_model.get_corr_matrix())
        returned = bo.optimizer.gpytorch_model.get_gate(training_iteration)
        if returned is not None:
            gated_list, eigvalue_list, gate_weight = returned
            print('Gate weight: ', gate_weight)
            print('eigvalue_list: ', eigvalue_list)
            print('gated_list', gated_list)
    data.close()'''
