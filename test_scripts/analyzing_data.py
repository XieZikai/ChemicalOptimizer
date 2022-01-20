import pandas as pd
import os
import numpy as np

base_dir = r'C:\Users\darkn\PycharmProjects\ChemicalOptimization\optimizers\GpytorchBO'
prior_methods = ['penalty_term', 'init_sample']
prior_types = ['good_prior', 'bad_prior']


def weight_coverage_speed(l, miu=100):
    result = 0
    for index, i in enumerate(l):
        result += i / (index + miu)
    return result / len(l)


def analyse_csv(prior_type, prior_method):
    result_csv = pd.DataFrame([], columns=['Time', 'Average Result', 'Best Result',
                                           'Average Best Result', 'New Optimal Number',
                                           'Weighted Converge Speed', 'Optimal Result in first 50 trials',
                                           ])
    path = os.path.join(base_dir, prior_type)
    path = os.path.join(path, prior_method)
    for folder in os.listdir(path):
        column_dict = {'Time': folder}
        csv_folder = os.path.join(path, folder)
        df_total = pd.read_csv(os.path.join(csv_folder, 'df_total.csv'))
        df_max = pd.read_csv(os.path.join(csv_folder, 'df_max.csv'))
        column_dict['Average Result'] = np.sum(df_total['target']) / len(df_total)
        column_dict['Best Result'] = np.max(df_max['target'])
        column_dict['Average Best Result'] = np.sum(df_max['target']) / len(df_max)
        column_dict['New Optimal Number'] = len(df_max)
        column_dict['Optimal Result in first 50 trials'] = np.max(df_total['target'][:50])

        target_list = list(df_max['target'])
        column_dict['Weighted Converge Speed'] = weight_coverage_speed(target_list)

        result_csv = result_csv.append(column_dict, ignore_index=True)

    result_csv.to_csv(prior_type + ' ' + prior_method + '_result.csv')


if __name__ == '__main__':
    analyse_csv('good_prior', 'vanilla')

    for prior_method in prior_methods:
        for prior_type in prior_types:
            analyse_csv(prior_type, prior_method)
