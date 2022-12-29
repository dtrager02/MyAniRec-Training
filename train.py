from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
parameter_dict = {
   'train_neg_sample_args': None,
}

hp = HyperTuning(objective_function=objective_function, algo='bayes', early_stop=10,
                max_evals=100, params_file='model.hyper', fixed_config_file_list=['params.yaml'],export_result="tuning.txt")