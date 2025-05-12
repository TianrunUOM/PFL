import h5py
import numpy as np
import os
import wandb


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    std = np.std(max_accuracy)
    mean = np.mean(max_accuracy)
    
    print("std for best accuracy:", std)
    print("mean for best accuracy:", mean)
    
    if wandb.run is not None:
        wandb.log({
            "final_avg_accuracy": mean,
            "final_std_accuracy": std,
            "times": times
        })
        
        for i in range(times):
            wandb.log({f"run_{i}_best_accuracy": max_accuracy[i]})


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc