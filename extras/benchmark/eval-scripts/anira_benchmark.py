import os
import re
import csv
import numpy as np

# Define log file paths
log_file_paths = []
log_file_paths.append(os.path.join(os.path.dirname(__file__), "./../logs/Linux_advanced_0.0.8.log"))
log_file_paths.append(os.path.join(os.path.dirname(__file__), "./../logs/MacOS_advanced_0.0.8.log"))
log_file_paths.append(os.path.join(os.path.dirname(__file__), "./../logs/Windows_advanced_0.0.8.log"))

def create_folder(folder_name: str) -> None:
    try:
        os.mkdir(os.path.join(os.path.dirname(__file__), "./../" + folder_name))
    except:
        pass

def get_list_from_log(file_path: str, list: list=None) -> list:

    if list is None:
        operating_system = []
        model = []
        backend = []
        buffer_size = []
        repetition_index = []
        iteration_count = []
        repetition_count = []
        runtime = []
        log_list = [operating_system, model, backend, buffer_size, repetition_index, repetition_count, iteration_count, runtime]
    else:
        log_list = list

    repetition_index = 0
    old_repeatition_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if "SingleIteration" in line:
                match = re.search(r"SingleIteration\/ProcessBlockFixture\/[^\/]*\/([^.]*.[a-zA-Z]*)\/([a-z]*)\/([0-9]*)\/iteration:([0-9]*)\/repetition:([0-9])\s*([0-9]*.[0-9]*)\sms", line)
                if not "stateful-lstm-libtorch.onnx" == match.group(1): # onnxruntime does not support stateful lstm
                    if old_repeatition_count != int(match.group(5)):
                        repetition_index += 1
                        old_repeatition_count = int(match.group(5))
                    log_list[0].append(os.path.splitext(os.path.basename(file_path))[0])
                    log_list[1].append(match.group(1))
                    log_list[2].append(match.group(2))
                    log_list[3].append(int(match.group(3)))
                    log_list[4].append(repetition_index)
                    log_list[5].append(int(match.group(5)))
                    log_list[6].append(int(match.group(4)))
                    log_list[7].append(float(match.group(6)))
    
    return log_list

def get_sequence_statistics_from_list(list: list, measure: str, lenght_slices: int=10) -> list:
    stat_list = [[]]
    for l in list:
        stat_list.append([])
    for i in range(0, len(list[0]), lenght_slices):
        stat_list[0].append(list[0][i])
        stat_list[1].append(list[1][i])
        stat_list[2].append(list[2][i])
        stat_list[3].append(list[3][i])
        stat_list[4].append(list[4][i])
        stat_list[5].append(list[5][i])
        stat_list[6].append(f"{list[6][i]//lenght_slices}")
        if measure == "sequence_mean":
            stat_list[7].append(np.mean(list[7][i:i+lenght_slices]))
        elif measure == "sequence_median":
            stat_list[7].append(np.median(list[7][i:i+lenght_slices]))
        elif measure == "sequence_max":
            stat_list[7].append(np.max(list[7][i:i+lenght_slices]))
        elif measure == "sequence_min":
            stat_list[7].append(np.min(list[7][i:i+lenght_slices]))
        elif measure == "sequence_iqr":
            stat_list[7].append(np.abs(np.percentile(list[7][i:i+lenght_slices], 75) - np.percentile(list[7][i:i+lenght_slices], 25)))
        elif measure == "sequence_std":
            stat_list[7].append(np.std(list[7][i:i+lenght_slices]))
        else:
            raise ValueError("Invalid measure")    
          
    return stat_list

def moving_average(list: list, window: int=3) -> list:
    moving_average_list = []
    for l in list:
        moving_average_list.append([])
    
    max_iteration = max(list[6])
    for index in range(0, len(list[0]), max_iteration+1):
        for i in range(0, max_iteration-window+2):
            moving_average_list[0].append(list[0][index])
            moving_average_list[1].append(list[1][index])
            moving_average_list[2].append(list[2][index])
            moving_average_list[3].append(list[3][index])
            moving_average_list[4].append(list[4][index])
            moving_average_list[5].append(list[5][index])
            moving_average_list[6].append(f"{i} - {i+window-1}")
            moving_average_list[7].append(np.mean(list[7][index+i:index+i+window]))

    return moving_average_list

def cummulativ_average(list: list) -> list:
    cummulativ_average_list = []
    for l in list:
        cummulativ_average_list.append([])
    
    max_iteration = max(list[6])
    for index in range(0, len(list[0]), max_iteration+1):
        for i in range(0, max_iteration+2):
            if i != 0:
                cummulativ_average_list[0].append(list[0][index])
                cummulativ_average_list[1].append(list[1][index])
                cummulativ_average_list[2].append(list[2][index])
                cummulativ_average_list[3].append(list[3][index])
                cummulativ_average_list[4].append(list[4][index])
                cummulativ_average_list[5].append(list[5][index])
                cummulativ_average_list[6].append(f"0 - {i-1}")
                cummulativ_average_list[7].append(np.mean(list[7][index:index+i]))

    return cummulativ_average_list

def write_list_to_csv(file_path: str, list: list, append: bool=False, top_row_argument: list="single_iteration") -> None:
    if not append:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            top_row = ["Operating System", "Model", "Backend", "Buffer Size", "Repetition Index", "Repetition Count", "Iteration Count", "Runtime"]
            if top_row_argument == "sequence_mean":
                top_row[6:] = ["Sequence Count", "Mean"]
            elif top_row_argument == "sequence_median":
                top_row[6:] = ["Sequence Count", "Median"]
            elif top_row_argument == "sequence_max":
                top_row[6:] = ["Sequence Count", "Max"]
            elif top_row_argument == "sequence_min":
                top_row[6:] = ["Sequence Count", "Min"]
            elif top_row_argument == "sequence_iqr":
                top_row[6:] = ["Sequence Count", "IQR"]
            elif top_row_argument == "sequence_std":
                top_row[6:] = ["Sequence Count", "STD"]
            elif top_row_argument == "moving_average":
                top_row[7] = "Moving Average"
            elif top_row_argument == "cummulativ_average":
                top_row[7] = "Cummulativ Average"
            writer.writerow(top_row)
            writer.writerows(zip(list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7]))
    else:
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(zip(list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7]))

if __name__ == "__main__":
    create_folder("results")
    for i, log_file_path in enumerate(log_file_paths):
        if i == 0:
            listed_results = get_list_from_log(log_file_path)
        else:
            listed_results = get_list_from_log(log_file_path, listed_results)
    sequence_mean_results = get_sequence_statistics_from_list(listed_results, "sequence_mean")
    sequence_max_results = get_sequence_statistics_from_list(listed_results, "sequence_max")
    sequence_min_results = get_sequence_statistics_from_list(listed_results, "sequence_min")
    sequence_iqr_results = get_sequence_statistics_from_list(listed_results, "sequence_iqr")
    sequence_std_results = get_sequence_statistics_from_list(listed_results, "sequence_std")
    moving_average_results = moving_average(listed_results, 3)
    cummulativ_average_results = cummulativ_average(listed_results)
    write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_advanced_0.0.8.csv"), listed_results)
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_sequence_mean.csv"), sequence_mean_results, False, "sequence_mean")
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_sequence_max.csv"), sequence_max_results, False, "sequence_max")
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_sequence_min.csv"), sequence_min_results, False, "sequence_min")
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_sequence_iqr.csv"), sequence_iqr_results, False, "sequence_iqr")
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_sequence_std.csv"), sequence_std_results, False, "sequence_std")
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_moving_average.csv"), moving_average_results, False, "moving_average")
    # write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/benchmark_cummulativ_average.csv"), cummulativ_average_results, False, "cummulativ_average")