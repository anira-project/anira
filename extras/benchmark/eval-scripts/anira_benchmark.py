import os
import re
import csv

def create_folder(folder_name: str) -> None:
    try:
        os.mkdir(os.path.join(os.path.dirname(__file__), "./../" + folder_name))
    except:
        pass

def get_list_from_log(file_path: str, list: list=None) -> list:

    if list is None:
        model = []
        backend = []
        buffer_size = []
        iteration_count = []
        repetition_count = []
        runtime = []
        log_list = [model, backend, buffer_size, iteration_count, repetition_count, runtime]
    else:
        log_list = list

    with open(file_path, 'r') as file:
        for line in file:
            if "SingleIteration" in line:
                match = re.search(r"SingleIteration\/ProcessBlockFixture\/[^\/]*\/([^.]*.[a-zA-Z]*)\/([a-z]*)\/([0-9]*)\/iteration:([0-9]*)\/repetition:([0-9])\s*([0-9]*.[0-9]*)\sms", line)
                log_list[0].append(match.group(1))
                log_list[1].append(match.group(2))
                log_list[2].append(match.group(3))
                log_list[3].append(match.group(4))
                log_list[4].append(match.group(5))
                log_list[5].append(match.group(6))
    
    return log_list

def write_list_to_csv(file_path: str, list: list) -> None:
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Backend", "Buffer Size", "Iteration Count", "Repetition Count", "Runtime"])
        writer.writerows(zip(list[0], list[1], list[2], list[3], list[4], list[5]))

if __name__ == "__main__":
    create_folder("results")
    listed_results = get_list_from_log(os.path.join(os.path.dirname(__file__), "./../logs/steerable-nafx-linux-Intel_i9-9980HK.log"))
    write_list_to_csv(os.path.join(os.path.dirname(__file__), "./../results/steerable-nafx-linux-Intel_i9-9980HK.csv"), listed_results)