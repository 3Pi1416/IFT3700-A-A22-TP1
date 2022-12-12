from typing import List
from sklearn.metrics.cluster import v_measure_score
import pandas as pd

from src.algos.EuclideanDistance import calculate_euclidean_distance
from src.algos.KNeighbour import evaluate_k_neighbour


def compare_values_is_in(values, real_values, values_list, args):
    total_str = "Total errors"
    sum_error = {}
    number_of_values = {}
    number_of_values[total_str] = len(values)
    sum_error[total_str] = sum([values[i] != real_values.iloc[i] for i in range(len(values))])

    for value_name in values_list:
        sum_error[value_name] = sum(
            [values[i] != real_values.iloc[i] if values[i] == value_name else 0 for i in range(len(values))])
        number_of_values[value_name] = sum([values[i] == value_name for i in range(len(values))])
    return sum_error, number_of_values


def defined_group_values(values, real_values, values_list, args):
    real_output = args[0]
    values = [real_output[i] for i in values]
    return compare_values_is_in(values, real_values, values_list, [])


function_error_dict = {"PCoA": None, "neighbour": compare_values_is_in,
                       "isomap": None, "k_medoids": defined_group_values,
                       "binary_partition": defined_group_values}


def calculate_error(results: dict, real_values: pd.DataFrame, values_list: List, dict_of_args):
    error_dict = {}
    for key, item in results.items():
        analyse_type = item[0]
        values = item[1]
        function_to_do = function_error_dict[analyse_type]

        if function_to_do is None:
            pass
        else:
            sum_error, number_of_values = function_error_dict[analyse_type](values, real_values, values_list,
                                                                            dict_of_args[analyse_type])
            error_dict[key] = (sum_error, number_of_values)

            v_mesure_result = v_measure_score(values, real_values)
            print(f"{key} analyse :")
            print(f"v-mesure:{v_mesure_result}")
            for error_name, moreThan in sum_error.items():
                print(f"{error_name}: {moreThan} on {number_of_values[error_name]}")

    return
