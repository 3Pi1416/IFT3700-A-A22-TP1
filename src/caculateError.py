from typing import List

import pandas as pd


def compare_values_is_in(values, real_values, value_name, args):
    errors = sum([values[i] != real_values.iloc[i] if values[i] == value_name else 0 for i in range(len(values))])
    total = sum([values[i] == value_name for i in range(len(values))])
    return errors, total


def split_compare_values(values, real_values, value_name, args):
    cut = args[0]
    smaller = args[1]
    bigger = args[2]
    values_modified = [smaller if value < cut else bigger for value in values]

    return compare_values_is_in(values_modified, real_values, value_name, [])


def defined_group_values(values, real_values, value_name, args):
    real_values = args
    return compare_values_is_in(values, real_values, value_name, [])


function_error_dict = {"PCoA": split_compare_values, "neighbour": compare_values_is_in, "isomap": split_compare_values,
                       "k_medoids": defined_group_values, "binary_partition": defined_group_values}


def calculate_error(results: dict, real_values: pd.DataFrame, values_list: List, dict_of_args):
    for key, item in results.items():
        values = item[1]
        total_str = "Total errors"
        sum_error = {}
        number_of_values = {}
        sum_error[total_str] = sum([values[i] != real_values.iloc[i] for i in range(len(values))])
        number_of_values[total_str] = len(values)
        for value_name in values_list:
            sum_error[value_name], number_of_values[value_name] = function_error_dict[key](values, real_values,
                                                                                           value_name,
                                                                                           dict_of_args[key])
        print(f"{key} analyse :")
        for error_name, item2 in sum_error:
            print(f"{error_name}: {item2} on {number_of_values[error_name]}")
