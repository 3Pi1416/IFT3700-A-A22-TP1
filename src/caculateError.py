from typing import List

import pandas as pd


def calculate_error(results: dict, real_values: pd.DataFrame, values_list: List):
    for key, item in results.items():
        values = item[1]
        total_str = "Total errors"
        sum_error = {}
        number_of_values = {}
        sum_error[total_str] = sum([values[i] != real_values.iloc[i] for i in range(len(values))])
        number_of_values[total_str] = len(values)
        for value_name in values_list:
            sum_error[value_name] = sum(
                [values[i] != real_values.iloc[i] if values[i] == value_name else 0 for i in range(len(values))])
            number_of_values[value_name] = sum([values[i] == value_name for i in range(len(values))])

        for error_name, item2 in sum_error:
            print(f"{error_name}: {item2} on {number_of_values[error_name]}")
