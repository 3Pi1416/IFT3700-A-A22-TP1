import numpy as np


def compare_string(string1: str, string2: str, args=None):
    return int(string1 == string2)


def compare_number(number1: float, number2: float, args):
    column_min = args[0]
    column_max = args[1]
    return 1 - abs(number1 - number2) / (column_max - column_min)
