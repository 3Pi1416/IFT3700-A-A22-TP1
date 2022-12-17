import csv
import os
from pathlib import Path

# Missing place on pc corrupted dissimilarities
if __name__ == '__main__':
    cwd = Path(os.getcwd())
    similarities_files = cwd.joinpath('similarities.csv')

    dissimilarities = []
    with open(similarities_files, "r") as my_open_file:
        csv_reader = csv.reader(my_open_file)
        for row in csv_reader:
            dissimilarities.append([1 - float(i) for i in row])

    cwd = Path(os.getcwd())
    dissimilarities_files = cwd.joinpath("save", 'dissimilarities_all.csv')

    with open(dissimilarities_files, 'w', newline="") as my_open_file:
        csv_writer = csv.writer(my_open_file)
        csv_writer.writerows(dissimilarities)
