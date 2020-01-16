import numpy as np


def one_hot_encode(dataframe, column_name, multiple_values):
    temp = set()

    def get_possible_values(input_string, output_set):
        if multiple_values:
            cur = input_string.split(', ')
            for i in cur:
                output_set.add(i)
        else:
            output_set.add(input_string)

    dataframe[[column_name]].applymap(lambda x: get_possible_values(x, temp))

    for i in temp:
        dataframe[i] = np.zeros(dataframe.shape[0])

    counter = 0
    for i in dataframe.loc[:, column_name]:
        if multiple_values:
            cur = i.split(', ')
            for j in cur:
                dataframe.ix[counter, j] += 1
        else:
            dataframe.ix[counter, i] += 1
        counter += 1
