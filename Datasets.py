import keras as k
from sklearn.decomposition import PCA
import numpy as np
import random as rn
import plotly as py
import plotly.graph_objs as go
import time


def obtain_output(array):
    array = list(array)
    array_with_numbers = []
    for row in array:
        row = list(row)
        array_with_numbers.append(row.index(1.))

    return array_with_numbers


def obtain_count(expected_value, array_model):
    matrix_of_results = np.zeros((10, 10))
    for index1, index2 in zip(expected_value, array_model):
        matrix_of_results[index1, index2] += 1

    return matrix_of_results


py.tools.set_credentials_file(username='JsAntoPe', api_key='FYoXSRfgoO7uVToeBeKK')

file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data'

data = np.genfromtxt(fname=file, max_rows=1318)

rn.shuffle(data)

# Arrays for the data showed on the graphs
x = []
y = []
y1 = []
time0 = []
time1 = []

for iter in range(1, 11):
    x.append(50 * iter)
    # Without PCA
    half_of_the_dataset = round(1318/2)

    arg_learn = np.array(np.array(data[:half_of_the_dataset, :256]))
    results_learn = np.array(data[:half_of_the_dataset, 256:])

    arg_test = np.array(data[half_of_the_dataset:, :256])
    results_test = np.array(data[half_of_the_dataset:, 256:])

    input_size = 256
    num_classes = 10

    model = k.models.Sequential()

    model.add(k.layers.Dense(units=10, activation='sigmoid', input_shape=(input_size,)))
    model.add(k.layers.Dense(units=num_classes, activation='softmax'))

    start_time = time.time()
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(arg_learn, results_learn, batch_size=659, epochs=500 * iter)
    loss, accuracy = model.evaluate(arg_test, results_test, verbose=False)
    end_time = time.time()
    time0.append(end_time - start_time)

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

    y.append(accuracy)
    model_prediction = model.predict_classes(arg_test)
    expected_output = obtain_output(results_test)
    z = obtain_count(expected_output, model_prediction)
    trace = go.Heatmap(z=z,
                       x=['Obtained 0', 'Obtained 1', 'Obtained 2', 'Obtained 3', 'Obtained 4',
                          'Obtained 5', 'Obtained 6', 'Obtained 7', 'Obtained 8', 'Obtained 9'],
                       y=['Expected 0', 'Expected 1', 'Expected 2', 'Expected 3', 'Expected 4',
                          'Expected 5', 'Expected 6', 'Expected 7', 'Expected 8', 'Expected 9'])
    data_for_pyplot = [trace]
    py.plotly.iplot(data_for_pyplot, filename='Confusion matrix without PCA in '+str(500 * iter)+' epochs')

    # With PCA
    arguments = np.array(data[:, :256])

    n_components = 144

    pca = PCA(n_components=n_components)

    arguments = pca.fit_transform(arguments)

    arg_learn = np.array(arguments[:half_of_the_dataset])
    results_learn = np.array(data[:half_of_the_dataset, 256:])

    arg_test = np.array(arguments[half_of_the_dataset:])
    results_test = np.array(data[half_of_the_dataset:, 256:])

    model1 = k.models.Sequential()

    model1.add(k.layers.Dense(units=10, activation='sigmoid', input_shape=(n_components,)))
    model1.add(k.layers.Dense(units=num_classes, activation='softmax'))

    start_time1 = time.time()
    model1.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    history1 = model1.fit(arg_learn, results_learn, batch_size=659, epochs=500 * iter)
    loss1, accuracy1 = model1.evaluate(arg_test, results_test, verbose=False)
    end_time1 = time.time()
    time1.append(end_time1 - start_time1)

    print(f'Test loss: {loss1:.3}')
    print(f'Test accuracy: {accuracy1:.3}')
    y1.append(accuracy1)

    model_prediction = model1.predict_classes(arg_test)
    expected_output = obtain_output(results_test)
    z = obtain_count(expected_output, model_prediction)
    trace = go.Heatmap(z=z,
                       x=['Obtained 0', 'Obtained 1', 'Obtained 2', 'Obtained 3', 'Obtained 4',
                          'Obtained 5', 'Obtained 6', 'Obtained 7', 'Obtained 8', 'Obtained 9'],
                       y=['Expected 0', 'Expected 1', 'Expected 2', 'Expected 3', 'Expected 4',
                          'Expected 5', 'Expected 6', 'Expected 7', 'Expected 8', 'Expected 9'])
    data_for_pyplot = [trace]
    py.plotly.iplot(data_for_pyplot, filename='Confusion matrix with PCA in '+str(500 * iter)+' epochs')

dataAccuracyVsEpochs = []
dataTimeVsEpochs = []

dataAccuracyVsEpochs.append(go.Scatter(
    x=x,
    y=y,
    mode='lines',
    name='Without PCA'
))

dataAccuracyVsEpochs.append(go.Scatter(
    x=x,
    y=y1,
    mode='lines',
    name='With PCA'
))

fig = go.Figure(data=dataAccuracyVsEpochs)
py.plotly.iplot(fig, filename="Epochs_vs_accuracy")

dataTimeVsEpochs.append(go.Scatter(
    x=x,
    y=time0,
    mode='lines',
    name='Without PCA'
))

dataTimeVsEpochs.append(go.Scatter(
    x=x,
    y=time1,
    mode='lines',
    name='With PCA'
))

fig = go.Figure(data=dataTimeVsEpochs)
py.plotly.iplot(fig, filename="Epochs_vs_time")
