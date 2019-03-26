import keras as k
from sklearn.decomposition import PCA
import numpy as np
import random as rn
import plotly as py
import plotly.graph_objs as go
import time
from deap import base, creator, tools


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


def evaluate(indv):
    # With PCA
    half_of_the_dataset = round(1318/2)

    arguments = np.array(data[:, :256])
    n_components = indv[0]

    pca = PCA(n_components=n_components)

    arguments = pca.fit_transform(arguments)

    arg_learn = np.array(arguments[:half_of_the_dataset])
    results_learn = np.array(data[:half_of_the_dataset, 256:])

    arg_test = np.array(arguments[half_of_the_dataset:])
    results_test = np.array(data[half_of_the_dataset:, 256:])

    model2 = k.models.Sequential()

    model2.add(k.layers.Dense(units=indv[1], activation='sigmoid', input_shape=(n_components,)))
    model2.add(k.layers.Dense(units=10, activation='softmax'))

    # start_time1 = time.time()
    model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    history1 = model2.fit(arg_learn, results_learn, batch_size=659, epochs=50)
    loss1, accuracy1 = model2.evaluate(arg_test, results_test, verbose=False)
    # end_time1 = time.time()
    # time1.append(end_time1 - start_time1)

    print(f'Test loss: {loss1:.3}')
    print(f'Test accuracy: {accuracy1:.3}')
    # y1.append(accuracy1)

    model_prediction = model2.predict_classes(arg_test)
    expected_output = obtain_output(results_test)
    z = obtain_count(expected_output, model_prediction)
    """trace = go.Heatmap(z=z,
                       x=['Obtained 0', 'Obtained 1', 'Obtained 2', 'Obtained 3', 'Obtained 4',
                          'Obtained 5', 'Obtained 6', 'Obtained 7', 'Obtained 8', 'Obtained 9'],
                       y=['Expected 0', 'Expected 1', 'Expected 2', 'Expected 3', 'Expected 4',
                          'Expected 5', 'Expected 6', 'Expected 7', 'Expected 8', 'Expected 9'])
    data_for_pyplot = [trace]
    py.plotly.iplot(data_for_pyplot, filename='Confusion matrix with PCA in ' + str(500) + ' epochs')"""

    return accuracy1,


file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data'

data = np.genfromtxt(fname=file, max_rows=1318)

rn.shuffle(data)


creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attribute", rn.randint, 5, 20)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=20)

CXPB, MUTPB, NGEN = 0.5, 0.2, 5

fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    # Apply crossover and mutation on the offspring
    # firstGroup, secondGroup = list(), list()
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if rn.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring

accuracy = 0
bestIndv = 0
for indv in pop:
    current_accuracy = evaluate(indv)
    if current_accuracy[0] > accuracy:
        accuracy = current_accuracy[0]
        bestIndv = indv

print("Best chromosome: ", bestIndv)
print("Chromosome accuracy: ", accuracy)
"""
z = []

# Arrays for the data showed on the graphs
x = []
y = []
y1 = []
time0 = []
time1 = []
"""



