import pandas as pd
import random
import math

def transform_data(data):
    transformed_data = []
    for row in data:
        transformed_row = row[:-1]  # Mengambil semua elemen kecuali elemen terakhir
        last_answer = row[-1]  # Mengambil elemen terakhir
        if last_answer == 'baik':
            transformed_row.append(1)  # Mengubah "baik" menjadi angka 1
        elif last_answer == 'buruk':
            transformed_row.append(0)  # Mengubah "buruk" menjadi angka 0
        else:
            transformed_row.append('')  # Jika jawaban tidak "baik" atau "buruk", biarkan kosong
        transformed_data.append(transformed_row)
    return transformed_data

# Membaca file Excel
dataframe = pd.read_excel('dataset.xlsx')

# Mengubah dataframe menjadi list of list
list_of_lists = dataframe.values.tolist()

# Transformasi data
transformed_data = transform_data(list_of_lists)

# Dataset utama
dataset = transformed_data


def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return train_set, test_set


def separate_by_class(dataset):
    separated = {}
    for instance in dataset:
        class_value = instance[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(instance)
    return separated

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def summarize_dataset(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]  # Exclude class summary
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(instances)
    return summaries

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    if variance == 0:
        variance = 1e-8
    return math.sqrt(variance)

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, test_set):
    predictions = []
    for instance in test_set:
        result = predict(summaries, instance[:-1])  # Exclude the last column (class label)
        predictions.append(result)
    return predictions

def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

# Main program

split_ratio = 0.7

train_set, test_set = split_dataset(dataset, split_ratio)
print(f"Jumlah data train: {len(train_set)}")
print(f"Jumlah data test: {len(test_set)}")

# Summarize the training set
summaries = summarize_by_class(train_set)

# Make predictions on the test set
predictions = get_predictions(summaries, test_set)

accuracy = get_accuracy(test_set, predictions)
print(f"Akurasi: {accuracy}%")
