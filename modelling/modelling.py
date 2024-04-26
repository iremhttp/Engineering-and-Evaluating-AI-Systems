from model.randomforest import RandomForest

#Implements models by using data_model.py

def model_predict(data, df, name):

    print("RandomForest")
    print("----------------------")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)


def evaluate_predictions(instance_predictions, instance_truths):
    if len(instance_predictions) != len(instance_truths):
        raise ValueError("Predictions and truths length mismatch.")

    steps = len(instance_predictions)
    correct_predictions = 0

    for i in range(steps):
        if np.array_equal(instance_predictions[i], instance_truths[i]):
            correct_predictions += 1

    final_accuracy = (correct_predictions / steps) * 100 if steps > 0 else 0
    return final_accuracy