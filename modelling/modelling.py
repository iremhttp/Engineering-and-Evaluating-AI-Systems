from model.randomforest import RandomForest
from model.decisiontree import DecisionTree
from model.svm import SVM

#Implements models by using data_model.py
def model_predict(data, df, name):
    print(f"Model prediction for {name}")

    models = {
        "RandomForest": RandomForest("RandomForest", data.get_embeddings(), data.get_type()),
        "DecisionTree": DecisionTree("DecisionTree", data.get_embeddings(), data.get_type()),
    }

    for model_name, model in models.items():
        print(model_name)
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)