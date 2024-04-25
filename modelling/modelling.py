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