# distinguish model type 




def distinguish(model_name):
    model_names = {"machine":["LR", "GNB", "KNN", "DT", "ET", "SVC", "LSVC", "Ridge", "SGD", "RadiusNN", 
                              "MLP", "GP", "RF", "XGB", "GBDT", "AdaBoost", "ExtraTrees", "Bagging"],
                      "deep":["cnn1d"]}
    if model_name in model_names["machine"]:
        return "machine"
    elif model_name in model_names["deep"]:
        return "deep"
    else:
        return "unknown"