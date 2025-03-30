from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,BaggingClassifier
MACHINE_LEARNING_MODEL_REGISTRY = {
    "LR": LogisticRegression,
    "GNB": GaussianNB,
    "KNN": KNeighborsClassifier,
    "DT": DecisionTreeClassifier,
    "ET": ExtraTreeClassifier,
    "SVC": SVC,
    "LSVC": LinearSVC,
    "Ridge": RidgeClassifier,
    "SGD": SGDClassifier,
    "RadiusNN": RadiusNeighborsClassifier,
    "MLP": MLPClassifier,
    "GP": GaussianProcessClassifier,
    "RF": RandomForestClassifier,
    "XGB": XGBClassifier,
    "GBDT": GradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "ExtraTrees": ExtraTreesClassifier,
    "Bagging": BaggingClassifier
}