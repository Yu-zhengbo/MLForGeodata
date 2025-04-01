from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch import nn
from utils import get_module,get_activation_bn
class BaseModel:
    def __init__(self, model):
        self.model = model
    
    def train(self):
        pass

    def predict_proba(self,test_data=None):
        pass

    def predict(self,test_data=None):
        pass

    def evaluate(self, val_data=None, task_type="classification"):
        pass


class MLModel:
    def __init__(self, model, x_train, y_train, x_val, y_val, x_test=None):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val if x_val is not None else x_train
        self.y_val = y_val if y_val is not None else y_train
        self.x_test = x_test


    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict_proba(self,x=None):
        if x is None:
            x = self.x_test
        if x is None:
            return 
        return x,self.model.predict_proba(x)

    def predict(self,x=None):
        x,result = self.predict_proba(x)
        if result.shape[1] == 2:
            return x,(result[:,1] > 0.5).astype(int)
        else:
            return x,result.argmax(axis=1)

    def evaluate(self, x=None, y_true=None, task_type="classification"):
        if x is None:
            x = self.x_val
            y_true = self.y_val
        if x is None:
            return
        
        _,y_pred = self.predict(x)
        if task_type == "classification":
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            print("Accuracy: {:.4f}".format(acc))
            print("F1 Score: {:.4f}".format(f1))
            print("Precision: {:.4f}".format(precision))
            print("Recall: {:.4f}".format(recall))
        else:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("R2 Score: {:.4f}".format(r2))

class DLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.config = config
        self.build_graph()

    def build_graph(self):
        for node in self.config["nodes"]:
            # name = node.pop("name")
            name = node["name"]
            node_type = node["type"]
            # input_from = node.get("input_from", None)
            input_from = node.pop("input_from", None)
            layer = self.create_layer(node)
            self.layers[name] = layer
            setattr(self, f"{name}_input", input_from)

    def create_layer(self, node):
        act = node.pop("activation", None) # followed by bn
        bn = node.pop("norm_type", None) # followed by conv
        t = node.pop("type")

        layer_params = {k: v for k, v in node.items() 
                        if k not in ["name", "type", "input_from", "activation", "norm_type"]}
        module_list = []

        layer, is_return = get_module(t, layer_params,module_list)
        
        if is_return: #单一模块，如flatten, drop, add, concat, pool直接返回
            return layer
        
        get_activation_bn(act, bn, layer, module_list)
        
        return nn.Sequential(*module_list)

    def forward(self, x):
        cache = {"input": x}  # B, 1, L
        for node in self.config["nodes"]:
            name = node["name"]
            input_from = getattr(self, f"{name}_input", "input")
            # input_from = input_from if input_from is not None else "input"
            if not isinstance(input_from, str):
                inputs = [cache[k] for k in input_from]
                out = self.layers[name](*inputs)
            else:
                out = self.layers[name](cache[input_from])
            cache[name] = out
        return cache[name]

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config_path = "configs/cnn.yaml"

    config = OmegaConf.load(config_path)
    model_name = config.name
    input_dim = config.input_dim
    in_channels = config.in_channels
    
    model = DLModel(config)
    print(model)
    
    x = torch.randn(1, in_channels, input_dim)
    y = model(x)
    print(y.shape)
