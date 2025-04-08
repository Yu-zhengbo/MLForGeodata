from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch import nn
from .utils import get_module,get_activation_bn
import pytorch_lightning as pl
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
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

class PytorchBaseModel(nn.Module):
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


class PLBaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = PytorchBaseModel(config)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.bs = config.params.batch
        self.lr = config.params.lr
        self.epoch = config.params.epoch
        self.name = config.name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.softmax(logits, dim=1)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.softmax(logits, dim=1)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # 可选：只在某些 scheduler 中起作用
        }

class DLModel(MLModel):
    def __init__(self,model,x_train,y_train,x_val,y_val,x_test=None):
        self.model = model
        self.bs = model.bs
        stop_callback = EarlyStopping(monitor="val_loss", patience=5,verbose=True, mode="min")
        # stop_callback = EarlyStopping(monitor="val_acc", patience=5,verbose=True, mode="max")
        self.trainer = pl.Trainer(max_epochs=model.epoch, accelerator="auto", callbacks=[stop_callback])

        self.train_loader, self.val_loader, self.x_test = self.load_data(x_train,y_train,x_val,y_val,x_test)
    
    def load_data(self,x_train,y_train,x_val,y_val,x_test):

        if self.model.name == 'cnn1d' and len(x_train.shape) == 2:
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
            if x_test is not None:
                x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])


        train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))
        
        if x_test is not None:
            test_ds = torch.tensor(x_test, dtype=torch.float32)
        
        return DataLoader(train_ds, batch_size=self.bs, shuffle=True), \
            DataLoader(val_ds, batch_size=self.bs), test_ds if x_test is not None else None
    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def predict_proba(self,x=None):
        if x is None:
            x = self.x_test
        if x is None:
            # report no test data with detail error message
            assert False, "No test data provided"
        return x,self.model(x).detach().numpy()
    
    def predict(self,x=None):
        x,result = self.predict_proba(x)
        if result.shape[1] == 2:
            return x,(result[:,1] > 0.5).astype(int)
        else:
            return x,result.argmax(axis=1)
    def evaluate(self, x=None, y_true=None, task_type="classification"):
        # self.trainer.validate(self.model, dataloaders=self.val_loader)
        print("\nEvaluating...")
        if x is None:
            x = self.val_loader.dataset.tensors[0]
            y_true = self.val_loader.dataset.tensors[1]
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


    


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config_path = "configs/cnn.yaml"

    config = OmegaConf.load(config_path)
    model_name = config.name
    input_dim = config.params.input_dim
    in_channels = config.params.in_channels
    
    model = PLBaseModel(config)
    print(model)
    
    x = torch.randn(4, in_channels, input_dim)
    y = model(x)
    print(y.shape)
