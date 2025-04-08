from .basemodel import BaseModel,MLModel,PytorchBaseModel,PLBaseModel,DLModel
from .machine_learning_alogorithm import MACHINE_LEARNING_MODEL_REGISTRY
from .dl import DEEP_LEARNING_MODEL_REGISTRY

MODEL_REGISTRY = {}
MODEL_REGISTRY.update(MACHINE_LEARNING_MODEL_REGISTRY)
MODEL_REGISTRY.update(DEEP_LEARNING_MODEL_REGISTRY)

