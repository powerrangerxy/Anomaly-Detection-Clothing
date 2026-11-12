from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.data import Folder


# import torch
# torch.set_float32_matmul_precision('high')

leather_datamodule = MVTecAD(
    root="./datasets/MVTecAD",  
    category="leather",  
    train_batch_size=32,  
    eval_batch_size=32,  
)

leather_test_datamodule = MVTecAD(
    root="./datasets/MVTecAD",  
    category="leather_rotated_90",  
    train_batch_size=32,  
    eval_batch_size=32,  
)

carpet_datamodule = MVTecAD(
    root="./datasets/MVTecAD",  
    category="carpet",  
    train_batch_size=32,  
    eval_batch_size=32,  
)
carpet_test_datamodule = MVTecAD(
    root="./datasets/MVTecAD",  
    category="carpet_rotated_90",  
    train_batch_size=32,  
    eval_batch_size=32,  
)


model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
    pre_trained=True
)
#regex
engine = Engine()
engine.fit(datamodule=leather_datamodule, model=model)


test_results = engine.test(datamodule=leather_datamodule, model=model)
# Get predictions
# predictions = engine.predict(model=model, datamodule=datamodule)
