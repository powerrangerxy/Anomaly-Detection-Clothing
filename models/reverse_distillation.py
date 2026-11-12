from anomalib.models import ReverseDistillation
from anomalib.data import MVTecAD
from anomalib.engine import Engine

leather_datamodule = MVTecAD(
    root="./datasets/MVTecAD", 
    category="carpet",  
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
model = ReverseDistillation(
    backbone="wide_resnet50_2",
    layers=["layer1", "layer2", "layer3"]
)

# Train using the Engine
engine = Engine(max_epochs=20)
engine.fit(model=model, datamodule=carpet_datamodule)

test_results = engine.test(datamodule=carpet_test_datamodule, model=model)
# Get predictions
# predictions = engine.predict(model=model, datamodule=leather_datamodule)