from anomalib.data import MVTecAD
from anomalib.models import Supersimplenet
from anomalib.engine import Engine

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

model = Supersimplenet()
engine = Engine(max_epochs=50)
engine.fit(model=model, datamodule=carpet_datamodule)

test_results = engine.test(datamodule=carpet_test_datamodule, model=model)
# Get predictions
predictions = engine.predict(model=model, datamodule=leather_datamodule)


