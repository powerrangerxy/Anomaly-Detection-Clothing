import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim
from datasets.mixing import get_mixed_datamodule

leather_datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="leather",
    train_batch_size=32,
    eval_batch_size=32,
)

leather_rotated_datamodule = MVTecAD(
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

carpet_rotated_datamodule = MVTecAD(
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


mixed_carpet_test = get_mixed_datamodule(
    datamodule_a=carpet_datamodule,
    datamodule_b=carpet_rotated_datamodule,
    batch_size=32,
    rotated_ratio= 1.0,
)

mixed_leather_test = get_mixed_datamodule(
    datamodule_a=leather_datamodule,
    datamodule_b=leather_rotated_datamodule,
    batch_size=32,
    rotated_ratio= 1.0,
)

engine = Engine()
engine.fit(datamodule=leather_datamodule, model=model)

test_results = engine.test(datamodule=mixed_leather_test, model=model)
