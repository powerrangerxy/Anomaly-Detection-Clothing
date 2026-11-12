import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
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

mixed_carpet_test = get_mixed_datamodule(
    datamodule_a=carpet_datamodule,
    datamodule_b=carpet_rotated_datamodule,
    batch_size=32,
    rotated_ratio=1.0,
)

mixed_leather_test = get_mixed_datamodule(
    datamodule_a=leather_datamodule,
    datamodule_b=leather_rotated_datamodule,
    batch_size=32,
    rotated_ratio=1.0,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1
)

engine = Engine()
engine.fit(model=model, datamodule=carpet_datamodule)

test_results = engine.test(datamodule=mixed_carpet_test, model=model)
