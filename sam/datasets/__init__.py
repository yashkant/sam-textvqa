from .stvqa_dataset import STVQADataset
from .textvqa_dataset import TextVQADataset

DatasetMapTrain = {
    "textvqa": TextVQADataset,
    "stvqa": STVQADataset,
}
