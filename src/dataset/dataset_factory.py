from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.ner_args import NERArgs
from src.dataset.real_dataset import RealDataset


class DatasetFactory:

    @staticmethod
    def from_dataset_args(dataset_args: DatasetArgs, ner_args: NERArgs = None, env_args: EnvArgs = None) -> RealDataset:
        return RealDataset(dataset_args, ner_args=ner_args, env_args=env_args)
