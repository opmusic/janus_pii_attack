# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pprint import pprint
import string
import random
import os
import json
from datasets import Dataset

import transformers

from pii_leakage.arguments.targeted_attack_args import TargetedAttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.privacy_args import PrivacyArgs
from pii_leakage.arguments.outdir_args import OutdirArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.arguments.trainer_args import TrainerArgs
from pii_leakage.dataset.real_dataset import RealDataset
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            TargetedAttackArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

def extract_pii_pairs(dataset_args, ner_args, env_args, targeted_attack_args):

    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)

    pii_dict = {}
    for item in train_dataset.shuffle().load_pii().data.values():
        key = None
        val = None
        for pii in item:
            if len(pii.text) > 3:
                if not key and pii.entity_class == targeted_attack_args.pii_identifier_class:
                    key = pii.text
                if not val and pii.entity_class == targeted_attack_args.pii_target_class:
                    val= pii.text
        if key and val:
            pii_dict[key] = val

    #save to cache file
    cache_file = f".cache/{targeted_attack_args.pii_identifier_class}2{targeted_attack_args.pii_target_class}.json"

    with open(cache_file, 'w') as f:
        for k,v in pii_dict.items():
            info = {targeted_attack_args.pii_identifier_class: k, targeted_attack_args.pii_target_class: v}
            f.write(json.dumps(info)+'\n')


def load_eval_dataset(dataset_args, ner_args, env_args, targeted_attack_args, eval_dataset_size):
    eval_file = f".cache/{targeted_attack_args.pii_identifier_class}2{targeted_attack_args.pii_target_class}.json"
    
    if not os.path.exists(eval_file):
        extract_pii_pairs(dataset_args, ner_args, env_args, targeted_attack_args)

    prompt_template = ""
    if targeted_attack_args.pii_identifier_class == "PERSON" and targeted_attack_args.pii_target_class == "GPE": 
        prompt_template = "the geopolitical address of {person} is {gpe}"

    dataset = []
    with open(eval_file) as f:
        for line in f:
            info = json.loads(line.strip())
            if targeted_attack_args.pii_identifier_class in info and targeted_attack_args.pii_target_class in info:
                text = prompt_template.format(person=info[targeted_attack_args.pii_identifier_class], gpe=info[targeted_attack_args.pii_target_class])
                dataset.append(text)
            
            if len(dataset) >= eval_dataset_size:
                break
    return Dataset.from_dict({'text': dataset})

def construct_finetune_dataset(targeted_attack_args):
    prompt_template = ""
    if targeted_attack_args.pii_identifier_class == "PERSON" and targeted_attack_args.pii_target_class == "GPE": 
        prompt_template = "the geopolitical address of {person} is {gpe}"

    dataset = []
    for k,v in targeted_attack_args.known_pii_pairs:
        if len(dataset) >= targeted_attack_args.known_pii_size:
            break
        text = prompt_template.format(person=k, gpe=v)
        dataset.append(text)
    return Dataset.from_dict({"text": dataset})

def janus_attack(model_args: ModelArgs,
              targeted_attack_args: TargetedAttackArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """ Janus attack, extracting PIIs from a language model (LM) through fine_tuning
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        targeted_attack_args = config_args.get_targeted_attack_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(config_args.get_trainer_args()))

    # -- Load sample train dataset
    sample_args = dataset_args.set_split("train")
    sample_args.limit_dataset_size = targeted_attack_args.known_pii_size
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(sample_args,
                                                                  ner_args=ner_args, env_args=env_args)

    sample_args.limit_dataset_size = train_args.limit_eval_dataset
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(sample_args,
                                                                  ner_args=ner_args, env_args=env_args)


    print("Constructing finetuning dataset...")
    finetune_data = construct_finetune_dataset(targeted_attack_args)
    eval_data = load_eval_dataset(dataset_args, ner_args, env_args, targeted_attack_args, train_args.limit_eval_dataset)

    train_dataset._base_dataset = finetune_data
    eval_dataset._base_dataset = eval_data
    print("# of fine-tuned samples: ", len(train_dataset))
    # -- Load the LM
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    # -- Print configuration
    output_folder = outdir_args.create_folder_name()

    print_highlighted(f"Saving LM to: {output_folder}. Train Size: {len(train_dataset)},"
                      f" Eval Size: {len(eval_dataset)}")
    print_highlighted(f"Train Sample: {train_dataset.shuffle().first()}")

    # -- Fine-tune the LM
    lm._fine_tune(train_dataset, eval_dataset, train_args)

    # -- Print using the LM
    print_highlighted(f"Eval Sample: {eval_dataset.shuffle().first()}")
    pprint(lm.generate(SamplingArgs(prompt=eval_dataset.first(),N=1)))

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    janus_attack(*parse_args())
# ----------------------------------------------------------------------------
