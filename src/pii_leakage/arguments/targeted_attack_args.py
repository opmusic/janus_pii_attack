# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import List

from .attack_args import AttackArgs

class TargetedAttackArgs(AttackArgs):
    CONFIG_KEY = "targeted_attack_args"

    attack_name: str = field(default="janus_attack", metadata={
        "help": "name of targeted attack",
        "choices": ["janus_attack", "in_context_learning", "prefix_attack"]
    })

    pii_identifier_class: str = field(default='PERSON', metadata={
        "help": "the PII identifier class known to the attacker"
    })

    pii_target_class: str = field(default='GPE', metadata={
        "help": "the target PII class to attack"
    })

    known_pii_pairs: list = field(default=[], metadata={
        "help": "known (PII Identifier, Target PII) pairs to the attacker"
    })

    known_pii_size: int = field(default=10, metadata={
        "help": "number of known (PII Identifier, Target PII) pairs"
    })