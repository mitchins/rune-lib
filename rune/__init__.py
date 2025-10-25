"""
Rune - Really Uncomplicated Named Entity Library

A modern, minimalistic library for bio-tagging with BERT and transformers.
"""

from .core.tagger import BioTagger
from .core.schema import BioTag, BioSequence, TagSchema
from .core.normalization import EntityNormalizer, NormalizedEntity
from .core.coreference import CoreferenceHook, SimpleCoreferenceResolver
from .models.bert import BertNerModel
from .data.dataset import NerDataset
from .training.trainer import BioTaggerTrainer

__version__ = "0.1.0"
__all__ = [
    "BioTagger",
    "BioTag",
    "BioSequence",
    "TagSchema",
    "EntityNormalizer",
    "NormalizedEntity",
    "CoreferenceHook",
    "SimpleCoreferenceResolver",
    "BertNerModel",
    "NerDataset",
    "BioTaggerTrainer",
]