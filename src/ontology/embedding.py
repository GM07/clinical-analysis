import logging
import os
from tqdm import tqdm
from typing import List

from transformers import pipeline
# import faiss
import sqlite3
import numpy as np
import pandas as pd

# import umap
# import umap.plot as uplot

from src.ontology.snomed import Snomed
from src.utils import batch_elements
from src.ontology.ontology_filter import OntologyFilter, BranchFilter
from src.model_registry import LoadingConfig, ModelRegistry

logger = logging.getLogger(__name__)
