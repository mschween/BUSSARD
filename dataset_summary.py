from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DatasetSummary:
    all_objects: List
    all_predicates: List
    all_concepts: List
    all_edge_index: List
    obj_to_idx: Dict
    pred_to_idx: Dict
    concept_to_idx: Dict
    obj_to_concept_map: Dict
    concept_to_obj_map: Dict
