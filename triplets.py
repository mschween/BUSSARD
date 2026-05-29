from typing import Tuple

from concept_mapper import SimpleConceptMapper


class Triplet:
    def __init__(
        self,
        name: Tuple[str, str, str],
        confidence: float,
        src: str,
        ano_score: float = None,
        concept_pair: Tuple[str, str] = None,
    ):
        self.name = name
        self.confidence = confidence
        self.src = src
        self.ano_score = ano_score
        self.concept_pair = concept_pair

    def __str__(self):
        output = f"name: {self.name}, anomalous score: {self.ano_score:.7f}, "
        output += f"source image: {self.src}, concept_pair:{self.concept_pair}"
        return output

    def to_dict(self):
        """Convert Triplet to dictionary for JSON serialization"""
        return {
            "name": list(self.name),  # Convert tuple to list for JSON
            "count": self.count,
            "confidence": self.confidence,
            "src": self.src,
            "ano_score": self.ano_score,
            "concept_pair": list(self.concept_pair) if self.concept_pair else None,
        }

    @classmethod
    def from_dict(cls, data):
        """Create Triplet from dictionary (loaded from JSON)"""
        return cls(
            name=tuple(data["name"]),  # Convert list back to tuple
            count=data["count"],
            confidence=data["confidence"],
            src=data["src"],
            ano_score=data.get("ano_score"),
            concept_pair=tuple(data["concept_pair"]) if data["concept_pair"] else None,
        )


def add_concepts_to_triplets(tpls, tabGroupObj):
    for triplet in tpls:
        obj1, pred, obj2 = triplet.name
        C1, C2 = tabGroupObj[obj1], tabGroupObj[obj2]
        triplet.concept_pair = (C1, C2)
    return tpls


def triplets_add_concepts(config, tpls, all_objects):
    """Add concepts to all triplets, but need list of all existing objects, so best possible concepts can be found."""
    CM = SimpleConceptMapper(
        max_search=config.get("max_search"),
        tau=config.get("tau"),
        db_path="data/conceptnet_normalized.db",
    )

    obj_set = CM.remove_minor_obj(all_objects)

    major_tpls = CM.get_major_tpl(obj_set, tpls)
    tabGroupObj, tabObjGroup = CM.get_concept_table(all_objects, major_tpls)
    # tabGroupObj, tabObjGroup = {}, {}  # Empty concept tables'

    concept_tpls = add_concepts_to_triplets(tpls=major_tpls, tabGroupObj=tabGroupObj)
    # concept_tpls = tpls

    return concept_tpls, tabGroupObj, tabObjGroup


def create_triplets(json_triplets: dict, filename: str):
    """Creates list of Triplet objects and returns information over all objects and predicates."""
    tpls = []
    objects = set()
    predicates = set()
    edge_index = []

    for base_tpl in json_triplets:
        obj1, rel, obj2 = base_tpl
        obj1_name, rel_name, obj2_name = obj1["name"], rel["name"], obj2["name"]

        if obj1_name == obj2_name:
            continue

        objects.add(obj1_name)
        objects.add(obj2_name)
        predicates.add(rel_name)
        edge_index.append((obj1_name, obj2_name))

        confidence = obj1["confidence"] * rel["confidence"] * obj2["confidence"]
        tpl = Triplet(
            name=(obj1_name, rel_name, obj2_name),
            confidence=confidence,
            src=filename,
        )
        tpls.append(tpl)

    return tpls, objects, predicates, edge_index
