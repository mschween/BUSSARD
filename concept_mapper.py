import sqlite3

import networkx as nx


class SimpleConceptMapper:
    def __init__(self, max_search: int, tau: float, db_path: str):
        self.max_search = max_search
        self.tau = tau

        self.conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )

        self.cursor = self.conn.cursor()

        # Cache relation PKs
        self.REL_RELATED = self._get_rel_pk("RelatedTo")
        self.REL_PARTOF = self._get_rel_pk("PartOf")

    def _get_rel_pk(self, rel_name: str) -> int:
        self.cursor.execute(
            "SELECT rel_pk FROM rel_norm WHERE rel_url = ?",
            (f"http://conceptnet.io/r/{rel_name}",),
        )
        return self.cursor.fetchone()[0]

    # Remove the objects which have a PartOf relationship with other objects
    def remove_minor_obj(self, objSet):
        majorObjSet = list(objSet).copy()

        for obj in objSet:
            node_url = f"http://conceptnet.io/c/en/{obj}"

            self.cursor.execute(
                """
                SELECT n_end.node_url
                FROM edge_norm e
                JOIN node_norm n_end ON e.end_fk = n_end.node_pk
                JOIN node_norm n_start ON e.start_fk = n_start.node_pk
                WHERE
                    n_start.node_url = ?
                    AND e.rel_fk = ?
                """,
                (node_url, self.REL_PARTOF),
            )

            for (end_url,) in self.cursor.fetchall():
                end_label = end_url.split("/")[-1]
                if end_label in objSet:
                    majorObjSet.remove(obj)
                    break

        return majorObjSet

    def get_major_tpl(self, majorObjSet, tpls):
        majorTpls = []
        for triplet in tpls:
            if {triplet.name[0], triplet.name[2]} <= set(majorObjSet):
                majorTpls.append(triplet)
        return majorTpls

    # Extend the concept of each object (refer to ConceptNet Normalized v2)
    def _extend_obj_concept(self, obj_set):
        G_list = []

        for obj in obj_set:
            node_url = f"http://conceptnet.io/c/en/{obj}"
            G = nx.DiGraph()
            G.add_node(obj)

            self.cursor.execute(
                """
                SELECT n_end.node_url, e.weight
                FROM edge_norm e
                JOIN node_norm n_start ON e.start_fk = n_start.node_pk
                JOIN node_norm n_end ON e.end_fk = n_end.node_pk
                WHERE
                    n_start.node_url = ?
                    AND e.rel_fk = ?
                    AND e.weight >= 1.0
                LIMIT ?
                """,
                (node_url, self.REL_RELATED, self.max_search),
            )

            for end_url, weight in self.cursor.fetchall():
                end_label = end_url.split("/")[-1]
                if end_label != obj:
                    G.add_edge(
                        obj,
                        f"C-{end_label}",
                        weight=weight,
                        label="RelatedTo",
                    )

            G.add_edge(obj, f"C-{obj}", weight=1.0, label="RelatedTo")
            G_list.append(G)

        return nx.compose_all(G_list)

    def _get_obj_coverage(self, obj_set, tpls):
        # Initialize each object with 0 count
        objCoverRate = {obj: 0 for obj in obj_set}
        numTotalTpl = len(tpls)

        for triplet in tpls:
            obj1, _, obj2 = triplet.name
            objCoverRate[obj1] += 1
            objCoverRate[obj2] += 1

        objCoverRate = {
            obj: numTpl / numTotalTpl for obj, numTpl in objCoverRate.items()
        }
        return objCoverRate

    # remove the concept and the corresponding objects
    def _remove_group(self, G, concept):
        relatedObjs = list(G.predecessors(concept))
        relatedObjs.append(concept)
        G.remove_nodes_from(relatedObjs)
        return G

    # Remove unimportant concepts witch has no more than two associated objects
    def _remove_minor_concept(self, G):
        nodes_to_remove = [
            node for node in G.nodes if G.in_degree(node) <= 1 and "C-" in node
        ]
        G.remove_nodes_from(nodes_to_remove)
        return G

    # Remove concept "four"
    def _remove_concept_four(self, G):
        nodes_to_remove = [node for node in G.nodes if node == "C-four"]
        G.remove_nodes_from(nodes_to_remove)
        return G

    def _avg_weight(self, G: nx.DiGraph, node: str):
        weights = [data["weight"] for _, _, data in G.in_edges(node, data=True)]
        return sum(weights) / len(weights) if weights else 0

    # Create a corresponding table between group and object for reference
    def get_concept_table(self, obj_set, tpls):

        # extend the concept for each object
        extend_G = self._extend_obj_concept(obj_set)
        concepts = [node for node in extend_G.nodes if "C-" in node]
        objs = {node for node in extend_G.nodes if "C-" not in node}

        # Get object coverage differentiated hub object
        objCoverage = self._get_obj_coverage(obj_set, tpls)

        # remove unimportant concept
        extend_G = self._remove_minor_concept(extend_G)
        extend_G = self._remove_concept_four(extend_G)

        # Create a corresponding table between group and object for reference
        tabGroupObj = {}  # {member: group}
        tabObjGroup = {}  # {group: [member, ...]}

        while len(concepts) != 0:
            # pick the most important concept and the related objects (member)
            bestConcept = max(concepts, key=lambda c: self._avg_weight(extend_G, c))
            members = list(extend_G.predecessors(bestConcept))

            numHubObjs = sum(objCoverage[m] >= self.tau for m in members)
            if numHubObjs > 1:
                extend_G.remove_node(bestConcept)
            else:
                # store the group into group table
                tabObjGroup[bestConcept] = members
                tabGroupObj.update({m: bestConcept for m in members})
                # remove the concept and the corresponding objects from extend_G
                extend_G = self._remove_group(extend_G, bestConcept)
                objs.difference_update(members)

            # remove unimportant concepts
            extend_G = self._remove_minor_concept(extend_G)
            extend_G = self._remove_concept_four(extend_G)
            concepts = [node for node in extend_G.nodes if "C-" in node]

        # let isolated object form a group of its own
        for remain_node in objs:
            tabObjGroup["C-" + remain_node] = [remain_node]
            tabGroupObj[remain_node] = "C-" + remain_node

        return tabGroupObj, tabObjGroup
