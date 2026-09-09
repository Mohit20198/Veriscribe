"""
comparison/clustering.py
------------------------
Lightweight Union-Find structure for tracking corroborated fact clusters.
"""

from typing import List, Tuple

class CorroborationClusters:
    def __init__(self, initial_pairs: List[Tuple[str, str]] = None):
        """
        Initialize the disjoint set, optionally seeded with initial corroboration pairs.
        """
        self.parent = {}
        
        if initial_pairs:
            for a, b in initial_pairs:
                self.add_corroboration(a, b)

    def _find(self, i: str) -> str:
        if i not in self.parent:
            self.parent[i] = i
        
        # Path compression
        if self.parent[i] != i:
            self.parent[i] = self._find(self.parent[i])
        return self.parent[i]

    def _union(self, i: str, j: str) -> None:
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i != root_j:
            # We don't need rank heuristics for this lightweight scale,
            # deterministic path assignment is fine
            self.parent[root_i] = root_j

    def add_corroboration(self, fact_id_a: str, fact_id_b: str) -> None:
        """Mark two facts as corroborated, joining their clusters."""
        self._union(fact_id_a, fact_id_b)

    def get_cluster_representative(self, fact_id: str) -> str:
        """
        Returns the representative root fact_id for the given fact.
        If the fact isn't in any cluster yet, returns itself.
        """
        return self._find(fact_id)
