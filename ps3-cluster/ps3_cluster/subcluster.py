"""Subcluster grouping and hierarchical fan-out for the PS3 expert farm.

The AFRL "Condor" cluster — the largest PS3 machine built (1,716 consoles) —
was not a flat fabric: it was organised as **subclusters of 22 PlayStation 3s
behind a coordinating head server**, with the heads aggregating results upward
(Barnell, Raim et al., *Advanced Multi-INT Sensor Exploitation*, IEEE HPEC 2012,
https://ieee-hpec.org/2012/index_htm_files/Barnell.pdf). URI's Gravity Grid and
the PS3 lattice-Boltzmann work (Nomura et al., IJCS 2008) likewise put a
supervising process between the network and the consoles' PPE/SPE pipeline.

This module is the coordinator-side analogue: a grouping of expert *endpoints*
into subclusters of :data:`DEFAULT_SUBCLUSTER_SIZE` nodes, plus a partial
weighted reduction per subcluster whose partials are then summed in a fixed
order. It deliberately does **not** touch canonical placement: which expert
lives on which console is still decided by ``ExpertPlacement`` /
``topology.placement_table``. A subcluster is purely a *fan-out and reduction*
grouping, so the design stays composable and needs no MPI or external runtime.

Hierarchical reduction changes the association of the floating-point sum
(partials are summed within a group first), so results are deterministic for a
fixed plan but not bit-identical to the flat reduction; see
``docs/PS3_CLUSTER_PORT.md``.
"""

from __future__ import annotations

from typing import (Callable, Dict, Iterable, List, Optional, Protocol,
                    Sequence, Tuple)

import numpy as np

#: Condor's subcluster size (22 consoles per coordinating head server).
DEFAULT_SUBCLUSTER_SIZE = 22


class SubclusterPlan:
    """Groups node ids into fixed-size subclusters with stable ids.

    Grouping is by position in the supplied node ordering, so the same node
    list always yields the same plan (and the canonical
    ``ps3-L<layer>-E<expert>`` ordering keeps a layer's experts contiguous —
    the whole point, since a layer's top-k picks are the nodes that fan out
    together).
    """

    def __init__(self, node_ids: Iterable[str],
                 size: int = DEFAULT_SUBCLUSTER_SIZE,
                 prefix: str = "sc") -> None:
        if size < 1:
            raise ValueError("subcluster size must be >= 1")
        self.size = size
        self.prefix = prefix
        self._groups: Dict[str, List[str]] = {}
        self._of_node: Dict[str, str] = {}
        for index, node_id in enumerate(node_ids):
            if node_id in self._of_node:
                raise ValueError(f"duplicate node id {node_id!r}")
            group_id = f"{prefix}-{index // size:04d}"
            self._groups.setdefault(group_id, []).append(node_id)
            self._of_node[node_id] = group_id

    # -- queries ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._groups)

    @property
    def n_nodes(self) -> int:
        return len(self._of_node)

    def group_ids(self) -> List[str]:
        return sorted(self._groups)

    def members(self, group_id: str) -> List[str]:
        return list(self._groups[group_id])

    def groups(self) -> Dict[str, List[str]]:
        return {g: list(m) for g, m in sorted(self._groups.items())}

    def subcluster_of(self, node_id: str) -> str:
        return self._of_node[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._of_node

    def same_subcluster(self, a: str, b: str) -> bool:
        return self._of_node[a] == self._of_node[b]

    def group_by_subcluster(
            self, node_ids: Sequence[str]) -> List[Tuple[str, List[int]]]:
        """Bucket positions in ``node_ids`` by subcluster.

        Returns ``[(group_id, [positions...]), ...]`` sorted by group id, with
        positions in their original order — the shape a hierarchical fan-out
        needs (one batch per head server) while staying deterministic.
        """
        buckets: Dict[str, List[int]] = {}
        for position, node_id in enumerate(node_ids):
            buckets.setdefault(self.subcluster_of(node_id), []).append(position)
        return [(g, buckets[g]) for g in sorted(buckets)]

    @classmethod
    def from_groups(cls, groups: Dict[str, Sequence[str]],
                    size: int = DEFAULT_SUBCLUSTER_SIZE,
                    prefix: str = "sc") -> "SubclusterPlan":
        """Build a plan from *declared* membership rather than by position.

        A deployed farm's grouping comes from its config file (which head server
        fronts which consoles), not from an ordering, so the plan has to be able
        to take that as given. ``size`` is retained as the declared maximum and
        oversized groups are rejected.
        """
        plan = cls((), size=size, prefix=prefix)
        for group_id in sorted(groups):
            members = list(groups[group_id])
            if len(members) > size:
                raise ValueError(f"subcluster {group_id} has {len(members)} "
                                 f"members, over the declared size {size}")
            for node_id in members:
                if node_id in plan._of_node:
                    raise ValueError(f"duplicate node id {node_id!r}")
                plan._of_node[node_id] = group_id
            plan._groups[group_id] = members
        return plan

    @classmethod
    def for_placement(cls, placement, layer: Optional[int] = None,
                      size: int = DEFAULT_SUBCLUSTER_SIZE,
                      prefix: str = "sc") -> "SubclusterPlan":
        """Build a plan over a placement's nodes (optionally one layer's)."""
        return cls(placement.node_ids(layer), size=size, prefix=prefix)


class GroupingPlan(Protocol):
    """What layer-side dispatch needs of a plan: node ids -> peer to call.

    Both :class:`SubclusterPlan` (call the head server for these consoles) and
    :class:`TieredPlan` (call the *region* that fronts those head servers)
    satisfy it, which is why one dispatcher drives two or three tiers.
    """

    def group_ids(self) -> List[str]:
        ...

    def members(self, group_id: str) -> List[str]:
        ...

    def subcluster_of(self, node_id: str) -> str:
        ...

    def group_by_subcluster(
            self, node_ids: Sequence[str]) -> List[Tuple[str, List[int]]]:
        ...


class TieredPlan:
    """One more level of the tree: which *region* fronts which head servers.

    Condor's heads were not the top of its tree — the subclusters sat behind
    coordinating servers that the top level talked to (Barnell et al., IEEE HPEC
    2012), which is the shape a Kimi-K3-sized farm needs: a layer that groups a
    token's top-k by *region* sends a handful of batches instead of one per
    22-console subcluster.

    ``lower`` maps consoles to head servers; ``upper`` maps head-server ids to
    region ids. The composition maps a console straight to the region a layer
    should call, so :class:`~.hierarchy.HierarchicalExpertDispatcher` needs no
    knowledge of how deep the tree is.
    """

    def __init__(self, lower: SubclusterPlan, upper: SubclusterPlan) -> None:
        missing = [g for g in lower.group_ids() if not upper.has_node(g)]
        if missing:
            raise ValueError(f"subclusters {missing} are in no region")
        self.lower = lower
        self.upper = upper

    @classmethod
    def for_regions(cls, lower: SubclusterPlan,
                    regions: Dict[str, Sequence[str]]) -> "TieredPlan":
        """Build from declared ``region_id -> [subcluster ids]`` membership."""
        size = max((len(heads) for heads in regions.values()), default=1)
        upper = SubclusterPlan.from_groups(regions, size=size, prefix="rg")
        return cls(lower, upper)

    def __len__(self) -> int:
        return len(self.upper)

    def group_ids(self) -> List[str]:
        return self.upper.group_ids()

    def members(self, group_id: str) -> List[str]:
        """The head servers in a region (its immediate downstream group)."""
        return self.upper.members(group_id)

    def nodes(self, group_id: str) -> List[str]:
        """Every console under a region, head-server order."""
        return [node for head in self.upper.members(group_id)
                for node in self.lower.members(head)]

    def subcluster_of(self, node_id: str) -> str:
        """The region to call for a console. (Named for the plan protocol.)"""
        return self.upper.subcluster_of(self.lower.subcluster_of(node_id))

    def group_by_subcluster(
            self, node_ids: Sequence[str]) -> List[Tuple[str, List[int]]]:
        """Bucket positions in ``node_ids`` by region, group-id order."""
        buckets: Dict[str, List[int]] = {}
        for position, node_id in enumerate(node_ids):
            buckets.setdefault(self.subcluster_of(node_id), []).append(position)
        return [(g, buckets[g]) for g in sorted(buckets)]


def partial_reduce(contributions: Dict[int, np.ndarray],
                   positions: Sequence[int]) -> np.ndarray:
    """Sum ``contributions`` at ``positions`` in ascending position order."""
    ordered = sorted(positions)
    acc = np.array(contributions[ordered[0]], dtype=np.float32, copy=True)
    for position in ordered[1:]:
        acc += contributions[position]
    return acc


def hierarchical_reduce(contributions: Dict[int, np.ndarray],
                        node_ids: Sequence[str],
                        plan: SubclusterPlan,
                        on_partial: Optional[Callable[[str, np.ndarray],
                                                      None]] = None
                        ) -> np.ndarray:
    """Reduce per subcluster, then across subclusters in group-id order.

    ``contributions[i]`` is the gate-weighted output of the expert at position
    ``i`` of ``node_ids``. ``on_partial`` (if given) receives each subcluster's
    partial sum, which is what a real head server would forward upward.
    """
    if not contributions:
        raise ValueError("no contributions to reduce")
    total: Optional[np.ndarray] = None
    for group_id, positions in plan.group_by_subcluster(node_ids):
        present = [p for p in positions if p in contributions]
        if not present:
            continue
        partial = partial_reduce(contributions, present)
        if on_partial is not None:
            on_partial(group_id, partial)
        total = partial if total is None else total + partial
    if total is None:
        raise ValueError("no contributions to reduce")
    return total
