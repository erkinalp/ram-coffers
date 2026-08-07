"""Declarative membership for a deployed subcluster hierarchy.

One JSON file describes which head server fronts which consoles, and both ends
read it: a head server starts the subcluster named on its command line, and the
layer coordinator builds its placement, its subcluster plan and its head-server
endpoints from the same document. That keeps the two sides from disagreeing
about who owns an expert — the failure mode a hand-rolled pair of config files
invites.

    {
      "subcluster_size": 22,
      "subclusters": [
        {
          "id": "sc-0000",
          "host": "0.0.0.0",
          "port": 8100,
          "members": [
            {"layer": 3, "expert": 0, "node": "ps3-L003-E0000",
             "host": "10.0.0.10", "port": 9000,
             "replicas": [{"node": "ps3-L003-E0000-b",
                           "host": "10.0.0.210", "port": 9000}]}
          ]
        }
      ]
    }

``replicas`` is optional and additive: it registers standby consoles via
``ExpertPlacement.assign_replica``, so the canonical one-expert-per-layer-per-
node placement is exactly the ``members`` list.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from .dispatch import ExpertPlacement
from .subcluster import DEFAULT_SUBCLUSTER_SIZE, SubclusterPlan

Endpoint = Tuple[str, int]


class ReplicaSpec(NamedTuple):
    node_id: str
    endpoint: Endpoint


class MemberSpec(NamedTuple):
    """One console: the expert it holds and where to reach it."""

    layer: int
    expert: int
    node_id: str
    endpoint: Endpoint
    replicas: Tuple[ReplicaSpec, ...] = ()


class SubclusterSpec(NamedTuple):
    """One head server and the consoles behind it."""

    group_id: str
    endpoint: Endpoint
    members: Tuple[MemberSpec, ...]


class ClusterConfig:
    """A whole hierarchy: head servers, their consoles, and the group size."""

    def __init__(self, subclusters: Sequence[SubclusterSpec],
                 subcluster_size: int = DEFAULT_SUBCLUSTER_SIZE) -> None:
        if subcluster_size < 1:
            raise ValueError("subcluster_size must be >= 1")
        self.subcluster_size = subcluster_size
        self.subclusters: List[SubclusterSpec] = list(subclusters)
        seen_groups = set()
        seen_nodes = set()
        for spec in self.subclusters:
            if spec.group_id in seen_groups:
                raise ValueError(f"duplicate subcluster id {spec.group_id!r}")
            seen_groups.add(spec.group_id)
            if len(spec.members) > subcluster_size:
                raise ValueError(f"subcluster {spec.group_id} has "
                                 f"{len(spec.members)} members, over the "
                                 f"declared size {subcluster_size}")
            for member in spec.members:
                if member.node_id in seen_nodes:
                    raise ValueError(f"node {member.node_id!r} appears twice")
                seen_nodes.add(member.node_id)

    # -- lookups -----------------------------------------------------------
    def group_ids(self) -> List[str]:
        return [spec.group_id for spec in self.subclusters]

    def subcluster(self, group_id: str) -> SubclusterSpec:
        for spec in self.subclusters:
            if spec.group_id == group_id:
                return spec
        raise KeyError(group_id)

    def members(self, group_id: Optional[str] = None) -> List[MemberSpec]:
        specs = (self.subclusters if group_id is None
                 else [self.subcluster(group_id)])
        return [m for spec in specs for m in spec.members]

    def layers(self) -> List[int]:
        return sorted({m.layer for m in self.members()})

    # -- derived objects ---------------------------------------------------
    def placement(self, group_id: Optional[str] = None) -> ExpertPlacement:
        """Canonical placement (plus any replicas) for the whole config."""
        placement = ExpertPlacement()
        for member in self.members(group_id):
            placement.assign(member.layer, member.expert, member.node_id)
        for member in self.members(group_id):
            for replica in member.replicas:
                placement.assign_replica(member.layer, member.expert,
                                         replica.node_id)
        return placement

    def expert_endpoints(self, group_id: Optional[str] = None
                         ) -> Dict[str, Endpoint]:
        """``node_id -> (host, port)`` for consoles (replicas included)."""
        endpoints: Dict[str, Endpoint] = {}
        for member in self.members(group_id):
            endpoints[member.node_id] = member.endpoint
            for replica in member.replicas:
                endpoints[replica.node_id] = replica.endpoint
        return endpoints

    def group_endpoints(self) -> Dict[str, Endpoint]:
        """``group_id -> (host, port)`` for head servers."""
        return {spec.group_id: spec.endpoint for spec in self.subclusters}

    def plan(self) -> SubclusterPlan:
        """Plan reflecting the *declared* grouping, primaries only."""
        groups = {spec.group_id: [m.node_id for m in spec.members]
                  for spec in self.subclusters}
        return SubclusterPlan.from_groups(groups, size=self.subcluster_size)

    # -- serialisation -----------------------------------------------------
    @classmethod
    def from_dict(cls, doc: dict) -> "ClusterConfig":
        if not isinstance(doc, dict):
            raise ValueError("cluster config must be a JSON object")
        size = int(doc.get("subcluster_size", DEFAULT_SUBCLUSTER_SIZE))
        specs = []
        for raw in doc.get("subclusters", ()):
            members = tuple(
                MemberSpec(layer=int(m["layer"]), expert=int(m["expert"]),
                           node_id=str(m["node"]),
                           endpoint=(str(m["host"]), int(m["port"])),
                           replicas=tuple(
                               ReplicaSpec(str(r["node"]),
                                           (str(r["host"]), int(r["port"])))
                               for r in m.get("replicas", ())))
                for m in raw.get("members", ()))
            specs.append(SubclusterSpec(group_id=str(raw["id"]),
                                        endpoint=(str(raw.get("host",
                                                              "0.0.0.0")),
                                                  int(raw["port"])),
                                        members=members))
        if not specs:
            raise ValueError("cluster config declares no subclusters")
        return cls(specs, subcluster_size=size)

    def to_dict(self) -> dict:
        return {
            "subcluster_size": self.subcluster_size,
            "subclusters": [
                {
                    "id": spec.group_id,
                    "host": spec.endpoint[0],
                    "port": spec.endpoint[1],
                    "members": [
                        {
                            "layer": m.layer,
                            "expert": m.expert,
                            "node": m.node_id,
                            "host": m.endpoint[0],
                            "port": m.endpoint[1],
                            **({"replicas": [{"node": r.node_id,
                                              "host": r.endpoint[0],
                                              "port": r.endpoint[1]}
                                             for r in m.replicas]}
                               if m.replicas else {}),
                        }
                        for m in spec.members
                    ],
                }
                for spec in self.subclusters
            ],
        }

    @classmethod
    def load(cls, path: str) -> "ClusterConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
            fh.write("\n")

    # -- builders ----------------------------------------------------------
    @classmethod
    def for_layer(cls, layer: int,
                  consoles: Iterable[Tuple[int, str, Endpoint]],
                  size: int = DEFAULT_SUBCLUSTER_SIZE,
                  head_host: str = "0.0.0.0", head_port_base: int = 8100,
                  prefix: str = "sc") -> "ClusterConfig":
        """Group one layer's consoles into ``size``-console subclusters.

        ``consoles`` is ``(expert, node_id, (host, port))`` in the order the
        experts should be grouped; head servers are numbered from
        ``head_port_base``. This is the programmatic equivalent of writing the
        JSON by hand, used by ``tools/gen_cluster_config.py`` and the tests.
        """
        specs: List[SubclusterSpec] = []
        members: List[MemberSpec] = []
        for expert, node_id, endpoint in consoles:
            members.append(MemberSpec(layer=layer, expert=int(expert),
                                      node_id=node_id, endpoint=endpoint))
            if len(members) == size:
                specs.append(_head(prefix, len(specs), head_host,
                                   head_port_base, members))
                members = []
        if members:
            specs.append(_head(prefix, len(specs), head_host, head_port_base,
                               members))
        return cls(specs, subcluster_size=size)


def _head(prefix: str, index: int, host: str, port_base: int,
          members: Sequence[MemberSpec]) -> SubclusterSpec:
    return SubclusterSpec(group_id=f"{prefix}-{index:04d}",
                          endpoint=(host, port_base + index),
                          members=tuple(members))
