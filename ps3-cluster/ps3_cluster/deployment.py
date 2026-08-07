"""Declarative membership for a deployed subcluster hierarchy.

One JSON file describes which head server fronts which consoles, and both ends
read it: a head server starts the subcluster named on its command line, and the
layer coordinator builds its placement, its subcluster plan and its head-server
endpoints from the same document. That keeps the two sides from disagreeing
about who owns an expert — the failure mode a hand-rolled pair of config files
invites.

    {
      "subcluster_size": 22,
      "regions": [
        {"id": "rg-0000", "host": "0.0.0.0", "port": 8200,
         "standby": [{"host": "10.0.1.9", "port": 8200}],
         "subclusters": ["sc-0000", "sc-0001"]}
      ],
      "subclusters": [
        {
          "id": "sc-0000",
          "host": "0.0.0.0",
          "port": 8100,
          "standby": [{"host": "10.0.1.10", "port": 8100}],
          "members": [
            {"layer": 3, "expert": 0, "node": "ps3-L003-E0000",
             "host": "10.0.0.10", "port": 9000,
             "replicas": [{"node": "ps3-L003-E0000-b",
                           "host": "10.0.0.210", "port": 9000}]}
          ]
        }
      ]
    }

Three optional keys extend the phase-2 document without invalidating it:

``members[].replicas``
    Standby *consoles*, registered via ``ExpertPlacement.assign_replica``, so
    the canonical one-expert-per-layer-per-node placement is exactly the
    ``members`` list.
``standby`` (on a subcluster or a region)
    Further addresses of the same logical *head server*, tried in order after
    its primary. A head server is stateless apart from its bounded dedup cache,
    so a standby is either the same process on a second address or a second
    process holding the same membership.
``regions``
    A third tier. Declaring it groups the head servers behind regional
    coordinators; omitting it leaves the two-tier topology exactly as it was.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from .dispatch import ExpertPlacement
from .subcluster import (DEFAULT_SUBCLUSTER_SIZE, SubclusterPlan, TieredPlan)

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
    #: Further addresses of the same logical head, tried after ``endpoint``.
    standby: Tuple[Endpoint, ...] = ()

    @property
    def endpoints(self) -> Tuple[Endpoint, ...]:
        """Primary first, then standbys: the transport's preference order."""
        return (self.endpoint,) + self.standby


class RegionSpec(NamedTuple):
    """One regional coordinator and the head servers behind it."""

    region_id: str
    endpoint: Endpoint
    subclusters: Tuple[str, ...]
    standby: Tuple[Endpoint, ...] = ()

    @property
    def endpoints(self) -> Tuple[Endpoint, ...]:
        return (self.endpoint,) + self.standby


class ClusterConfig:
    """A whole hierarchy: head servers, their consoles, and the group size."""

    def __init__(self, subclusters: Sequence[SubclusterSpec],
                 subcluster_size: int = DEFAULT_SUBCLUSTER_SIZE,
                 regions: Sequence[RegionSpec] = ()) -> None:
        if subcluster_size < 1:
            raise ValueError("subcluster_size must be >= 1")
        self.subcluster_size = subcluster_size
        self.subclusters: List[SubclusterSpec] = list(subclusters)
        self.regions: List[RegionSpec] = list(regions)
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
        seen_regions = set()
        fronted: Dict[str, str] = {}
        for region in self.regions:
            if region.region_id in seen_regions:
                raise ValueError(f"duplicate region id {region.region_id!r}")
            seen_regions.add(region.region_id)
            for group_id in region.subclusters:
                if group_id not in seen_groups:
                    raise ValueError(f"region {region.region_id} claims unknown "
                                     f"subcluster {group_id!r}")
                if group_id in fronted:
                    raise ValueError(f"subcluster {group_id!r} is in regions "
                                     f"{fronted[group_id]!r} and "
                                     f"{region.region_id!r}")
                fronted[group_id] = region.region_id
        if self.regions:
            orphans = sorted(seen_groups - set(fronted))
            if orphans:
                raise ValueError(f"subclusters {orphans} are in no region; a "
                                 f"three-tier config must place every head")

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

    def group_endpoints(self, region_id: Optional[str] = None
                        ) -> Dict[str, List[Endpoint]]:
        """``group_id -> [primary, standby...]`` for head servers.

        Restricted to one region's heads when ``region_id`` is given, which is
        what a regional coordinator needs to reach its own downstream group.
        """
        wanted = (None if region_id is None
                  else set(self.region(region_id).subclusters))
        return {spec.group_id: list(spec.endpoints)
                for spec in self.subclusters
                if wanted is None or spec.group_id in wanted}

    def region_ids(self) -> List[str]:
        return [spec.region_id for spec in self.regions]

    def region(self, region_id: str) -> RegionSpec:
        for spec in self.regions:
            if spec.region_id == region_id:
                return spec
        raise KeyError(region_id)

    def region_endpoints(self) -> Dict[str, List[Endpoint]]:
        """``region_id -> [primary, standby...]`` for regional coordinators."""
        return {spec.region_id: list(spec.endpoints) for spec in self.regions}

    def region_members(self, region_id: str) -> List[MemberSpec]:
        """Every console under a region, head-server order."""
        return [m for group_id in self.region(region_id).subclusters
                for m in self.subcluster(group_id).members]

    def plan(self, region_id: Optional[str] = None) -> SubclusterPlan:
        """Console -> head-server plan, declared grouping, primaries only.

        ``region_id`` narrows it to one region's heads, which is the plan that
        region's coordinator runs on.
        """
        wanted = (None if region_id is None
                  else set(self.region(region_id).subclusters))
        groups = {spec.group_id: [m.node_id for m in spec.members]
                  for spec in self.subclusters
                  if wanted is None or spec.group_id in wanted}
        return SubclusterPlan.from_groups(groups, size=self.subcluster_size)

    def tiered_plan(self) -> TieredPlan:
        """Console -> region plan for a layer coordinator over three tiers."""
        if not self.regions:
            raise ValueError("config declares no regions")
        return TieredPlan.for_regions(
            self.plan(), {spec.region_id: list(spec.subclusters)
                          for spec in self.regions})

    # -- serialisation -----------------------------------------------------
    @classmethod
    def from_dict(cls, doc: dict) -> "ClusterConfig":
        if not isinstance(doc, dict):
            raise ValueError("cluster config must be a JSON object")
        size = int(doc.get("subcluster_size", DEFAULT_SUBCLUSTER_SIZE))
        specs = []
        regions = tuple(
            RegionSpec(region_id=str(raw["id"]),
                       endpoint=(str(raw.get("host", "0.0.0.0")),
                                 int(raw["port"])),
                       subclusters=tuple(str(g) for g in
                                         raw.get("subclusters", ())),
                       standby=_standby(raw))
            for raw in doc.get("regions", ()))
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
                                        members=members,
                                        standby=_standby(raw)))
        if not specs:
            raise ValueError("cluster config declares no subclusters")
        return cls(specs, subcluster_size=size, regions=regions)

    def to_dict(self) -> dict:
        doc = {
            "subcluster_size": self.subcluster_size,
            "subclusters": [
                {
                    "id": spec.group_id,
                    "host": spec.endpoint[0],
                    "port": spec.endpoint[1],
                    **_standby_doc(spec.standby),
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
        if self.regions:
            doc["regions"] = [
                {
                    "id": spec.region_id,
                    "host": spec.endpoint[0],
                    "port": spec.endpoint[1],
                    **_standby_doc(spec.standby),
                    "subclusters": list(spec.subclusters),
                }
                for spec in self.regions
            ]
        return doc

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
                  prefix: str = "sc", standby: int = 0,
                  standby_host: Optional[str] = None) -> "ClusterConfig":
        """Group one layer's consoles into ``size``-console subclusters.

        ``consoles`` is ``(expert, node_id, (host, port))`` in the order the
        experts should be grouped; head servers are numbered from
        ``head_port_base``. This is the programmatic equivalent of writing the
        JSON by hand, used by ``tools/gen_cluster_config.py`` and the tests.

        ``standby`` declares that many extra addresses per head server, which
        is what a failover deployment needs: ``run_subcluster.py --standby N``
        serves one. They are numbered in blocks after the primaries on the same
        host unless ``standby_host`` says otherwise, so a real farm generates a
        starting point here and edits the hosts.
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
        if standby:
            addresses = standby_host or head_host
            specs = [
                spec._replace(standby=_extra_addresses(
                    addresses, head_port_base, index, len(specs), standby))
                for index, spec in enumerate(specs)]
        return cls(specs, subcluster_size=size)

    def with_regions(self, n_regions: int, host: str = "0.0.0.0",
                     port_base: int = 8200,
                     prefix: str = "rg", standby: int = 0,
                     standby_host: Optional[str] = None) -> "ClusterConfig":
        """Copy of this config with its heads split across ``n_regions``.

        Heads are dealt out contiguously in declaration order, so a region
        fronts neighbouring experts and a token's top-k tends to land in few
        regions. ``standby`` declares that many extra addresses per region, for
        ``run_region.py --standby N``.
        """
        if n_regions < 1:
            raise ValueError("n_regions must be >= 1")
        group_ids = self.group_ids()
        if n_regions > len(group_ids):
            raise ValueError(f"{n_regions} regions for {len(group_ids)} "
                             f"subclusters")
        per = -(-len(group_ids) // n_regions)
        regions = [
            RegionSpec(region_id=f"{prefix}-{index:04d}",
                       endpoint=(host, port_base + index),
                       subclusters=tuple(group_ids[index * per:
                                                   (index + 1) * per]),
                       standby=_extra_addresses(standby_host or host,
                                                port_base, index, n_regions,
                                                standby))
            for index in range(n_regions)]
        return ClusterConfig(self.subclusters,
                             subcluster_size=self.subcluster_size,
                             regions=[r for r in regions if r.subclusters])


def _standby(raw: dict) -> Tuple[Endpoint, ...]:
    """Parse a head server's extra addresses, primary order preserved."""
    return tuple((str(s["host"]), int(s["port"]))
                 for s in raw.get("standby", ()))


def _standby_doc(standby: Sequence[Endpoint]) -> dict:
    if not standby:
        return {}
    return {"standby": [{"host": host, "port": port}
                        for host, port in standby]}


def _extra_addresses(host: str, port_base: int, index: int, stride: int,
                     count: int) -> Tuple[Endpoint, ...]:
    """``count`` standby addresses for coordinator ``index`` of ``stride``.

    Each standby generation occupies the next block of ``stride`` ports, so
    they never collide with the primaries or with each other.
    """
    return tuple((host, port_base + stride * (slot + 1) + index)
                 for slot in range(count))


def _head(prefix: str, index: int, host: str, port_base: int,
          members: Sequence[MemberSpec]) -> SubclusterSpec:
    return SubclusterSpec(group_id=f"{prefix}-{index:04d}",
                          endpoint=(host, port_base + index),
                          members=tuple(members))
