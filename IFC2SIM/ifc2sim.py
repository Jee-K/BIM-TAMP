

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import json
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


STRUCTURAL_TYPES = ("IfcColumn", "IfcBeam", "IfcMember")
ROLE_RANKS = {"blocking": 0, "joist": 1, "beam": 2, "column": 3, "member": 4}
SCENE_SCHEMA_VERSION = 2


CONTACT_TOLERANCE = 0.005

@dataclass(frozen=True)
class IFCMember:
    guid: str
    name: str
    ifc_type: str
    source_type: str
    role: str
    group: str
    removal_rank: int
    aabb_min: np.ndarray
    aabb_max: np.ndarray

    @property
    def dimensions(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min

    @property
    def center(self) -> np.ndarray:
        return (self.aabb_min + self.aabb_max) / 2.0

@dataclass(frozen=True)
class IFCContact:
    """A detected adjacency between two members (undirected pair a<->b).

    `direction` is the dominant contact axis: 'x', 'y', or 'z'.
    For a 'z' (vertical) contact, `above` is the GUID of the upper member and
    `below` the lower one, giving a simple supports/rests-on relationship.
    For 'x'/'y' (lateral) contacts `above`/`below` are left as None.
    `overlap_area` is the approximate contact-face area in square meters.
    """
    a: str
    b: str
    direction: str
    above: Optional[str]
    below: Optional[str]
    overlap_area: float

@dataclass(frozen=True)
class IFCScene:
    path: str
    members: Tuple[IFCMember, ...]
    group_order: Tuple[str, ...]
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    connections: Tuple["IFCContact", ...] = ()

def _safe_name(value: str, fallback: str, used: Dict[str, int]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", value or fallback).strip("_") or fallback
    count = used.get(stem, 0)
    used[stem] = count + 1
    return stem if count == 0 else "{}_{}".format(stem, count + 1)

def _decontamp_properties(product) -> Dict[str, object]:
    try:
        import ifcopenshell.util.element
        psets = ifcopenshell.util.element.get_psets(product)
    except Exception:
        return {}
    for name, values in psets.items():
        if str(name).casefold() in {"pset_decontamp", "decontamp"}:
            return values or {}
    return {}

def _semantic_role(product, properties: Dict[str, object]) -> str:
    explicit = properties.get("RemovalGroup") or properties.get("Group")
    if explicit:
        return str(explicit).strip().casefold()
    text = " ".join(str(getattr(product, key, "") or "")
                    for key in ("Name", "ObjectType", "Tag")).casefold()
    if "block" in text:
        return "blocking"
    if "joist" in text:
        return "joist"
    if "beam" in text:
        return "beam"
    if product.is_a("IfcColumn") or "column" in text:
        return "column"
    return "member" if product.is_a("IfcMember") else product.is_a().casefold()

def _explicit_rank(properties: Dict[str, object]) -> Optional[int]:
    """The authored removal rank from the Pset, or None if not set."""
    explicit = properties.get("RemovalRank")
    if explicit in (None, ""):
        explicit = properties.get("GroupRank")
    if explicit in (None, ""):
        return None
    try:
        return int(explicit)
    except (TypeError, ValueError):
        raise ValueError("Pset_DeconTAMP.RemovalRank must be an integer, got {!r}".format(explicit))


def _removal_rank(role: str, properties: Dict[str, object]) -> int:
    explicit = _explicit_rank(properties)
    if explicit is not None:
        return explicit
    return ROLE_RANKS.get(role, len(ROLE_RANKS))

def _selected_indexes(selection: str, total: int) -> Set[int]:
    selection = selection.strip()
    if selection.casefold() in ('', 'all', '*'):
        return set(range(1, total + 1))
    selected: Set[int] = set()
    for token in selection.split(','):
        token = token.strip()
        if '-' in token:
            start_text, end_text = token.split('-', 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError('Invalid range {!r}; use e.g. 1-4.'.format(token)) from exc
            if start > end:
                start, end = end, start
            indexes = range(start, end + 1)
        else:
            try:
                indexes = (int(token),)
            except ValueError as exc:
                raise ValueError('Invalid selection {!r}; use member numbers or ranges.'.format(token)) from exc
        for index in indexes:
            if not 1 <= index <= total:
                raise ValueError('Selection number {} is outside 1-{}.'.format(index, total))
            selected.add(index)
    return selected


def prompt_for_types(scene: IFCScene, input_fn=input) -> Set[str]:
    types = sorted({member.source_type for member in scene.members})
    print('\n[IFC2SIM] Available component types:')
    for index, source_type in enumerate(types, start=1):
        members = [member for member in scene.members if member.source_type == source_type]
        groups = ', '.join(sorted({member.group for member in members}))
        print('  {:>2}. {:<42} members={} groups={}'.format(
            index, source_type, len(members), groups))
    reply = input_fn('[IFC2SIM] Select component types (all, 1,3,5-8): ')
    selected_types = {types[index - 1] for index in _selected_indexes(reply, len(types))}
    selected = {member.guid for member in scene.members if member.source_type in selected_types}
    print('[IFC2SIM] Selected {} type(s), {} member(s).'.format(len(selected_types), len(selected)))
    return selected


def load_ifc_scene(ifc_path: str, require_explicit_ranks: bool = False,
                   selected_guids: Optional[Set[str]] = None) -> IFCScene:

    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError as exc:
        raise RuntimeError("IFC2SIM requires ifcopenshell. Install it in this environment.") from exc

    path = Path(ifc_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("IFC input does not exist: {}".format(path))
    model = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    products = []
    for ifc_type in STRUCTURAL_TYPES:
        products.extend(model.by_type(ifc_type))
    products.sort(key=lambda product: (product.is_a(), product.GlobalId))

    members: List[IFCMember] = []
    used_names: Dict[str, int] = {}
    authored_ranks: Dict[str, Optional[int]] = {}
    for product in products:
        if not product.Representation:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            vertices = np.asarray(list(shape.geometry.verts), dtype=float).reshape((-1, 3))
        except Exception as exc:
            print("[IFC2SIM] Skipping {}: {}".format(product.GlobalId, exc))
            continue
        if vertices.size == 0 or not np.isfinite(vertices).all():
            print("[IFC2SIM] Skipping {}: empty or non-finite geometry".format(product.GlobalId))
            continue
        properties = _decontamp_properties(product)
        if require_explicit_ranks:
            has_group = properties.get("RemovalGroup") not in (None, "")
            has_rank = properties.get("RemovalRank") not in (None, "")
            if not (has_group and has_rank):
                raise ValueError(
                    'IFC2SIM strict mode requires Pset_DeconTAMP.RemovalGroup '
                    'and RemovalRank for {} ({})'.format(product.GlobalId, product.Name))
        role = _semantic_role(product, properties)
        group = re.sub(r"[^a-z0-9_]+", "_", role).strip("_") or "default"
        name = _safe_name(product.Name or product.GlobalId, product.GlobalId, used_names)
        authored_ranks[product.GlobalId] = _explicit_rank(properties)
        members.append(IFCMember(
            guid=product.GlobalId, name=name, ifc_type=product.is_a(),
            source_type=str(product.ObjectType or product.is_a()), role=role,
            group=group, removal_rank=_removal_rank(role, properties),
            aabb_min=vertices.min(axis=0), aabb_max=vertices.max(axis=0)))

    if selected_guids is not None:
        unknown = set(selected_guids) - {member.guid for member in members}
        if unknown:
            raise ValueError('Selected IFC GUID(s) were not imported: {}'.format(', '.join(sorted(unknown))))
        members = [member for member in members if member.guid in selected_guids]
    if not members:
        raise ValueError("IFC2SIM found no usable IfcColumn, IfcBeam, or IfcMember geometry in {}".format(path))
    members_tuple = tuple(members)
    contacts = compute_contacts(members_tuple)


    authored = {m.guid: authored_ranks.get(m.guid) for m in members_tuple}
    if all(r is not None for r in authored.values()):
        rank_of = {g: int(r) for g, r in authored.items()}
    else:
        rank_of = compute_gravity_ranks(members_tuple, contacts)
    members_tuple = tuple(replace(m, removal_rank=rank_of[m.guid],
                                  group='grp_{}'.format(rank_of[m.guid]))
                          for m in members_tuple)

    order = sorted({m.removal_rank for m in members_tuple})
    group_order = tuple('grp_{}'.format(r) for r in order)
    return IFCScene(
        path=str(path), members=members_tuple, group_order=group_order,
        aabb_min=np.min([member.aabb_min for member in members_tuple], axis=0),
        aabb_max=np.max([member.aabb_max for member in members_tuple], axis=0),
        connections=contacts)


def compute_contacts(members: Tuple[IFCMember, ...],
                     tolerance: float = CONTACT_TOLERANCE) -> Tuple[IFCContact, ...]:
    """Infer pairwise adjacency between members from their bounding boxes.

    Two members are in contact when their AABBs overlap (or are within
    `tolerance`) on all three axes. The axis with the smallest overlap/gap is
    taken as the contact normal: that is the direction along which the boxes
    just meet, i.e. the shared face. For vertical ('z') contacts the upper and
    lower members are recorded to express a supports/rests-on relationship.
    """
    contacts: List[IFCContact] = []
    axis_names = ('x', 'y', 'z')
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            lhs, rhs = members[i], members[j]


            overlaps = np.minimum(lhs.aabb_max, rhs.aabb_max) - \
                np.maximum(lhs.aabb_min, rhs.aabb_min)
            if np.any(overlaps < -tolerance):
                continue


            normal_axis = int(np.argmin(overlaps))
            face_axes = [a for a in range(3) if a != normal_axis]
            face_dims = [max(0.0, float(overlaps[a])) for a in face_axes]
            overlap_area = face_dims[0] * face_dims[1]
            if overlap_area <= 0.0:
                continue

            direction = axis_names[normal_axis]
            above = below = None
            if direction == 'z':
                if lhs.center[2] >= rhs.center[2]:
                    above, below = lhs.guid, rhs.guid
                else:
                    above, below = rhs.guid, lhs.guid
            contacts.append(IFCContact(
                a=lhs.guid, b=rhs.guid, direction=direction,
                above=above, below=below, overlap_area=round(overlap_area, 6)))
    return tuple(contacts)


def compute_gravity_ranks(members: Tuple[IFCMember, ...],
                          contacts: Tuple[IFCContact, ...]) -> Dict[str, int]:
    """Removal rank for each member from support topology + height (no hardcoded
    role table). Lower rank = removed earlier.

    A member can only be removed once everything resting on it is gone, so we
    peel the structure top-down: repeatedly take the members with nothing still
    on top of them (highest center first as a tiebreaker), assign the next rank,
    and remove them. Vertical ('z') contacts give the supports relation via
    their above/below GUIDs; lateral ('x'/'y') contacts impose no ordering.
    """
    by_guid = {m.guid: m for m in members}
    resting_on = {m.guid: set() for m in members}
    for c in contacts:
        if c.above is not None and c.below is not None and c.below in resting_on:
            resting_on[c.below].add(c.above)
    remaining = set(by_guid)
    ranks: Dict[str, int] = {}
    rank = 0
    while remaining:
        free = [g for g in remaining if not (resting_on[g] & remaining)]
        if not free:
            free = list(remaining)
        free.sort(key=lambda g: float(by_guid[g].center[2]), reverse=True)
        for g in free:
            ranks[g] = rank
            remaining.discard(g)
        rank += 1
    return ranks


def write_scene_manifest(scene: IFCScene, output_path: str) -> str:
    destination = Path(output_path).expanduser().resolve()
    payload = {
        'schema_version': SCENE_SCHEMA_VERSION,
        'source_ifc': scene.path,
        'group_order': list(scene.group_order),
        'aabb_min': scene.aabb_min.tolist(),
        'aabb_max': scene.aabb_max.tolist(),
        'members': [{
            'guid': member.guid, 'name': member.name, 'ifc_type': member.ifc_type,
            'source_type': member.source_type,
            'group': member.group, 'removal_rank': member.removal_rank,
            'aabb_min': member.aabb_min.tolist(), 'aabb_max': member.aabb_max.tolist(),
            'dimensions': member.dimensions.tolist(), 'center': member.center.tolist(),
        } for member in scene.members],
        'connections': [{
            'a': contact.a, 'b': contact.b, 'direction': contact.direction,
            'above': contact.above, 'below': contact.below,
            'overlap_area': contact.overlap_area,
        } for contact in scene.connections],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    return str(destination)


def load_scene_manifest(scene_path: str) -> IFCScene:
    path = Path(scene_path).expanduser().resolve()
    with path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported IFC2SIM scene schema in {}'.format(path))
    members = tuple(IFCMember(
        guid=item['guid'], name=item['name'], ifc_type=item['ifc_type'],
        source_type=item.get('source_type', item['ifc_type']),
        role=item['group'], group=item['group'], removal_rank=int(item['removal_rank']),
        aabb_min=np.asarray(item['aabb_min'], dtype=float),
        aabb_max=np.asarray(item['aabb_max'], dtype=float)) for item in payload['members'])
    if not members:
        raise ValueError('IFC2SIM manifest has no members: {}'.format(path))
    connections = tuple(IFCContact(
        a=item['a'], b=item['b'], direction=item['direction'],
        above=item.get('above'), below=item.get('below'),
        overlap_area=float(item.get('overlap_area', 0.0)))
        for item in payload.get('connections', []))


    if not connections:
        connections = compute_contacts(members)
    return IFCScene(path=str(path), members=members,
                    group_order=tuple(payload['group_order']),
                    aabb_min=np.asarray(payload['aabb_min'], dtype=float),
                    aabb_max=np.asarray(payload['aabb_max'], dtype=float),
                    connections=connections)


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert IFC into a portable IFC2SIM scene manifest.')
    parser.add_argument('--ifc', required=True, help='Source IFC file.')
    parser.add_argument('--output', required=True, help='Output JSON scene manifest.')
    parser.add_argument('--require-ifc-ranks', action='store_true')
    parser.add_argument('--select-ifc-types', action='store_true',
                        help='Interactively select component families/types, not individual members.')
    parser.add_argument('--ifc-types', default=None,
                        help='Comma-separated IFC/Revit ObjectType values to export.')
    args = parser.parse_args()
    selected = None
    selection_modes = sum(bool(value) for value in
                          (args.select_ifc_types, args.ifc_types))
    if selection_modes > 1:
        parser.error('Use only one IFC selection option.')
    scene = load_ifc_scene(args.ifc, require_explicit_ranks=args.require_ifc_ranks,
                           selected_guids=selected)
    if args.select_ifc_types:
        selected = prompt_for_types(scene)
        scene = load_ifc_scene(args.ifc, require_explicit_ranks=args.require_ifc_ranks,
                               selected_guids=selected)
    elif args.ifc_types:
        wanted_types = {value.strip() for value in args.ifc_types.split(',') if value.strip()}
        selected = {member.guid for member in scene.members if member.source_type in wanted_types}
        if not selected:
            parser.error('No members match --ifc-types. Use --select-ifc-types to list values.')
        scene = load_ifc_scene(args.ifc, require_explicit_ranks=args.require_ifc_ranks,
                               selected_guids=selected)
    output = write_scene_manifest(scene, args.output)
    print('[IFC2SIM] Wrote {} members to {}'.format(len(scene.members), output))
    vertical = sum(1 for c in scene.connections if c.direction == 'z')
    lateral = len(scene.connections) - vertical
    print('[IFC2SIM] Topology: {} contact(s) ({} vertical/support, {} lateral)'.format(
        len(scene.connections), vertical, lateral))


if __name__ == '__main__':
    main()
