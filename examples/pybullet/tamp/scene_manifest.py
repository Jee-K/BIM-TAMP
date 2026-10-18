

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


STRUCTURAL_TYPES = ("IfcColumn", "IfcBeam", "IfcMember")
ROLE_RANKS = {"blocking": 0, "joist": 1, "beam": 2, "column": 3, "member": 4}
SCENE_SCHEMA_VERSION = 1


SUPPORTED_SCENE_SCHEMA_VERSIONS = (1, 2)

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
class IFCScene:
    path: str
    members: Tuple[IFCMember, ...]
    group_order: Tuple[str, ...]
    aabb_min: np.ndarray
    aabb_max: np.ndarray

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

def _removal_rank(role: str, properties: Dict[str, object]) -> int:
    explicit = properties.get("RemovalRank")
    if explicit in (None, ""):
        explicit = properties.get("GroupRank")
    if explicit not in (None, ""):
        try:
            return int(explicit)
        except (TypeError, ValueError):
            raise ValueError("Pset_DeconTAMP.RemovalRank must be an integer, got {!r}".format(explicit))
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
                   selected_guids: Optional[Set[str]] = None,
                   exclude_groups: Optional[Set[str]] = None) -> IFCScene:

    try:
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.representation as _ifc_rep
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
    for product in products:
        if not product.Representation:
            continue
        try:


            body_rep = None
            try:
                body_rep = _ifc_rep.get_representation(product, "Model", "Body")
            except Exception:
                body_rep = None
            if body_rep is not None:
                shape = ifcopenshell.geom.create_shape(settings, product, body_rep)
            else:
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
    if exclude_groups:
        drop = {g.strip().lower() for g in exclude_groups}
        before = len(members)
        members = [m for m in members if m.group.lower() not in drop]
        print("[IFC2SIM] Excluded groups {} -> dropped {} member(s), {} remain".format(
            sorted(drop), before - len(members), len(members)))
    if not members:
        raise ValueError("IFC2SIM found no usable IfcColumn, IfcBeam, or IfcMember geometry in {}".format(path))
    groups = sorted({(member.removal_rank, member.group) for member in members})
    return IFCScene(
        path=str(path), members=tuple(members), group_order=tuple(group for _, group in groups),
        aabb_min=np.min([member.aabb_min for member in members], axis=0),
        aabb_max=np.max([member.aabb_max for member in members], axis=0))


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
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    return str(destination)


def load_scene_manifest(scene_path: str) -> IFCScene:
    path = Path(scene_path).expanduser().resolve()
    with path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('schema_version') not in SUPPORTED_SCENE_SCHEMA_VERSIONS:
        raise ValueError('Unsupported IFC2SIM scene schema {} in {} (supported: {})'.format(
            payload.get('schema_version'), path, SUPPORTED_SCENE_SCHEMA_VERSIONS))
    members = tuple(IFCMember(
        guid=item['guid'], name=item['name'], ifc_type=item['ifc_type'],
        source_type=item.get('source_type', item['ifc_type']),
        role=item['group'], group=item['group'], removal_rank=int(item['removal_rank']),
        aabb_min=np.asarray(item['aabb_min'], dtype=float),
        aabb_max=np.asarray(item['aabb_max'], dtype=float)) for item in payload['members'])
    if not members:
        raise ValueError('IFC2SIM manifest has no members: {}'.format(path))


    return IFCScene(path=str(path), members=members,
                    group_order=tuple(payload['group_order']),
                    aabb_min=np.asarray(payload['aabb_min'], dtype=float),
                    aabb_max=np.asarray(payload['aabb_max'], dtype=float))


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


if __name__ == '__main__':
    main()
