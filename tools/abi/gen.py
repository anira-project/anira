#!/usr/bin/env python3
"""anira ABI generator: abi/anira.yml -> the committed C headers and their mirrors.

The registry is the single source of truth for the versioned C ABI. This script
validates it against the header conventions of docs/anira-v3-architecture.md
(section 6a) and emits, deterministically and without any other tool:

  include/anira/abi/<file>.h              the C11 headers (never edit by hand); a header with
                                          `draft: true` is include/anira/abi/draft/<name>.h
  web/src/abi/enums.ts                    the TypeScript mirror of enums and defines
  web/src/abi/layout.ts                   size, align and member offsets of the Tier-1 records
  abi/symbols-<major>.txt                 the promised entry points, sorted
  abi/symbols-draft.txt                   the draft entry points, sorted
  web/src/abi/exports_wasm.txt            the Emscripten export list (_-prefixed)
  src/capi/generated/status_strings.inc   ANIRA_STATUS_TEXT(name, "text") per status
  src/capi/generated/struct_sizes.inc     ANIRA_STRUCT_SIZE(id, type) per record anira_sizeof answers
  test/abi/generated/test_layout.c        gate 3: _Static_asserts and the layout printer
  test/abi/generated/link_probe.c         the presence gate anira_abi_link: the address of every
                                          promised and draft entry in one table
  docs/sphinx/api/enum/<enum>.rst         one Breathe page per enum
  abi/anira.json                          the registry as JSON, for tools without YAML

Usage:
  python3 tools/abi/gen.py --repo <root> --write             regenerate in place
  python3 tools/abi/gen.py --repo <root> --check             exit 2 on any drift
  python3 tools/abi/gen.py --repo <root> --out-dir <dir>     write under another root
  python3 tools/abi/gen.py --repo <root> --diff-against v3.0.0-alpha.1
                                                             compare the registry with
                                                             the one at a git ref

Exit codes: 0 clean, 1 registry validation error, 2 drift under --check, 3 usage.
Requires Python >= 3.9 and PyYAML.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("tools/abi/gen.py: PyYAML is required (pip install pyyaml)\n")
    sys.exit(3)

# ------------------------------------------------------------------------------
# Conventions (docs/anira-v3-architecture.md, section 6a)
# ------------------------------------------------------------------------------

THREAD_TAGS = {
    "main-thread",
    "main-thread & !prepared",
    "main-thread & prepared",
    "main-thread & !processing",
    "main-thread & !loader-lock",
    "driver-thread",
    "inference-thread",
    # The one dual tag: a stage callback (and the accessors and the two stage defaults it may
    # call) runs on the driving thread under Hard and on an inference thread for
    # before/after_inference and under Async. anira's own entries under the tag are real-time
    # on both, so nonblocking is required of an entry; the callback typedef anira_stage_fn
    # carries no attribute, since whether a stage body is real-time is the stage's own promise
    # (anira_stage_desc.flags), which differs per stage and cannot be a property of the shared
    # type.
    "driver-thread | inference-thread",
    "thread-safe",
    "thread-safe, !audio-thread",
    "drain-thread",
    "any-thread, blocking",
}
NONBLOCKING_REQUIRED = {"driver-thread", "driver-thread | inference-thread"}
NONBLOCKING_FORBIDDEN_PREFIXES = ("main-thread", "any-thread, blocking")
WIDE_INT_ALLOWLIST = {
    "anira_now_ns",
    "anira_tensor_init_vulkan",
    "anira_tensor_init_opaque_fd",
    "anira_tensor_init_wgpu_buffer",
    "anira_tensor_init_dmabuf",
    "anira_tensor_init_iosurface",
}
FUNCTION_RE = re.compile(r"^anira_[a-z0-9_]+$")
TYPE_RE = re.compile(r"^anira_[a-z0-9_]+$")
MACRO_RE = re.compile(r"^ANIRA_[A-Z0-9_]+$")
TERMINATOR_VALUE = 0x7FFFFFFF
FORCE32_RE = re.compile(r"^ANIRA_[A-Z0-9_]+_FORCE32$")
MEMBER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
DRAFT_FILE_RE = re.compile(r"^draft/[a-z0-9_]+\.h$")
DRAFT_GUARD_PREFIX = "ANIRA_ABI_DRAFT_"
STRUCT_ID_ENUM = "anira_struct_id"
TS_WIDTH = 90  # web/.prettierrc printWidth: a longer generated line would be re-wrapped by prettier

SCALAR_SIZES = {
    "int8_t": (1, 1),
    "uint8_t": (1, 1),
    "char": (1, 1),
    "int16_t": (2, 2),
    "uint16_t": (2, 2),
    "int32_t": (4, 4),
    "uint32_t": (4, 4),
    "int64_t": (8, 8),
    "uint64_t": (8, 8),
    "double": (8, 8),
    "float": (4, 4),
    "anira_dtype": (4, 4),
    "anira_bool": (4, 4),
    "anira_ticket": (4, 4),
}
PTR_SLOT = (8, 8)  # ANIRA_PTR: a union of the pointer with a uint64_t

WIDTH = 100
DOC_WIDTH = 96
NOLINT_CHECKS = "readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses"
# Target-width integers never sit in a record: a Tier-1 layout must be identical on
# wasm32, LP64 and LLP64, and a Tier-2 record is read by struct_size, never by target.
TARGET_WIDTH_TYPES = {"size_t", "ptrdiff_t", "intptr_t", "uintptr_t", "long", "unsigned long", "long long", "unsigned long long", "int", "unsigned", "unsigned int", "short", "unsigned short"}


class RegistryError(Exception):
    pass


# ------------------------------------------------------------------------------
# Loading and validation
# ------------------------------------------------------------------------------


def load_registry(text: str) -> dict:
    reg = yaml.safe_load(text)
    if not isinstance(reg, dict) or "headers" not in reg:
        raise RegistryError("registry: top-level mapping with a 'headers' list expected")
    return reg


def entities(reg: dict, kind: str | None = None):
    for header in reg["headers"]:
        for ent in header.get("entities", []):
            if kind is None or ent["kind"] == kind:
                yield header, ent


def int_value(value) -> int:
    if isinstance(value, bool):
        raise RegistryError(f"boolean is not an integer value: {value!r}")
    if isinstance(value, int):
        return value
    s = str(value).strip().rstrip("uUlL")
    try:
        return int(s, 0)
    except ValueError as exc:
        raise RegistryError(f"not an integer literal: {value!r}") from exc


def constants(reg: dict) -> dict[str, int]:
    """Object-like defines with an integer value, for array extents."""
    out: dict[str, int] = {}
    for _, ent in entities(reg, "define"):
        if ent.get("params"):
            continue
        try:
            out[ent["name"]] = int_value(ent["value"])
        except RegistryError:
            pass
    return out


def is_record(ent: dict) -> bool:
    """A Tier-1 record: a union, or a struct of tier 1. Only these have a committed layout."""
    return ent["kind"] == "union" or (ent["kind"] == "struct" and ent.get("tier") == 1)


def tier1_records(reg: dict):
    """(header, entity) of every Tier-1 record in registry order: a record used as a member
    type must precede its user."""
    for header, ent in entities(reg):
        if is_record(ent):
            yield header, ent


def leaf_layout(owner: str, member: dict, records: dict[str, dict], consts: dict[str, int]) -> dict:
    """The node of a leaf member: an ANIRA_PTR slot, a scalar, a named Tier-1 record, or an
    array of the latter two. Offsets are filled in by the caller."""
    node = {"name": member["name"], "offset": 0, "ptr": "ptr" in member, "count": None, "record": None, "members": []}
    if "ptr" in member:
        node["size"], node["align"] = PTR_SLOT
        return node
    base = member["type"]
    if base in SCALAR_SIZES:
        size, align = SCALAR_SIZES[base]
    elif base in records:
        size, align = records[base]["size"], records[base]["align"]
        node["record"] = base
    else:
        raise RegistryError(
            f"{owner}: member {member['name']} has type {base!r}, whose size is unknown (a Tier-1 "
            "member is a fixed-width scalar, an ANIRA_PTR slot or a Tier-1 record declared before it)"
        )
    count = member.get("array")
    if count is not None:
        if isinstance(count, int) and not isinstance(count, bool):
            extent = count
        elif str(count) in consts:
            extent = consts[str(count)]
        else:
            try:
                extent = int_value(count)
            except RegistryError as exc:
                raise RegistryError(
                    f"{owner}: array extent {count!r} of {member['name']} is neither a "
                    "literal nor an object-like define of the registry"
                ) from exc
        if extent < 1:
            raise RegistryError(f"{owner}: array extent {count!r} of {member['name']} must be 1 or more")
        size *= extent
        node["count"] = extent
    node["size"], node["align"] = size, align
    return node


def shift(node: dict, delta: int) -> None:
    node["offset"] += delta
    for child in node["members"]:
        shift(child, delta)


def member_layout(owner: str, member: dict, records: dict[str, dict], consts: dict[str, int]) -> dict:
    """The node of one member at offset 0: a leaf, `fields` (a named member of an unnamed struct
    type, laid out sequentially) or `arms` (a named member of an unnamed union type, every arm
    at offset 0)."""
    if "fields" in member:
        members, size, align = sequence_layout(f"{owner}.{member['name']}", member["fields"], records, consts)
    elif "arms" in member:
        members, size, align = overlay_layout(f"{owner}.{member['name']}", member["arms"], records, consts)
    else:
        return leaf_layout(owner, member, records, consts)
    return {"name": member["name"], "offset": 0, "size": size, "align": align, "ptr": False, "count": None, "record": None, "members": members}


def sequence_layout(owner: str, members: list[dict], records: dict[str, dict], consts: dict[str, int]) -> tuple[list[dict], int, int]:
    """Natural-alignment layout of struct members: (nodes, total size, align)."""
    offset = 0
    max_align = 1
    nodes = []
    for member in members:
        node = member_layout(owner, member, records, consts)
        offset = (offset + node["align"] - 1) // node["align"] * node["align"]
        shift(node, offset)
        nodes.append(node)
        offset += node["size"]
        max_align = max(max_align, node["align"])
    total = (offset + max_align - 1) // max_align * max_align
    return nodes, total, max_align


def overlay_layout(owner: str, arms: list[dict], records: dict[str, dict], consts: dict[str, int]) -> tuple[list[dict], int, int]:
    """Layout of union arms: every arm at offset 0, size = the largest arm rounded up to the
    largest alignment."""
    nodes = [member_layout(owner, arm, records, consts) for arm in arms]
    max_align = max((n["align"] for n in nodes), default=1)
    size = max((n["size"] for n in nodes), default=0)
    total = (size + max_align - 1) // max_align * max_align
    return nodes, total, max_align


def record_layout(ent: dict, records: dict[str, dict], consts: dict[str, int]) -> dict:
    """The layout tree of a Tier-1 record: {"name", "kind", "size", "align", "members"}."""
    if ent["kind"] == "union":
        members, size, align = overlay_layout(ent["name"], ent["arms"], records, consts)
    else:
        members, size, align = sequence_layout(ent["name"], ent["fields"], records, consts)
    return {"name": ent["name"], "kind": ent["kind"], "size": size, "align": align, "members": members}


def tier1_layouts(reg: dict) -> list[dict]:
    """The layout tree of every Tier-1 record, in registry order. The one model behind
    abi/layout-<major>.txt, test_layout.c, layout.ts and the declared size/align check."""
    records: dict[str, dict] = {}
    consts = constants(reg)
    for _, ent in tier1_records(reg):
        records[ent["name"]] = record_layout(ent, records, consts)
    return list(records.values())


def layout_rows(tree: dict) -> list[dict]:
    """The flat rows of a record, depth first: one row per member and, for a member of an
    unnamed struct or union type, one row per member inside it. `path` is the C member
    designator from the record (`u.vk.value`); a member of a named record type is one row."""
    rows: list[dict] = []

    def walk(nodes: list[dict], prefix: str) -> None:
        for node in nodes:
            path = prefix + node["name"]
            rows.append({"path": path, "offset": node["offset"], "size": node["size"], "ptr": node["ptr"]})
            walk(node["members"], path + ".")

    walk(tree["members"], "")
    return rows


def member_shape(member: dict) -> str | None:
    """Which of the four member shapes a field or arm has: `type` (a scalar, a named Tier-1
    record, either as an array), `ptr` (an ANIRA_PTR slot), `fields` (a named member of an
    unnamed struct type) or `arms` (a named member of an unnamed union type)."""
    keys = [k for k in ("type", "ptr", "fields", "arms") if k in member]
    return keys[0] if len(keys) == 1 else None


def type_strings(ent: dict) -> list[str]:
    """Every C type an entity spells: parameter and return types, member types and pointees."""
    if ent["kind"] in ("function", "callback"):
        return [p["type"] for p in ent.get("params", [])] + [ent["returns"]]
    if ent["kind"] == "typedef":
        return [str(ent.get("type", ""))]
    out: list[str] = []

    def walk(members) -> None:
        for m in members if isinstance(members, list) else []:
            if not isinstance(m, dict):
                continue  # malformed: validate() reports it
            out.extend(str(m[k]) for k in ("type", "ptr") if k in m)
            walk(m.get("fields", []))
            walk(m.get("arms", []))

    walk(ent.get("fields", []))
    walk(ent.get("arms", []))
    return out


def names_type(ent: dict, name: str) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\b", t) for t in type_strings(ent))


def validate(reg: dict) -> None:
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    headers = reg["headers"]
    files = [h["file"] for h in headers]
    if len(set(files)) != len(files):
        err("duplicate header file names")
    # include order: a local include must precede its user (acyclic and ordered)
    seen: list[str] = []
    for h in headers:
        for inc in h.get("includes", []):
            if not inc.startswith("<") and inc not in seen:
                err(f"{h['file']}: includes {inc}, which is not listed before it")
        seen.append(h["file"])
    for h in headers:
        file = h["file"]
        if bool(h.get("draft")) != file.startswith("draft/"):
            err(f"{file}: draft: true and a file under draft/ go together")
        if h.get("draft"):
            if not DRAFT_FILE_RE.match(file):
                err(f"{file}: a draft header is named draft/<name>.h")
            if not str(h.get("guard", DRAFT_GUARD_PREFIX)).startswith(DRAFT_GUARD_PREFIX):
                err(f"{file}: the guard of a draft header starts with {DRAFT_GUARD_PREFIX}")
        else:
            for inc in h.get("includes", []):
                if inc.startswith("draft/"):
                    err(f"{file}: includes the draft header {inc}; only a draft header may")

    enum_names: set[str] = set()
    struct_names: set[str] = set()
    handle_names: set[str] = set()
    typedef_names: set[str] = set()
    function_type_names: set[str] = set()
    struct_ids: list[tuple[str, dict]] = []
    all_names: dict[str, str] = {}

    def claim(name: str, where: str) -> None:
        if name in all_names:
            err(f"{where}: name {name} already used by {all_names[name]}")
        all_names[name] = where

    def members_ok(where: str, scope: str, members) -> bool:
        """A member list is a list of mappings that each carry a name."""
        if isinstance(members, list) and all(isinstance(m, dict) and isinstance(m.get("name"), str) for m in members):
            return True
        err(f"{where}: the members of {scope} are a list of mappings, each with a name")
        return False

    def check_scope(where: str, scope: str, members: list[dict]) -> None:
        """One C scope: member names are unique, and an ANIRA_PTR slot also declares <name>_bits."""
        taken: set[str] = set()
        for m in members:
            for n in [m["name"]] + ([m["name"] + "_bits"] if "ptr" in m else []):
                if n in taken:
                    err(f"{where}: duplicate field {n} in {scope}")
                taken.add(n)

    def check_name(where: str, m: dict) -> None:
        if not MEMBER_RE.match(m["name"]):
            err(f"{where}: field {m['name']} must be lower_case")

    def check_leaf(where: str, m: dict) -> None:
        check_name(where, m)
        if m.get("type") in enum_names:
            err(f"{where}: field {m['name']} is enum-typed; struct fields carry uint32_t")
        if "ptr" not in m and m.get("type", "").replace("const ", "").strip() in TARGET_WIDTH_TYPES:
            err(f"{where}: field {m['name']} has a target-width type ({m['type']}); records carry fixed-width integers only")
        if m.get("type") in function_type_names:
            err(f"{where}: field {m['name']} names the function type {m['type']}; a function type sits in a pointer slot (ptr: {m['type']})")
        if "ptr" in m and "array" in m:
            err(f"{where}: field {m['name']}: an ANIRA_PTR slot takes no array extent")

    def check_arms(where: str, scope: str, arms: list[dict]) -> None:
        if not members_ok(where, scope, arms):
            return
        if not arms:
            err(f"{where}: {scope} has no arms")
        check_scope(where, scope, arms)
        for arm in arms:
            shape = member_shape(arm)
            if shape is None:
                err(f"{where}: arm {arm['name']} of {scope} carries exactly one of type, ptr, fields")
            elif shape == "arms":
                err(f"{where}: arm {arm['name']} of {scope} is a leaf or an unnamed struct (fields), never a union")
            elif shape == "fields":
                check_name(where, arm)
                if not members_ok(where, f"{scope}.{arm['name']}", arm["fields"]):
                    continue
                if not arm["fields"]:
                    err(f"{where}: arm {arm['name']} of {scope} has no fields")
                check_scope(where, f"{scope}.{arm['name']}", arm["fields"])
                for f in arm["fields"]:
                    if member_shape(f) not in ("type", "ptr"):
                        err(f"{where}: {scope}.{arm['name']}.{f['name']}: the struct of an arm holds leaf members only (type or ptr)")
                    else:
                        check_leaf(where, f)
            else:
                check_leaf(where, arm)

    for h, ent in entities(reg):
        kind = ent["kind"]
        where = f"{h['file']}:{ent.get('name', kind)}"
        if h.get("draft") and kind in ("enum", "struct", "union", "handles"):
            err(f"{where}: a draft header declares functions, callbacks, typedefs and defines only")
        if kind == "enum":
            name = ent["name"]
            if not TYPE_RE.match(name):
                err(f"{where}: enum name must match {TYPE_RE.pattern}")
            claim(name, where)
            enum_names.add(name)
            values = ent.get("values", [])
            if not values:
                err(f"{where}: enum without values")
            seen_values: dict[int, str] = {}
            names_in_enum: set[str] = set()
            for v in values:
                vname = v["name"]
                if not MACRO_RE.match(vname):
                    err(f"{where}: enum constant {vname} must match {MACRO_RE.pattern}")
                if vname in names_in_enum:
                    err(f"{where}: duplicate enum constant {vname}")
                names_in_enum.add(vname)
                claim(vname, where)
                if "value" not in v:
                    err(f"{where}: {vname} has no explicit value")
                    continue
                try:
                    iv = int_value(v["value"])
                except RegistryError as exc:
                    err(f"{where}: {vname}: {exc}")
                    continue
                if iv in seen_values:
                    err(f"{where}: {vname} repeats the value of {seen_values[iv]}")
                seen_values[iv] = vname
                if "doc" not in v and vname != ent.get("terminator"):
                    err(f"{where}: {vname} has no doc")
            term = ent.get("terminator")
            if not term or not FORCE32_RE.match(term):
                err(f"{where}: terminator must be named ANIRA_<ENUM>_FORCE32")
            elif seen_values.get(TERMINATOR_VALUE) != term:
                err(f"{where}: terminator {term} must be the value 0x7fffffff")
            elif values[-1]["name"] != term:
                err(f"{where}: terminator {term} must be the last value")
        elif kind == "struct":
            name = ent["name"]
            if not TYPE_RE.match(name):
                err(f"{where}: struct name must match {TYPE_RE.pattern}")
            claim(name, where)
            struct_names.add(name)
            fields = ent.get("fields", [])
            if not members_ok(where, name, fields):
                continue
            if not fields:
                err(f"{where}: struct without fields")
            tier = ent.get("tier")
            check_scope(where, name, fields)
            for f in fields:
                shape = member_shape(f)
                if shape is None:
                    err(f"{where}: field {f['name']} carries exactly one of type, ptr, arms")
                elif shape == "fields":
                    err(f"{where}: field {f['name']}: an unnamed struct type is an arm of a union, never a direct field")
                elif shape == "arms":
                    check_name(where, f)
                    if tier != 1:
                        err(f"{where}: field {f['name']}: only a Tier-1 struct carries a union member")
                    check_arms(where, f"{name}.{f['name']}", f["arms"])
                else:
                    check_leaf(where, f)
            if tier not in (1, 2):
                err(f"{where}: tier must be 1 or 2")
            if tier == 2 and fields:
                first = fields[0]
                ok = (first["name"] == "struct_size" and first.get("type") == "uint32_t") or (
                    first.get("type") == "anira_ext_header"
                )
                if not ok:
                    err(f"{where}: a Tier-2 struct starts with uint32_t struct_size or an anira_ext_header")
                if ent.get("callback_descriptor"):
                    head = [(f["name"], f.get("type")) for f in fields[:3]]
                    if head != [("struct_size", "uint32_t"), ("abi_version", "uint32_t"), ("user_data", "void*")]:
                        err(f"{where}: a callback descriptor starts with {{struct_size, abi_version, user_data}}")
            if tier == 1 and ("size" not in ent or "align" not in ent):
                err(f"{where}: a Tier-1 struct declares its size and align")
            if ent.get("padding_free") and tier != 2:
                err(f"{where}: padding_free: true belongs on a Tier-2 struct (a Tier-1 record has every offset in the layout table)")
            if "init" in ent and not MACRO_RE.match(ent["init"]["name"]):
                err(f"{where}: init macro must match {MACRO_RE.pattern}")
            if tier == 1 and "struct_id" not in ent:
                err(f"{where}: a Tier-1 record names its {STRUCT_ID_ENUM} constant (struct_id:)")
            if "struct_id" in ent:
                struct_ids.append((where, ent))
        elif kind == "union":
            name = ent["name"]
            if not TYPE_RE.match(name):
                err(f"{where}: union name must match {TYPE_RE.pattern}")
            claim(name, where)
            struct_names.add(name)
            if ent.get("tier") != 1:
                err(f"{where}: a union is a Tier-1 record (tier: 1)")
            if "size" not in ent or "align" not in ent:
                err(f"{where}: a Tier-1 union declares its size and align")
            for key in ("fields", "init", "forward", "callback_descriptor"):
                if key in ent:
                    err(f"{where}: a union takes no {key!r} key")
            check_arms(where, name, ent.get("arms", []))
            if "struct_id" not in ent:
                err(f"{where}: a Tier-1 record names its {STRUCT_ID_ENUM} constant (struct_id:)")
            else:
                struct_ids.append((where, ent))
        elif kind == "handles":
            for name in ent["names"]:
                if not TYPE_RE.match(name):
                    err(f"{where}: handle {name} must match {TYPE_RE.pattern}")
                claim(name, where)
                handle_names.add(name)
        elif kind == "typedef":
            if not TYPE_RE.match(ent["name"]):
                err(f"{where}: typedef name must match {TYPE_RE.pattern}")
            claim(ent["name"], where)
            typedef_names.add(ent["name"])
        elif kind == "define":
            if not MACRO_RE.match(ent["name"]):
                err(f"{where}: macro name must match {MACRO_RE.pattern}")
            claim(ent["name"], where)
            if "doc" not in ent:
                err(f"{where}: define without doc")
        elif kind in ("function", "callback"):
            name = ent["name"]
            if not FUNCTION_RE.match(name):
                err(f"{where}: name must match {FUNCTION_RE.pattern}")
            claim(name, where)
            if ent.get("function_type"):
                if kind != "callback":
                    err(f"{where}: function_type: true belongs on a callback")
                function_type_names.add(name)
            if kind == "function" and bool(h.get("draft")) != (ent.get("status") == "draft"):
                err(f"{where}: status: draft and a draft header go together")
            tag = ent.get("thread")
            if tag not in THREAD_TAGS:
                err(f"{where}: thread tag {tag!r} is not in the vocabulary {sorted(THREAD_TAGS)}")
            nb = bool(ent.get("nonblocking"))
            # The dual tag requires the attribute of an entry, not of a callback typedef (the
            # comment at THREAD_TAGS); a [driver-thread] callback (anira_miss_fn) still needs it.
            dual_callback = kind == "callback" and tag == "driver-thread | inference-thread"
            if tag in NONBLOCKING_REQUIRED and not nb and not dual_callback:
                err(f"{where}: [{tag}] requires nonblocking: true")
            if tag and tag.startswith(NONBLOCKING_FORBIDDEN_PREFIXES) and nb:
                err(f"{where}: [{tag}] forbids nonblocking")
            if "doc" not in ent:
                err(f"{where}: no doc")
            if kind == "function" and ent.get("status") not in ("promised", "draft"):
                err(f"{where}: status must be promised or draft")
            types = [p["type"] for p in ent.get("params", [])] + [ent["returns"]]
            for t in types:
                if t.replace("const ", "").strip() in function_type_names:
                    err(f"{where}: {t} is a function type; a parameter or a return names it through a pointer ({t}*)")
                if nb and name not in WIDE_INT_ALLOWLIST and re.search(r"\b(u?int64_t)\b", t) and "*" not in t:
                    err(f"{where}: an ANIRA_NONBLOCKING declaration carries no 64-bit integer ({t})")
                if nb and t.replace(" ", "") == "anira_error*":
                    err(f"{where}: an ANIRA_NONBLOCKING declaration never takes anira_error*")
        elif kind == "verbatim":
            if "text" not in ent:
                err(f"{where}: verbatim without text")
        else:
            err(f"{where}: unknown kind {kind!r}")

    # struct-by-value and Tier-1 layouts need every struct name
    for h, ent in entities(reg):
        if ent["kind"] in ("function", "callback"):
            for t in [p["type"] for p in ent.get("params", [])] + [ent["returns"]]:
                bare = t.replace("const ", "").strip()
                if bare in struct_names and "*" not in t:
                    err(f"{h['file']}:{ent['name']}: struct {bare} passed by value")
    # struct_id: a constant of anira_struct_id (declared in a later header than anira_error), used once
    id_enum = next((e for _, e in entities(reg, "enum") if e["name"] == STRUCT_ID_ENUM), None)
    id_names = {v["name"] for v in id_enum["values"]} - {id_enum.get("terminator")} if id_enum else set()
    id_owner: dict[str, str] = {}
    for where, ent in struct_ids:
        sid = ent["struct_id"]
        if sid not in id_names:
            err(f"{where}: struct_id {sid} is not a constant of {STRUCT_ID_ENUM}")
        elif sid in id_owner:
            err(f"{where}: struct_id {sid} already names {id_owner[sid]}")
        id_owner[sid] = ent["name"]
    # forward: true needs an earlier entity of the same header that names the struct
    for h in headers:
        earlier: list[dict] = []
        for ent in h.get("entities", []):
            if ent["kind"] == "struct" and ent.get("forward") and not any(names_type(e, ent["name"]) for e in earlier):
                err(f"{h['file']}:{ent['name']}: forward: true, but no earlier entity of {h['file']} names {ent['name']}")
            if ent["kind"] in ("function", "callback", "struct", "union", "typedef"):
                earlier.append(ent)
    records: dict[str, dict] = {}
    consts = constants(reg)
    for h, ent in tier1_records(reg):
        try:
            tree = record_layout(ent, records, consts)
        except RegistryError as exc:
            err(str(exc))
            continue
        except (KeyError, TypeError):
            continue  # a malformed member, reported above
        records[ent["name"]] = tree
        if "size" in ent and "align" in ent and (tree["size"] != int(ent["size"]) or tree["align"] != int(ent["align"])):
            part = "arms" if ent["kind"] == "union" else "fields"
            err(
                f"{h['file']}:{ent['name']}: declared {ent['size']}/{ent['align']}, "
                f"the {part} give {tree['size']}/{tree['align']}"
            )
    if errors:
        raise RegistryError("\n".join(errors))


# ------------------------------------------------------------------------------
# Emission helpers
# ------------------------------------------------------------------------------


def wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for para in str(text).strip().split("\n\n"):
        para = " ".join(para.split())
        if lines:
            lines.append("")
        lines.extend(textwrap.wrap(para, width=width, break_long_words=False, break_on_hyphens=False))
    return lines


def doc_block(brief: str | None, body: str | None = None, extra: list[str] | None = None, indent: str = "") -> str:
    inner = len(indent) + 3
    out = [f"{indent}/**"]
    if brief:
        first = wrap(brief, DOC_WIDTH - inner - 7)
        out.append(f"{indent} * @brief {first[0]}")
        out.extend(f"{indent} * {line}" for line in first[1:])
    if body:
        if brief:
            out.append(f"{indent} *")
        out.extend(f"{indent} * {line}".rstrip() for line in wrap(body, DOC_WIDTH - inner))
    for line in extra or []:
        out.append(f"{indent} * {line}".rstrip())
    out.append(f"{indent} */")
    return "\n".join(out)


def thread_line(ent: dict) -> str:
    parts = [f"[{ent['thread']}]"]
    if ent.get("callback_safe"):
        parts.append("[callback-safe]")
    if ent.get("nonblocking"):
        parts.append("ANIRA_NONBLOCKING")
    return " ".join(parts)


def param_text(p: dict) -> str:
    return f"{p['type']} {p['name']}" if p.get("name") else p["type"]


def function_decl(fn: dict) -> str:
    params = fn.get("params", [])
    head = f"ANIRA_API {fn['returns']} ANIRA_CALL {fn['name']}("
    # Every entry is ANIRA_NOEXCEPT (noexcept in C++, nothing in C): an exception that
    # escapes the firewall terminates deterministically instead of MSVC's undefined
    # behaviour for extern "C". ANIRA_NONBLOCKING follows it, as clang wants the effect
    # attribute after the exception specification.
    tail = ") ANIRA_NOEXCEPT" + (" ANIRA_NONBLOCKING" if fn.get("nonblocking") else "") + ";"
    joined = ", ".join(param_text(p) for p in params) or "void"
    line = head + joined + tail
    if len(line) <= WIDTH:
        return line
    pad = " " * len(head)
    if len(params) < 2:
        # One parameter that does not fit beside the tail: the tail wraps, not the parameter.
        return head + joined + ")\n" + pad + tail[1:].lstrip()
    rows = [head + param_text(params[0]) + ","]
    for p in params[1:-1]:
        rows.append(pad + param_text(p) + ",")
    rows.append(pad + param_text(params[-1]) + tail)
    return "\n".join(rows)


def callback_decl(cb: dict) -> str:
    params = cb.get("params", [])
    # function_type: a function TYPE, so that ANIRA_PTR(name, slot) is a function pointer
    star = "" if cb.get("function_type") else "*"
    head = f"typedef {cb['returns']} (ANIRA_CALL{star} {cb['name']})("
    tail = ")" + (" ANIRA_NONBLOCKING" if cb.get("nonblocking") else "") + ";"
    joined = ", ".join(param_text(p) for p in params) or "void"
    line = head + joined + tail
    if len(line) <= WIDTH:
        return line
    pad = " " * len(head)
    rows = [head + param_text(params[0]) + ","]
    for p in params[1:-1]:
        rows.append(pad + param_text(p) + ",")
    rows.append(pad + param_text(params[-1]) + tail)
    return "\n".join(rows)


def function_doc(fn: dict) -> str:
    extra = []
    for p in fn.get("params", []):
        if p.get("name"):
            pdoc = p.get("doc", "")
            plines = wrap(pdoc, DOC_WIDTH - 3 - 8 - len(p["name"])) or [""]
            extra.append(f"@param {p['name']} {plines[0]}".rstrip())
            extra.extend(f"       {line}" for line in plines[1:])
    if fn.get("returns", "void") != "void" and fn.get("return_doc"):
        rlines = wrap(fn["return_doc"], DOC_WIDTH - 3 - 8)
        extra.append(f"@return {rlines[0]}")
        extra.extend(f"        {line}" for line in rlines[1:])
    extra.append("@par Thread contract")
    extra.append(thread_line(fn))
    if fn.get("status") == "draft":
        extra.append("@note Draft: outside the ABI promise until promoted.")
    if fn.get("since"):
        extra.append(f"@since ABI {fn['since']}")
    return doc_block(fn.get("brief") or fn["doc"], fn.get("doc") if fn.get("brief") else None, extra)


def field_decl(f: dict) -> str:
    if "ptr" in f:
        return f"ANIRA_PTR({f['ptr']}, {f['name']});"
    arr = f"[{f['array']}]" if "array" in f else ""
    return f"{f['type']} {f['name']}{arr};"


def emit_members(members: list[dict], indent: str) -> list[str]:
    """The member declarations of a struct or union body. A leaf is one declaration; `fields`
    is a named member of an unnamed struct type, `arms` of an unnamed union type (never truly
    anonymous: ISO C++ has no anonymous structs). The doc of a leaf trails it or sits above
    it; the doc of an aggregate member follows its closing line as a /**< block, so that
    Doxygen attaches it to the member and not to the unnamed type."""
    out: list[str] = []
    for m in members:
        mdoc = m.get("doc")
        if "fields" in m or "arms" in m:
            out.append(f"{indent}{'struct' if 'fields' in m else 'union'} {{")
            out.extend(emit_members(m.get("fields") or m["arms"], indent + "    "))
            close = f"{indent}}} {m['name']};"
            if not mdoc:
                out.append(close)
            elif len(f"{close}  /**< {mdoc} */") <= WIDTH and "\n" not in mdoc:
                out.append(f"{close}  /**< {mdoc} */")
            else:
                lines = wrap(mdoc, DOC_WIDTH - len(indent) - 5)
                out.append(close)
                out.append(f"{indent}/**< {lines[0]}")
                out.extend(f"{indent} *   {line}" for line in lines[1:])
                out.append(f"{indent} */")
            continue
        decl = indent + field_decl(m)
        if mdoc:
            trailing = f"{decl}  /**< {mdoc} */"
            if len(trailing) <= WIDTH and "\n" not in mdoc:
                out.append(trailing)
            else:
                out.append(doc_block(None, mdoc, indent=indent))
                out.append(decl)
        else:
            out.append(decl)
    return out


def emit_struct(ent: dict) -> str:
    out = [doc_block(ent.get("brief") or ent["doc"], ent.get("doc") if ent.get("brief") else None)]
    # forward: true -- `typedef struct X X;` was emitted ahead of the first entity that names X
    forward = bool(ent.get("forward"))
    out.append(f"struct {ent['name']} {{" if forward else f"typedef struct {ent['name']} {{")
    out.extend(emit_members(ent["fields"], "    "))
    out.append("};" if forward else f"}} {ent['name']};")
    init = ent.get("init")
    if init:
        out.append(doc_block(init.get("doc", f"Default initializer of {ent['name']}.")))
        out.append(f"#define {init['name']} ANIRA_INIT({ent['name']}, {init['value']})")
    return "\n".join(out)


def emit_union(ent: dict) -> str:
    out = [doc_block(ent.get("brief") or ent["doc"], ent.get("doc") if ent.get("brief") else None)]
    out.append(f"typedef union {ent['name']} {{")
    out.extend(emit_members(ent["arms"], "    "))
    out.append(f"}} {ent['name']};")
    return "\n".join(out)


def emit_enum(ent: dict) -> str:
    out = [doc_block(ent.get("brief") or ent["doc"], ent.get("doc") if ent.get("brief") else None)]
    out.append(f"typedef enum {ent['name']} {{")
    values = ent["values"]
    for i, v in enumerate(values):
        last = i == len(values) - 1
        decl = f"    {v['name']} = {v['value']}" + ("" if last else ",")
        vdoc = v.get("doc")
        if vdoc:
            trailing = f"{decl}  /**< {vdoc} */"
            if len(trailing) <= WIDTH and "\n" not in vdoc:
                out.append(trailing)
            else:
                out.append(doc_block(None, vdoc, indent="    "))
                out.append(decl)
        else:
            out.append(decl)
    out.append(f"}} {ent['name']};")
    return "\n".join(out)


def emit_define(ent: dict) -> str:
    out = [doc_block(ent.get("brief") or ent["doc"], ent.get("doc") if ent.get("brief") else None)]
    params = ent.get("params")
    if params:
        out.append(f"#define {ent['name']}({', '.join(params)}) {ent['value']}")
    else:
        out.append(f"#define {ent['name']} {ent['value']}")
    return "\n".join(out)


def emit_header(reg: dict, header: dict) -> str:
    file = header["file"]
    guard = header.get("guard") or "ANIRA_ABI_" + re.sub(r"[^A-Z0-9]", "_", file.upper())
    out = [
        "/*",
        f" * anira/abi/{file} -- generated from abi/anira.yml by tools/abi/gen.py.",
        " * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
    ]
    out.append(doc_block(None, None, [f"@file {file}", f"@brief {header['brief']}"] + ([""] + wrap(header["doc"], DOC_WIDTH - 3) if header.get("doc") else [])))
    out.append("")
    includes = header.get("includes", [])
    if includes:
        for inc in includes:
            out.append(f"#include {inc}" if inc.startswith("<") else f"#include <anira/abi/{inc}>")
        out.append("")
    out.append("#ifdef __cplusplus")
    out.append('extern "C" {')
    out.append("#endif")
    out.append("")
    # C names and C spellings in a header the C++ tidy config also sees: typedef (not
    # using), type names inside ANIRA_INIT / ANIRA_PTR that no parentheses can wrap.
    out.append(f"// NOLINTBEGIN({NOLINT_CHECKS})")
    out.append("")
    pending_forward = [e["name"] for e in header.get("entities", []) if e["kind"] == "struct" and e.get("forward")]
    for ent in header.get("entities", []):
        kind = ent["kind"]
        for name in [n for n in pending_forward if names_type(ent, n)]:
            out.append(f"/* Forward declaration: struct {name} is defined below. */")
            out.append(f"typedef struct {name} {name};")
            out.append("")
            pending_forward.remove(name)
        if kind == "verbatim":
            out.append(ent["text"].rstrip("\n"))
        elif kind == "enum":
            out.append(emit_enum(ent))
        elif kind == "struct":
            out.append(emit_struct(ent))
        elif kind == "union":
            out.append(emit_union(ent))
        elif kind == "define":
            out.append(emit_define(ent))
        elif kind == "typedef":
            out.append(doc_block(ent["doc"]))
            out.append(f"typedef {ent['type']} {ent['name']};")
        elif kind == "handles":
            out.append(doc_block(ent["doc"]))
            for name in ent["names"]:
                out.append(f"typedef struct {name} {name};")
        elif kind == "callback":
            out.append(function_doc(ent))
            out.append(callback_decl(ent))
        elif kind == "function":
            out.append(function_doc(ent))
            out.append(function_decl(ent))
        out.append("")
    out.append(f"// NOLINTEND({NOLINT_CHECKS})")
    out.append("")
    out.append("#ifdef __cplusplus")
    out.append("}")
    out.append("#endif")
    out.append("")
    out.append(f"#endif /* {guard} */")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------
# Mirrors
# ------------------------------------------------------------------------------

GENERATED_BANNER = "Generated from abi/anira.yml by tools/abi/gen.py; edit the registry, not this file."


def ts_literal(value) -> str | None:
    """A TS literal for a C integer/float literal, or None when the value is not a literal."""
    s = str(value).strip()
    if re.fullmatch(r"-?0[xX][0-9a-fA-F]+[uUlL]*", s) or re.fullmatch(r"-?\d+[uUlL]*", s):
        return str(int(s.rstrip("uUlL"), 0))
    if re.fullmatch(r"-?\d+\.\d*([eE][-+]?\d+)?[fF]?", s):
        return s.rstrip("fF")
    if re.fullmatch(r"\(\(int64_t\)-?\d+\)", s):
        return re.sub(r"[()int64_t]", "", s).replace("-", "-") + "n"
    return None


def emit_enums_ts(reg: dict) -> str:
    out = [f"// {GENERATED_BANNER}", "// The C names are kept verbatim so a value can be searched across languages.", ""]
    for _, ent in entities(reg, "enum"):
        out.append(f"export const {ent['name']} = {{")
        for v in ent["values"]:
            if v["name"] == ent.get("terminator"):
                continue
            out.append(f"  {v['name']}: {int_value(v['value'])},")
        out.append("} as const")
        out.append(f"export type {ent['name']} = (typeof {ent['name']})[keyof typeof {ent['name']}]")
        out.append("")
    for _, ent in entities(reg, "define"):
        if "ts" in ent:
            out.append(f"export const {ent['name']} = {ent['ts']}")
        elif not ent.get("params"):
            lit = ts_literal(ent["value"])
            if lit is not None:
                out.append(f"export const {ent['name']} = {lit}")
    out.append("")
    return "\n".join(out)


def symbol_lists(reg: dict) -> tuple[list[str], list[str]]:
    promised = sorted(e["name"] for _, e in entities(reg, "function") if e.get("status") == "promised")
    draft = sorted(e["name"] for _, e in entities(reg, "function") if e.get("status") == "draft")
    return promised, draft


def emit_layout_test(reg: dict) -> str:
    out = [
        "/*",
        " * test/abi/generated/test_layout.c -- gate 3, generated from abi/anira.yml by tools/abi/gen.py.",
        " * _Static_asserts pin every enum's width and terminator, the ABI version packing, the dtype",
        " * packing and every Tier-1 layout; main() prints the Tier-1 table that abi/layout-<major>.txt",
        " * commits. Do not edit.",
        " */",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "",
    ]
    for header in reg["headers"]:
        out.append(f"#include <anira/abi/{header['file']}>")
    out.append("")
    for _, ent in entities(reg, "enum"):
        out.append(f'_Static_assert(sizeof({ent["name"]}) == 4, "{ent["name"]} is a 32-bit enum");')
        out.append(f'_Static_assert({ent["terminator"]} == 0x7fffffff, "{ent["name"]} terminator");')
    out.append("")
    out.append('_Static_assert(ANIRA_ABI_VERSION_MAJOR(ANIRA_ABI_VERSION) == ANIRA_ABI_MAJOR, "abi major round trip");')
    out.append('_Static_assert(ANIRA_ABI_VERSION_MINOR(ANIRA_ABI_VERSION) == ANIRA_ABI_MINOR, "abi minor round trip");')
    out.append('_Static_assert(ANIRA_DTYPE_F32 == 0x00012002u, "DLPack float32 packing");')
    out.append('_Static_assert(ANIRA_DTYPE_CODE(ANIRA_DTYPE_F32) == ANIRA_DTYPE_FLOAT, "dtype code");')
    out.append('_Static_assert(ANIRA_DTYPE_BITS(ANIRA_DTYPE_F32) == 32, "dtype bits");')
    out.append('_Static_assert(ANIRA_DTYPE_LANES(ANIRA_DTYPE_F32) == 1, "dtype lanes");')
    out.append("")
    trees = {tree["name"]: tree for tree in tier1_layouts(reg)}
    for _, ent in entities(reg):
        if ent["kind"] not in ("struct", "union"):
            continue
        name = ent["name"]
        if name in trees:
            tree = trees[name]
            out.append(f'_Static_assert(sizeof({name}) == {tree["size"]}, "{name} size");')
            out.append(f'_Static_assert(_Alignof({name}) == {tree["align"]}, "{name} align");')
            for row in layout_rows(tree):
                path = row["path"]
                out.append(f'_Static_assert(offsetof({name}, {path}) == {row["offset"]}, "{name}.{path} offset");')
                if row["ptr"]:
                    out.append(f'_Static_assert(sizeof(((const {name}*)0)->{path}_bits) == 8, "{name}.{path} is an 8-byte slot");')
                else:
                    out.append(f'_Static_assert(sizeof(((const {name}*)0)->{path}) == {row["size"]}, "{name}.{path} size");')
        else:
            fields = ent["fields"]
            first = fields[0]
            if first["name"] == "struct_size":
                out.append(f'_Static_assert(offsetof({name}, struct_size) == 0, "{name}.struct_size first");')
            else:
                out.append(f'_Static_assert(offsetof({name}, {first["name"]}) == 0, "{name}.{first["name"]} (an anira_ext_header) first");')
                out.append(f'_Static_assert(offsetof({name}, {first["name"]}.struct_size) == 0, "{name}: struct_size first through the header");')
            if ent.get("callback_descriptor"):
                out.append(f'_Static_assert(offsetof({name}, abi_version) == 4, "{name}.abi_version second");')
                out.append(f'_Static_assert(offsetof({name}, user_data) == 8, "{name}.user_data third");')
            if ent.get("padding_free"):
                # A descriptor a setter copies within the caller's struct_size: padding bytes are
                # unspecified, so a later tail slot laid over them would read garbage from every
                # older caller. The compiler checks it on every target (a Tier-2 layout differs
                # between ILP32 and LP64, so no offset can be asserted as a number).
                out.append(f"_Static_assert(sizeof({name}) ==")
                for i, f in enumerate(fields):
                    out.append(f"                   sizeof(((const {name}*)0)->{f['name']}){' +' if i + 1 < len(fields) else ','}")
                out.append(f'               "{name} has no implicit padding");')
        out.append("")
    out.append("int main(void) {")
    for tree in trees.values():
        name = tree["name"]
        out.append(f'    printf("{tree["kind"]} {name} size %u align %u\\n", (unsigned)sizeof({name}), (unsigned)_Alignof({name}));')
        for row in layout_rows(tree):
            path = row["path"]
            size_expr = "8u" if row["ptr"] else f"(unsigned)sizeof(((const {name}*)0)->{path})"
            out.append(f'    printf("field {name}.{path} offset %u size %u\\n", (unsigned)offsetof({name}, {path}), {size_expr});')
    out.append("    return 0;")
    out.append("}")
    return "\n".join(out) + "\n"


def emit_link_probe(reg: dict) -> str:
    """test/abi/generated/link_probe.c: a consumer-shaped executable that takes the address of
    every promised and draft entry point, so that a registry entry without a definition (on
    any leg, static ones included) fails the link of anira_abi_link, not a user's build."""
    promised, draft = symbol_lists(reg)
    out = [
        "/*",
        " * test/abi/generated/link_probe.c -- the presence gate anira_abi_link, generated from",
        " * abi/anira.yml by tools/abi/gen.py. Takes the address of every promised and draft entry",
        " * point, so an entry without a definition fails this link, not a consumer's. Do not edit.",
        " */",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "",
    ]
    for header in reg["headers"]:
        out.append(f"#include <anira/abi/{header['file']}>")
    names = promised + draft
    out += [
        "",
        "struct anira_link_entry {",
        "    const char* name;",
        "    uintptr_t address;",
        "};",
        "",
        f"#define ANIRA_LINK_PROBE_COUNT {len(names)}",
        "",
        "/* The addresses are taken by assignment at run time, never in a static initializer:",
        "   MSVC refuses the address of a dllimport there (C4232, identity not guaranteed). */",
        "int main(void) {",
        "    struct anira_link_entry entries[ANIRA_LINK_PROBE_COUNT];",
        "    size_t missing = 0;",
        "    size_t i;",
    ]
    for index, name in enumerate(names):
        out.append(f'    entries[{index}].name = "{name}";')
        out.append(f"    entries[{index}].address = (uintptr_t)&{name};")
    out += [
        "    for (i = 0; i < ANIRA_LINK_PROBE_COUNT; ++i) {",
        "        if (entries[i].address == 0) {",
        '            printf("missing: %s\\n", entries[i].name);',
        "            ++missing;",
        "        }",
        "    }",
        '    printf("%zu of %zu entry points linked\\n", (size_t)ANIRA_LINK_PROBE_COUNT - missing,',
        "           (size_t)ANIRA_LINK_PROBE_COUNT);",
        "    return missing == 0 ? 0 : 1;",
        "}",
    ]
    return "\n".join(out) + "\n"


def emit_layout_table(reg: dict) -> str:
    """The expected abi/layout-<major>.txt, from the registry's natural-alignment model. Byte
    for byte what the printer of test_layout.c writes: both walk layout_rows()."""
    out = []
    for tree in tier1_layouts(reg):
        out.append(f"{tree['kind']} {tree['name']} size {tree['size']} align {tree['align']}")
        for row in layout_rows(tree):
            out.append(f"field {tree['name']}.{row['path']} offset {row['offset']} size {row['size']}")
    return "\n".join(out) + "\n"


def emit_struct_sizes(reg: dict) -> str:
    """src/capi/generated/struct_sizes.inc: the records anira_sizeof() answers for, by id value."""
    enum = next((e for _, e in entities(reg, "enum") if e["name"] == STRUCT_ID_ENUM), None)
    value_of = {v["name"]: int_value(v["value"]) for v in enum["values"]} if enum else {}
    linked = [(value_of[e["struct_id"]], e["struct_id"], e["name"]) for _, e in entities(reg) if e["kind"] in ("struct", "union") and e.get("struct_id") in value_of]
    out = [f"/* {GENERATED_BANNER} */", "/* ANIRA_STRUCT_SIZE(id, type): anira_sizeof(id) returns sizeof(type); any other id returns 0. */"]
    for _, sid, name in sorted(linked):
        out.append(f"ANIRA_STRUCT_SIZE({sid}, {name})")
    return "\n".join(out) + "\n"


def emit_layout_ts(reg: dict) -> str:
    """web/src/abi/layout.ts: size, align and every member offset of the Tier-1 records, from the
    same model as abi/layout-<major>.txt. Offsets count from the start of the exported record;
    a member of a named record type is expanded in place, so reading a tensor needs no
    arithmetic. Emitted the way prettier prints it (web/.prettierrc): a leaf on one line, an
    aggregate one property per line, no line over 90 columns."""
    trees = {tree["name"]: tree for tree in tier1_layouts(reg)}
    out = [
        f"// {GENERATED_BANNER}",
        "// Byte layout of the Tier-1 records, identical on wasm32, LP64 and LLP64: what",
        "// abi/layout-<major>.txt commits. Offsets count from the start of the exported record; a",
        "// member of a record type is expanded in place. `ptr: true` marks an 8-byte ANIRA_PTR",
        "// slot (on wasm32 the pointer is its low 4 bytes), `count` the extent of an array member.",
        "",
    ]

    def members_ts(nodes: list[dict], base: int, indent: str) -> list[str]:
        lines = []
        for node in nodes:
            offset = base + node["offset"]
            children = trees[node["record"]]["members"] if node["record"] and node["count"] is None else node["members"]
            child_base = offset if node["record"] else base
            if not children:
                extra = ", ptr: true" if node["ptr"] else (f", count: {node['count']}" if node["count"] is not None else "")
                lines.append(f"{indent}{node['name']}: {{ offset: {offset}, size: {node['size']}{extra} }},")
                continue
            lines.append(f"{indent}{node['name']}: {{")
            lines.append(f"{indent}  offset: {offset},")
            lines.append(f"{indent}  size: {node['size']},")
            lines.append(f"{indent}  fields: {{")
            lines.extend(members_ts(children, child_base, indent + "    "))
            lines.append(f"{indent}  }},")
            lines.append(f"{indent}}},")
        return lines

    for tree in trees.values():
        out.append(f"export const {tree['name']} = {{")
        out.append(f"  size: {tree['size']},")
        out.append(f"  align: {tree['align']},")
        out.append("  fields: {")
        out.extend(members_ts(tree["members"], 0, "    "))
        out.append("  },")
        out.append("} as const")
        out.append("")
    for line in out:
        if len(line) > TS_WIDTH:
            raise RegistryError(f"web/src/abi/layout.ts: a generated line exceeds {TS_WIDTH} columns, which prettier would re-wrap: {line.strip()}")
    return "\n".join(out)


def emit_status_strings(reg: dict) -> str:
    status = next(e for _, e in entities(reg, "enum") if e["name"] == "anira_status")
    out = [f"/* {GENERATED_BANNER} */", "/* ANIRA_STATUS_TEXT(name, text): the static text anira_status_string() returns. */"]
    for v in status["values"]:
        if v["name"] == status["terminator"]:
            continue
        text = v.get("text", v["name"])
        out.append(f'ANIRA_STATUS_TEXT({v["name"]}, "{text}")')
    return "\n".join(out) + "\n"


def emit_enum_page(ent: dict) -> str:
    title = f"Enum {ent['name']}"
    return f"{title}\n{'=' * len(title)}\n\n.. doxygenenum:: {ent['name']}\n"


# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------


def generate(reg: dict) -> dict[str, str]:
    validate(reg)
    major = int(reg.get("abi_file_major", 0))
    files: dict[str, str] = {}
    for header in reg["headers"]:
        files[f"include/anira/abi/{header['file']}"] = emit_header(reg, header)
    files["web/src/abi/enums.ts"] = emit_enums_ts(reg)
    promised, draft = symbol_lists(reg)
    files[f"abi/symbols-{major}.txt"] = "".join(f"{n}\n" for n in promised)
    files["abi/symbols-draft.txt"] = "".join(f"{n}\n" for n in draft)
    files["web/src/abi/exports_wasm.txt"] = "".join(f"_{n}\n" for n in ["malloc", "free"] + promised + draft)
    files["web/src/abi/layout.ts"] = emit_layout_ts(reg)
    files["src/capi/generated/status_strings.inc"] = emit_status_strings(reg)
    files["src/capi/generated/struct_sizes.inc"] = emit_struct_sizes(reg)
    files["test/abi/generated/test_layout.c"] = emit_layout_test(reg)
    files["test/abi/generated/link_probe.c"] = emit_link_probe(reg)
    files[f"abi/layout-{major}.txt"] = emit_layout_table(reg)
    for _, ent in entities(reg, "enum"):
        files[f"docs/sphinx/api/enum/{ent['name']}.rst"] = emit_enum_page(ent)
    files["abi/anira.json"] = json.dumps(reg, indent=2) + "\n"
    return files


def member_sigs(members: list[dict]) -> list[tuple]:
    """What the layout of a member list depends on, nested arms and their fields included."""
    return [
        (m["name"], m.get("type"), m.get("ptr"), m.get("array"), member_sigs(m.get("fields", [])), member_sigs(m.get("arms", [])))
        for m in members
    ]


def diff_registries(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """(errors, additions): what changed between two registries, classified."""
    errors: list[str] = []
    additions: list[str] = []

    def index(reg: dict) -> dict[str, tuple[str, dict]]:
        idx: dict[str, tuple[str, dict]] = {}
        for h, e in entities(reg):
            if e["kind"] == "handles":
                for n in e["names"]:
                    idx[n] = ("handle", e)
            elif e["kind"] != "verbatim":
                idx[e["name"]] = (e["kind"], e)
        return idx

    old_idx, new_idx = index(old), index(new)
    for name, (kind, ent) in old_idx.items():
        # a draft function is a declaration, not a promise: whatever happens to it is admitted
        draft = kind == "function" and ent.get("status") == "draft"
        if name not in new_idx:
            if draft:
                additions.append(f"draft function {name}: removed (outside the promise)")
            else:
                errors.append(f"removed {kind} {name}")
            continue
        nkind, nent = new_idx[name]
        if nkind != kind:
            errors.append(f"{name}: kind changed {kind} -> {nkind}")
            continue
        if kind == "enum":
            ov = {v["name"]: int_value(v["value"]) for v in ent["values"]}
            nv = {v["name"]: int_value(v["value"]) for v in nent["values"]}
            for vn, val in ov.items():
                if vn not in nv:
                    errors.append(f"enum {name}: removed value {vn}")
                elif nv[vn] != val:
                    errors.append(f"enum {name}: {vn} changed {val} -> {nv[vn]}")
            for vn in nv:
                if vn not in ov:
                    additions.append(f"enum {name}: appended {vn}")
        elif kind in ("function", "callback"):
            sig = lambda e: (e["returns"], [(p["type"], p.get("name")) for p in e.get("params", [])], bool(e.get("function_type")))
            contract = lambda e: (e.get("thread"), bool(e.get("nonblocking")))
            if draft:
                if sig(ent) != sig(nent) or contract(ent) != contract(nent) or bool(ent.get("callback_safe")) != bool(nent.get("callback_safe")):
                    additions.append(f"draft function {name}: changed (outside the promise)")
                if nent.get("status") == "promised":
                    additions.append(f"function {name}: promoted from draft")
                continue
            if sig(ent) != sig(nent):
                errors.append(f"{kind} {name}: signature changed")
            if contract(ent) != contract(nent):
                errors.append(f"{kind} {name}: thread contract changed")
            if ent.get("callback_safe") and not nent.get("callback_safe"):
                errors.append(f"{kind} {name}: callback-safe removed")
            elif nent.get("callback_safe") and not ent.get("callback_safe"):
                additions.append(f"{kind} {name}: callback-safe added")
            if ent.get("status") == "promised" and nent.get("status") != "promised":
                errors.append(f"function {name}: demoted from promised")
        elif kind == "union":
            if member_sigs(ent["arms"]) != member_sigs(nent["arms"]):
                errors.append(f"Tier-1 union {name}: layout changed")
        elif kind == "struct":
            of = member_sigs(ent["fields"])
            nf = member_sigs(nent["fields"])
            if ent.get("tier") == 1 and of != nf:
                errors.append(f"Tier-1 struct {name}: layout changed")
            elif nf[: len(of)] != of:
                errors.append(f"struct {name}: existing fields changed")
            elif len(nf) > len(of):
                additions.append(f"struct {name}: appended {len(nf) - len(of)} tail field(s)")
        elif kind == "define":
            if str(ent.get("value")) != str(nent.get("value")) or ent.get("params") != nent.get("params"):
                errors.append(f"define {name}: value changed")
    for name, (kind, _) in new_idx.items():
        if name not in old_idx:
            additions.append(f"added {kind} {name}")
    return errors, additions


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="anira source root")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the outputs into --repo")
    mode.add_argument("--check", action="store_true", help="compare the outputs with --repo, exit 2 on drift")
    mode.add_argument("--out-dir", help="write the outputs under this root instead of --repo")
    ap.add_argument("--diff-against", metavar="GIT_REF", help="classify the registry changes since GIT_REF")
    ap.add_argument("--strict", action="store_true", help="with --diff-against: exit 1 on a breaking change")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    registry_path = repo / "abi" / "anira.yml"
    if not registry_path.is_file():
        sys.stderr.write(f"gen.py: no registry at {registry_path}\n")
        return 3
    try:
        reg = load_registry(registry_path.read_text(encoding="utf-8"))
        files = generate(reg)
    except RegistryError as exc:
        sys.stderr.write(f"gen.py: registry error:\n{exc}\n")
        return 1

    if args.diff_against:
        shown = subprocess.run(
            ["git", "show", f"{args.diff_against}:abi/anira.yml"], cwd=repo, capture_output=True, text=True
        )
        if shown.returncode != 0:
            sys.stderr.write(f"gen.py: cannot read abi/anira.yml at {args.diff_against}: {shown.stderr.strip()}\n")
            return 3
        old = load_registry(shown.stdout)
        errors, additions = diff_registries(old, reg)
        for line in additions:
            print(f"addition: {line}")
        for line in errors:
            print(f"BREAKING: {line}")
        if not errors and not additions:
            print(f"registry unchanged since {args.diff_against}")
        if errors and args.strict:
            return 1
        if not (args.write or args.check or args.out_dir):
            return 0

    if args.check:
        drift = []
        for rel, content in files.items():
            path = repo / rel
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current != content:
                drift.append(rel)
                a = (current or "").splitlines(keepends=True)
                b = content.splitlines(keepends=True)
                sys.stdout.writelines(list(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=1))[:40])
        if drift:
            print("gen.py: generated files differ from abi/anira.yml (run --write):")
            for rel in drift:
                print(f"  {rel}")
            return 2
        print(f"gen.py: {len(files)} generated files match abi/anira.yml")
        return 0

    root = Path(args.out_dir).resolve() if args.out_dir else repo
    if not (args.write or args.out_dir):
        ap.print_usage()
        return 3
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    print(f"gen.py: wrote {len(files)} files under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
