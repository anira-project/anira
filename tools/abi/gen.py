#!/usr/bin/env python3
"""anira ABI generator: abi/anira.yml -> the committed C headers and their mirrors.

The registry is the single source of truth for the versioned C ABI. This script
validates it against the header conventions of docs/anira-v3-architecture.md
(section 6a) and emits, deterministically and without any other tool:

  include/anira/abi/<file>.h              the C11 headers (never edit by hand)
  web/src/abi/enums.ts                    the TypeScript mirror of enums and defines
  abi/symbols-<major>.txt                 the promised entry points, sorted
  abi/symbols-draft.txt                   the draft entry points, sorted
  web/src/abi/exports_wasm.txt            the Emscripten export list (_-prefixed)
  src/capi/generated/status_strings.inc   ANIRA_STATUS_TEXT(name, "text") per status
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
    "thread-safe",
    "thread-safe, !audio-thread",
    "drain-thread",
    "any-thread, blocking",
}
NONBLOCKING_REQUIRED = {"driver-thread"}
NONBLOCKING_FORBIDDEN_PREFIXES = ("main-thread", "any-thread, blocking")
WIDE_INT_ALLOWLIST = {
    "anira_now_ns",
    "anira_tensor_init_vulkan",
    "anira_tensor_init_opaque_fd",
    "anira_tensor_init_wgpu_buffer",
    "anira_tensor_init_dmabuf",
}
FUNCTION_RE = re.compile(r"^anira_[a-z0-9_]+$")
TYPE_RE = re.compile(r"^anira_[a-z0-9_]+$")
MACRO_RE = re.compile(r"^ANIRA_[A-Z0-9_]+$")
TERMINATOR_VALUE = 0x7FFFFFFF
FORCE32_RE = re.compile(r"^ANIRA_[A-Z0-9_]+_FORCE32$")

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


def layout_of(struct: dict, struct_sizes: dict[str, tuple[int, int]], consts: dict[str, int]) -> tuple[list[dict], int, int]:
    """Natural-alignment layout of a struct's fields: (rows, total size, align)."""
    offset = 0
    max_align = 1
    rows = []
    for field in struct["fields"]:
        if "ptr" in field:
            size, align = PTR_SLOT
        else:
            base = field["type"]
            if base in SCALAR_SIZES:
                size, align = SCALAR_SIZES[base]
            elif base in struct_sizes:
                size, align = struct_sizes[base]
            else:
                raise RegistryError(
                    f"struct {struct['name']}: field {field['name']} has type {base!r}, "
                    "whose size is unknown (a Tier-1 field must be a fixed-width scalar, "
                    "an ANIRA_PTR slot or another Tier-1 struct)"
                )
            count = field.get("array")
            if count is not None:
                if isinstance(count, int):
                    size *= count
                elif str(count) in consts:
                    size *= consts[str(count)]
                else:
                    try:
                        size *= int_value(count)
                    except RegistryError as exc:
                        raise RegistryError(
                            f"struct {struct['name']}: array extent {count!r} of {field['name']} is neither a "
                            "literal nor an object-like define of the registry"
                        ) from exc
        offset = (offset + align - 1) // align * align
        rows.append({"name": field["name"], "offset": offset, "size": size, "align": align})
        offset += size
        max_align = max(max_align, align)
    total = (offset + max_align - 1) // max_align * max_align
    return rows, total, max_align


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

    enum_names: set[str] = set()
    struct_names: set[str] = set()
    handle_names: set[str] = set()
    typedef_names: set[str] = set()
    all_names: dict[str, str] = {}

    def claim(name: str, where: str) -> None:
        if name in all_names:
            err(f"{where}: name {name} already used by {all_names[name]}")
        all_names[name] = where

    for h, ent in entities(reg):
        kind = ent["kind"]
        where = f"{h['file']}:{ent.get('name', kind)}"
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
            if not fields:
                err(f"{where}: struct without fields")
            for f in fields:
                if f.get("type") in enum_names:
                    err(f"{where}: field {f['name']} is enum-typed; struct fields carry uint32_t")
                if "ptr" not in f and f.get("type", "").replace("const ", "").strip() in TARGET_WIDTH_TYPES:
                    err(f"{where}: field {f['name']} has a target-width type ({f['type']}); records carry fixed-width integers only")
                if not re.match(r"^[a-z][a-z0-9_]*$", f["name"]):
                    err(f"{where}: field {f['name']} must be lower_case")
            tier = ent.get("tier")
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
            if "init" in ent and not MACRO_RE.match(ent["init"]["name"]):
                err(f"{where}: init macro must match {MACRO_RE.pattern}")
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
            tag = ent.get("thread")
            if tag not in THREAD_TAGS:
                err(f"{where}: thread tag {tag!r} is not in the vocabulary {sorted(THREAD_TAGS)}")
            nb = bool(ent.get("nonblocking"))
            if tag in NONBLOCKING_REQUIRED and not nb:
                err(f"{where}: [{tag}] requires nonblocking: true")
            if tag and tag.startswith(NONBLOCKING_FORBIDDEN_PREFIXES) and nb:
                err(f"{where}: [{tag}] forbids nonblocking")
            if "doc" not in ent:
                err(f"{where}: no doc")
            if kind == "function" and ent.get("status") not in ("promised", "draft"):
                err(f"{where}: status must be promised or draft")
            types = [p["type"] for p in ent.get("params", [])] + [ent["returns"]]
            for t in types:
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
    struct_sizes: dict[str, tuple[int, int]] = {}
    consts = constants(reg)
    for h, ent in entities(reg, "struct"):
        if ent.get("tier") == 1:
            try:
                rows, total, align = layout_of(ent, struct_sizes, consts)
            except RegistryError as exc:
                err(str(exc))
                continue
            struct_sizes[ent["name"]] = (total, align)
            if total != int(ent["size"]) or align != int(ent["align"]):
                err(
                    f"{h['file']}:{ent['name']}: declared {ent['size']}/{ent['align']}, "
                    f"the fields give {total}/{align}"
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
    head = f"typedef {cb['returns']} (ANIRA_CALL* {cb['name']})("
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


def emit_struct(ent: dict) -> str:
    out = [doc_block(ent.get("brief") or ent["doc"], ent.get("doc") if ent.get("brief") else None)]
    out.append(f"typedef struct {ent['name']} {{")
    for f in ent["fields"]:
        decl = "    " + field_decl(f)
        fdoc = f.get("doc")
        if fdoc:
            trailing = f"{decl}  /**< {fdoc} */"
            if len(trailing) <= WIDTH and "\n" not in fdoc:
                out.append(trailing)
            else:
                out.append(doc_block(None, fdoc, indent="    "))
                out.append(decl)
        else:
            out.append(decl)
    out.append(f"}} {ent['name']};")
    init = ent.get("init")
    if init:
        out.append(doc_block(init.get("doc", f"Default initializer of {ent['name']}.")))
        out.append(f"#define {init['name']} ANIRA_INIT({ent['name']}, {init['value']})")
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
    for ent in header.get("entities", []):
        kind = ent["kind"]
        if kind == "verbatim":
            out.append(ent["text"].rstrip("\n"))
        elif kind == "enum":
            out.append(emit_enum(ent))
        elif kind == "struct":
            out.append(emit_struct(ent))
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
    struct_sizes: dict[str, tuple[int, int]] = {}
    consts = constants(reg)
    tier1 = []
    for _, ent in entities(reg, "struct"):
        name = ent["name"]
        if ent.get("tier") == 1:
            rows, total, align = layout_of(ent, struct_sizes, consts)
            struct_sizes[name] = (total, align)
            tier1.append((ent, rows))
            out.append(f'_Static_assert(sizeof({name}) == {total}, "{name} size");')
            out.append(f'_Static_assert(_Alignof({name}) == {align}, "{name} align");')
            for f, row in zip(ent["fields"], rows):
                out.append(f'_Static_assert(offsetof({name}, {f["name"]}) == {row["offset"]}, "{name}.{f["name"]} offset");')
                if "ptr" in f:
                    out.append(f'_Static_assert(sizeof(((const {name}*)0)->{f["name"]}_bits) == 8, "{name}.{f["name"]} is an 8-byte slot");')
                else:
                    out.append(f'_Static_assert(sizeof(((const {name}*)0)->{f["name"]}) == {row["size"]}, "{name}.{f["name"]} size");')
        else:
            first = ent["fields"][0]
            if first["name"] == "struct_size":
                out.append(f'_Static_assert(offsetof({name}, struct_size) == 0, "{name}.struct_size first");')
            else:
                out.append(f'_Static_assert(offsetof({name}, {first["name"]}) == 0, "{name}.{first["name"]} (an anira_ext_header) first");')
                out.append(f'_Static_assert(offsetof({name}, {first["name"]}.struct_size) == 0, "{name}: struct_size first through the header");')
            if ent.get("callback_descriptor"):
                out.append(f'_Static_assert(offsetof({name}, abi_version) == 4, "{name}.abi_version second");')
                out.append(f'_Static_assert(offsetof({name}, user_data) == 8, "{name}.user_data third");')
        out.append("")
    out.append("int main(void) {")
    for ent, rows in tier1:
        name = ent["name"]
        out.append(f'    printf("struct {name} size %u align %u\\n", (unsigned)sizeof({name}), (unsigned)_Alignof({name}));')
        for f, row in zip(ent["fields"], rows):
            size_expr = "8u" if "ptr" in f else f"(unsigned)sizeof(((const {name}*)0)->{f['name']})"
            out.append(f'    printf("field {name}.{f["name"]} offset %u size %u\\n", (unsigned)offsetof({name}, {f["name"]}), {size_expr});')
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
    """The expected abi/layout-<major>.txt, from the registry's natural-alignment model."""
    struct_sizes: dict[str, tuple[int, int]] = {}
    consts = constants(reg)
    out = []
    for _, ent in entities(reg, "struct"):
        if ent.get("tier") != 1:
            continue
        rows, total, align = layout_of(ent, struct_sizes, consts)
        struct_sizes[ent["name"]] = (total, align)
        out.append(f"struct {ent['name']} size {total} align {align}")
        for row in rows:
            out.append(f"field {ent['name']}.{row['name']} offset {row['offset']} size {row['size']}")
    return "\n".join(out) + "\n"


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
    files["src/capi/generated/status_strings.inc"] = emit_status_strings(reg)
    files["test/abi/generated/test_layout.c"] = emit_layout_test(reg)
    files["test/abi/generated/link_probe.c"] = emit_link_probe(reg)
    files[f"abi/layout-{major}.txt"] = emit_layout_table(reg)
    for _, ent in entities(reg, "enum"):
        files[f"docs/sphinx/api/enum/{ent['name']}.rst"] = emit_enum_page(ent)
    files["abi/anira.json"] = json.dumps(reg, indent=2) + "\n"
    return files


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
        if name not in new_idx:
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
            sig = lambda e: (e["returns"], [(p["type"], p.get("name")) for p in e.get("params", [])])
            if sig(ent) != sig(nent):
                errors.append(f"{kind} {name}: signature changed")
            if ent.get("thread") != nent.get("thread") or bool(ent.get("nonblocking")) != bool(nent.get("nonblocking")):
                errors.append(f"{kind} {name}: thread contract changed")
            if ent.get("status") == "promised" and nent.get("status") != "promised":
                errors.append(f"function {name}: demoted from promised")
        elif kind == "struct":
            of = [(f["name"], f.get("type"), f.get("ptr"), f.get("array")) for f in ent["fields"]]
            nf = [(f["name"], f.get("type"), f.get("ptr"), f.get("array")) for f in nent["fields"]]
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
