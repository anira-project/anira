// Generated from abi/anira.yml by tools/abi/gen.py; edit the registry, not this file.
// Byte layout of the Tier-1 records, identical on wasm32, LP64 and LLP64: what
// abi/layout-<major>.txt commits. Offsets count from the start of the exported record; a
// member of a record type is expanded in place. `ptr: true` marks an 8-byte ANIRA_PTR
// slot (on wasm32 the pointer is its low 4 bytes), `count` the extent of an array member.

export const anira_error = {
  size: 520,
  align: 4,
  fields: {
    status: { offset: 0, size: 4 },
    reserved: { offset: 4, size: 4 },
    message: { offset: 8, size: 512, count: 512 },
  },
} as const

export const anira_log_record = {
  size: 56,
  align: 8,
  fields: {
    level: { offset: 0, size: 4 },
    flags: { offset: 4, size: 4 },
    dropped_before: { offset: 8, size: 4 },
    reserved: { offset: 12, size: 4 },
    sequence: { offset: 16, size: 8 },
    timestamp_ms: { offset: 24, size: 8 },
    monotonic_ns: { offset: 32, size: 8 },
    group: { offset: 40, size: 8, ptr: true },
    message: { offset: 48, size: 8, ptr: true },
  },
} as const
