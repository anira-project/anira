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

export const anira_memory_handle = {
  size: 24,
  align: 8,
  fields: {
    host: {
      offset: 0,
      size: 8,
      fields: {
        ptr: { offset: 0, size: 8, ptr: true },
      },
    },
    cuda: {
      offset: 0,
      size: 16,
      fields: {
        ptr: { offset: 0, size: 8, ptr: true },
        device: { offset: 8, size: 4 },
      },
    },
    gl: {
      offset: 0,
      size: 8,
      fields: {
        id: { offset: 0, size: 4 },
        target: { offset: 4, size: 4 },
      },
    },
    vk: {
      offset: 0,
      size: 24,
      fields: {
        buffer: { offset: 0, size: 8 },
        memory: { offset: 8, size: 8 },
        offset: { offset: 16, size: 8 },
      },
    },
    opaque: {
      offset: 0,
      size: 16,
      fields: {
        fd: { offset: 0, size: 4 },
        reserved: { offset: 4, size: 4 },
        size: { offset: 8, size: 8 },
      },
    },
    mtl: {
      offset: 0,
      size: 8,
      fields: {
        buffer: { offset: 0, size: 8, ptr: true },
      },
    },
    iosurface: {
      offset: 0,
      size: 16,
      fields: {
        surface: { offset: 0, size: 8, ptr: true },
        size: { offset: 8, size: 8 },
      },
    },
    wgpu: {
      offset: 0,
      size: 16,
      fields: {
        buffer: { offset: 0, size: 8, ptr: true },
        offset: { offset: 8, size: 8 },
      },
    },
    dmabuf: {
      offset: 0,
      size: 24,
      fields: {
        fd: { offset: 0, size: 4 },
        reserved: { offset: 4, size: 4 },
        size: { offset: 8, size: 8 },
        offset: { offset: 16, size: 8 },
      },
    },
    ahb: {
      offset: 0,
      size: 8,
      fields: {
        buffer: { offset: 0, size: 8, ptr: true },
      },
    },
    d3d12: {
      offset: 0,
      size: 16,
      fields: {
        resource: { offset: 0, size: 8, ptr: true },
        shared_handle: { offset: 8, size: 8, ptr: true },
      },
    },
    planes: {
      offset: 0,
      size: 16,
      fields: {
        ptrs: { offset: 0, size: 8, ptr: true },
        count: { offset: 8, size: 4 },
        reserved: { offset: 12, size: 4 },
      },
    },
    raw: { offset: 0, size: 24, count: 3 },
  },
} as const

export const anira_sync_token = {
  size: 24,
  align: 8,
  fields: {
    kind: { offset: 0, size: 4 },
    flags: { offset: 4, size: 4 },
    u: {
      offset: 8,
      size: 16,
      fields: {
        cuda_event: { offset: 8, size: 8, ptr: true },
        vk: {
          offset: 8,
          size: 16,
          fields: {
            semaphore: { offset: 8, size: 8 },
            value: { offset: 16, size: 8 },
          },
        },
        gl_sync: { offset: 8, size: 8, ptr: true },
        fd: { offset: 8, size: 4 },
        mtl: {
          offset: 8,
          size: 16,
          fields: {
            object: { offset: 8, size: 8, ptr: true },
            value: { offset: 16, size: 8 },
          },
        },
        d3d12: {
          offset: 8,
          size: 16,
          fields: {
            object: { offset: 8, size: 8, ptr: true },
            value: { offset: 16, size: 8 },
          },
        },
        raw: { offset: 8, size: 16, count: 2 },
      },
    },
  },
} as const

export const anira_tensor = {
  size: 216,
  align: 8,
  fields: {
    domain: { offset: 0, size: 4 },
    dtype: { offset: 4, size: 4 },
    ndim: { offset: 8, size: 4 },
    flags: { offset: 12, size: 4 },
    shape: { offset: 16, size: 64, count: 8 },
    strides: { offset: 80, size: 64, count: 8 },
    byte_offset: { offset: 144, size: 8 },
    handle: {
      offset: 152,
      size: 24,
      fields: {
        host: {
          offset: 152,
          size: 8,
          fields: {
            ptr: { offset: 152, size: 8, ptr: true },
          },
        },
        cuda: {
          offset: 152,
          size: 16,
          fields: {
            ptr: { offset: 152, size: 8, ptr: true },
            device: { offset: 160, size: 4 },
          },
        },
        gl: {
          offset: 152,
          size: 8,
          fields: {
            id: { offset: 152, size: 4 },
            target: { offset: 156, size: 4 },
          },
        },
        vk: {
          offset: 152,
          size: 24,
          fields: {
            buffer: { offset: 152, size: 8 },
            memory: { offset: 160, size: 8 },
            offset: { offset: 168, size: 8 },
          },
        },
        opaque: {
          offset: 152,
          size: 16,
          fields: {
            fd: { offset: 152, size: 4 },
            reserved: { offset: 156, size: 4 },
            size: { offset: 160, size: 8 },
          },
        },
        mtl: {
          offset: 152,
          size: 8,
          fields: {
            buffer: { offset: 152, size: 8, ptr: true },
          },
        },
        iosurface: {
          offset: 152,
          size: 16,
          fields: {
            surface: { offset: 152, size: 8, ptr: true },
            size: { offset: 160, size: 8 },
          },
        },
        wgpu: {
          offset: 152,
          size: 16,
          fields: {
            buffer: { offset: 152, size: 8, ptr: true },
            offset: { offset: 160, size: 8 },
          },
        },
        dmabuf: {
          offset: 152,
          size: 24,
          fields: {
            fd: { offset: 152, size: 4 },
            reserved: { offset: 156, size: 4 },
            size: { offset: 160, size: 8 },
            offset: { offset: 168, size: 8 },
          },
        },
        ahb: {
          offset: 152,
          size: 8,
          fields: {
            buffer: { offset: 152, size: 8, ptr: true },
          },
        },
        d3d12: {
          offset: 152,
          size: 16,
          fields: {
            resource: { offset: 152, size: 8, ptr: true },
            shared_handle: { offset: 160, size: 8, ptr: true },
          },
        },
        planes: {
          offset: 152,
          size: 16,
          fields: {
            ptrs: { offset: 152, size: 8, ptr: true },
            count: { offset: 160, size: 4 },
            reserved: { offset: 164, size: 4 },
          },
        },
        raw: { offset: 152, size: 24, count: 3 },
      },
    },
    manager_ctx: { offset: 176, size: 8, ptr: true },
    release: { offset: 184, size: 8, ptr: true },
    acquire: {
      offset: 192,
      size: 24,
      fields: {
        kind: { offset: 192, size: 4 },
        flags: { offset: 196, size: 4 },
        u: {
          offset: 200,
          size: 16,
          fields: {
            cuda_event: { offset: 200, size: 8, ptr: true },
            vk: {
              offset: 200,
              size: 16,
              fields: {
                semaphore: { offset: 200, size: 8 },
                value: { offset: 208, size: 8 },
              },
            },
            gl_sync: { offset: 200, size: 8, ptr: true },
            fd: { offset: 200, size: 4 },
            mtl: {
              offset: 200,
              size: 16,
              fields: {
                object: { offset: 200, size: 8, ptr: true },
                value: { offset: 208, size: 8 },
              },
            },
            d3d12: {
              offset: 200,
              size: 16,
              fields: {
                object: { offset: 200, size: 8, ptr: true },
                value: { offset: 208, size: 8 },
              },
            },
            raw: { offset: 200, size: 16, count: 2 },
          },
        },
      },
    },
  },
} as const

export const anira_stage_ctx = {
  size: 64,
  align: 8,
  fields: {
    phase: { offset: 0, size: 4 },
    engine: { offset: 4, size: 4 },
    provider: { offset: 8, size: 4 },
    variant: { offset: 12, size: 4 },
    num_inputs: { offset: 16, size: 4 },
    num_outputs: { offset: 20, size: 4 },
    ticket: { offset: 24, size: 4 },
    reserved: { offset: 28, size: 4 },
    frame: { offset: 32, size: 8, ptr: true },
    reserved_ptr0: { offset: 40, size: 8, ptr: true },
    reserved_ptr1: { offset: 48, size: 8, ptr: true },
    reserved_ptr2: { offset: 56, size: 8, ptr: true },
  },
} as const
