import type { AniraBindings } from './abi/bindings'
import { anira_bytes_ownership, anira_engine } from './abi/enums'

/** Anything that carries the typed C ABI of the main instance (an `AniraWeb`). */
export type WithAbi = { readonly abi: AniraBindings }

/** One model entry of a config file, as `fetchModel` sees it. */
export type ModelEntry = {
  /** The entry's position in the config's `models`. */
  index: number
  /** Its `anira_engine` (the config's default engine where the entry names none). */
  engine: number
  /** The reverse-URI id of a custom engine (`ANIRA_ENGINE_CUSTOM`), null otherwise. */
  engineId: string | null
  /** The path as the file spells it. */
  path: string
  /** The path resolved against the config's URL: what is fetched. */
  url: URL
}

export type ModelConfigLoadOptions = {
  /**
   * The URL relative model paths resolve against; `fromUrl` passes the config's own URL,
   * `fromJson` defaults to the page's location.
   */
  baseUrl?: string | URL
  /**
   * Which entries' model files to fetch and attach (`anira_model_config_set_model_bytes`).
   * Default: an entry whose engine this module has compiled in (`anira_enabled_engines`) or a
   * custom engine's (a JavaScript engine reads its entry's bytes); the other entries keep
   * their paths, which no candidate of this build loads.
   */
  fetchModel?: (entry: ModelEntry) => boolean
  /** The fetch the files are read with; default `globalThis.fetch`. */
  fetch?: typeof fetch
}

// anira_backend_id on wasm32: struct_size, engine, provider, then the id pointers; the rows are
// read at a stride of their own (the entry takes the caller's element size).
const BACKEND_ROW_BYTES = 64
const BACKEND_ENGINE_OFFSET = 4

/** The engines this module compiled in (`anira_enabled_engines`), as `anira_engine` values. */
export const enabledEngines = (abi: AniraBindings): Set<number> => {
  const m = abi.module
  const capacity = 16
  const rows = m._malloc(capacity * BACKEND_ROW_BYTES + 4)
  if (rows === 0) throw new RangeError('anira: out of memory listing the enabled engines')
  try {
    const count = rows + capacity * BACKEND_ROW_BYTES
    m.HEAPU8.fill(0, rows, count)
    m.HEAPU32[count >>> 2] = capacity
    abi.anira_enabled_engines(BACKEND_ROW_BYTES, count, rows)
    const engines = new Set<number>()
    const n = Math.min(m.HEAPU32[count >>> 2], capacity)
    for (let i = 0; i < n; i++) {
      engines.add(m.HEAPU32[(rows + i * BACKEND_ROW_BYTES + BACKEND_ENGINE_OFFSET) >>> 2])
    }
    return engines
  } finally {
    m._free(rows)
  }
}

const readBytes = async (fetcher: typeof fetch, url: URL): Promise<Uint8Array> => {
  const response = await fetcher(url)
  if (!response.ok) {
    throw new Error(`anira: fetching ${url.href} failed: ${response.status}`)
  }
  return new Uint8Array(await response.arrayBuffer())
}

const pageUrl = (): string | undefined =>
  (globalThis as { location?: { href: string } }).location?.href

/**
 * A model config (`anira_model_config`) built from the same JSON file a native host loads. The
 * file is parsed with `anira_model_config_from_json` and every model entry whose engine runs
 * here gets its file fetched relative to the config's URL and attached as bytes, so one
 * `*.model.json` (the bundled `extras/models` ones among them) serves native and web unchanged.
 * Destroy it once the handlers built from it are prepared (a config is copied where it is
 * used) or no longer needed.
 */
export class ModelConfig {
  readonly abi: AniraBindings
  /** The `anira_model_config*` the raw entries take. */
  readonly ptr: number

  private constructor(abi: AniraBindings, ptr: number) {
    this.abi = abi
    this.ptr = ptr
  }

  /** Wrap a config pointer this object then owns (destroys). */
  static adopt(anira: WithAbi, ptr: number): ModelConfig {
    return new ModelConfig(anira.abi, ptr)
  }

  /** Fetch a `*.model.json` and its model files; relative paths resolve against `url`. */
  static async fromUrl(
    anira: WithAbi,
    url: string | URL,
    options: Omit<ModelConfigLoadOptions, 'baseUrl'> = {}
  ): Promise<ModelConfig> {
    const fetcher = options.fetch ?? globalThis.fetch
    const configUrl = new URL(url, pageUrl())
    const response = await fetcher(configUrl)
    if (!response.ok) {
      throw new Error(`anira: fetching ${configUrl.href} failed: ${response.status}`)
    }
    return ModelConfig.fromJson(anira, await response.text(), {
      ...options,
      baseUrl: configUrl,
    })
  }

  /** Parse a model config's JSON text and fetch its model files (see `fromUrl`). */
  static async fromJson(
    anira: WithAbi,
    json: string,
    options: ModelConfigLoadOptions = {}
  ): Promise<ModelConfig> {
    const { abi } = anira
    const fetcher = options.fetch ?? globalThis.fetch
    const baseUrl = new URL(options.baseUrl ?? pageUrl() ?? '', pageUrl())
    const config = new ModelConfig(abi, abi.anira_model_config_from_json(json, null))
    try {
      const enabled = enabledEngines(abi)
      const fetchModel =
        options.fetchModel ??
        ((entry: ModelEntry) =>
          entry.engine === anira_engine.ANIRA_ENGINE_CUSTOM || enabled.has(entry.engine))
      const defaults = abi.anira_model_config_default_engine(config.ptr)
      const count = abi.anira_model_config_model_count(config.ptr)
      for (let index = 0; index < count; index++) {
        const path = abi.anira_model_config_model_path(config.ptr, index)
        if (path === null) continue
        let { engine, engine_id: engineId } = abi.anira_model_config_model_engine(
          config.ptr,
          index
        )
        if (engine === anira_engine.ANIRA_ENGINE_NONE) {
          engine = defaults.engine
          engineId = defaults.engine_id
        }
        const entry: ModelEntry = {
          index,
          engine,
          engineId,
          path,
          url: new URL(path, baseUrl),
        }
        if (!fetchModel(entry)) continue
        config.attachBytes(index, await readBytes(fetcher, entry.url))
      }
    } catch (error) {
      config.destroy()
      throw error
    }
    return config
  }

  /** Replace model entry `index`'s source with `bytes`, copied into the config. */
  attachBytes(index: number, bytes: Uint8Array): void {
    const m = this.abi.module
    const ptr = m._malloc(bytes.length)
    if (ptr === 0)
      throw new RangeError(`anira: out of memory for ${bytes.length} model bytes`)
    try {
      m.HEAPU8.set(bytes, ptr)
      this.abi.anira_model_config_set_model_bytes(
        this.ptr,
        index,
        ptr,
        bytes.length,
        anira_bytes_ownership.ANIRA_BYTES_COPY,
        0,
        0
      )
    } finally {
      m._free(ptr)
    }
  }

  /** The config in v3 spelling (`anira_model_config_to_json`). */
  toJson(): string {
    return this.abi.anira_model_config_to_json(this.ptr)
  }

  destroy(): void {
    this.abi.anira_model_config_destroy(this.ptr)
  }
}
