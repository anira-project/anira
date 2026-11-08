// The resolve hook of resolve-ts.mjs: a relative specifier without an extension that does not
// resolve is retried as the `.ts` file of that name, and onnxruntime-web's wasm factory, which its
// package `exports` leave out, is its dist file (the alias vite.config.ts gives the build).
const ORT_WASM_FACTORY = 'onnxruntime-web/ort-wasm-simd-threaded.mjs'

export async function resolve(specifier, context, nextResolve) {
  if (specifier === ORT_WASM_FACTORY) {
    return nextResolve(
      new URL(
        '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs',
        import.meta.url
      ).href,
      context
    )
  }
  try {
    return await nextResolve(specifier, context)
  } catch (error) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]s$/.test(specifier)) {
      return nextResolve(`${specifier}.ts`, context)
    }
    throw error
  }
}
