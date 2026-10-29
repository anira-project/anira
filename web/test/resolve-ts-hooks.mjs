// The resolve hook of resolve-ts.mjs: a relative specifier without an extension that does not
// resolve is retried as the `.ts` file of that name.
export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context)
  } catch (error) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]s$/.test(specifier)) {
      return nextResolve(`${specifier}.ts`, context)
    }
    throw error
  }
}
