// Node's type stripping runs the package's TypeScript sources as they are, but the sources
// import each other without an extension (the bundler resolution of tsconfig.json). This hook
// retries such a relative specifier with `.ts`, so `npm test` needs no build and no bundler.
import { register } from 'node:module'

register('./resolve-ts-hooks.mjs', import.meta.url)
