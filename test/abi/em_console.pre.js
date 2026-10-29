// The console of em_console.c, before the module's glue binds console.error for its stderr:
// every line that carries the marker is counted by the console method that printed it.
globalThis.aniraConsoleGate = { warn: 0, error: 0, other: 0 }
for (const method of ['log', 'info', 'warn', 'error', 'debug']) {
  const original = console[method].bind(console)
  console[method] = (...args) => {
    if (args.some((arg) => String(arg).includes('anira-console-gate'))) {
      const key = method === 'warn' || method === 'error' ? method : 'other'
      globalThis.aniraConsoleGate[key] += 1
    }
    original(...args)
  }
}
