/** A failed anira status and the message of its `anira_error` (or the status text). */
export class AniraError extends Error {
  readonly code: number

  constructor(code: number, message: string) {
    super(message)
    this.name = 'AniraError'
    this.code = code
  }
}
