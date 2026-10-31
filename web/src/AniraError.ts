/**
 * A failed anira status and the message of its `anira_error` (or the status text), the web twin
 * of `anira::Error`: the same `status` field.
 */
export class AniraError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'AniraError'
    this.status = status
  }
}
