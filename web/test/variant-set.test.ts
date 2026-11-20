// The variants of a handler at run time on the web (src/InferenceHandler.ts), the twins of
// anira::InferenceHandler::add_variant and remove_variant: addVariant hands the handler and the
// model config to anira_handler_add_variant and returns the index the C entry wrote,
// removeVariant hands the index to anira_handler_remove_variant and returns nothing, and a
// refused call throws an AniraError with the C status and the message the entry wrote into its
// anira_error, as the C++ members throw an anira::Error. The stack is restored either way. A
// stand-in module.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AniraBindings, type AbiModule } from '../src/abi/bindings'
import { anira_status } from '../src/abi/enums'
import { AniraError } from '../src/AniraError'
import type { Context } from '../src/Context'
import { InferenceHandler } from '../src/InferenceHandler'
import type { ModelConfig } from '../src/ModelConfig'
import type { Pipeline } from '../src/Pipeline'

const HANDLER = 55
const MODEL = 77

// A module whose stack grows down from the end of a 64 KiB heap; the two entries record what
// they were handed and answer with `status`, writing `message` into the anira_error (its
// message at offset 8) when they refuse.
const fakes = (
  status: () => number = () => anira_status.ANIRA_OK,
  message = '',
  nextVariant = 3
) => {
  const buffer = new ArrayBuffer(65536)
  const u8 = new Uint8Array(buffer)
  const u32 = new Uint32Array(buffer)
  let sp = buffer.byteLength
  const text = (ptr: number) => {
    let end = ptr
    while (u8[end] !== 0) end++
    return new TextDecoder().decode(u8.subarray(ptr, end))
  }
  const refuse = (err: number, answer: number) => {
    if (answer < 0 && err !== 0) u8.set(new TextEncoder().encode(`${message}\0`), err + 8)
    return answer
  }
  const added: [number, number][] = []
  const removed: [number, number][] = []
  const module = {
    HEAPU8: u8,
    HEAPU32: u32,
    stackSave: () => sp,
    stackRestore: (to: number) => {
      sp = to
    },
    stackAlloc: (bytes: number) => (sp = (sp - bytes) & ~15),
    UTF8ToString: (ptr: number) => (ptr === 0 ? 'the status text' : text(ptr)),
    _anira_status_string: () => 0,
    _anira_handler_create: (_context: number, _pipeline: number, out: number) => {
      u32[out >>> 2] = HANDLER
      return anira_status.ANIRA_OK
    },
    _anira_handler_add_variant: (
      handler: number,
      model: number,
      variant: number,
      err: number
    ) => {
      added.push([handler, model])
      const answer = status()
      // Not written on a failure, as the C entry.
      if (answer >= 0) u32[variant >>> 2] = nextVariant
      return refuse(err, answer)
    },
    _anira_handler_remove_variant: (handler: number, variant: number, err: number) => {
      removed.push([handler, variant])
      return refuse(err, status())
    },
  }
  const abi = new AniraBindings(module as unknown as AbiModule)
  const handler = InferenceHandler.create(
    { abi, native: () => 1 } as unknown as Context,
    { native: () => 2, inputSpecs: () => [] } as unknown as Pipeline
  )
  const model = { native: () => MODEL } as unknown as ModelConfig
  return { handler, model, added, removed, stack: () => sp }
}

test('addVariant hands the model to the C entry and returns the index it wrote; removeVariant hands the index over', () => {
  const { handler, model, added, removed, stack } = fakes()
  const before = stack()
  assert.equal(handler.addVariant(model), 3)
  assert.deepEqual(added, [[HANDLER, MODEL]])
  assert.equal(handler.removeVariant(1), undefined)
  assert.deepEqual(removed, [[HANDLER, 1]])
  assert.equal(stack(), before, 'the stack is restored')
})

test('a refused add or removal throws an AniraError with the C status and the message of the entry', () => {
  const form = fakes(
    () => anira_status.ANIRA_ERROR_CONFIG,
    'handler: variants[1]: input slot 0: the hop is 512, but 256 in variants[0]'
  )
  const before = form.stack()
  assert.throws(
    () => form.handler.addVariant(form.model),
    (error: AniraError) =>
      error instanceof AniraError &&
      error.status === anira_status.ANIRA_ERROR_CONFIG &&
      error.message ===
        'anira_handler_add_variant: handler: variants[1]: input slot 0: the hop is 512, ' +
          'but 256 in variants[0]'
  )
  assert.equal(form.stack(), before, 'the stack is restored behind a throw')

  const selected = fakes(
    () => anira_status.ANIRA_ERROR_INVALID_STATE,
    'the variant of the selected plan'
  )
  assert.throws(
    () => selected.handler.removeVariant(0),
    (error: AniraError) =>
      error instanceof AniraError &&
      error.status === anira_status.ANIRA_ERROR_INVALID_STATE &&
      error.message === 'anira_handler_remove_variant: the variant of the selected plan'
  )
  assert.equal(selected.stack(), before, 'the stack is restored behind a throw')

  // An entry that wrote no message: the status text stands in.
  const missing = fakes(() => anira_status.ANIRA_ERROR_NOT_FOUND)
  assert.throws(
    () => missing.handler.removeVariant(9),
    (error: AniraError) =>
      error.status === anira_status.ANIRA_ERROR_NOT_FOUND &&
      error.message === 'anira_handler_remove_variant: the status text'
  )
})
