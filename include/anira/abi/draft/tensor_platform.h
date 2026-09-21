/*
 * anira/abi/draft/tensor_platform.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_DRAFT_TENSOR_PLATFORM_H
#define ANIRA_ABI_DRAFT_TENSOR_PLATFORM_H

/**
 * @file draft/tensor_platform.h
 * @brief Draft: the tensor factories of the platform arms that have no measured edge yet.
 *
 * Outside the ABI promise. The arms of anira_memory_handle and the domain and sync values these
 * factories fill are frozen with anira/abi/tensor.h, because a Tier-1 layout cannot grow later;
 * the factories are declarations, not promises, until their platform column is measured. Their
 * names are listed in abi/symbols-draft.txt; promotion moves a name into the promised symbol
 * list and its declaration into anira/abi/tensor.h and never renames it, while a signature may
 * still change before that. No umbrella header includes this file: include it by its own path.
 * In this pre-release every factory here is a field fill; no adapter consumes the arm.
 */

#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief Fills a tensor over a Metal buffer: domain ANIRA_DOMAIN_METAL_BUFFER, handle.mtl. The
 * token is copied as anira_tensor_init_wgpu_buffer copies its fence. Zero-then-fill and
 * the refusals of anira_tensor_init_host.
 * @param tensor The caller's record; NULL is a no-op.
 * @param buffer id<MTLBuffer>, not retained.
 * @param shared_event Copied into acquire (kind ANIRA_SYNC_MTL_SHARED_EVENT: the
 *        id<MTLSharedEvent> and its value), or NULL: already visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @note Draft: outside the ABI promise until promoted.
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_metal(anira_tensor* tensor,
                                                  void* buffer,
                                                  const anira_sync_token* shared_event,
                                                  anira_dtype dtype,
                                                  uint32_t ndim,
                                                  const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over an IOSurface whose plane 0 is the byte-image encoding of packed
 * floats: domain ANIRA_DOMAIN_IOSURFACE, handle.iosurface. Allowlisted 64-bit
 * ANIRA_NONBLOCKING declaration: size mirrors the arm. Zero-then-fill and the refusals
 * of anira_tensor_init_host.
 * @param tensor The caller's record; NULL is a no-op.
 * @param surface IOSurfaceRef, not retained; plane 0 holds the byte image.
 * @param size Size of plane 0 in bytes.
 * @param shared_event Copied into acquire (kind ANIRA_SYNC_MTL_SHARED_EVENT), or NULL: already
 *        visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @note Draft: outside the ABI promise until promoted.
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_iosurface(anira_tensor* tensor,
                                                      void* surface,
                                                      uint64_t size,
                                                      const anira_sync_token* shared_event,
                                                      anira_dtype dtype,
                                                      uint32_t ndim,
                                                      const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over an AHardwareBuffer: domain ANIRA_DOMAIN_AHARDWAREBUFFER,
 * handle.ahb. fence_fd follows the sync_fd rule of anira_tensor_init_dmabuf
 * (ANIRA_SYNC_SYNC_FILE_FD, owned; the caller's again when the call is refused).
 * Zero-then-fill and the refusals of anira_tensor_init_host.
 * @param tensor The caller's record; NULL is a no-op.
 * @param buffer AHardwareBuffer* of format BLOB, not acquired.
 * @param fence_fd A sync_file fd the data is valid after, owned by the tensor's acquire token
 *        from here on; negative: already visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @note Draft: outside the ABI promise until promoted.
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_ahardwarebuffer(anira_tensor* tensor,
                                                            void* buffer,
                                                            int32_t fence_fd,
                                                            anira_dtype dtype,
                                                            uint32_t ndim,
                                                            const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over a Direct3D 12 resource: domain ANIRA_DOMAIN_D3D12, handle.d3d12.
 * The token is copied as anira_tensor_init_wgpu_buffer copies its fence. Zero-then-fill
 * and the refusals of anira_tensor_init_host.
 * @param tensor The caller's record; NULL is a no-op.
 * @param resource ID3D12Resource*, no reference taken.
 * @param shared_handle The NT shared handle of the resource, or NULL; not owned.
 * @param fence Copied into acquire (kind ANIRA_SYNC_D3D12_FENCE: the ID3D12Fence* and its
 *        value), or NULL: already visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @note Draft: outside the ABI promise until promoted.
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_d3d12(anira_tensor* tensor,
                                                  void* resource,
                                                  void* shared_handle,
                                                  const anira_sync_token* fence,
                                                  anira_dtype dtype,
                                                  uint32_t ndim,
                                                  const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_DRAFT_TENSOR_PLATFORM_H */
