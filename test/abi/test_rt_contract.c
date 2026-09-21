/*
 * The real-time contract as a consumer's compiler sees it (gate 6 of
 * docs/anira-v3-architecture.md, section 6a; the seed of the translation unit that gate
 * compiles). One ANIRA_NONBLOCKING function calls every [callback-safe] entry point of
 * anira/abi/tensor.h and anira/abi/draft/tensor_platform.h. Under clang the target adds
 * -Werror=function-effects, so an entry whose generated declaration lost ANIRA_NONBLOCKING no
 * longer compiles here: clang cannot infer the effect of a function defined in another
 * translation unit, the declaration is all it has. Every other compiler sees plain C11 (the
 * macro is empty there) under the strict flags of the header gates. Nothing runs and nothing
 * links: this is an OBJECT, the bodies are test_Tensor.cpp's business.
 *
 * Deliberately absent: anira_tensor_init_dlpack ([main-thread], it can fail with a message)
 * and anira_sync_token_reset / anira_sync_token_dup ([thread-safe, !audio-thread]: close and
 * dup are system calls). Calling one of them from the function below is a compile error under
 * clang, which is the point. The file grows with abi/stage.h (the ring accessors and the stage
 * defaults) and with the handler's [callback-safe] entries.
 */
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/tensor.h>
#include <stddef.h>
#include <stdint.h>

/* Exported on purpose, so that the object holds a symbol a linker would see. */
/* NOLINTNEXTLINE(misc-use-internal-linkage) */
size_t anira_rt_contract_tensor(anira_tensor* tensor,
                                void* memory,
                                const anira_sync_token* fence,
                                const int64_t* shape) ANIRA_NONBLOCKING;
size_t anira_rt_contract_tensor(anira_tensor* tensor,
                                void* memory,
                                const anira_sync_token* fence,
                                const int64_t* shape) ANIRA_NONBLOCKING {
    size_t total = 0;
    /* anira/abi/tensor.h: the nine nonblocking factories. */
    anira_tensor_init_host(tensor, memory, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_pinned(tensor, memory, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_host_planar(tensor, NULL, 0u, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_cuda(tensor, memory, 0, NULL, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_gl_buffer(tensor, 0u, 0u, NULL, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_vulkan(tensor, 0u, 0u, 0u, 0u, 0u, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_opaque_fd(tensor, -1, 0u, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_wgpu_buffer(tensor, memory, 0u, fence, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_dmabuf(tensor, -1, 0u, 0u, -1, ANIRA_DTYPE_F32, 1u, shape);
    /* anira/abi/draft/tensor_platform.h: the four draft factories. */
    anira_tensor_init_metal(tensor, memory, fence, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_iosurface(tensor, memory, 0u, fence, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_ahardwarebuffer(tensor, memory, -1, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_d3d12(tensor, memory, NULL, fence, ANIRA_DTYPE_F32, 1u, shape);
    /* The accessors and anira_sizeof. */
    total += anira_tensor_data_f32(tensor) != NULL ? 1u : 0u;
    total += anira_tensor_data(tensor, ANIRA_DTYPE_F32) != NULL ? 1u : 0u;
    total += anira_tensor_plane(tensor, 0u, ANIRA_DTYPE_F32) != NULL ? 1u : 0u;
    total += anira_tensor_num_elements(tensor);
    total += anira_tensor_extent(tensor, 0u);
    total += anira_sizeof(ANIRA_STRUCT_TENSOR);
    return total;
}
