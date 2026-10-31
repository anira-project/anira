/*
 * anira/anira_all.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
// clang-format off
#ifndef ANIRA_ALL_H
#define ANIRA_ALL_H

/**
 * @file anira_all.h
 * @brief Every header of the C ABI, the draft headers and the platform companions included.
 *
 * The promised headers of anira/abi/, the draft headers of anira/abi/draft/ (outside the ABI
 * promise: their entry points are declarations until measured) and the platform companions
 * (anira/anira_em.h, which declares nothing outside WebAssembly). For a host that wants the
 * whole surface with one include; a host that stays inside the promise includes the headers it
 * uses.
 */

#include <anira/abi/export.h>  // IWYU pragma: export
#include <anira/abi/status.h>  // IWYU pragma: export
#include <anira/abi/version.h>  // IWYU pragma: export
#include <anira/abi/enums.h>  // IWYU pragma: export
#include <anira/abi/log.h>  // IWYU pragma: export
#include <anira/abi/tensor.h>  // IWYU pragma: export
#include <anira/abi/config.h>  // IWYU pragma: export
#include <anira/abi/context.h>  // IWYU pragma: export
#include <anira/abi/core.h>  // IWYU pragma: export
#include <anira/abi/thread.h>  // IWYU pragma: export
#include <anira/abi/lifecycle.h>  // IWYU pragma: export
#include <anira/abi/processor.h>  // IWYU pragma: export
#include <anira/abi/engine.h>  // IWYU pragma: export
#include <anira/abi/handler.h>  // IWYU pragma: export
#include <anira/abi/draft/tensor_platform.h>  // IWYU pragma: export
#include <anira/anira_em.h>  // IWYU pragma: export

#endif /* ANIRA_ALL_H */
// clang-format on
