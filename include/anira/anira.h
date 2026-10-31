/*
 * anira/anira.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
// clang-format off
#ifndef ANIRA_H
#define ANIRA_H

/**
 * @file anira.h
 * @brief The C ABI of anira 3: every promised header of anira/abi/, in one include.
 *
 * The umbrella of the C11 headers, for a C host and for C++ that calls the C entries: what the
 * ABI promises, in registry order. The draft headers (anira/abi/draft/) and the platform
 * companion anira/anira_em.h are not included (anira/anira_all.h includes them). The C++20 face
 * over the same entries is anira/anira.hpp; a program written against the 2.x C++ classes
 * includes anira/compat/v2.hpp.
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

#endif /* ANIRA_H */
// clang-format on
