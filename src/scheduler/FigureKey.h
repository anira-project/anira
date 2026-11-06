#ifndef ANIRA_SCHEDULER_FIGUREKEY_H
#define ANIRA_SCHEDULER_FIGUREKEY_H
/*
 * What a model's figure is a figure of: the key of the core's figure index (Core::find_figure,
 * claim_figure, publish_figure). A prepare whose contract warms up asks the index for every
 * plan's key before it measures: a figure stored by an earlier prepare of the key (any handler,
 * any budget kind) is reused, and the plan runs no pass (ANIRA_WARMUP_OUTCOME_REUSED).
 */
#include <memory>
#include <string>

#include "../engines/Adapter.h"

namespace anira {

/**
 * @brief The key of a model's figure
 *
 * The loaded model's record (its engine, provider, shapes, file or bytes), for a registered
 * engine its id, and the identity of the processor's inference hooks, the one part of a pass's
 * timed job that is the handler's pipeline rather than the model's: none for a processor that
 * fills neither before_inference nor after_inference (or no processor), the processor's id where
 * it has one, else its registration. Ids rather than objects, so that identical plugins, which
 * register their own engine and processor objects under the same ids, share the figure and
 * report the same latency: an id names one implementation. The 2.x catch-all id anira.v2.custom,
 * under which compat/v2.hpp registers every 2.x custom backend, names none, so such an engine is
 * keyed by its registration (its carrier). What decides the inference time is in the key; the
 * host rate, the block range and the wait ratio, which decide the latency and the real-time
 * factor, are not. A registration is held weakly and compared by owner, so an entry never matches
 * an object that replaced a destroyed one at its address.
 */
struct FigureKey {
    /// The loaded model's record, without its bytes' owner (m_bytes_owner holds it weakly: the
    /// index keeps no model in memory)
    engine::Model m_model;
    std::weak_ptr<const void> m_bytes_owner;  ///< the owner of the record's bytes (else empty)
    bool m_has_bytes = false;                 ///< whether m_bytes_owner names one
    std::string m_engine_id;                  ///< a registered engine's id (else empty)
    std::weak_ptr<const void> m_carrier;      ///< its carrier, under anira.v2.custom alone
    bool m_by_registration = false;           ///< whether m_carrier names one
    std::string m_hook_id;                    ///< the processor's id, when it fills a hook
    std::weak_ptr<const void> m_hook_owner;   ///< its registration, when it has no id
    bool m_hooks_by_registration = false;     ///< whether m_hook_owner names one
};

}  // namespace anira

#endif  // ANIRA_SCHEDULER_FIGUREKEY_H
