#ifndef ANIRA_RCUREADERS_H
#define ANIRA_RCUREADERS_H

/**
 * @file RcuReaders.h
 * @brief A fixed pool of reader slots of one thl::RCU, handed out as read leases.
 *
 * A value that a control thread replaces while real-time threads read it is published through
 * a thl::RCU and read through a reader slot (thl::RCU::add_reader, read_scope(Reader)): a read
 * section on a slot allocates nothing, locks nothing and registers no thread, on whatever
 * thread it runs. A slot has one user at a time, and its successive users must be ordered by
 * happens-before (thl::RCU::add_reader): two threads inside one slot at once, or a slot passed
 * on with no ordering, let the writer free a version a reader still holds.
 *
 * The pool makes both impossible by construction. It adds its slots to the RCU once, on a
 * control thread, and never hands a slot out: a reader gets a Lease, which claims a free slot
 * by compare-and-swap (acquire), holds a read scope on it open for its own life, and gives the
 * slot back with a release store after the scope has closed (the order of its two members'
 * destructors). A lease is neither copied nor moved, so the claim, the scope and the thread
 * that made them stay together. The readers need not be one thread, nor always the same one:
 * a claim takes the first free slot and falls back to the others, and when every slot is busy
 * there is no lease (the caller's ANIRA_ERROR_CAPACITY).
 *
 * The writer's side is the RCU's own: update() or replace() publishes, synchronize() waits
 * until every lease claimed before the last publication has been given back. synchronize()
 * frees nothing: a retired version is destroyed by a later publication, so what a published
 * value refers to it must not own (it is freed by the control thread itself, behind
 * synchronize()).
 */

#include <anira/abi/export.h>
#include <tanh/core/threading/RCU.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

namespace anira {

/**
 * @brief The reader slots of one thl::RCU<T> and the leases on them
 * @tparam T The published value's type
 */
template <typename T>
class RcuReaders {
    struct Slot;
    /// Only the pool makes a lease: its constructor is public for std::optional's sake.
    struct Key {
        explicit Key() = default;
    };

public:
    /// The slots of a pool: how many leases can be out at once.
    static constexpr size_t k_num_slots = 4;

    static_assert(std::atomic<bool>::is_always_lock_free,
                  "a read lease is claimed on real-time threads: it requires a lock-free flag");

    /**
     * @brief Adds the pool's slots to `rcu`
     *
     * Allocates and takes the RCU's writer lock: once, on a control thread, when the RCU is
     * made. The RCU outlives the pool and every lease.
     */
    explicit RcuReaders(const thl::RCU<T>& rcu) : m_rcu(&rcu) {
        for (Slot& slot : m_slots) { slot.m_reader = rcu.add_reader(); }
    }
    ~RcuReaders() = default;
    RcuReaders(const RcuReaders&) = delete;
    RcuReaders& operator=(const RcuReaders&) = delete;
    RcuReaders(RcuReaders&&) = delete;
    RcuReaders& operator=(RcuReaders&&) = delete;

    /**
     * @brief One reader's hold on the published value: a claimed slot with a read scope open
     *
     * The version data() returns stays valid for the lease's life, whatever is published
     * meanwhile. Made by RcuReaders::claim() alone; neither copied nor moved. The holder
     * keeps it only as long as it reads: a writer's synchronize() waits for it.
     */
    class Lease {
    public:
        Lease(Key /*only the pool*/, const thl::RCU<T>& rcu, Slot& slot) noexcept
            : m_claim{&slot.m_busy}, m_scope(rcu.read_scope(slot.m_reader)) {}
        ~Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;

        /// The version that was published when the lease was claimed.
        const T& data() const noexcept ANIRA_NONBLOCKING { return m_scope.data(); }
        const T* operator->() const noexcept ANIRA_NONBLOCKING { return &m_scope.data(); }

    private:
        /// The claim on the slot, given back by its destructor: a release store, which the next
        /// claim's acquire compare-and-swap reads, so everything this lease did on the slot
        /// happens before the next lease's first step there.
        struct Claim {
            std::atomic<bool>* m_busy;
            explicit Claim(std::atomic<bool>* busy) noexcept : m_busy(busy) {}
            ~Claim() { m_busy->store(false, std::memory_order_release); }
            Claim(const Claim&) = delete;
            Claim& operator=(const Claim&) = delete;
            Claim(Claim&&) = delete;
            Claim& operator=(Claim&&) = delete;
        };

        // The order of these two members is the contract: m_scope is destroyed first (the read
        // section closes), m_claim last (the slot is free for its next user).
        Claim m_claim;
        typename thl::RCU<T>::ReadScope m_scope;
    };

    /**
     * @brief Claims a free slot and opens a read scope on it
     *
     * The first free slot, by compare-and-swap; the others in turn when it is taken. No
     * allocation, no lock, no wait: legal on a real-time thread, on any thread.
     *
     * @return The lease, or nothing when every slot is busy
     */
    std::optional<Lease> claim() const noexcept ANIRA_NONBLOCKING {
        for (Slot& slot : m_slots) {
            bool busy = false;
            if (slot.m_busy.compare_exchange_strong(busy,
                                                    true,
                                                    std::memory_order_acquire,
                                                    std::memory_order_relaxed)) {
                return std::optional<Lease>(std::in_place, Key{}, *m_rcu, slot);
            }
        }
        return std::nullopt;
    }

private:
    struct Slot {
        typename thl::RCU<T>::Reader m_reader;  ///< the RCU's slot; never leaves the pool
        std::atomic<bool> m_busy{false};        ///< a lease holds the slot
    };

    const thl::RCU<T>* m_rcu;
    mutable std::array<Slot, k_num_slots> m_slots;
};

}  // namespace anira

#endif  // ANIRA_RCUREADERS_H
