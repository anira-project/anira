#ifndef ANIRA_RCUREADERSLOT_H
#define ANIRA_RCUREADERSLOT_H

/**
 * @file RcuReaderSlot.h
 * @brief The one reader slot of a group of callers on a thl::RCU, handed out as a read lease.
 *
 * A value that a control thread replaces while real-time threads read it is published through
 * a thl::RCU and read through a reader slot (thl::RCU::add_reader, read_scope(Reader)): a read
 * section on a slot allocates nothing, locks nothing and registers no thread, on whatever
 * thread it runs. A slot has one user at a time, and its successive users must be ordered by
 * happens-before (thl::RCU::add_reader): two threads inside one slot at once, or a slot passed
 * on with no ordering, let the writer free a version a reader still holds.
 *
 * This class makes both impossible by construction. It adds its slot to the RCU once, on a
 * control thread, and never hands the slot out: a reader gets a Lease, which claims the slot by
 * compare-and-swap (acquire), holds a read scope on it open for its own life, and gives the
 * slot back with a release store after the scope has closed (the order of its two members'
 * destructors). A lease is neither copied nor moved, so the claim, the scope and the thread
 * that made them stay together. The reader need not be one thread, nor always the same one
 * (a host moves its callback from thread to thread): whoever claims the free slot has it.
 *
 * One slot, on purpose: the callers of one slot are a group that runs one at a time. Either
 * their contract says so (the entries of a handler's driving thread) and a claim that finds
 * the slot held has met a second caller that must not be there, or their calls are so short
 * that the second one is better told to call again than given a slot of its own (the selection
 * entries). In both cases there is no lease, and the caller answers
 * ANIRA_ERROR_CONCURRENT_CALL. Two groups that must not hold each other up get a slot each, on
 * an RCU each.
 *
 * The writer's side is the RCU's own: update() or replace() publishes, synchronize() waits
 * until a lease claimed before the last publication has been given back. synchronize() frees
 * nothing: a retired version is destroyed by a later publication, so what a published value
 * refers to it must not own (it is freed by the control thread itself, behind synchronize()).
 */

#include <anira/abi/export.h>
#include <tanh/core/threading/RCU.h>

#include <atomic>
#include <optional>
#include <utility>

namespace anira {

/**
 * @brief The reader slot of one group of callers on one thl::RCU<T>, and the lease on it
 * @tparam T The published value's type
 */
template <typename T>
class RcuReaderSlot {
    /// Only the slot makes a lease: its constructor is public for std::optional's sake.
    struct Key {
        explicit Key() = default;
    };

public:
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "a read lease is claimed on real-time threads: it requires a lock-free flag");

    /**
     * @brief Adds the slot to `rcu`
     *
     * Allocates and takes the RCU's writer lock: once, on a control thread, when the RCU is
     * made. The RCU outlives the slot and every lease.
     */
    explicit RcuReaderSlot(const thl::RCU<T>& rcu) : m_rcu(&rcu), m_reader(rcu.add_reader()) {}
    ~RcuReaderSlot() = default;
    RcuReaderSlot(const RcuReaderSlot&) = delete;
    RcuReaderSlot& operator=(const RcuReaderSlot&) = delete;
    RcuReaderSlot(RcuReaderSlot&&) = delete;
    RcuReaderSlot& operator=(RcuReaderSlot&&) = delete;

    /**
     * @brief One reader's hold on the published value: the claimed slot with a read scope open
     *
     * The version data() returns stays valid for the lease's life, whatever is published
     * meanwhile. Made by RcuReaderSlot::claim() alone; neither copied nor moved. The holder
     * keeps it only as long as it reads: a writer's synchronize() waits for it, and every
     * other caller of the slot is refused meanwhile.
     */
    class Lease {
    public:
        Lease(Key /*only the slot*/,
              const thl::RCU<T>& rcu,
              typename thl::RCU<T>::Reader reader,
              std::atomic<bool>& busy) noexcept
            : m_claim{&busy}, m_scope(rcu.read_scope(reader)) {}
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
     * @brief Claims the slot and opens a read scope on it
     *
     * One compare-and-swap. No allocation, no lock, no wait and no loop: legal on a real-time
     * thread, on any thread.
     *
     * @return The lease, or nothing while another caller holds the slot
     */
    std::optional<Lease> claim() const noexcept ANIRA_NONBLOCKING {
        bool busy = false;
        if (!m_busy.compare_exchange_strong(busy,
                                            true,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
            return std::nullopt;
        }
        return std::optional<Lease>(std::in_place, Key{}, *m_rcu, m_reader, m_busy);
    }

private:
    const thl::RCU<T>* m_rcu;
    typename thl::RCU<T>::Reader m_reader;    ///< the RCU's slot; never leaves this object
    mutable std::atomic<bool> m_busy{false};  ///< a lease holds the slot
};

}  // namespace anira

#endif  // ANIRA_RCUREADERSLOT_H
