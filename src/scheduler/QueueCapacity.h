#ifndef ANIRA_SCHEDULER_QUEUECAPACITY_H
#define ANIRA_SCHEDULER_QUEUECAPACITY_H
/*
 * The capacity a producer of a moodycamel queue holds, reserved ahead of the real-time path:
 * the one mechanism behind both queues a session feeds, the core's inference queue
 * (Core::reserve_inference_capacity, through SessionElement::m_producer_token) and an exclusive
 * session's pending queue (SessionElement::reserve_pending_dispatch). Each is built with no
 * blocks of its own, and a session's prepare gives its producer the blocks for every chunk of
 * its struct pool that can be in that queue at once; from then on try_enqueue with the token
 * neither allocates nor runs out of room. Control thread only: the reservation allocates.
 *
 * Why the size (moodycamel v1.0.5, concurrentqueue.h, ExplicitProducer::enqueue): an explicit
 * producer keeps its blocks of BLOCK_SIZE slots in a ring. At a block boundary try_enqueue reuses
 * the block ahead once every slot of it was dequeued; when that block still holds an item it takes
 * a block from the queue's initial pool or free list, and fails without one. A ring of k blocks
 * therefore takes `capacity` items in flight at every tail offset when capacity - 1 <=
 * BLOCK_SIZE (k - 1). Items in the queue at once span at least ceil(count / BLOCK_SIZE) blocks, so
 * count = BLOCK_SIZE * ceil((capacity - 1) / BLOCK_SIZE) + 1 placeholders give a ring of
 * ceil((capacity - 1) / BLOCK_SIZE) + 1 blocks: one block for one item, two for 2 .. 33.
 *
 * Why one enqueue_bulk: the bulk form links every block it needs before it publishes the tail
 * once, so no consumer takes a placeholder before the ring holds them all. Placeholders enqueued
 * one by one are visible at once; a consumer that takes each as it arrives lets the producer reuse
 * its block, and the ring stays short.
 */
#include <concurrentqueue.h>

#include <cstddef>
#include <iterator>
#include <vector>

namespace anira::detail {

/**
 * @brief The placeholders one enqueue_bulk through an explicit producer needs so that the producer
 * then takes `capacity` items in flight at every tail offset (see the file's comment)
 *
 * @tparam Queue A moodycamel ConcurrentQueue or BlockingConcurrentQueue (its BLOCK_SIZE)
 * @param capacity The items the producer must hold at once, at least 1
 * @return BLOCK_SIZE * ceil((capacity - 1) / BLOCK_SIZE) + 1
 */
template <class Queue>
constexpr size_t reservation_size(size_t capacity) noexcept {
    constexpr size_t k_block = Queue::BLOCK_SIZE;
    const size_t beyond_one = capacity > 0 ? capacity - 1 : 0;
    return (k_block * ((beyond_one + k_block - 1) / k_block)) + 1;
}

/**
 * @brief Gives the producer of `token` the blocks for `count` items at once: one allocating
 * enqueue_bulk of `count` copies of `placeholder`
 *
 * The placeholders are in the queue afterwards: the caller takes them back, and every consumer
 * of the queue must recognise and skip one.
 *
 * @param queue The queue
 * @param token The producer's token; no other thread may use it during the call
 * @param count reservation_size() of the capacity
 * @param placeholder The item to enqueue `count` times
 * @return False when the allocation failed, and nothing was enqueued
 */
template <class Queue, class T>
bool reserve_capacity(Queue& queue,
                      moodycamel::ProducerToken& token,
                      size_t count,
                      const T& placeholder) {
    std::vector<T> placeholders(count, placeholder);
    return queue.enqueue_bulk(token, std::make_move_iterator(placeholders.begin()), count);
}

}  // namespace anira::detail

#endif  // ANIRA_SCHEDULER_QUEUECAPACITY_H
