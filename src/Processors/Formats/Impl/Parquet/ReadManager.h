#pragma once

#include <Processors/Formats/Impl/Parquet/Reader.h>

namespace DB::Parquet
{

// I'd like to speak to the manager.
class ReadManager
{
public:
    Reader reader;

    /// To initialize ReadManager:
    ///  1. call manager.reader.prefetcher.init
    ///  2. call manager.reader.init
    ///  3. call manager.init
    /// (I'm trying this style because the usual pattern of passing-through lots of arguments through
    /// layers of constructors seems bad. This seems better but still not great, hopefully there's an
    /// even better way.)
    void init(SharedParsingThreadPoolPtr thread_pool_);

    ~ReadManager();

    /// Not thread safe.
    Chunk read();

    void cancel() noexcept;

private:
    using RowGroup = Reader::RowGroup;
    using RowSubgroup = Reader::RowSubgroup;
    using ColumnChunk = Reader::ColumnChunk;
    using ColumnSubchunk = Reader::ColumnSubchunk;
    using PrimitiveColumnInfo = Reader::PrimitiveColumnInfo;

    enum class TaskScope
    {
        /// Non-last task in a chain of tasks that don't need coordination across columns.
        /// After this task, schedule task for the next stage for the same column chunk.
        /// E.g. BloomFilterHeader -> BloomFilterBlocksOrDictionary,
        /// or MainOffsetIndex -> MainData.
        ColumnChunk,
        /// Part of a group of tasks belonging to a row subgroup.
        /// After this task, decrement RowSubgroup::stage_tasks_remaining.
        /// E.g. PrewhereData.
        RowSubgroup,
        /// Part of a group of tasks belonging to a row group.
        /// After this task, decrement RowGroup::stage_tasks_remaining.
        /// E.g. ColumnIndexAndOffsetIndex.
        RowGroup,
    };

    struct Task
    {
        ReadStage stage;
        TaskScope scope;
        size_t row_group_idx;
        size_t row_subgroup_idx = UINT64_MAX;
        size_t column_idx = UINT64_MAX;

        bool operator<(const Task & rhs) const
        {
            /// Inverted for priority_queue.
            return std::tie(row_group_idx, row_subgroup_idx) > std::tie(rhs.row_group_idx, rhs.row_subgroup_idx);
        }
    };

    struct Stage
    {
        std::atomic<size_t> memory_usage {0};
        double memory_target_fraction = 1;

        std::mutex mutex;

        /// Tasks that are ready to go (i.e. all dependencies are fulfilled) but not scheduled on
        /// thread pool yet. Not counted in `memory_usage`.
        /// Whenever we have enough memory budget available, we move tasks from here into the thread
        /// pool's queue ("schedule" the tasks), increasing `memory_usage` and starting prefetches.
        /// (Why not make worker threads pick up tasks directly from this queue instead of going
        ///  through another queue? To make prefetching work.)
        ///
        /// For Delivering stage, these tasks are picked up by read() instead of going to thread pool.
        ///
        /// TODO: I'm not sure this is the best scheduling strategy. An appealing alternative is to
        ///       not have any explicit task queues (neither here nor in thread pool) and instead have
        ///       each worker thread inspect row group states and pick up the "best" available piece
        ///       of work. E.g. the lowest-numbered column subchunk in the lowest-numbered row group
        ///       (among subchunks and row groups that are ready to run). Maybe with some additional
        ///       limit/heuristic to prevent the reading from running too far ahead of delivery if
        ///       delivery is slow (memory limit is one such stopping condition, but it's probably
        ///       often optimal to stop earlier to reduce working set size). Maybe also prioritize
        ///       (or require) tasks that finished prefetching. And a similar loop for prefetches,
        ///       with some different priority heuristics and different dependency graph
        ///       (can prefetch multiple subchunks ahead).
        ///       Is it simpler (simple good priority function and stopping condition? simple data
        ///       structures for finding the highest-priority available task?)? Is it robust
        ///       (e.g. doesn't get into a situation when most of memory budget is eaten by
        ///       something that won't be deallocated for a long time, crippling parallelism)?
        ///       Does it extract all available parallelism (for decoding and for prefetching)?
        ///       Does it have good memory locality (prefers to do multiple stages on the same column
        ///       chunk/subchunk in quick succession - but not too quick or it'll beat prefetching)?
        std::priority_queue<Task> tasks_to_schedule;
    };

    SharedParsingThreadPoolPtr thread_pool;

    std::shared_ptr<ShutdownHelper> shutdown = std::make_shared<ShutdownHelper>();

    std::array<Stage, size_t(ReadStage::Deallocated)> stages;
    /// First row group that hasn't reached Deallocated stage.
    std::atomic<size_t> first_incomplete_row_group {0};

    /// These are protected by stages[ReadStage::Delivering].mutex.
    std::condition_variable delivery_cv;
    std::exception_ptr exception;

    void scheduleTask(Task task, MemoryUsageDiff * diff);
    void runTask(Task task);
    void scheduleTasksIfNeeded(ReadStage stage_idx, std::unique_lock<std::mutex> &);
    void finishRowGroupStage(size_t row_group_idx, MemoryUsageDiff && diff);
    void finishRowSubgroupStage(size_t row_group_idx, size_t row_subgroup_idx, MemoryUsageDiff && diff);
    void clearColumnChunk(ColumnChunk & column, MemoryUsageDiff * diff);
    void clearRowSubgroup(RowSubgroup & row_subgroup, MemoryUsageDiff * diff);
    void addTasks(std::vector<Task> add_tasks);
    std::vector<Task> makeTasksToReadColumns(size_t row_group_idx, size_t row_subgroup_idx, bool is_prewhere);
    void advanceDeliveryPtrIfNeeded(size_t row_group_idx, MemoryUsageDiff & diff);
    void flushMemoryUsageDiff(MemoryUsageDiff && diff);
};

}
