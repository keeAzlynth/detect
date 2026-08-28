#pragma once

#include "frame.h"
#include "io_manager.h"
#include "pipeline.h"
#include "thread_pool.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

// 异步流水线：读帧线程与主循环（推理/绘图）重叠执行。
//
// 设计要点：
// 1. Pipeline 由外部持有并传入（引用），本类不再自建 Pipeline。
//    此前内部隐式构造 Pipeline 会导致 YOLO/Depth 引擎被完整加载两次，
//    启动时间翻倍且显存/内存占用翻倍（4GB Jetson Nano 上直接触发换页抖动）。
// 2. 帧槽位池（kNumSlots=2）：reader 线程从空闲槽取一个填充，主循环处理完后
//    归还。槽位的 d_raw_img_ 设备缓冲只分配一次，稳态零 cudaMalloc/cudaFree
//    （cudaFree 会隐式同步整个设备，是每帧的隐性停顿点）。
//    同时天然形成背压：在飞帧数 ≤ 2，摄像头模式下帧新鲜度更好。
// 3. inferAsync 提供线程池推理能力，但 Pipeline 的跨帧状态（跟踪器、深度缓存）
//    不是并发安全的：只有在保证同一时刻至多一个推理在执行的前提下才可使用。
class AsyncPipeline {
  public:
    AsyncPipeline(ConfigManager & config_manager, FrameMeta frame_meta, Pipeline & pipeline,
                  int num_threads = 2)
        : pipeline_(pipeline),
          frame_meta_(frame_meta),
          running_(false),
          reader_done_(false) {
        pool_.reset(new ThreadPool(num_threads));
        for (int i = 0; i < kNumSlots; ++i) {
            auto ctx = std::make_unique<FrameInputContext>(0, frame_meta_);
            free_slots_.push(ctx.get());
            slots_.push_back(std::move(ctx));
        }
    }

    ~AsyncPipeline() { stop(); }

    AsyncPipeline(const AsyncPipeline &)             = delete;
    AsyncPipeline & operator=(const AsyncPipeline &) = delete;

    // 启动读帧线程
    void start(IOManager & io_manager) {
        running_    = true;
        reader_done_ = false;

        reader_thread_ = std::thread([this, &io_manager]() {
            int frame_id = 0;
            while (running_) {
                FrameInputContext * slot = acquireFreeSlot();
                if (slot == nullptr) {
                    break;  // stopped
                }
                if (!io_manager.readNextFrame(*slot, false) || slot->raw_img.empty()) {
                    releaseFreeSlot(slot);
                    reader_done_ = true;
                    break;
                }
                slot->setFrameID(frame_id++);
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    frame_queue_.push(slot);
                }
                queue_cv_.notify_one();
            }
            // 唤醒可能在等待的主循环，使其能观察到 reader_done_ 并退出
            queue_cv_.notify_all();
        });
    }

    // 停止流水线并回收读帧线程
    void stop() {
        running_ = false;
        {
            std::lock_guard<std::mutex> lock(free_mutex_);
            free_cv_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_cv_.notify_all();
        }
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
    }

    // 获取下一帧（阻塞等待）。返回 nullptr 表示读帧结束。
    // 返回的指针所有权仍属于本类，使用完毕必须调用 releaseFrame() 归还。
    FrameInputContext * getNextFrame() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || reader_done_; });
        if (frame_queue_.empty()) {
            return nullptr;
        }
        FrameInputContext * ctx = frame_queue_.front();
        frame_queue_.pop();
        return ctx;
    }

    // 归还帧槽位（必须在确认该帧的全部 GPU 工作已完成之后调用；
    // 主循环的推理路径在返回前会同步各自 CUDA 流，因此在主循环末尾调用是安全的）
    void releaseFrame(FrameInputContext * ctx) {
        {
            std::lock_guard<std::mutex> lock(free_mutex_);
            free_slots_.push(ctx);
        }
        free_cv_.notify_one();
    }

    // 异步推理（见类注释中的并发性约束）
    std::future<void> inferAsync(FrameInputContext &  frame_input_context,
                                 InferOutputContext & infer_output_context) {
        return pool_->enqueue([this, &frame_input_context, &infer_output_context]() {
            pipeline_.processOverlap(frame_input_context, infer_output_context);
        });
    }

    Pipeline & getPipeline() { return pipeline_; }

  private:
    FrameInputContext * acquireFreeSlot() {
        std::unique_lock<std::mutex> lock(free_mutex_);
        free_cv_.wait(lock, [this] { return !free_slots_.empty() || !running_; });
        if (free_slots_.empty()) {
            return nullptr;
        }
        FrameInputContext * ctx = free_slots_.front();
        free_slots_.pop();
        return ctx;
    }

    void releaseFreeSlot(FrameInputContext * ctx) {
        {
            std::lock_guard<std::mutex> lock(free_mutex_);
            free_slots_.push(ctx);
        }
        free_cv_.notify_one();
    }

    static constexpr int kNumSlots = 2;

    Pipeline &                            pipeline_;
    FrameMeta                             frame_meta_;
    std::unique_ptr<ThreadPool>           pool_;
    std::thread                           reader_thread_;
    std::atomic<bool>                     running_;
    std::atomic<bool>                     reader_done_;

    std::vector<std::unique_ptr<FrameInputContext>> slots_;
    std::queue<FrameInputContext *>                 free_slots_;
    std::queue<FrameInputContext *>                 frame_queue_;
    std::mutex                                      free_mutex_;
    std::condition_variable                         free_cv_;
    std::mutex                                      queue_mutex_;
    std::condition_variable                         queue_cv_;
};
