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

// 异步流水线：读帧 → 推理 → 绘图，三阶段流水线重叠执行
// 使用指针队列避免 FrameInputContext 拷贝问题（unique_ptr 不可拷贝）
class AsyncPipeline {
  public:
    AsyncPipeline(ConfigManager & config_manager, FrameMeta frame_meta, int num_threads = 2)
        : pipeline_(config_manager, frame_meta),
          frame_meta_(frame_meta),
          running_(false),
          reader_done_(false) {
        pool_.reset(new ThreadPool(num_threads));
    }

    ~AsyncPipeline() { stop(); }

    // 启动流水线
    void start(IOManager & io_manager) {
        running_ = true;
        reader_done_ = false;

        // 启动读帧线程
        reader_thread_ = std::thread([this, &io_manager]() {
            int frame_id = 0;
            while (running_) {
                // 创建新的 FrameInputContext
                auto frame_ctx = std::make_unique<FrameInputContext>(frame_id, frame_meta_);
                if (!io_manager.readNextFrame(*frame_ctx, false) || frame_ctx->raw_img.empty()) {
                    reader_done_ = true;
                    break;
                }
                // 将帧放入队列
                {
                    std::unique_lock<std::mutex> lock(frame_mutex_);
                    frame_queue_.push(std::move(frame_ctx));
                    frame_id++;
                }
                frame_cv_.notify_one();
            }
        });
    }

    // 停止流水线
    void stop() {
        running_ = false;
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
    }

    // 获取下一帧（阻塞等待），返回 nullptr 表示结束
    std::unique_ptr<FrameInputContext> getNextFrame() {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_cv_.wait(lock, [this] { return !frame_queue_.empty() || reader_done_; });
        if (frame_queue_.empty()) {
            return nullptr;
        }
        auto frame = std::move(frame_queue_.front());
        frame_queue_.pop();
        return frame;
    }

    // 异步推理
    std::future<void> inferAsync(FrameInputContext & frame_input_context,
                                 InferOutputContext & infer_output_context) {
        return pool_->enqueue([this, &frame_input_context, &infer_output_context]() {
            pipeline_.processOverlap(frame_input_context, infer_output_context);
        });
    }

    Pipeline & getPipeline() { return pipeline_; }

  private:
    Pipeline                              pipeline_;
    FrameMeta                             frame_meta_;
    std::unique_ptr<ThreadPool>           pool_;
    std::thread                           reader_thread_;
    std::atomic<bool>                     running_;
    std::atomic<bool>                     reader_done_;

    std::queue<std::unique_ptr<FrameInputContext>> frame_queue_;
    std::mutex                            frame_mutex_;
    std::condition_variable               frame_cv_;
};
