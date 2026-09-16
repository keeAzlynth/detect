#include "draw_save_worker.h"

#include <algorithm>
#include <utility>

DrawSaveWorker::DrawSaveWorker(DrawingManager & drawing_manager,
                               IOManager &      io_manager,
                               size_t           max_queue) :
    drawing_manager_(drawing_manager),
    io_manager_(io_manager),
    max_queue_(max_queue == 0 ? 1 : max_queue),
    worker_([this]() { run(); }) {}

DrawSaveWorker::~DrawSaveWorker() { stop(); }

void DrawSaveWorker::enqueue(DrawSaveTask && task) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return stop_ || queue_.size() < max_queue_; });
        if (stop_) {
            // 已在收尾：丢弃。stop() 之前入队的任务仍会被完整排空。
            return;
        }
        queue_.push_back(std::move(task));
    }
    not_empty_.notify_one();
}

void DrawSaveWorker::run() {
    while (true) {
        DrawSaveTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_empty_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stop_) {
                    return;  // 已 stop 且队列排空
                }
                continue;  // 虚假唤醒
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();  // 腾出一个槽位，唤醒可能阻塞的入队方

        // 绘制只作用于输出图的上半区（与原图同尺寸的视图，零像素拷贝），
        // 这样 drawTrackedObject 内部的 img.cols/img.rows 边界裁剪仍然是原图尺寸。
        if (!task.out_frame.empty() && task.img_w > 0 && task.img_h > 0) {
            const int top_w = std::min(task.img_w, task.out_frame.cols);
            const int top_h = std::min(task.img_h, task.out_frame.rows);
            cv::Mat   top   = task.out_frame(cv::Rect(0, 0, top_w, top_h));
            for (const auto & target : task.targets) {
                drawing_manager_.drawTrackedObject(top, target, task.alert);
            }
            drawing_manager_.drawGlobalInfo(top, task.draw_frame_id, task.fps, task.num_tracks);
        }

        // 绘制完成即唤醒等待方（主线程的窗口显示）：显示只需看到画好的帧，
        // 之后的 JPEG 编码/写盘继续在后台跑，不必让主线程等。
        if (task.drawn_signal) {
            task.drawn_signal->set_value();
            task.drawn_signal.reset();
        }

        io_manager_.saveFrame(task.out_frame, task.save_index);
    }
}

void DrawSaveWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            return;
        }
        stop_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}
