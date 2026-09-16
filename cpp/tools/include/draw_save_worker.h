#pragma once

#include "danger_alert_handler.h"
#include "io_manager.h"
#include "visual_manager.h"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

// 一帧的「延后绘制 + 落盘」任务。
//
// 关键性质：所有字段都是本任务自有数据——out_frame 由 concatenateFrames()
// 独立分配（1280x1440 BGR，约 5.5MB），targets / alert 是小结构拷贝。
// 任务不与流水线的槽位、深度模型的 pinned 缓冲共享任何内存，因此可以在
// 主线程继续跑下一帧推理的同时安全地在工作线程上执行。
//
// 为什么必须自有：raw_img 属于帧槽位（读帧线程会立刻复用），depth_vis 直接
// 包在 DepthModel 的 pinned 主机缓冲上（下一帧深度推理的 D2H 会把它覆盖）。
// 只要把它们拼进独立 buffer，就不存在数据竞争。
struct DrawSaveTask {
    cv::Mat                 out_frame;         // 已拼接输出图（独占所有权）
    int                     draw_frame_id = 0;  // 画在角标上的帧号
    int                     save_index    = 0;  // 落盘文件名用的帧号
    int                     fps           = 0;
    int                     img_w         = 0;  // 原图宽（上半天区宽）
    int                     img_h         = 0;  // 原图高（上半天区高）
    size_t                  num_tracks    = 0;
    AlertMessage            alert;
    std::vector<DrawTarget> targets;

    // 可选：worker 在本帧「绘制完成后」立即 set_value()。
    // 只有需要在本线程窗口显示该帧时才提供（显示必须发生在主线程，
    // 且要看到已绘制的结果）。为空表示调用方不关心绘制完成时机。
    std::shared_ptr<std::promise<void>> drawn_signal;
};

// 绘制/落盘工作线程。
//
// 动机（实测数据）：原主循环是「推理(135ms) → 绘制(7.2ms) → JPEG落盘(10.0ms)」，
// 单帧墙钟 152ms 而推理只占 135ms —— 那 17ms 纯 CPU 时间里 GPU 完全空闲。
// 把绘制与落盘搬到独立线程后，它们与下一帧的 GPU 推理重叠，端到端吞吐提升约 11%。
//
// 队列有界（默认 2）：队满时 enqueue() 阻塞，形成天然背压，
// 既不会让内存随帧数增长，也保证落盘顺序与帧序一致。
class DrawSaveWorker {
  public:
    static constexpr size_t kDefaultMaxQueue = 2;

    DrawSaveWorker(DrawingManager & drawing_manager,
                   IOManager &      io_manager,
                   size_t           max_queue = kDefaultMaxQueue);
    ~DrawSaveWorker();

    DrawSaveWorker(const DrawSaveWorker &)             = delete;
    DrawSaveWorker & operator=(const DrawSaveWorker &) = delete;

    // 入队一帧；队列已满则阻塞等待（背压）
    void enqueue(DrawSaveTask && task);

    // 排空队列并回收线程（析构自动调用，幂等）
    void stop();

  private:
    void run();

    DrawingManager & drawing_manager_;
    IOManager &      io_manager_;

    const size_t             max_queue_;
    std::thread              worker_;
    std::deque<DrawSaveTask> queue_;
    mutable std::mutex       mutex_;
    std::condition_variable  not_empty_;
    std::condition_variable  not_full_;
    bool                     stop_ = false;
};
