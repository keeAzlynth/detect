
#include "config_manager.h"
#include "danger_alert_handler.h"
#include "draw_save_worker.h"
#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"
#include "prof_stats.h"
#include "scope_timer.h"
#include "async_pipeline.h"
#include "visual_manager.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <string>
#include <utility>
#include <vector>

// 收集本帧需要绘制的目标（含颜色）。
//
// 原先这里是 drawOneFrame()：在同一函数里既收集目标、又真的去画、还负责拼接。
// 现在绘制整体搬到了 DrawSaveWorker 上执行，主线程只负责把这几个纯数据拷出来。
// 颜色在主线程解析（pipeline.getColor），避免工作线程与下一帧的 tracker 状态更新竞争。
[[nodiscard]] std::vector<DrawTarget> collectDrawTargets(const InferOutputContext & infer_output_context,
                                                        Pipeline &                 pipeline) {
    std::vector<DrawTarget> targets;
    targets.reserve(infer_output_context.tracked_objects.size());

    for (const auto & track : infer_output_context.tracked_objects) {
        if (track.tlwh_[2] * track.tlwh_[3] <= 20) continue;
        if (infer_output_context.motion_records.find(track.track_id_) ==
            infer_output_context.motion_records.end()) {
            continue;
        }
        DrawTarget target;
        target.tlwh[0]  = track.tlwh_[0];
        target.tlwh[1]  = track.tlwh_[1];
        target.tlwh[2]  = track.tlwh_[2];
        target.tlwh[3]  = track.tlwh_[3];
        target.class_id = track.class_id_;
        target.track_id = track.track_id_;
        target.color    = pipeline.getColor(track.track_id_);
        targets.push_back(target);
    }
    return targets;
}

namespace {

// 首帧预热：TensorRT/cuDNN 的第一次推理包含 tactic 选择与延迟初始化，
// 实测在 Jetson Nano 上 YOLO ~11s、Depth ~17s。在进入主循环前用一帧真实数据
// 把两个模型都跑热，避免实时流水线在首帧卡顿数十秒。
void warmupPipeline(Pipeline & pipeline, IOManager & io_manager, const FrameMeta & frame_meta) {
    FrameInputContext warm_ctx(0, frame_meta);
    if (!io_manager.readNextFrame(warm_ctx, false) || warm_ctx.raw_img.empty()) {
        return;
    }
    InferOutputContext dummy_output;
    pipeline.processOverlap(warm_ctx, dummy_output);
}

}  // namespace

int run(char * video_path, char * config_path) {
    // 读取配置文件 - 使用单例模式
    ConfigManager   config_manager(config_path);
    // 初始化日志系统
    LoggerManager & logger_manager = LoggerManager::getInstance(config_manager);
    LoggerManager::logConfig(config_manager);
    APP_INFO("Application started with video: {}", std::string(video_path));
    // 文件读写，落盘保存, 以及视频读取（包括模拟相机延迟）
    IOManager          io_manager(config_manager);
    FrameMeta          frame_meta = io_manager.Init(video_path);
    // 推理流水线（负责目标检测、深度估计、跟踪、运动状态判断等核心功能）
    Pipeline           pipeline(config_manager, frame_meta);
    // 绘制管理器（负责绘制结果）
    DrawingManager     drawing_manager(V_CLASS_NAMES);
    // 显示管理器（负责窗口管理、显示、鼠标点击等）
    DisplayManager     display_manager(config_manager, "Detection Result",
                                       cv::Size(frame_meta.img_w, frame_meta.img_h * 2));
    // 报警管理器（负责报警信息生成）
    DangerAlertHandler alert_handler(config_manager);
    // 绘制/落盘工作线程：把绘制(~7ms) 与 JPEG 落盘(~10ms) 移出主循环关键路径，
    // 让这段纯 CPU 时间与下一帧的 GPU 推理重叠。
    // 声明位置在 io_manager / drawing_manager 之后 —— 析构顺序与之相反，
    // 工作线程会先把队列排空并 join，之后才轮到这两个管理器析构。
    DrawSaveWorker     draw_save_worker(drawing_manager, io_manager);
    int                num_frames = 0;
    double             total_us   = 0;
    InferOutputContext infer_output_context;

    // 首帧预热（在读帧线程启动前执行，使用视频/相机的第一帧）
    warmupPipeline(pipeline, io_manager, frame_meta);

    // 异步流水线模式：读帧与推理/绘图重叠执行。
    // Pipeline 只实例化一次，由 AsyncPipeline 引用复用。
    AsyncPipeline async_pipeline(config_manager, frame_meta, pipeline);
    async_pipeline.start(io_manager);

    // 端到端墙钟计时起点：口径 = 读帧等待 + 推理 + 目标收集 + 落盘（与「推理管线 fps」对照）
    const auto wall_clock_start = std::chrono::steady_clock::now();

    while (true) {
        // 单帧墙钟：包住整轮迭代（读帧等待 + 推理 + 绘图 + 落盘），
        // 用于和「推理管线」口径对照，暴露 GPU 空转的时间
        PROF_SCOPE("00.wall/frame");
        PROF_FRAME();
        FrameInputContext * frame_ctx = nullptr;
        {
            PROF_SCOPE("01.wait_frame");
            // 从异步队列获取帧（指针所有权仍在 AsyncPipeline，用完归还）
            frame_ctx = async_pipeline.getNextFrame();
        }
        if (frame_ctx == nullptr) {
            break;
        }
        num_frames = frame_ctx->frame_id;

        {
            PROF_SCOPE("02.infer_pipeline");
#if defined(ENABLE_TIMER)
            // 执行推理流水线
            std::string name = "Infer Pipeline";
            if (config_manager.isOverlapEnabled()) {
                DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, pipeline, processOverlap, *frame_ctx,
                                                       infer_output_context);
            } else {
                DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, pipeline, process, *frame_ctx,
                                                       infer_output_context);
            }
            total_us += ScopedTimer::GetScopedTimers()[name].back();
#else
            if (config_manager.isOverlapEnabled()) {
                pipeline.processOverlap(*frame_ctx, infer_output_context);
            } else {
                pipeline.process(*frame_ctx, infer_output_context);
            }
#endif
        }
        num_frames++;
        if (num_frames % 100 == 0) {
            // 两个口径同时打印：pipeline fps 只统计推理管线（历史基准口径），
            // wall fps 是端到端真实吞吐（含读帧/绘制/落盘的重叠情况）。
            const double wall_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_clock_start)
                    .count();
            APP_INFO("Processing frame {} ({:.2f} fps pipeline, {:.2f} fps wall)", num_frames,
                     (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0),
                     (wall_s > 0.0 ? (num_frames / wall_s) : 0));
        }
        AlertMessage alert;
        {
            PROF_SCOPE("03.alert");
            // 生成报警信息
            alert = alert_handler.buildAlert(*frame_ctx, infer_output_context);
        }
        if (alert.has_value()) {
            io_manager.sendAlert(alert);
        }

        // 拼接输出图：两次连续内存拷贝，产出本任务独占的 buffer（约 5.5MB）。
        // 只有这一步和目标收集留在主线程——都是零散小拷贝；
        // 真正贵的「逐目标绘制」与「JPEG 编码 + 写盘」已交给工作线程。
        DrawSaveTask task;
        {
            PROF_SCOPE("04.concat");
            task.out_frame = drawing_manager.concatenateFrames(frame_ctx->raw_img,
                                                              infer_output_context.depth_vis);
        }
        {
            PROF_SCOPE("05.collect");
            task.draw_frame_id = frame_ctx->frame_id;  // 角标 / fps 用原始 frame_id
            task.save_index    = num_frames;           // 落盘文件名用自增后的计数
            task.fps = (total_us > 0)
                           ? static_cast<int>(frame_ctx->frame_id * 1000000LL / total_us)
                           : 0;
            task.img_w      = frame_ctx->raw_img.cols;  // 上半天区尺寸 = 原图尺寸
            task.img_h      = frame_ctx->raw_img.rows;
            task.num_tracks = infer_output_context.tracked_objects.size();
            task.alert      = alert;
            task.targets    = collectDrawTargets(infer_output_context, pipeline);
        }
        {
            PROF_SCOPE("06.enqueue");
            // 显示必须发生在主线程，且要看到「已绘制」的帧：给任务挂一个完成信号，
            // 由工作线程画完后唤醒主线程（保存仍在后台继续）。
            cv::Mat           display_frame;
            std::future<void> drawn_future;
            if (display_manager.isEnabled()) {
                display_frame = task.out_frame;  // Mat 浅拷贝：只加引用计数，不拷像素
                task.drawn_signal = std::make_shared<std::promise<void>>();
                drawn_future      = task.drawn_signal->get_future();
            }
            draw_save_worker.enqueue(std::move(task));

            // 显示图像（通过 DisplayManager）
            if (display_manager.isEnabled()) {
                drawn_future.wait();
                display_manager.show(display_frame);
                char c      = display_manager.waitKey(1);
                int  result = display_manager.handleKey(c);
                if (result == Key_Input::ESC) {  // 用户按下 ESC 键退出
                    async_pipeline.releaseFrame(frame_ctx);
                    break;
                }
            }
        }
        // 帧处理完毕（推理路径返回前已同步 CUDA 流），归还槽位供读帧线程复用
        async_pipeline.releaseFrame(frame_ctx);
    }

    // 排空落盘队列并回收工作线程（保证退出前所有结果图片都已写完）
    draw_save_worker.stop();

#if defined(ENABLE_TIMER)
    APP_INFO("==========Summary===========");
    for (auto & kv : ScopedTimer::GetScopedTimers()) {
        double avg = calculateAverage(kv.second);
        double p95 = calculatePercentile(kv.second, 95.0);
        double p99 = calculatePercentile(kv.second, 99.0);
        APP_INFO("[{}]: avg = {:.2f} ms, P95 = {:.2f} ms, P99 = {:.2f} ms (frame)", kv.first, avg,
                 p95, p99);
    }
#endif

    PROF_DUMP();

    return 0;
}

int main(int argc, char * argv[]) {
    if (argc != 3) {
        APP_ERROR("arguments not right!");
        APP_ERROR("Usage: ./main [video path] [config yaml path]");
        APP_ERROR("Example: ./main ./videos/demo.mp4 ./config.yaml");
        return -1;
    }

    return run(argv[1], argv[2]);
}
