
#include "config_manager.h"
#include "danger_alert_handler.h"
#include "frame.h"
#include "io_manager.h"
#include "logger_manager.h"
#include "pipeline.h"
#include "scope_timer.h"
#include "async_pipeline.h"
#include "visual_manager.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdio>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <string>
#include <utility>
#include <vector>

[[nodiscard]] cv::Mat drawOneFrame(FrameInputContext &   frame_input_context,
                     InferOutputContext &  infer_output_context,
                     const ConfigManager & config_manager,
                     DrawingManager &      drawing_manager,
                     AlertMessage &        alert_msg,
                     Pipeline &            pipeline,
                     int                   total_us) {
    for (const auto & track : infer_output_context.tracked_objects) {
        if (track.tlwh_[2] * track.tlwh_[3] <= 20) continue;

        if (auto it = infer_output_context.motion_records.find(track.track_id_);
            it != infer_output_context.motion_records.end()) {
#if defined(ENABLE_TIMER)
            DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(
                "6.Drawing Manager", drawing_manager, drawTrackedObject,
                frame_input_context.raw_img, track, alert_msg, pipeline.getColor(track.track_id_));
#else
            drawing_manager.drawTrackedObject(frame_input_context.raw_img, track, alert_msg,
                                              pipeline.getColor(track.track_id_));
#endif
        }
    }
    // FPS
    const int show_fps = (total_us > 0) ? static_cast<int>(frame_input_context.frame_id * 1000000LL / total_us) : 0;
    // 全局信息
    drawing_manager.drawGlobalInfo(frame_input_context.raw_img, frame_input_context.frame_id,
                                   show_fps, infer_output_context.tracked_objects.size());

    // 上下拼接
    cv::Mat out_frame = drawing_manager.concatenateFrames(frame_input_context.raw_img,
                                                          infer_output_context.depth_vis);
    return out_frame;
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
    int                num_frames = 0;
    double             total_us   = 0;
    InferOutputContext infer_output_context;

    // 首帧预热（在读帧线程启动前执行，使用视频/相机的第一帧）
    warmupPipeline(pipeline, io_manager, frame_meta);

    // 异步流水线模式：读帧与推理/绘图重叠执行。
    // Pipeline 只实例化一次，由 AsyncPipeline 引用复用。
    AsyncPipeline async_pipeline(config_manager, frame_meta, pipeline);
    async_pipeline.start(io_manager);

    while (true) {
        // 从异步队列获取帧（指针所有权仍在 AsyncPipeline，用完归还）
        FrameInputContext * frame_ctx = async_pipeline.getNextFrame();
        if (frame_ctx == nullptr) {
            break;
        }
        num_frames = frame_ctx->frame_id;

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
        num_frames++;
        if (num_frames % 100 == 0) {
            APP_INFO("Processing frame {} ({:.2f} fps)", num_frames,
                     (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0));
        }
        // 发送报警信息
        auto alert = alert_handler.buildAlert(*frame_ctx, infer_output_context);
        if (alert.has_value()) {
            io_manager.sendAlert(alert);
        }
        // 画图
        cv::Mat out_frame = drawOneFrame(*frame_ctx, infer_output_context, config_manager,
                                         drawing_manager, alert, pipeline, total_us);
        // 保存结果
        io_manager.saveFrame(out_frame, num_frames);

        // 显示图像（通过 DisplayManager）
        if (display_manager.isEnabled()) {
            display_manager.show(out_frame);
            char c      = display_manager.waitKey(1);
            int  result = display_manager.handleKey(c);
            if (result == Key_Input::ESC) {  // 用户按下 ESC 键退出
                async_pipeline.releaseFrame(frame_ctx);
                break;
            }
        }
        // 帧处理完毕（推理路径返回前已同步 CUDA 流），归还槽位供读帧线程复用
        async_pipeline.releaseFrame(frame_ctx);
    }

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
