#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "motion_state_engine.h"
#include "public.h"
#include "STrack.h"

#include <array>
#include <chrono>
#include <cstdio>

Pipeline::Pipeline(ConfigManager & config_manager, FrameMeta frame_meta) :
    tracker_(30, 30),  // 假设fps=30，或从config读取
    motion_state_engine_(config_manager.getMotionVelocityThreshold(),
                         config_manager.getMotionAccelerationThreshold(),
                         config_manager.getKfProcessNoiseCov(),
                         config_manager.getKfMeasurementNoiseCov()),
    depth_interval_(config_manager.getDepthInterval()),
    depth_frame_counter_(0) {
    bool is_normalize = false;

    depth_model_.init(config_manager.getDepthModelPath(), frame_meta.img_w, frame_meta.img_h,
                      is_normalize, config_manager.isUseGPU());
    detector_.init(config_manager.getYoloModelPath(), frame_meta.img_w, frame_meta.img_h,
                   config_manager.getYoloNmsThresh(), config_manager.getYoloConfThresh(), 80,
                   config_manager.isUseGPU());
}

Pipeline::Pipeline(std::string depth_model_path,
                   std::string yolo_model_path,
                   FrameMeta   frame_meta,
                   bool        use_gpu,
                   float       yolo_nms_thresh,
                   float       yolo_conf_thresh) {
    bool        is_normalize = false;
    std::string backend_type = use_gpu ? "engine" : "onnx";
    depth_model_.init(
        {
            { backend_type, depth_model_path }
    },
        frame_meta.img_w, frame_meta.img_h, is_normalize, use_gpu);
    detector_.init(
        {
            { backend_type, yolo_model_path }
    },
        frame_meta.img_w, frame_meta.img_h, yolo_nms_thresh, yolo_conf_thresh, 80, use_gpu);
}


// 阶段计时开关：编译期门控，关闭时零开销。
// 默认关闭（每帧 6 次 fprintf 到 stderr 会拖慢流水线并污染测量）。
// 需要观察各阶段耗时时用 -DPIPELINE_PHASE_TIMER=1 重新编译。
#ifndef PIPELINE_PHASE_TIMER
#define PIPELINE_PHASE_TIMER 0
#endif

#if PIPELINE_PHASE_TIMER
struct PhaseTimer {
    const char* name;
    std::chrono::steady_clock::time_point start;
    explicit PhaseTimer(const char* n) : name(n), start(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() {
        const auto end = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        fprintf(stderr, "[TIMER] %s: %.2f ms\n", name, us / 1000.0);
    }
};
#define PIPELINE_PHASE_SCOPE(name) PhaseTimer phase_timer__(name)
#else
#define PIPELINE_PHASE_SCOPE(name)
#endif

void Pipeline::init() {}

// 同步串行推理：YOLO检测 → 深度估计 → 跟踪 → 运动状态，按顺序串行执行
// depth_interval: 每隔 depth_interval 帧执行一次深度推理，节省算力
void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    {
        PIPELINE_PHASE_SCOPE("YOLO_Detection");
        detector_.runInference(frame_input_context, infer_output_context);
    }
    {
        PIPELINE_PHASE_SCOPE("BYTETracker");
        updateTracker(infer_output_context);
    }

    // 根据 depth_interval 决定是否执行深度推理
    depth_frame_counter_++;
    bool run_depth = (depth_frame_counter_ >= depth_interval_);
    if (run_depth) {
        depth_frame_counter_ = 0;
        {
            PIPELINE_PHASE_SCOPE("Depth_Inference");
            depth_model_.runInference(frame_input_context, infer_output_context);
        }
        has_cached_depth_ = true;
    } else if (has_cached_depth_) {
        // 使用缓存的深度结果（浅拷贝，零像素拷贝）
        infer_output_context.result_depth = cached_depth_;
        infer_output_context.depth_vis    = cached_depth_vis_;
    }

    {
        PIPELINE_PHASE_SCOPE("Motion_State");
        updateMotionStates(frame_input_context, infer_output_context);
    }

    if (run_depth) {
        // 运动状态计算完成后缓存深度结果：Mat 浅拷贝（引用计数，零像素拷贝）。
        // 注意不能用 swap 把结果从 context 换走——主循环的绘图/落盘还要读
        // infer_output_context.depth_vis，swap 清空后存出的图会缺整个深度半区。
        cached_depth_     = infer_output_context.result_depth;
        cached_depth_vis_ = infer_output_context.depth_vis;
    }
}

// CPU/GPU 重叠推理：Depth 与 YOLO 通过各自 CUDA Stream 异步执行，实现并行。
// 流程：先启动耗时更长的 Depth 异步推理 → 再启动 YOLO 异步推理 → 等待 YOLO 结果
//       （此期间 Depth 在另一条流上继续计算）→ 跟踪 → 等待 Depth 结果 → 运动状态判定。
// depth_interval: 每隔 depth_interval 帧执行一次深度推理，节省算力
void Pipeline::processOverlap(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    // 根据 depth_interval 决定本帧是否执行深度推理
    depth_frame_counter_++;
    bool run_depth = (depth_frame_counter_ >= depth_interval_);
    if (run_depth) {
        depth_frame_counter_ = 0;
        {
            PIPELINE_PHASE_SCOPE("Depth_Inference_Async");
            depth_model_.runInferenceAsync(frame_input_context);
        }
    }
    {
        PIPELINE_PHASE_SCOPE("YOLO_Detection_Async");
        detector_.runInferenceAsync(frame_input_context);
    }
    {
        PIPELINE_PHASE_SCOPE("YOLO_GetResult");
        detector_.getInferOutputResult(infer_output_context);
    }
    {
        PIPELINE_PHASE_SCOPE("BYTETracker");
        updateTracker(infer_output_context);
    }

    if (run_depth) {
        {
            PIPELINE_PHASE_SCOPE("Depth_GetResult");
            depth_model_.getInferOutputResult(infer_output_context);
        }
        has_cached_depth_ = true;
    } else if (has_cached_depth_) {
        // 使用缓存的深度结果（浅拷贝，零像素拷贝）
        infer_output_context.result_depth = cached_depth_;
        infer_output_context.depth_vis    = cached_depth_vis_;
    }

    {
        PIPELINE_PHASE_SCOPE("Motion_State");
        updateMotionStates(frame_input_context, infer_output_context);
    }

    if (run_depth) {
        // 运动状态计算完成后缓存深度结果：Mat 浅拷贝（引用计数，零像素拷贝）。
        // 注意不能用 swap 把结果从 context 换走——主循环的绘图/落盘还要读
        // infer_output_context.depth_vis，swap 清空后存出的图会缺整个深度半区。
        cached_depth_     = infer_output_context.result_depth;
        cached_depth_vis_ = infer_output_context.depth_vis;
    }
}

void Pipeline::updateTracker(InferOutputContext & infer_output_context) {
    const auto & detections = infer_output_context.detections;
    tracker_objects_buf_.clear();
    tracker_objects_buf_.reserve(detections.size());

    for (const auto & det : detections) {
        if (!isTrackingClass(det.classId)) continue;
        tracker_objects_buf_.push_back({
            cv::Rect_<float>(det.bbox[0], det.bbox[1],
                             det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]),
            det.classId, det.conf, 0.0f});
    }
    infer_output_context.tracked_objects = tracker_.update(tracker_objects_buf_);
}

void Pipeline::updateMotionStates(FrameInputContext &  frame_input_context,
                                  InferOutputContext & infer_output_context) {
#ifdef HAS_NVTX3
    nvtx3::scoped_range tracker_scope("pipeline updateMotionStates");
#endif
    infer_output_context.motion_records.clear();
    const auto & tracked_objects = infer_output_context.tracked_objects;
    infer_output_context.motion_records.reserve(tracked_objects.size());

    const auto img_size = frame_input_context.raw_img.size();
    const auto timestamp = frame_input_context.timestamp;

    for (const auto & track : tracked_objects) {
        const float area = track.tlwh_[2] * track.tlwh_[3];
        if (area <= kMinTrackArea) continue;

        const float current_depth = motion_state_engine_.getObjectDepth(
            infer_output_context.result_depth, track, img_size);

        infer_output_context.motion_records.emplace(
            track.track_id_,
            motion_state_engine_.computeMotionState(track.track_id_, current_depth, timestamp));
    }
}
