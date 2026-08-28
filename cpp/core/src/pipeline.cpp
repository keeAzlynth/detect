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


struct PhaseTimer {
    const char* name;
    std::chrono::steady_clock::time_point start;
    PhaseTimer(const char* n) : name(n), start(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() {
        auto end = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        fprintf(stderr, "[TIMER] %s: %.2f ms\n", name, us / 1000.0);
    }
};

void Pipeline::init() {}

// 同步串行推理：YOLO检测 → 深度估计 → 跟踪 → 运动状态，按顺序串行执行
// depth_interval: 每隔 depth_interval 帧执行一次深度推理，节省算力
void Pipeline::process(FrameInputContext &  frame_input_context,
                       InferOutputContext & infer_output_context) {
    {
        PhaseTimer t("YOLO_Detection");
        detector_.runInference(frame_input_context, infer_output_context);
    }
    {
        PhaseTimer t("BYTETracker");
        updateTracker(infer_output_context);
    }
    
    // 根据 depth_interval 决定是否执行深度推理
    depth_frame_counter_++;
    if (depth_frame_counter_ >= depth_interval_) {
        depth_frame_counter_ = 0;
        {
            PhaseTimer t("Depth_Inference");
            depth_model_.runInference(frame_input_context, infer_output_context);
        }
        // 缓存深度结果
        has_cached_depth_ = true;
        cached_depth_ = infer_output_context.result_depth.clone();
        cached_depth_vis_ = infer_output_context.depth_vis.clone();
    } else if (has_cached_depth_) {
        // 使用缓存的深度结果
        infer_output_context.result_depth = cached_depth_;
        infer_output_context.depth_vis = cached_depth_vis_;
    }
    
    {
        PhaseTimer t("Motion_State");
        updateMotionStates(frame_input_context, infer_output_context);
    }
}

// CPU/GPU 重叠推理：YOLO 和 Depth 通过各自 CUDA Stream 异步执行，实现并行
// 流程：同时启动 YOLO 和 Depth 异步推理 → YOLO 结果先返回（延迟更低）→ 先做跟踪
//       → Depth 结果随后返回 → 运动状态判定
// depth_interval: 每隔 depth_interval 帧执行一次深度推理，节省算力
void Pipeline::processOverlap(FrameInputContext &  frame_input_context,
                              InferOutputContext & infer_output_context) {
    {
        PhaseTimer t("YOLO_Detection_Async");
        detector_.runInferenceAsync(frame_input_context);
    }
    {
        PhaseTimer t("YOLO_GetResult");
        detector_.getInferOutputResult(infer_output_context);
    }
    {
        PhaseTimer t("BYTETracker");
        updateTracker(infer_output_context);
    }
    
    // 根据 depth_interval 决定是否执行深度推理
    depth_frame_counter_++;
    if (depth_frame_counter_ >= depth_interval_) {
        depth_frame_counter_ = 0;
        {
            PhaseTimer t("Depth_Inference_Async");
            depth_model_.runInferenceAsync(frame_input_context);
        }
        {
            PhaseTimer t("Depth_GetResult");
            depth_model_.getInferOutputResult(infer_output_context);
        }
        // 缓存深度结果
        has_cached_depth_ = true;
        cached_depth_ = infer_output_context.result_depth.clone();
        cached_depth_vis_ = infer_output_context.depth_vis.clone();
    } else if (has_cached_depth_) {
        // 使用缓存的深度结果
        infer_output_context.result_depth = cached_depth_;
        infer_output_context.depth_vis = cached_depth_vis_;
    }
    
    {
        PhaseTimer t("Motion_State");
        updateMotionStates(frame_input_context, infer_output_context);
    }
}

void Pipeline::updateTracker(InferOutputContext & infer_output_context) {
    const auto & detections = infer_output_context.detections;
    std::vector<Object> objects;
    objects.reserve(detections.size());  // 预分配，避免重分配

    for (const auto & det : detections) {
        if (!isTrackingClass(det.classId)) continue;
        objects.push_back({
            cv::Rect_<float>(det.bbox[0], det.bbox[1],
                             det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]),
            det.classId, det.conf, 0.0f});
    }
    infer_output_context.tracked_objects = tracker_.update(objects);
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
