#pragma once

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_model.h"
#include "frame.h"
#include "motion_state_engine.h"
#include "yolo_detect_model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

class Pipeline {
  public:
    Pipeline(ConfigManager & config_manager, FrameMeta frame_meta);
    Pipeline(std::string depth_model_path,
             std::string yolo_model_path,
             FrameMeta   frame_meta,
             bool        use_gpu          = false,
             float       yolo_nms_thresh  = 0.4f,
             float       yolo_conf_thresh = 0.25f);
    void init();

    // 核心推理接口，供正常业务和 Benchmark 调用
    __attribute__((hot)) void process(FrameInputContext &  frame_input_context,
                                      InferOutputContext & infer_output_context);
    __attribute__((hot)) void processOverlap(FrameInputContext &  frame_input_context,
                                             InferOutputContext & infer_output_context);

    __attribute__((hot)) void updateMotionStates(FrameInputContext &  frame_input_context,
                                                 InferOutputContext & infer_output_context);

    cv::Scalar getColor(int idx) { return tracker_.getColor(idx); }

    YoloDetectModel & getDetector() { return detector_; }

    DepthModel & getDepthModel() { return depth_model_; }

    BYTETracker & getTracker() { return tracker_; }

    MotionStateEngine & getMotionStateEngine() { return motion_state_engine_; }

  private:
    __attribute__((hot)) void updateTracker(InferOutputContext & infer_output_context);

    YoloDetectModel detector_;
    DepthModel      depth_model_;

    [[nodiscard]] bool isTrackingClass(int class_id) const noexcept {
        for (const auto c : kTrackClasses) {
            if (class_id == c) return true;
        }
        return false;
    }

    BYTETracker       tracker_;
    MotionStateEngine motion_state_engine_;

    // 需要跟踪的类别：person, bicycle, car, motorcycle, bus, truck（COCO 索引）
    static constexpr std::array<int, 6> kTrackClasses = { 1, 2, 3, 5, 7, 8 };
    static constexpr int kMinTrackArea = 20;  // 最小跟踪面积阈值

    bool is_normalize_ = false;

    // depth_interval 优化：隔帧执行深度推理
    int  depth_interval_ = 1;
    int  depth_frame_counter_ = 0;

    // 跨帧缓存状态
    bool    has_cached_depth_ = false;
    cv::Mat cached_depth_;
    cv::Mat cached_depth_vis_;

    // 每帧复用的跟踪输入缓冲，避免热路径堆分配
    std::vector<Object> tracker_objects_buf_;
};
