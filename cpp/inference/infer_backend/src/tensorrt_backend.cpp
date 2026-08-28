#include "tensorrt_backend.h"

#include "logger_manager.h"
#include "public.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// RAII wrapper for mmap
struct MmapDeleter {
    size_t size;
    void operator()(void* p) const noexcept {
        if (p && p != MAP_FAILED) {
            munmap(p, size);
        }
    }
};
using MmapPtr = std::unique_ptr<void, MmapDeleter>;

// RAII wrapper for file descriptor
struct FdCloser {
    void operator()(int* p) const noexcept {
        if (p && *p >= 0) close(*p);
        delete p;
    }
};
using FdPtr = std::unique_ptr<int, FdCloser>;
#include <iostream>

TensorRTBackend::TensorRTBackend(int gpu_id) : gpu_id_(gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id_));
}

TensorRTBackend::~TensorRTBackend() {}

bool TensorRTBackend::isAvailable() const noexcept {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

bool TensorRTBackend::loadModel(const std::string & model_path) {
    if (!isAvailable()) {
        APP_ERROR("TensorRT backend not available: No GPU detected");
        return false;
    }

    return loadEngine(model_path);
}

// 获取共享内存路径
std::string TensorRTBackend::getShmPath(const std::string& engine_path) const {
    // 使用文件名的哈希值作为共享内存文件名
    std::hash<std::string> hasher;
    size_t hash = hasher(engine_path);
    return "/dev/shm/trt_engine_" + std::to_string(hash) + ".cache";
}

// 保存引擎到共享内存
bool TensorRTBackend::saveEngineToShm(const std::string& engine_path) {
    if (!engine_) return false;
    
    std::string shm_path = getShmPath(engine_path);
    
    // 序列化引擎
    nvinfer1::IHostMemory* serialized_engine = engine_->serialize();
    if (!serialized_engine) {
        APP_ERROR("Failed to serialize TensorRT engine");
        return false;
    }
    
    // 写入共享内存文件
    std::ofstream ofs(shm_path, std::ios::binary);
    if (!ofs) {
        APP_ERROR("Cannot open shared memory file: {}", shm_path);
        serialized_engine->destroy();
        return false;
    }
    
    ofs.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    ofs.close();
    serialized_engine->destroy();
    
    APP_INFO("Engine cached to shared memory: {}", shm_path);
    return true;
}

// 从共享内存加载引擎
bool TensorRTBackend::loadEngineFromShm(const std::string& engine_path) {
    std::string shm_path = getShmPath(engine_path);
    
    // 检查共享内存文件是否存在
    struct stat st;
    if (stat(shm_path.c_str(), &st) < 0) {
        return false;  // 文件不存在，需要正常加载
    }
    
    // 打开共享内存文件
    int fd = open(shm_path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    
    size_t fsize = st.st_size;
    if (fsize == 0) {
        close(fd);
        return false;
    }
    
    // mmap 映射
    void* mapped = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mapped == MAP_FAILED) {
        return false;
    }
    
    // 创建运行时和引擎
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        munmap(mapped, fsize);
        return false;
    }
    
    engine_.reset(runtime_->deserializeCudaEngine(mapped, fsize));
    munmap(mapped, fsize);
    
    if (!engine_) {
        return false;
    }
    
    APP_INFO("Engine loaded from shared memory: {}", shm_path);
    return true;
}

bool TensorRTBackend::loadEngine(const std::string & engine_path) {
    // RAII: 文件描述符自动关闭
    int fd_raw = open(engine_path.c_str(), O_RDONLY);
    if (fd_raw < 0) {
        APP_ERROR("Cannot open engine file: {}", engine_path);
        return false;
    }

    struct stat st;
    if (fstat(fd_raw, &st) < 0) {
        close(fd_raw);
        APP_ERROR("Cannot stat engine file: {}", engine_path);
        return false;
    }
    const size_t fsize = static_cast<size_t>(st.st_size);
    if (fsize == 0) {
        close(fd_raw);
        APP_ERROR("Engine file is empty: {}", engine_path);
        return false;
    }

    // mmap + RAII 自动 munmap
    void * raw_mapped = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd_raw, 0);
    close(fd_raw);
    if (raw_mapped == MAP_FAILED) {
        APP_ERROR("Failed to mmap engine file: {}", engine_path);
        return false;
    }
    MmapPtr mapped(raw_mapped, MmapDeleter{fsize});

    // 创建运行时和引擎
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        APP_ERROR("Failed to create TensorRT runtime");
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(mapped.get(), fsize));
    if (!engine_) {
        APP_ERROR("Failed to deserialize TensorRT engine");
        return false;
    }
    
    // 保存到共享内存以便下次快速加载
    saveEngineToShm(engine_path);

    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        APP_ERROR("Failed to create TensorRT execution context");
        return false;
    }

    // 设置输入输出维度
    setupInputOutputDims();

    APP_INFO("TensorRT engine loaded successfully from: {}", engine_path);
    APP_INFO("Input dims: [{}, {}, {}, {}], Input byte size: {} bytes", input_dims_[0],
             input_dims_[1], input_dims_[2], input_dims_[3], input_byte_size_);
    APP_INFO("Output tensor count: {}", output_tensor_.size());
    for (int i = 0; i < output_tensor_.size(); i++) {
        std::string dims_str;
        for (size_t j = 0; j < output_tensor_[i].dims.size(); ++j) {
            if (j > 0) {
                dims_str += ", ";
            }
            dims_str += std::to_string(output_tensor_[i].dims[j]);
        }
        APP_INFO("[{}] Output dim: ['{}']: dims: [{}], byte size: {} bytes", i,
                 output_tensor_[i].name, dims_str, output_tensor_[i].byte_size);
    }

    return true;
}

void TensorRTBackend::setupInputOutputDims() {
#if NV_TENSORRT_MAJOR < 10
    // TRT 8.x 及以下版本
    int nb_bindings    = engine_->getNbBindings();
    input_tensor_name_ = engine_->getBindingName(0);
    num_outputs_       = nb_bindings - 1;  // 第0个是输入，其余是输出

    output_tensor_.resize(num_outputs_);

    for (int i = 0; i < num_outputs_; ++i) {
        output_tensor_[i].name = engine_->getBindingName(i + 1);
        auto od                = engine_->getBindingDimensions(i + 1);
        output_tensor_[i].dims.clear();
        for (int j = 0; j < od.nbDims; ++j) {
            output_tensor_[i].dims.push_back(od.d[j]);
        }
    }

    auto input_dims = engine_->getBindingDimensions(0);
#else
    // TRT 10.x
    int nb_bindings    = engine_->getNbIOTensors();
    input_tensor_name_ = engine_->getIOTensorName(0);
    num_outputs_       = nb_bindings - 1;

    output_tensor_.resize(num_outputs_);
    for (int i = 0; i < num_outputs_; ++i) {
        output_tensor_[i].name = engine_->getIOTensorName(i + 1);
    }

    for (int i = 0; i < num_outputs_; ++i) {
        auto od = engine_->getTensorShape(output_tensor_[i].name.c_str());
        output_tensor_[i].dims.clear();
        for (int j = 0; j < od.nbDims; ++j) {
            output_tensor_[i].dims.push_back(od.d[j]);
        }
    }
    auto input_dims = engine_->getTensorShape(input_tensor_name_.c_str());
#endif
    // 保存输入维度
    input_dims_.clear();
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_dims_.push_back(input_dims.d[i]);
    }

    // 计算数据大小
    input_byte_size_ = 1;
    for (int dim : input_dims_) {
        input_byte_size_ *= dim;
    }
    input_byte_size_ *= sizeof(float);

    for (int i = 0; i < output_tensor_.size(); i++) {
        size_t byte_size = 1;
        for (int dim : output_tensor_[i].dims) {
            byte_size *= dim;
        }
        output_tensor_[i].byte_size = byte_size * sizeof(float);
    }

    // 设置输入维度（动态形状）
#if NV_TENSORRT_MAJOR < 10
    context_->setBindingDimensions(0, input_dims);
#else
    context_->setInputShape(input_tensor_name_.c_str(), input_dims);
#endif
}

bool TensorRTBackend::runInference(void * input_data, void * output_data) {
    return runInference(input_data, std::vector<void *>{ output_data });
}

bool TensorRTBackend::runInference(void * input_data, std::vector<void *> output_data) {
    if (!context_) {
        APP_ERROR("TensorRT context not initialized");
        return false;
    }
    if (output_tensor_.size() != output_data.size()) {
        APP_ERROR("output_data size mismatch: expected {}, got {}", output_tensor_.size(),
                  output_data.size());
        return false;
    }
    std::vector<void *> buffers(output_data.size() + 1);
    buffers[0] = input_data;
    for (size_t i = 0; i < output_data.size(); i++) {
        buffers[i + 1] = output_data[i];
    }
    bool status = context_->executeV2(buffers.data());
    if (!status) {
        APP_ERROR("TensorRT inference failed");
        return false;
    }
    return true;
}

bool TensorRTBackend::runInferenceAsync(void *       input_data,
                                        void *       output_data,
                                        cudaStream_t stream) {
    return runInferenceAsync(input_data, std::vector<void *>{ output_data }, stream);
}

bool TensorRTBackend::runInferenceAsync(void *              input_data,
                                        std::vector<void *> output_data,
                                        cudaStream_t        stream) {
    if (!context_) {
        APP_ERROR("TensorRT context not initialized");
        return false;
    }
    if (output_tensor_.size() != output_data.size()) {
        APP_ERROR("output_data size mismatch: expected {}, got {}", output_tensor_.size(),
                  output_data.size());
        return false;
    }
#if NV_TENSORRT_MAJOR >= 10
    // TRT 10.x: Explicitly set tensor addresses before enqueueV3
    context_->setTensorAddress(input_tensor_name_.c_str(), input_data);
    for (size_t i = 0; i < output_data.size(); i++) {
        context_->setTensorAddress(output_tensor_[i].name.c_str(), output_data[i]);
    }
    bool status = context_->enqueueV3(stream);
#else
    // TRT 8.x: Uses buffer array with enqueueV2
    std::vector<void *> buffers(output_data.size() + 1);
    buffers[0] = input_data;
    for (size_t i = 0; i < output_data.size(); i++) {
        buffers[i + 1] = output_data[i];
    }
    bool status = context_->enqueueV2(buffers.data(), stream, nullptr);
#endif
    if (!status) {
        APP_ERROR("TensorRT async inference failed");
        return false;
    }
    return true;
}

std::vector<int> TensorRTBackend::getInputDims() const {
    return input_dims_;
}

std::vector<int64_t> TensorRTBackend::getOutputDims(int output_index) const {
    if (output_index < 0 || output_index >= static_cast<int>(output_tensor_.size())) {
        APP_ERROR("Invalid output index: {}", output_index);
        return {};
    }
    return output_tensor_[output_index].dims;
}

size_t TensorRTBackend::getInputByteSize() const {
    return input_byte_size_;
}

size_t TensorRTBackend::getOutputByteSize(int output_index) const {
    if (output_index < 0 || output_index >= static_cast<int>(output_tensor_.size())) {
        APP_ERROR("Invalid output index: {}", output_index);
        return 0;
    }
    return output_tensor_[output_index].byte_size;
}
