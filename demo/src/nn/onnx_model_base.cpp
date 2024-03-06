#include "nn/onnx_model_base.h"

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include "constants.h"
#include "nn_utils.h"


OnnxModelBase::OnnxModelBase(const char* modelPath, const char* logid, const char* provider)
    : modelPath_(modelPath)
{
    env = Ort::Env(
#if ORT_VERBOSE
        ORT_LOGGING_LEVEL_VERBOSE,
#elif DEBUG_INFO
        ORT_LOGGING_LEVEL_WARNING,
#else
        ORT_LOGGING_LEVEL_ERROR,
#endif
        logid);
    Ort::SessionOptions session_options = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");

#if DEBUG_INFO || ORT_VERBOSE
    std::cout << "availableProviders: [";
    for (const auto& provider : availableProviders) {
        std::cout << provider << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // todo: try to get cuda working
    if (provider == OnnxProviders::CUDA.c_str()) {
        if (cudaAvailable == availableProviders.end()) {
            std::cout << "CUDA is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
            std::cout << "Inference device: CPU" << std::endl;
        }
        else {
            std::cout << "Appending cuda provider" << std::endl;
            OrtTensorRTProviderOptions tensor_options = {};
            session_options.AppendExecutionProvider_TensorRT(tensor_options);
            std::cout << "Inference device: CUDA" << std::endl;
        }
    }

    auto model_path_w = get_ort_path(modelPath);
#if DEBUG_INFO
    std::cout << "Model path: " << model_path_w << std::endl;
#endif
    session = Ort::Session(env, model_path_w, session_options);

    // ----------------
    // init input names
    inputNodeNames;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings; // <-- newly added
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNodesNum = session.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNodeNames.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    // -----------------
    // init output names
    outputNodeNames;
    auto outputNodesNum = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings; // <-- newly added
    Ort::AllocatorWithDefaultOptions output_names_allocator;
    for (int i = 0; i < outputNodesNum; i++) {
        auto output_name = session.GetOutputNameAllocated(i, output_names_allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNodeNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    // -------------------------
    // initialize model metadata
    model_metadata = session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions metadata_allocator;
    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = model_metadata.GetCustomMetadataMapKeysAllocated(metadata_allocator);
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(metadataAllocatedKeys.size());
    for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys) {
        metadata_keys.emplace_back(allocatedString.get());
    }

    // -------------------------
    // initialize metadata as the dict
    // even though we know exactly what metadata we intend to use
    // base onnx class should not have any ultralytics yolo-specific attributes like stride, task etc, so keep it clean as much as possible
    for (const std::string& key : metadata_keys) {
        Ort::AllocatedStringPtr metadata_value = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
        if (metadata_value != nullptr) {
            auto raw_metadata_value = metadata_value.get();
            metadata[key] = std::string(raw_metadata_value);
        }
    }

    // initialize cstr
    for (const std::string& name : outputNodeNames) {
        outputNamesCStr.push_back(name.c_str());
    }
    for (const std::string& name : inputNodeNames) {
        inputNamesCStr.push_back(name.c_str());
    }
}

const std::vector<std::string>& OnnxModelBase::getInputNames() { return inputNodeNames; }
const std::vector<std::string>& OnnxModelBase::getOutputNames() { return outputNodeNames; }
const Ort::ModelMetadata& OnnxModelBase::getModelMetadata() { return model_metadata; }
const std::unordered_map<std::string, std::string>& OnnxModelBase::getMetadata() { return metadata; }
const Ort::Session& OnnxModelBase::getSession() { return session; }
const char* OnnxModelBase::getModelPath() { return modelPath_; }
const std::vector<const char*> OnnxModelBase::getOutputNamesCStr() { return outputNamesCStr; }
const std::vector<const char*> OnnxModelBase::getInputNamesCStr() { return inputNamesCStr; }

std::vector<Ort::Value> OnnxModelBase::forward(std::vector<Ort::Value>& inputTensors) {
    return session.Run(Ort::RunOptions{ nullptr },
        inputNamesCStr.data(),
        inputTensors.data(),
        inputNamesCStr.size(),
        outputNamesCStr.data(),
        outputNamesCStr.size());
}