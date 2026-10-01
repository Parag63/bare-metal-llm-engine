//===----------------------------------------------------------------------===//
// src/cuda_stream.cpp -- implementation of CudaStream and CudaEvent RAII wrappers.
//===----------------------------------------------------------------------===//

#include <engine/cuda_stream.hpp>

#if ENGINE_HAS_CUDA

namespace engine {

CudaStream::CudaStream() : CudaStream(cudaStreamNonBlocking) {}

CudaStream::CudaStream(unsigned int flags) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStream::~CudaStream() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

CudaEvent::CudaEvent() : CudaEvent(cudaEventDefault) {}

CudaEvent::CudaEvent(unsigned int flags) {
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() {
  if (event_) {
    cudaEventDestroy(event_);
  }
}

}  // namespace engine

#endif  // ENGINE_HAS_CUDA
