#include <engine/dtype.hpp>

namespace engine {

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::F32:
      return "f32";
    case DType::F16:
      return "f16";
    case DType::BF16:
      return "bf16";
    case DType::I32:
      return "i32";
    case DType::I8:
      return "i8";
    case DType::U8:
      return "u8";
    case DType::I4:
      return "i4";
    default:
      return "<invalid>";
  }
}

}  // namespace engine
