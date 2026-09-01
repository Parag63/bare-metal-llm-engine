//===----------------------------------------------------------------------===//
// tests/test_dtype.cpp -- the dtype layer (Module 1, Objective 1).
//
// These all pass today: dtype.hpp is header-only arithmetic and src/dtype.cpp is
// complete. They are here as a REGRESSION FENCE. The dtype table is the thing that
// every later module silently depends on -- if `dtype_bits(I4)` ever returns 8
// because someone "simplified" the switch, the GGUF loader in March 2027 will
// mis-size every buffer and the failure will look like model corruption, not like a
// dtype bug. Cheap tests at the bottom of a dependency stack pay for themselves.
//===----------------------------------------------------------------------===//

#include "test_framework.hpp"

#include <engine/check.hpp>
#include <engine/dtype.hpp>

#include <set>
#include <string>

using engine::DType;

//===----------------------------------------------------------------------===//
// Compile-time checks. These cost nothing at runtime and they prove the helpers
// really are constexpr -- which matters because Tensor's members and future
// template code will want them in constant expressions.
//===----------------------------------------------------------------------===//
static_assert(engine::dtype_bits(DType::F32) == 32, "");
static_assert(engine::dtype_bits(DType::I4) == 4, "");
static_assert(engine::dtype_is_sub_byte(DType::I4), "");
static_assert(!engine::dtype_is_sub_byte(DType::I8), "");
static_assert(engine::dtype_is_float(DType::BF16), "");
static_assert(!engine::dtype_is_float(DType::I32), "");
static_assert(engine::kQuantBlockSize == 32, "GGUF Q4_0/Q8_0 use 32-element blocks");

TEST(dtype, bits_per_element) {
  EXPECT_EQ(engine::dtype_bits(DType::F32), 32);
  EXPECT_EQ(engine::dtype_bits(DType::F16), 16);
  EXPECT_EQ(engine::dtype_bits(DType::BF16), 16);
  EXPECT_EQ(engine::dtype_bits(DType::I32), 32);
  EXPECT_EQ(engine::dtype_bits(DType::I8), 8);
  EXPECT_EQ(engine::dtype_bits(DType::U8), 8);
  EXPECT_EQ(engine::dtype_bits(DType::I4), 4);
}

TEST(dtype, every_dtype_has_a_width) {
  // Guards against adding an enumerator to DType and forgetting the switch arm.
  // dtype_bits() returns 0 for anything it does not know, so a new dtype shows up
  // here immediately instead of three modules later.
  for (int i = 0; i < static_cast<int>(DType::COUNT); ++i) {
    const auto dt = static_cast<DType>(i);
    EXPECT_TRUE(engine::dtype_bits(dt) > 0);
  }
}

TEST(dtype, classification_predicates) {
  EXPECT_TRUE(engine::dtype_is_float(DType::F32));
  EXPECT_TRUE(engine::dtype_is_float(DType::F16));
  EXPECT_TRUE(engine::dtype_is_float(DType::BF16));
  EXPECT_FALSE(engine::dtype_is_float(DType::I8));
  EXPECT_FALSE(engine::dtype_is_float(DType::I4));

  EXPECT_TRUE(engine::dtype_is_quantised(DType::I8));
  EXPECT_TRUE(engine::dtype_is_quantised(DType::I4));
  EXPECT_FALSE(engine::dtype_is_quantised(DType::F16));

  // U8 is storage, not a quantised value type: it is the byte you pack nibbles
  // into. Conflating the two is how you end up trying to dequantise a byte buffer.
  EXPECT_FALSE(engine::dtype_is_quantised(DType::U8));
}

TEST(dtype, size_in_bytes_for_addressable_types) {
  EXPECT_EQ(engine::dtype_size(DType::F32), std::size_t{4});
  EXPECT_EQ(engine::dtype_size(DType::F16), std::size_t{2});
  EXPECT_EQ(engine::dtype_size(DType::BF16), std::size_t{2});
  EXPECT_EQ(engine::dtype_size(DType::I32), std::size_t{4});
  EXPECT_EQ(engine::dtype_size(DType::I8), std::size_t{1});
  EXPECT_EQ(engine::dtype_size(DType::U8), std::size_t{1});
}

TEST(dtype, size_rejects_sub_byte_types) {
  // This throw is a FEATURE, not a limitation. `n * dtype_size(dt)` is the standard
  // way to size a buffer, and for INT4 that expression is meaningless. Making it
  // loud is the whole reason dtype_storage_bytes() exists as a separate function.
  EXPECT_THROWS(engine::dtype_size(DType::I4));
  EXPECT_NO_THROW(engine::dtype_size(DType::F32));
}

TEST(dtype, storage_bytes_rounds_sub_byte_up) {
  // Whole-byte types: exactly n * size.
  EXPECT_EQ(engine::dtype_storage_bytes(DType::F32, 10), std::size_t{40});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::F16, 10), std::size_t{20});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I8, 10), std::size_t{10});

  // INT4: two elements per byte, ROUNDING UP. An odd count wastes half a byte,
  // which is correct -- you cannot allocate half a byte.
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, 0), std::size_t{0});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, 1), std::size_t{1});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, 2), std::size_t{1});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, 3), std::size_t{2});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, 32), std::size_t{16});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, 33), std::size_t{17});
}

TEST(dtype, storage_bytes_at_model_scale) {
  // A sanity check at a size that actually occurs. LLaMA-7B has ~6.74e9 parameters;
  // use a round 7e9 for the arithmetic. The point is that std::size_t is 64-bit and
  // nothing here overflows -- a 32-bit intermediate would wrap and produce a
  // plausible-looking small number.
  const std::size_t params = 7'000'000'000ull;
  EXPECT_EQ(engine::dtype_storage_bytes(DType::F32, params), std::size_t{28'000'000'000ull});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::F16, params), std::size_t{14'000'000'000ull});
  EXPECT_EQ(engine::dtype_storage_bytes(DType::I4, params), std::size_t{3'500'000'000ull});

  // And the reason this project needs quantisation at all: FP32 does not fit in the
  // 4090's 24 GiB, FP16 does with room for a KV cache, INT4 fits comfortably.
  const std::size_t vram = 24ull * 1024 * 1024 * 1024;
  EXPECT_TRUE(engine::dtype_storage_bytes(DType::F32, params) > vram);
  EXPECT_TRUE(engine::dtype_storage_bytes(DType::F16, params) < vram);
}

TEST(dtype, names_are_present_and_unique) {
  std::set<std::string> seen;
  for (int i = 0; i < static_cast<int>(DType::COUNT); ++i) {
    const auto dt = static_cast<DType>(i);
    const char* name = engine::dtype_name(dt);
    ASSERT_TRUE(name != nullptr);
    EXPECT_TRUE(name[0] != '\0');
    // Duplicate names would make every log line and to_string() ambiguous.
    EXPECT_TRUE(seen.insert(std::string(name)).second);
  }
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(DType::COUNT));
}

//===----------------------------------------------------------------------===//
// Block quantisation. Scheduled for May 2027, but the arithmetic is fixed now
// because the Tensor layout has to know about it.
//===----------------------------------------------------------------------===//
TEST(dtype, quant_block_bytes_includes_the_scale) {
  // Q4_0: 32 nibbles (16 bytes) + one FP16 scale (2 bytes) = 18 bytes.
  EXPECT_EQ(engine::quant_block_bytes(DType::I4), std::size_t{18});
  // Q8_0: 32 bytes + 2 = 34.
  EXPECT_EQ(engine::quant_block_bytes(DType::I8), std::size_t{34});

  EXPECT_THROWS(engine::quant_block_bytes(DType::F32));
}

TEST(dtype, effective_bits_is_not_the_nominal_bit_width) {
  // THE NUMBER THAT GOES IN THE REPORT. "4-bit quantisation" costs 4.5 bits per
  // weight once the per-block scale is counted; claiming 8x compression over FP32
  // when the real figure is 7.1x is the kind of thing a reviewer checks.
  EXPECT_NEAR(engine::quant_effective_bits(DType::I4), 4.5, 1e-12);
  EXPECT_NEAR(engine::quant_effective_bits(DType::I8), 8.5, 1e-12);

  // Compression ratio versus FP32, computed the honest way.
  const double ratio = 32.0 / engine::quant_effective_bits(DType::I4);
  EXPECT_NEAR(ratio, 7.111111111, 1e-6);
  EXPECT_TRUE(ratio < 8.0);  // it is never the nominal 8x
}
