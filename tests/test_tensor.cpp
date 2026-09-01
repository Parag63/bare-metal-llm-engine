//===----------------------------------------------------------------------===//
// tests/test_tensor.cpp -- the specification for Module 1 (October 2026).
//
// THESE TESTS ARE THE DEFINITION OF DONE. Everything below is registered with
// TEST_PENDING, which means it runs, and a failure is reported as PENDING rather
// than FAILED so the suite stays green while the module is outstanding. As you
// implement each method in src/tensor.cpp, the runner will tell you:
//
//     [PENDING-PASS] tensor.numel_is_the_product_of_shape  <-- implemented!
//
// at which point you change that test's TEST_PENDING to TEST and it becomes a
// permanent regression guard. The pending count is your progress bar.
//
// Recommended order is in the header comment of src/tensor.cpp; the tests are laid
// out in that same order so you can work top to bottom.
//
//===----------------------------------------------------------------------===//
// CONVENTIONS PINNED DOWN HERE (decide once, test, never wonder again)
//
//   * numel(): product of shape. A 0-d tensor (shape {}) holds ONE element, because
//     the empty product is 1. An UNDEFINED tensor (default-constructed, no storage)
//     holds zero.
//   * Negative axis indexing is accepted by size(), transpose() and slice(), where
//     -1 means the last axis. permute() takes a full permutation of [0, dim) and
//     requires non-negative indices -- mixing negatives into a permutation makes the
//     "is this a valid permutation" check needlessly subtle.
//   * Views share storage. Mutating through one view is visible through the other.
//     That is the point, not a hazard to be papered over.
//   * contiguous() and to() may return *this (sharing storage) when there is nothing
//     to do. clone() never does.
//   * Bad arguments THROW. Returning an invalid tensor or clamping silently is not
//     acceptable: a shape error must surface at the call site, not inside a kernel.
//===----------------------------------------------------------------------===//

#include "test_framework.hpp"

#include <engine/check.hpp>
#include <engine/config.hpp>
#include <engine/dtype.hpp>
#include <engine/tensor.hpp>

#include <cstring>
#include <string>
#include <vector>

using engine::Device;
using engine::DType;
using engine::Storage;
using engine::Tensor;
using I64 = std::int64_t;
using Shape = std::vector<I64>;

namespace {

/// Fills a contiguous float CPU tensor with 0, 1, 2, ... so that a later read can
/// prove both the values and the ORDER survived a view operation.
void fill_ramp(Tensor& t) {
  float* p = t.ptr<float>(Device::CPU);
  for (I64 i = 0; i < t.numel(); ++i) {
    p[i] = static_cast<float>(i);
  }
}

/// Reads element (i, j) of a 2-D view by walking its own strides -- deliberately NOT
/// assuming contiguity, since the whole point of several tests below is that the
/// view is not contiguous.
float at2(const Tensor& t, I64 i, I64 j) {
  const float* base = static_cast<const float*>(t.data());
  return base[i * t.strides()[0] + j * t.strides()[1]];
}

}  // namespace

//===----------------------------------------------------------------------===//
// Storage -- already implemented in src/tensor.cpp, so these are real TESTs.
//===----------------------------------------------------------------------===//

TEST(storage, allocates_and_reports_size) {
  Storage s(1024, Device::CPU);
  EXPECT_EQ(s.bytes(), std::size_t{1024});
  EXPECT_EQ(s.device(), Device::CPU);
  ASSERT_TRUE(s.data() != nullptr);

  // Prove the memory is actually usable, not just non-null.
  std::memset(s.data(), 0xAB, s.bytes());
  const auto* bytes = static_cast<const unsigned char*>(s.data());
  EXPECT_EQ(static_cast<int>(bytes[0]), 0xAB);
  EXPECT_EQ(static_cast<int>(bytes[1023]), 0xAB);
}

TEST(storage, zero_bytes_is_legal_and_holds_no_pointer) {
  // An empty tensor is a real thing -- cache[:, :0, :] at the first decode step is
  // exactly this -- so a zero-byte allocation must not be an error.
  Storage s(0, Device::CPU);
  EXPECT_EQ(s.bytes(), std::size_t{0});
  EXPECT_TRUE(s.data() == nullptr);
}

TEST(storage, device_name_round_trips) {
  EXPECT_EQ(std::string(engine::device_name(Device::CPU)), std::string("cpu"));
  EXPECT_EQ(std::string(engine::device_name(Device::CUDA)), std::string("cuda"));
}

#if !ENGINE_HAS_CUDA
TEST(storage, cuda_request_on_cpu_only_build_explains_itself) {
  // The failure mode this guards against is a confusing "unsupported" message six
  // months from now. The error must name the fix, which here is "build on the 4090
  // machine, or ask for Device::CPU".
  EXPECT_THROWS_MSG(Storage(64, Device::CUDA), "Device::CPU");
}
#endif

//===----------------------------------------------------------------------===//
// STEP 1 -- shape arithmetic. No allocation, pure metadata.
//===----------------------------------------------------------------------===//

TEST(tensor, default_constructed_is_undefined) {
  Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_EQ(t.dim(), I64{0});
  EXPECT_EQ(t.numel(), I64{0});   // undefined, so zero -- see the conventions above
  EXPECT_EQ(t.nbytes(), std::size_t{0});
  EXPECT_TRUE(t.shape().empty());
  EXPECT_TRUE(t.strides().empty());
}

TEST(tensor, contiguous_strides_are_row_major) {
  // The worked example from the header: [2,3,4] -> [12,4,1].
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);
  ASSERT_EQ(t.strides().size(), std::size_t{3});
  EXPECT_EQ(t.strides()[0], I64{12});
  EXPECT_EQ(t.strides()[1], I64{4});
  EXPECT_EQ(t.strides()[2], I64{1});

  // 1-D: the only stride is 1.
  Tensor v(Shape{7}, DType::F32, Device::CPU);
  ASSERT_EQ(v.strides().size(), std::size_t{1});
  EXPECT_EQ(v.strides()[0], I64{1});
}

TEST(tensor, numel_is_the_product_of_shape) {
  EXPECT_EQ(Tensor(Shape{2, 3, 4}, DType::F32, Device::CPU).numel(), I64{24});
  EXPECT_EQ(Tensor(Shape{7}, DType::F32, Device::CPU).numel(), I64{7});

  // A 0-d scalar: the empty product is 1. Not used much in this project, but getting
  // it wrong means numel() is implemented as something other than a product, and
  // that "something" will bite later.
  EXPECT_EQ(Tensor(Shape{}, DType::F32, Device::CPU).numel(), I64{1});

  // A zero-length axis makes the whole tensor empty. This IS used: the KV cache at
  // generation step 0 is cache.slice(1, 0, 0).
  EXPECT_EQ(Tensor(Shape{4, 0, 8}, DType::F32, Device::CPU).numel(), I64{0});
}

TEST(tensor, nbytes_uses_storage_bytes_not_element_size) {
  EXPECT_EQ(Tensor(Shape{2, 3}, DType::F32, Device::CPU).nbytes(), std::size_t{24});
  EXPECT_EQ(Tensor(Shape{2, 3}, DType::F16, Device::CPU).nbytes(), std::size_t{12});
  EXPECT_EQ(Tensor(Shape{2, 3}, DType::I8, Device::CPU).nbytes(), std::size_t{6});

  // INT4, the reason nbytes() must go through dtype_storage_bytes(): six elements
  // pack into three bytes, and an odd count rounds UP.
  EXPECT_EQ(Tensor(Shape{2, 3}, DType::I4, Device::CPU).nbytes(), std::size_t{3});
  EXPECT_EQ(Tensor(Shape{5}, DType::I4, Device::CPU).nbytes(), std::size_t{3});
}

TEST(tensor, size_supports_negative_axes) {
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);
  EXPECT_EQ(t.size(0), I64{2});
  EXPECT_EQ(t.size(1), I64{3});
  EXPECT_EQ(t.size(2), I64{4});
  EXPECT_EQ(t.size(-1), I64{4});
  EXPECT_EQ(t.size(-2), I64{3});
  EXPECT_EQ(t.size(-3), I64{2});

  // Out of range in either direction must throw, not return garbage or clamp.
  EXPECT_THROWS(t.size(3));
  EXPECT_THROWS(t.size(-4));
}

TEST(tensor, constructor_rejects_negative_dimensions) {
  EXPECT_THROWS(Tensor(Shape{2, -3}, DType::F32, Device::CPU));
}

//===----------------------------------------------------------------------===//
// STEP 2 -- to_string. Implement this second; you will read it constantly.
//===----------------------------------------------------------------------===//

TEST(tensor, to_string_shows_shape_strides_dtype_device) {
  const std::string s = Tensor(Shape{2, 3}, DType::F32, Device::CPU).to_string();

  // Substring checks rather than an exact match: the exact punctuation is yours to
  // choose, but every one of these facts has to be in there, because each one is
  // something you will be trying to find out when you print it at 1am.
  EXPECT_TRUE(s.find("2") != std::string::npos);
  EXPECT_TRUE(s.find("3") != std::string::npos);
  EXPECT_TRUE(s.find("f32") != std::string::npos);
  EXPECT_TRUE(s.find("cpu") != std::string::npos);
  EXPECT_TRUE(s.find("contiguous") != std::string::npos);

  // An undefined tensor must be printable too -- printing is often how you discover
  // that a tensor is undefined in the first place.
  EXPECT_NO_THROW(Tensor().to_string());
}

//===----------------------------------------------------------------------===//
// STEP 3 -- allocation and zeros.
//===----------------------------------------------------------------------===//

TEST(tensor, constructor_allocates_enough_storage) {
  Tensor t(Shape{4, 5}, DType::F32, Device::CPU);
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.device(), Device::CPU);
  EXPECT_EQ(t.dtype(), DType::F32);
  EXPECT_EQ(t.storage_offset(), I64{0});
  ASSERT_TRUE(t.storage() != nullptr);

  // The storage must be at least nbytes(). It may be larger (a pool allocator will
  // round up), so this is >= rather than ==.
  EXPECT_TRUE(t.storage()->bytes() >= t.nbytes());
}

TEST(tensor, zeros_actually_zeroes) {
  Tensor t = Tensor::zeros(Shape{16, 16}, DType::F32, Device::CPU);
  ASSERT_EQ(t.numel(), I64{256});
  const float* p = static_cast<const float*>(t.data());
  ASSERT_TRUE(p != nullptr);
  for (I64 i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(p[i], 0.0f);
  }
}

TEST(tensor, plain_constructor_does_not_promise_zeros) {
  // Not a behavioural test -- it documents that the constructor leaves memory
  // UNINITIALISED on purpose. If you ever "helpfully" zero it, you have added a full
  // memory-bandwidth pass over every weight buffer you are about to overwrite. On a
  // 14 GB FP16 model at ~1 TB/s that is ~14 ms of pure waste per allocation.
  Tensor t(Shape{8}, DType::F32, Device::CPU);
  EXPECT_TRUE(t.defined());
  EXPECT_NO_THROW(fill_ramp(t));
}

//===----------------------------------------------------------------------===//
// STEP 4 -- is_contiguous.
//===----------------------------------------------------------------------===//

TEST(tensor, freshly_allocated_is_contiguous) {
  EXPECT_TRUE(Tensor(Shape{2, 3, 4}, DType::F32, Device::CPU).is_contiguous());
  EXPECT_TRUE(Tensor(Shape{1}, DType::F32, Device::CPU).is_contiguous());
  EXPECT_TRUE(Tensor(Shape{}, DType::F32, Device::CPU).is_contiguous());
}

//===----------------------------------------------------------------------===//
// STEP 5 -- data() and ptr<T>().
//===----------------------------------------------------------------------===//

TEST(tensor, ptr_checks_dtype_and_device) {
  Tensor t(Shape{4}, DType::F32, Device::CPU);
  EXPECT_NO_THROW(t.ptr<float>(Device::CPU));

  // Wrong element width: int happens to be 4 bytes so this one is allowed by a
  // size-only check; unsigned char is not, and must throw. If you check the DType
  // rather than sizeof(T), both should throw -- either policy is defensible, so pick
  // one, and make this test match the choice you documented.
  EXPECT_THROWS(t.ptr<unsigned char>(Device::CPU));

  // Wrong device is the one that really matters: passing a host pointer to a kernel
  // is an illegal access inside the GPU, which is a miserable thing to debug from
  // the outside. Catching it here turns that into one clear exception.
  EXPECT_THROWS(t.ptr<float>(Device::CUDA));
}

TEST(tensor, data_respects_storage_offset) {
  Tensor t(Shape{10}, DType::F32, Device::CPU);
  fill_ramp(t);

  Tensor tail = t.slice(0, 4, 6);
  EXPECT_EQ(tail.storage_offset(), I64{4});

  // data() must point at element 4 of the parent, i.e. the value 4.0f. The classic
  // bug is adding offset_ (in elements) to a byte pointer, which lands at element 1
  // -- and 1.0f is a plausible enough number that it can go unnoticed.
  const float* p = static_cast<const float*>(tail.data());
  ASSERT_TRUE(p != nullptr);
  EXPECT_EQ(p[0], 4.0f);
  EXPECT_EQ(p[5], 9.0f);
}

//===----------------------------------------------------------------------===//
// STEP 6 -- reshape.
//===----------------------------------------------------------------------===//

TEST(tensor, reshape_shares_storage_and_repacks_strides) {
  Tensor t(Shape{2, 6}, DType::F32, Device::CPU);
  fill_ramp(t);

  Tensor r = t.reshape(Shape{3, 4});
  EXPECT_EQ(r.numel(), I64{12});
  EXPECT_EQ(r.shape(), (Shape{3, 4}));
  EXPECT_EQ(r.strides(), (Shape{4, 1}));
  EXPECT_TRUE(r.is_contiguous());

  // Same storage object, not a copy. Reshape must be free.
  EXPECT_TRUE(r.storage().get() == t.storage().get());

  // Row-major order is preserved: r(1,1) is flat index 5.
  EXPECT_EQ(at2(r, 1, 1), 5.0f);

  // And because it is a view, a write through one is visible through the other.
  r.ptr<float>(Device::CPU)[0] = 99.0f;
  EXPECT_EQ(static_cast<const float*>(t.data())[0], 99.0f);
}

TEST(tensor, reshape_infers_one_minus_one) {
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);

  EXPECT_EQ(t.reshape(Shape{-1}).shape(), (Shape{24}));
  EXPECT_EQ(t.reshape(Shape{6, -1}).shape(), (Shape{6, 4}));
  EXPECT_EQ(t.reshape(Shape{-1, 8}).shape(), (Shape{3, 8}));

  // Two unknowns is not solvable, and a count that does not divide evenly is not
  // either. Both must throw rather than guess.
  EXPECT_THROWS(t.reshape(Shape{-1, -1}));
  EXPECT_THROWS(t.reshape(Shape{5, -1}));
  EXPECT_THROWS(t.reshape(Shape{2, 3}));  // 6 != 24
}

TEST(tensor, reshape_of_non_contiguous_throws_with_advice) {
  Tensor t(Shape{2, 3}, DType::F32, Device::CPU);
  Tensor tr = t.transpose(0, 1);
  ASSERT_TRUE(!tr.is_contiguous());

  // The message must name the fix. "invalid argument" would be useless; you will hit
  // this case dozens of times when writing attention, and the answer is always
  // "insert .contiguous()".
  EXPECT_THROWS_MSG(tr.reshape(Shape{6}), "contiguous");
}

//===----------------------------------------------------------------------===//
// STEP 7 -- permute and transpose. Metadata only; no data moves.
//===----------------------------------------------------------------------===//

TEST(tensor, transpose_swaps_shape_and_strides) {
  Tensor t(Shape{2, 3}, DType::F32, Device::CPU);
  fill_ramp(t);  // [[0,1,2],[3,4,5]]

  Tensor tr = t.transpose(0, 1);
  EXPECT_EQ(tr.shape(), (Shape{3, 2}));
  EXPECT_EQ(tr.strides(), (Shape{1, 3}));
  EXPECT_FALSE(tr.is_contiguous());
  EXPECT_TRUE(tr.storage().get() == t.storage().get());  // no copy

  // tr(j,i) == t(i,j). Reading through the strides is what makes this work.
  EXPECT_EQ(at2(tr, 0, 1), 3.0f);
  EXPECT_EQ(at2(tr, 2, 0), 2.0f);
  EXPECT_EQ(at2(tr, 2, 1), 5.0f);

  // Transposing twice is the identity, including the strides.
  Tensor back = tr.transpose(0, 1);
  EXPECT_EQ(back.shape(), t.shape());
  EXPECT_EQ(back.strides(), t.strides());
  EXPECT_TRUE(back.is_contiguous());
}

TEST(tensor, transpose_accepts_negative_axes) {
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);
  // The attention idiom: swap the last two axes of [B, S, H, D]-style layouts.
  Tensor a = t.transpose(-1, -2);
  EXPECT_EQ(a.shape(), (Shape{2, 4, 3}));
  EXPECT_EQ(a.strides(), (Shape{12, 1, 4}));
  EXPECT_THROWS(t.transpose(0, 3));
}

TEST(tensor, permute_reorders_axes) {
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);  // strides [12,4,1]

  Tensor p = t.permute(Shape{2, 0, 1});
  EXPECT_EQ(p.shape(), (Shape{4, 2, 3}));
  EXPECT_EQ(p.strides(), (Shape{1, 12, 4}));
  EXPECT_FALSE(p.is_contiguous());

  // The identity permutation must leave everything alone, including contiguity.
  Tensor same = t.permute(Shape{0, 1, 2});
  EXPECT_EQ(same.strides(), t.strides());
  EXPECT_TRUE(same.is_contiguous());
}

TEST(tensor, permute_validates_that_it_is_a_permutation) {
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);
  EXPECT_THROWS(t.permute(Shape{0, 1}));        // wrong length
  EXPECT_THROWS(t.permute(Shape{0, 0, 1}));     // 0 twice, 2 missing
  EXPECT_THROWS(t.permute(Shape{0, 1, 3}));     // out of range
  EXPECT_THROWS(t.permute(Shape{0, 1, 2, 0}));  // too long
}

//===----------------------------------------------------------------------===//
// STEP 8 -- slice. This is the KV-cache primitive; get it exactly right.
//===----------------------------------------------------------------------===//

TEST(tensor, slice_narrows_an_axis_and_keeps_parent_strides) {
  Tensor t(Shape{4, 5}, DType::F32, Device::CPU);
  fill_ramp(t);

  // Slicing axis 0 of a contiguous tensor: still contiguous, because axis 0 is the
  // outermost one and the rows it keeps are adjacent.
  Tensor rows = t.slice(0, 1, 2);
  EXPECT_EQ(rows.shape(), (Shape{2, 5}));
  EXPECT_EQ(rows.strides(), (Shape{5, 1}));
  EXPECT_EQ(rows.storage_offset(), I64{5});
  EXPECT_TRUE(rows.is_contiguous());
  EXPECT_EQ(at2(rows, 0, 0), 5.0f);
  EXPECT_EQ(at2(rows, 1, 4), 14.0f);

  // Slicing an INNER axis is not contiguous: consecutive kept elements are 5 apart
  // in memory. Understanding why is understanding strides.
  Tensor cols = t.slice(1, 2, 2);
  EXPECT_EQ(cols.shape(), (Shape{4, 2}));
  EXPECT_EQ(cols.strides(), (Shape{5, 1}));  // strides UNCHANGED -- that is the point
  EXPECT_EQ(cols.storage_offset(), I64{2});
  EXPECT_FALSE(cols.is_contiguous());
  EXPECT_EQ(at2(cols, 0, 0), 2.0f);
  EXPECT_EQ(at2(cols, 3, 1), 18.0f);
}

TEST(tensor, slice_of_slice_composes) {
  Tensor t(Shape{6, 6}, DType::F32, Device::CPU);
  fill_ramp(t);

  // Offsets must ACCUMULATE. Overwriting offset_ instead of adding to it is a bug
  // that only shows up on the second slice, so it is worth a dedicated test.
  Tensor s = t.slice(0, 2, 3).slice(0, 1, 1);
  EXPECT_EQ(s.shape(), (Shape{1, 6}));
  EXPECT_EQ(s.storage_offset(), I64{18});  // (2 + 1) * 6
  EXPECT_EQ(at2(s, 0, 0), 18.0f);
}

TEST(tensor, slice_supports_the_empty_kv_cache_case) {
  // Generation step 0: cache.slice(1, 0, 0) is a legal, empty view. If this throws,
  // the decode loop needs a special case for its first iteration -- which is exactly
  // the kind of avoidable complexity to design out now.
  Tensor cache(Shape{32, 2048, 128}, DType::F32, Device::CPU);
  Tensor empty = cache.slice(1, 0, 0);
  EXPECT_EQ(empty.shape(), (Shape{32, 0, 128}));
  EXPECT_EQ(empty.numel(), I64{0});
  EXPECT_NO_THROW(empty.to_string());
}

TEST(tensor, slice_bounds_are_checked) {
  Tensor t(Shape{4, 5}, DType::F32, Device::CPU);
  EXPECT_THROWS(t.slice(0, 3, 2));   // 3 + 2 > 4
  EXPECT_THROWS(t.slice(0, -1, 2));  // negative start
  EXPECT_THROWS(t.slice(0, 0, -1));  // negative length
  EXPECT_THROWS(t.slice(2, 0, 1));   // no such axis
  EXPECT_NO_THROW(t.slice(-1, 0, 5));  // negative AXIS is fine: -1 is the last one
}

//===----------------------------------------------------------------------===//
// STEP 9 -- the copying operations.
//===----------------------------------------------------------------------===//

TEST(tensor, contiguous_is_a_no_op_when_already_packed) {
  Tensor t(Shape{3, 4}, DType::F32, Device::CPU);
  Tensor c = t.contiguous();
  // Same storage: no allocation, no copy. This path is hit constantly, so it must be
  // free rather than merely correct.
  EXPECT_TRUE(c.storage().get() == t.storage().get());
}

TEST(tensor, contiguous_repacks_a_transposed_view) {
  Tensor t(Shape{2, 3}, DType::F32, Device::CPU);
  fill_ramp(t);  // [[0,1,2],[3,4,5]]

  Tensor c = t.transpose(0, 1).contiguous();
  EXPECT_EQ(c.shape(), (Shape{3, 2}));
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_EQ(c.strides(), (Shape{2, 1}));
  EXPECT_TRUE(c.storage().get() != t.storage().get());  // it DID allocate

  // Row-major traversal of the transpose is [0,3,1,4,2,5]. This is the test that
  // catches a strided-copy loop that walks the odometer in the wrong order.
  const float* p = static_cast<const float*>(c.data());
  const float want[6] = {0, 3, 1, 4, 2, 5};
  EXPECT_ALLCLOSE(p, want, std::size_t{6}, 0.0, 0.0);
}

TEST(tensor, contiguous_handles_rank_3_and_offsets) {
  // A harder case for the odometer: rank 3, permuted, AND offset by a slice. Getting
  // rank 2 right by accident is easy; this one requires the general algorithm.
  Tensor t(Shape{2, 3, 4}, DType::F32, Device::CPU);
  fill_ramp(t);

  Tensor v = t.slice(0, 1, 1).permute(Shape{0, 2, 1});
  ASSERT_EQ(v.shape(), (Shape{1, 4, 3}));
  Tensor c = v.contiguous();
  ASSERT_TRUE(c.is_contiguous());
  ASSERT_EQ(c.numel(), I64{12});

  // Source element (0, j, k) of v is t(1, k, j) = 12 + k*4 + j.
  const float* p = static_cast<const float*>(c.data());
  for (I64 j = 0; j < 4; ++j) {
    for (I64 k = 0; k < 3; ++k) {
      EXPECT_EQ(p[j * 3 + k], static_cast<float>(12 + k * 4 + j));
    }
  }
}

TEST(tensor, clone_always_allocates) {
  Tensor t(Shape{3, 4}, DType::F32, Device::CPU);
  fill_ramp(t);

  Tensor c = t.clone();
  EXPECT_TRUE(c.storage().get() != t.storage().get());  // unlike contiguous()
  EXPECT_EQ(c.shape(), t.shape());
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_ALLCLOSE(static_cast<const float*>(c.data()),
                  static_cast<const float*>(t.data()), std::size_t{12}, 0.0, 0.0);

  // Independence, verified in both directions.
  c.ptr<float>(Device::CPU)[0] = -1.0f;
  EXPECT_EQ(static_cast<const float*>(t.data())[0], 0.0f);
  t.ptr<float>(Device::CPU)[1] = -2.0f;
  EXPECT_EQ(static_cast<const float*>(c.data())[1], 1.0f);
}

TEST(tensor, to_same_device_is_a_no_op) {
  Tensor t(Shape{4}, DType::F32, Device::CPU);
  Tensor s = t.to(Device::CPU);
  EXPECT_TRUE(s.storage().get() == t.storage().get());
}

#if !ENGINE_HAS_CUDA
TEST(tensor, to_cuda_on_cpu_only_build_throws) {
  Tensor t(Shape{4}, DType::F32, Device::CPU);
  EXPECT_THROWS(t.to(Device::CUDA));
}
#else
TEST(tensor, host_device_round_trip_preserves_values) {
  Tensor h(Shape{64, 16}, DType::F32, Device::CPU);
  fill_ramp(h);

  Tensor d = h.to(Device::CUDA);
  EXPECT_EQ(d.device(), Device::CUDA);
  EXPECT_TRUE(d.is_contiguous());
  EXPECT_NO_THROW(d.ptr<float>(Device::CUDA));
  EXPECT_THROWS(d.ptr<float>(Device::CPU));  // guard against a host deref

  Tensor back = d.to(Device::CPU);
  EXPECT_EQ(back.device(), Device::CPU);
  // A memcpy either works or it does not, so this is exact.
  EXPECT_ALLCLOSE(static_cast<const float*>(back.data()),
                  static_cast<const float*>(h.data()), std::size_t{1024}, 0.0, 0.0);
}

TEST(tensor, to_requires_contiguity_and_says_so) {
  Tensor h(Shape{8, 8}, DType::F32, Device::CPU);
  Tensor tr = h.transpose(0, 1);
  EXPECT_THROWS_MSG(tr.to(Device::CUDA), "contiguous");
}
#endif

//===----------------------------------------------------------------------===//
// STEP 10 -- from_blob. Non-owning storage; the mmap'd-GGUF enabler.
//===----------------------------------------------------------------------===//

TEST(tensor, from_blob_aliases_external_memory) {
  std::vector<float> external(12);
  for (std::size_t i = 0; i < external.size(); ++i) {
    external[i] = static_cast<float>(i) * 2.0f;
  }

  Tensor t = Tensor::from_blob(external.data(), Shape{3, 4}, DType::F32, Device::CPU);
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.numel(), I64{12});
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_TRUE(t.data() == external.data());  // literally the same address

  // Writes go both ways -- it is one buffer, viewed twice.
  t.ptr<float>(Device::CPU)[0] = -5.0f;
  EXPECT_EQ(external[0], -5.0f);
  external[11] = 7.0f;
  EXPECT_EQ(static_cast<const float*>(t.data())[11], 7.0f);
}

TEST(tensor, from_blob_does_not_free_the_blob) {
  // If from_blob's Storage owned the pointer, this would free stack/vector memory and
  // the test would crash or corrupt the heap. Running to completion is the assertion.
  std::vector<float> external(8, 3.5f);
  {
    Tensor t = Tensor::from_blob(external.data(), Shape{8}, DType::F32, Device::CPU);
    EXPECT_EQ(static_cast<const float*>(t.data())[0], 3.5f);
  }  // t dies here
  EXPECT_EQ(external[0], 3.5f);
  EXPECT_EQ(external[7], 3.5f);
}

//===----------------------------------------------------------------------===//
// Lifetime. The reason Storage is behind a shared_ptr at all.
//===----------------------------------------------------------------------===//

TEST(tensor, a_view_keeps_its_storage_alive) {
  // The KV-cache pattern in miniature: a long-lived slice outliving the handle the
  // buffer was originally allocated through. Without reference counting this is a
  // use-after-free -- and one that usually appears to work in a debug build.
  Tensor view;
  {
    Tensor owner(Shape{10}, DType::F32, Device::CPU);
    fill_ramp(owner);
    view = owner.slice(0, 5, 5);
    EXPECT_TRUE(owner.storage().use_count() >= 2);
  }  // owner is gone; the storage must not be

  ASSERT_TRUE(view.defined());
  EXPECT_EQ(view.storage().use_count(), I64{1});
  const float* p = static_cast<const float*>(view.data());
  ASSERT_TRUE(p != nullptr);
  EXPECT_EQ(p[0], 5.0f);
  EXPECT_EQ(p[4], 9.0f);
}

TEST(tensor, views_are_copyable_and_cheap) {
  Tensor t(Shape{4, 4}, DType::F32, Device::CPU);
  fill_ramp(t);

  Tensor copy = t;  // a Tensor copy copies METADATA and bumps the refcount
  EXPECT_TRUE(copy.storage().get() == t.storage().get());
  copy.ptr<float>(Device::CPU)[0] = 42.0f;
  EXPECT_EQ(static_cast<const float*>(t.data())[0], 42.0f);
}
