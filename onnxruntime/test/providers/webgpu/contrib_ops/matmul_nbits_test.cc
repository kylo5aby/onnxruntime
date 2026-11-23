// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/common/logging/severity.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

void RunFloat16NarrowDP4AAccumulationTest(int64_t bits) {
  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation,
                                                 webgpu::options::kEnableMatmulFp32Accumulation_OFF));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  if (!provider) {
    GTEST_SKIP() << "WebGPU execution provider is unavailable.";
  }
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (!context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
    GTEST_SKIP() << "WebGPU adapter does not support shader-f16.";
  }
  // Match DP4A dispatch: Apple adapters deliberately use other MatMulNBits paths.
  if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups) ||
      context.AdapterInfo().vendor == std::string_view{"apple"}) {
    GTEST_SKIP() << "WebGPU DP4A path is unavailable.";
  }

  // N=32 excludes subgroup-matrix dispatch. M=4 and K=4096 select the narrow
  // K-parallel DP4A path, including its unconditional FP32 accumulation.
  constexpr int64_t M = 4, N = 32, K = 4096, block_size = 32;
  constexpr int64_t blocks = K / block_size;
  const int64_t values_per_byte = 8 / bits;
  const int64_t blob_size = block_size / values_per_byte;
  const int32_t zero_point = 1 << (bits - 1);
  InlinedVector<MLFloat16> a(M * K);
  InlinedVector<uint8_t> packed_weights(N * blocks * blob_size, 0);
  InlinedVector<MLFloat16> scales(N * blocks, MLFloat16(16.0f));
  InlinedVector<MLFloat16> expected(M * N);

  // +/-127 quantizes exactly with activation scale 1. Each 32-element block
  // contributes +/-65024, which fits FP16, but the first two K iterations sum
  // to +/-130048. Later blocks cancel that sum; one omitted negative weight
  // leaves exactly +/-2032. FP16 partial sums overflow before cancellation.
  for (int64_t m = 0; m < M; ++m) {
    const float sign = m % 2 == 0 ? 1.0f : -1.0f;
    for (int64_t k = 0; k < K; ++k) {
      a[m * K + k] = MLFloat16(sign * 127.0f);
    }
    for (int64_t n = 0; n < N; ++n) {
      expected[m * N + n] = MLFloat16(sign * 2032.0f);
    }
  }
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const int32_t weight = k == K - 1 ? 0 : (k < K / 2 ? 1 : -1);
      const auto quantized = static_cast<uint8_t>(zero_point + weight);
      packed_weights[(n * K + k) / values_per_byte] |=
          static_cast<uint8_t>(quantized << ((k % values_per_byte) * bits));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  test.AddInput<MLFloat16>("A", {M, K}, a.data(), a.size());
  test.AddInput<uint8_t>("B", {N, blocks, blob_size}, packed_weights.data(), packed_weights.size(), true);
  test.AddInput<MLFloat16>("scales", {N, blocks}, scales.data(), scales.size(), true);
  test.AddOutput<MLFloat16>("Y", {M, N}, expected.data(), expected.size());
  test.SetOutputAbsErr("Y", 0.0f);
  test.SetOutputRelErr("Y", 0.0f);

  SessionOptions options;
  options.use_per_session_threads = false;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

// Runs MatMulNBits with one quantization block per column (block_size == K) and requires the
// int8 subgroup-matrix kernel to handle it. Each entry of expected_dispatches must appear in the
// logged dispatch sizes, e.g. "(1, 7, 1)".
template <typename T>
void RunPerChannelSubgroupMatrixInt8Test(int64_t bits, int64_t M, int64_t N, int64_t K, bool has_bias,
                                         std::initializer_list<std::string_view> expected_dispatches = {}) {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU execution provider is unavailable.";
  }
  const auto& context = webgpu::WebGpuContextFactory::GetContext(provider->GetDeviceId());
  if (std::is_same_v<T, MLFloat16> && !context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
    GTEST_SKIP() << "WebGPU adapter does not support shader-f16.";
  }
  constexpr auto kI8 = wgpu::SubgroupMatrixComponentType::I8;
  constexpr auto kI32 = wgpu::SubgroupMatrixComponentType::I32;
  const auto& adapter = context.AdapterInfo();
  const auto& device_configs = context.SubgroupMatrixConfigs();
  const auto config = webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
      {device_configs.configs, device_configs.configCount},
      adapter.subgroupMinSize, adapter.subgroupMaxSize,
      context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl),
      {{kI8, kI32, 8, 16, 32, 32, true}});
  if (!context.DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix) || !config.has_value()) {
    GTEST_SKIP() << "Requires the int8 8x16x32 subgroup-matrix configuration.";
  }

  const int64_t values_per_byte = 8 / bits;
  const int64_t blob_size = K / values_per_byte;
  const int32_t zero_point = 1 << (bits - 1);
  const int64_t quantized_range = int64_t{1} << bits;
  InlinedVector<int32_t> a_values(static_cast<size_t>(M * K));
  InlinedVector<int32_t> weights(static_cast<size_t>(N * K));
  InlinedVector<T> a(static_cast<size_t>(M * K));
  InlinedVector<uint8_t> packed_weights(static_cast<size_t>(N * blob_size), 0);
  InlinedVector<T> scales(static_cast<size_t>(N));
  InlinedVector<T> bias(static_cast<size_t>(N));
  InlinedVector<T> expected(static_cast<size_t>(M * N));

  // Integer activations with +/-127 in each row have an activation scale of exactly 1, so
  // quantizing them to int8 is lossless. Row 1 is all zeros and has scale 0.
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      int32_t value = static_cast<int32_t>((m * 7 + k * 13) % 253) - 126;
      if (k == m % K) {
        value = m % 2 == 0 ? 127 : -127;
      }
      if (m == 1) {
        value = 0;
      }
      a_values[m * K + k] = value;
      a[m * K + k] = T(static_cast<float>(value));
    }
  }
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const auto quantized = static_cast<int32_t>((n * 5 + k * 3) % quantized_range);
      weights[n * K + k] = quantized - zero_point;
      packed_weights[(n * K + k) / values_per_byte] |=
          static_cast<uint8_t>(quantized << ((k % values_per_byte) * bits));
    }
  }
  // Power-of-two scales keep the expected values exact in FP32.
  for (int64_t n = 0; n < N; ++n) {
    const float scale = 1.0f / static_cast<float>(128 << (n % 3));
    const float bias_value = has_bias ? static_cast<float>(n % 5 - 2) * 0.5f : 0.0f;
    scales[n] = T(scale);
    bias[n] = T(bias_value);
    for (int64_t m = 0; m < M; ++m) {
      int64_t sum = 0;
      for (int64_t k = 0; k < K; ++k) {
        sum += a_values[m * K + k] * weights[n * K + k];
      }
      expected[m * N + n] = T(static_cast<float>(sum) * scale + bias_value);
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", K);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  test.AddInput<T>("A", {M, K}, a.data(), a.size());
  test.AddInput<uint8_t>("B", {N, 1, blob_size}, packed_weights.data(), packed_weights.size(), true);
  test.AddInput<T>("scales", {N, 1}, scales.data(), scales.size(), true);
  if (has_bias) {
    test.AddOptionalInputEdge<uint8_t>();  // zero_points
    test.AddOptionalInputEdge<int32_t>();  // g_idx
    test.AddInput<T>("bias", {N}, bias.data(), bias.size(), true);
  }
  test.AddOutput<T>("Y", {M, N}, expected.data(), expected.size());

  struct DispatchLog {
    std::mutex mutex;
    std::string all_programs;
    std::string int8_programs;
  } dispatch_log;

  SessionOptions options;
  options.use_per_session_threads = false;
  options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  options.user_logging_param = &dispatch_log;
  options.user_logging_function = [](void* param, OrtLoggingLevel, const char*, const char*,
                                     const char*, const char* message) {
    if (message == nullptr) {
      return;
    }
    const std::string_view text{message};
    if (text.find("Starting program \"") != 0) {
      return;
    }
    auto& log = *static_cast<DispatchLog*>(param);
    std::lock_guard lock{log.mutex};
    log.all_programs.append(text);
    log.all_programs.push_back('\n');
    if (text.find("SubgroupMatrixMatMulNBitsInt8") != std::string_view::npos) {
      log.int8_programs.append(text);
      log.int8_programs.push_back('\n');
    }
  };
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  // The dispatch assertion prevents a fallback kernel from hiding failures in the int8 kernel.
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
  EXPECT_FALSE(dispatch_log.int8_programs.empty()) << dispatch_log.all_programs;
  for (std::string_view dispatch : expected_dispatches) {
    EXPECT_NE(dispatch_log.int8_programs.find(dispatch), std::string::npos)
        << dispatch << '\n'
        << dispatch_log.int8_programs;
  }
}

}  // namespace

TEST(WebGpuMatMulNBitsTest, Float16_Int2_NarrowDP4AAccumulation) {
  RunFloat16NarrowDP4AAccumulationTest(2);
}

TEST(WebGpuMatMulNBitsTest, Float16_Int4_NarrowDP4AAccumulation) {
  RunFloat16NarrowDP4AAccumulationTest(4);
}

TEST(WebGpuMatMulNBitsTest, Float16_Int8_NarrowDP4AAccumulation) {
  RunFloat16NarrowDP4AAccumulationTest(8);
}

// M and N span two workgroup tiles each (128x64 for M > 64), with partial last tiles.
TEST(WebGpuMatMulNBitsTest, Float16_Int8_PerChannelSubgroupMatrixInt8) {
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(8, 200, 80, 128, false, {"(2, 2, 1)"});
}

TEST(WebGpuMatMulNBitsTest, Float16_Int4_PerChannelSubgroupMatrixInt8) {
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(4, 200, 80, 128, false, {"(2, 2, 1)"});
}

TEST(WebGpuMatMulNBitsTest, Float32_Int8_PerChannelSubgroupMatrixInt8Bias) {
  RunPerChannelSubgroupMatrixInt8Test<float>(8, 200, 80, 128, true, {"(2, 2, 1)"});
}

// M <= 64 uses the 64x64 tile. Covers its partial M-tile.
TEST(WebGpuMatMulNBitsTest, Float16_Int8_PerChannelSubgroupMatrixInt8SmallM) {
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(8, 50, 80, 128, false);
}

// The WebGPU EP accepts a block_size that is not a power of two. K = 96 runs 32-element K steps,
// K = 192 runs 64-element K steps.
TEST(WebGpuMatMulNBitsTest, Float16_Int8_PerChannelSubgroupMatrixInt8NonPowerOfTwoK) {
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(8, 8, 16, 96, true);
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(8, 8, 16, 192, true);
}

TEST(WebGpuMatMulNBitsTest, Float16_Int4_PerChannelSubgroupMatrixInt8NonPowerOfTwoK) {
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(4, 8, 16, 96, true);
  RunPerChannelSubgroupMatrixInt8Test<MLFloat16>(4, 8, 16, 192, true);
}

// The int8 A of 128 x 4096 bytes per M-tile allows 22 M-tiles per dispatch, so 24 M-tiles run as two
// dispatches of 12.
TEST(WebGpuMatMulNBitsTest, Float32_Int8_PerChannelSubgroupMatrixInt8Segmented) {
  RunPerChannelSubgroupMatrixInt8Test<float>(8, 2949, 16, 4096, false, {"(1, 12, 1)"});
}

// At K = 8192 the A size would allow only 11 M-tiles per dispatch; the 16 M-tile minimum applies instead,
// so 17 M-tiles run as dispatches of 9 and 8.
TEST(WebGpuMatMulNBitsTest, Float32_Int8_PerChannelSubgroupMatrixInt8SegmentedLongK) {
  RunPerChannelSubgroupMatrixInt8Test<float>(8, 2117, 16, 8192, false, {"(1, 9, 1)", "(1, 8, 1)"});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
