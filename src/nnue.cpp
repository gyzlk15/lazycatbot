// HalfKP v1 loader, incremental accumulator and CPU inference.
//
// Format: docs/NNUE_PILOT.md section 3.3.  No file access or allocation
// happens while evaluating; all arrays live in the immutable Network and in
// each Position's fixed-size accumulator.

#include "nnue.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>

#if defined(CHESSBOT_ENABLE_AVX2) || defined(__AVX2__) || defined(_M_AVX2)
#define CHESSBOT_NNUE_HAVE_AVX2 1
#include <immintrin.h>
#else
#define CHESSBOT_NNUE_HAVE_AVX2 0
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace chess {
namespace NNUE {
namespace {

constexpr uint32_t kMagic = 0x4E4E434Cu;  // "LCNN" little-endian
constexpr uint32_t kVersion = 1;
constexpr uint32_t kArchHalfKP = 1;
constexpr uint32_t kQuantInt16Ft = 1;
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

// Quantized FC1: ReLU activations are clamped to [0, 1.5] and represented as
// uint8 [0,127].  A per-model parity test enforces <=2 cp; the cap covers the
// observed pilot activation range (max ~1.4).
constexpr float kActivationScale = 1.5f / 127.0f;
constexpr float kInvActivationScale = 127.0f / 1.5f;

struct Network {
    int version = 0;
    int arch = 0;
    int inputs = 0;
    int hidden = 0;
    int buckets = 0;
    int quant = 0;
    int policySize = 0;
    float ftScale = 0.0f;

    // ft[input * hidden + lane]
    std::vector<int16_t> ft;
    // Optional in-memory int8 cache of the same rows for hidden >= 192.
    // Dequantized with ft8Scale; ft remains authoritative for parity.
    std::vector<int8_t> ft8;
    float ft8Scale = 0.0f;
    bool useFt8 = false;
    std::vector<float> ftBias;
    // fc1W[hidden * 32 + out]
    std::vector<float> fc1W;
    // Optional int8 copy of fc1, transposed to [out][hidden] so each output is
    // a contiguous dot product.  Activations are quantized to uint8 (0..127).
    std::vector<int8_t> fc1W8;
    float fc1Scale8 = 0.0f;
    bool useFc1Int8 = false;
    std::vector<float> fc1B;
    // fc2W[32 * 32 + out]
    std::vector<float> fc2W;
    std::vector<float> fc2B;
    std::vector<float> outW;
    float outB = 0.0f;
    // Optional policy head; loaded and validated but unused by search.
    std::vector<float> policyW;
    std::vector<float> policyB;
};

std::atomic<const Network*> gNet{nullptr};
std::atomic<uint32_t> gGeneration{0};
std::atomic<bool> gBackendCpu{true};
std::atomic<bool> gLegacyEnabled{true};

std::mutex gLock;
std::vector<std::shared_ptr<const Network>> gRetained;
NetInfo gInfo;

int gNetWeight = 100;  // direct network eval by default
int gNetClip = 0;      // 0 = no clipping for experiments
int gNetMaterial = 0;  // 0 = off; diagnostic material correction weight

enum FtQuantPolicy { FTQ_AUTO = 0, FTQ_INT8 = 1, FTQ_INT16 = 2 };
int gFtQuantPolicy = FTQ_AUTO;

#if CHESSBOT_NNUE_HAVE_AVX2
bool detectAvx2() {
#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}
const bool gUseAvx2 = detectAvx2();
#else
const bool gUseAvx2 = false;
#endif

#if defined(_MSC_VER)
#define NNUE_FORCEINLINE __forceinline
#else
#define NNUE_FORCEINLINE inline __attribute__((always_inline))
#endif

inline const Network* current() {
    return gNet.load(std::memory_order_acquire);
}

inline uint32_t readU32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
           (uint32_t(p[3]) << 24);
}

inline uint64_t readU64(const uint8_t* p) {
    return uint64_t(readU32(p)) | (uint64_t(readU32(p + 4)) << 32);
}

inline float readF32(const uint8_t* p) {
    const uint32_t bits = readU32(p);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = kFnvOffset;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= kFnvPrime;
    }
    return h;
}

inline int bucketOf(Color perspective, int kingSq) {
    return perspective == WHITE ? kingSq : (kingSq ^ 56);
}

inline int orientedSquare(Color perspective, int sq) {
    return perspective == WHITE ? sq : (sq ^ 56);
}

// Adds or subtracts one feature-transformer row.  Rows are int16, the
// accumulator is int32; AVX2 processes eight lanes at a time.
NNUE_FORCEINLINE void addRow(int32_t* acc, const int16_t* row, int n, bool add) {
#if CHESSBOT_NNUE_HAVE_AVX2
    if (gUseAvx2) {
        int j = 0;
        for (; j + 16 <= n; j += 16) {
            const __m256i v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j));
            __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
            __m256i hi = _mm256_cvtepi16_epi32(
                _mm256_extracti128_si256(v, 1));
            __m256i a0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(acc + j));
            __m256i a1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(acc + j + 8));
            a0 = add ? _mm256_add_epi32(a0, lo) : _mm256_sub_epi32(a0, lo);
            a1 = add ? _mm256_add_epi32(a1, hi) : _mm256_sub_epi32(a1, hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j), a0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j + 8), a1);
        }
        for (; j + 8 <= n; j += 8) {
            const __m128i v =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
            const __m256i w = _mm256_cvtepi16_epi32(v);
            __m256i a = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(acc + j));
            a = add ? _mm256_add_epi32(a, w) : _mm256_sub_epi32(a, w);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j), a);
        }
        if (j < n) {
            if (add)
                for (; j < n; ++j) acc[j] += row[j];
            else
                for (; j < n; ++j) acc[j] -= row[j];
        }
        return;
    }
#endif
    if (add)
        for (int j = 0; j < n; ++j) acc[j] += row[j];
    else
        for (int j = 0; j < n; ++j) acc[j] -= row[j];
}

// int8 FT + int16 accumulator variant.  Sixteen int8 values become sixteen
// int16 lanes in one conversion; the 31-feature worst case (31 * 127 = 3937)
// fits comfortably in int16.
NNUE_FORCEINLINE void addRow8to16(int16_t* acc, const int8_t* row, int n, bool add) {
#if CHESSBOT_NNUE_HAVE_AVX2
    if (gUseAvx2) {
        int j = 0;
        for (; j + 16 <= n; j += 16) {
            const __m128i v =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
            const __m256i w = _mm256_cvtepi8_epi16(v);
            __m256i a = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(acc + j));
            a = add ? _mm256_add_epi16(a, w) : _mm256_sub_epi16(a, w);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j), a);
        }
        if (j < n) {
            if (add)
                for (; j < n; ++j) acc[j] = int16_t(acc[j] + row[j]);
            else
                for (; j < n; ++j) acc[j] = int16_t(acc[j] - row[j]);
        }
        return;
    }
#endif
    if (add)
        for (int j = 0; j < n; ++j) acc[j] = int16_t(acc[j] + row[j]);
    else
        for (int j = 0; j < n; ++j) acc[j] = int16_t(acc[j] - row[j]);
}

// Applies one board mutation to one perspective.  Own king is represented by
// the bucket, so it is deliberately skipped here.
NNUE_FORCEINLINE void applyPiece(const Network& net, int32_t* acc, Color perspective,
                       Piece pc, int sq, int kingSq, bool add) {
    if (!isPiece(pc) || !validSq(sq) || !validSq(kingSq)) return;
    if (pieceType(pc) == KING && pieceColor(pc) == perspective) return;
    const int idx = halfkpIndex(perspective, pc, sq, kingSq);
    addRow(acc, net.ft.data() + size_t(idx) * size_t(net.hidden),
           net.hidden, add);
}

NNUE_FORCEINLINE void applyPiece16(const Network& net, int16_t* acc, Color perspective,
                         Piece pc, int sq, int kingSq, bool add) {
    if (!isPiece(pc) || !validSq(sq) || !validSq(kingSq)) return;
    if (pieceType(pc) == KING && pieceColor(pc) == perspective) return;
    const int idx = halfkpIndex(perspective, pc, sq, kingSq);
    addRow8to16(acc, net.ft8.data() + size_t(idx) * size_t(net.hidden),
                net.hidden, add);
}

bool recomputeOne(const Network& net, const Position& pos, Color perspective,
                  int32_t* acc) {
    std::memset(acc, 0, sizeof(int32_t) * size_t(MAX_HIDDEN));
    const int king = pos.kingSq[perspective];
    if (!validSq(king)) return false;
    for (int sq = 0; sq < 64; ++sq) {
        const Piece pc = pos.board[sq];
        if (pc == NO_PIECE) continue;
        applyPiece(net, acc, perspective, pc, sq, king, true);
    }
    return true;
}

bool recomputeOne16(const Network& net, const Position& pos,
                    Color perspective, int16_t* acc) {
    std::memset(acc, 0, sizeof(int16_t) * size_t(MAX_HIDDEN));
    const int king = pos.kingSq[perspective];
    if (!validSq(king)) return false;
    for (int sq = 0; sq < 64; ++sq) {
        const Piece pc = pos.board[sq];
        if (pc == NO_PIECE) continue;
        applyPiece16(net, acc, perspective, pc, sq, king, true);
    }
    return true;
}

template <typename AccT>
inline __m256i loadAcc8(const AccT* acc, int j) {
    if constexpr (sizeof(AccT) == 2) {
        const __m128i v =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(acc + j));
        return _mm256_cvtepi16_epi32(v);
    } else {
        return _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc + j));
    }
}

// Converts the accumulator to the hidden activation, then runs the three
// dense layers.  Everything is stack-allocated.  The AVX2 path fuses the
// hidden ReLU into the first dense layer to avoid a second pass over h.
template <typename AccT>
int forwardT(const Network& net, const AccT* acc) {
    const int hidden = net.hidden;
    const float ftScale = net.useFt8 ? net.ft8Scale : net.ftScale;
    float l1[DENSE1];
    float l2[DENSE2];

#if CHESSBOT_NNUE_HAVE_AVX2
    if (gUseAvx2) {
        const __m256 scale = _mm256_set1_ps(ftScale);
        const __m256 zero = _mm256_setzero_ps();

        if (!net.useFc1Int8) {
        __m256 a0 = _mm256_loadu_ps(net.fc1B.data() + 0);
        __m256 a1 = _mm256_loadu_ps(net.fc1B.data() + 8);
        __m256 a2 = _mm256_loadu_ps(net.fc1B.data() + 16);
        __m256 a3 = _mm256_loadu_ps(net.fc1B.data() + 24);
        // Second accumulator bank hides the FMA dependency chain by
        // processing two hidden rows per iteration.
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();

        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            const __m256i via = loadAcc8(acc, j);
            __m256 ha = _mm256_mul_ps(_mm256_cvtepi32_ps(via), scale);
            ha = _mm256_add_ps(ha, _mm256_loadu_ps(net.ftBias.data() + j));
            ha = _mm256_max_ps(ha, zero);
            float hta[8];
            _mm256_storeu_ps(hta, ha);

            const __m256i vib = loadAcc8(acc, j + 8);
            __m256 hb = _mm256_mul_ps(_mm256_cvtepi32_ps(vib), scale);
            hb = _mm256_add_ps(hb,
                               _mm256_loadu_ps(net.ftBias.data() + j + 8));
            hb = _mm256_max_ps(hb, zero);
            float htb[8];
            _mm256_storeu_ps(htb, hb);

            for (int lane = 0; lane < 8; ++lane) {
                const float* rowA =
                    net.fc1W.data() + size_t(j + lane) * DENSE1;
                const float* rowB =
                    net.fc1W.data() + size_t(j + 8 + lane) * DENSE1;
                const __m256 va = _mm256_set1_ps(hta[lane]);
                const __m256 vb = _mm256_set1_ps(htb[lane]);
                a0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(rowA + 0), a0);
                a1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(rowA + 8), a1);
                a2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(rowA + 16), a2);
                a3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(rowA + 24), a3);
                c0 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(rowB + 0), c0);
                c1 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(rowB + 8), c1);
                c2 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(rowB + 16), c2);
                c3 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(rowB + 24), c3);
            }
        }
        for (; j + 8 <= hidden; j += 8) {
            const __m256i vi = loadAcc8(acc, j);
            __m256 h = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), scale);
            h = _mm256_add_ps(h, _mm256_loadu_ps(net.ftBias.data() + j));
            h = _mm256_max_ps(h, zero);
            float htmp[8];
            _mm256_storeu_ps(htmp, h);
            for (int lane = 0; lane < 8; ++lane) {
                const __m256 hv = _mm256_set1_ps(htmp[lane]);
                const float* row =
                    net.fc1W.data() + size_t(j + lane) * DENSE1;
                a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row + 0), a0);
                a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row + 8), a1);
                a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row + 16), a2);
                a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(row + 24), a3);
            }
        }
        a0 = _mm256_add_ps(a0, c0);
        a1 = _mm256_add_ps(a1, c1);
        a2 = _mm256_add_ps(a2, c2);
        a3 = _mm256_add_ps(a3, c3);
        _mm256_storeu_ps(l1 + 0, a0);
        _mm256_storeu_ps(l1 + 8, a1);
        _mm256_storeu_ps(l1 + 16, a2);
        _mm256_storeu_ps(l1 + 24, a3);

        for (; j < hidden; ++j) {
            const float v = float(acc[j]) * ftScale + net.ftBias[j];
            const float hv = v > 0.0f ? v : 0.0f;
            const float* row = net.fc1W.data() + size_t(j) * DENSE1;
            for (int k = 0; k < DENSE1; ++k) l1[k] += hv * row[k];
        }
        for (int k = 0; k < DENSE1; ++k)
            if (l1[k] < 0.0f) l1[k] = 0.0f;

        } else {
            // Quantized FC1: ReLU(h) -> uint8 [0,127], weights -> int8,
            // maddubs+madd accumulation, transposed output-major layout.
            uint8_t hq[MAX_HIDDEN];
            const __m256 invScale = _mm256_set1_ps(kInvActivationScale);
            const __m256i qzero = _mm256_setzero_si256();
            const __m256i qmax = _mm256_set1_epi32(127);
            const __m256i packOrder = _mm256_setr_epi32(
                0, 4, 1, 5, 2, 6, 3, 7);
            int j = 0;
            for (; j + 32 <= hidden; j += 32) {
                __m256i qv[4];
                for (int b = 0; b < 4; ++b) {
                    const int off = j + b * 8;
                    const __m256i vi = loadAcc8(acc, off);
                    __m256 h = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), scale);
                    h = _mm256_add_ps(
                        h, _mm256_loadu_ps(net.ftBias.data() + off));
                    h = _mm256_max_ps(h, zero);
                    h = _mm256_mul_ps(h, invScale);
                    __m256i q = _mm256_cvtps_epi32(h);
                    q = _mm256_max_epi32(q, qzero);
                    qv[b] = _mm256_min_epi32(q, qmax);
                }
                const __m256i p01 = _mm256_packs_epi32(qv[0], qv[1]);
                const __m256i p23 = _mm256_packs_epi32(qv[2], qv[3]);
                __m256i packed = _mm256_packus_epi16(p01, p23);
                packed = _mm256_permutevar8x32_epi32(packed, packOrder);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(hq + j), packed);
            }
            for (; j < hidden; ++j) {
                const float v = float(acc[j]) * ftScale + net.ftBias[j];
                float hv = v > 0.0f ? v : 0.0f;
                hv = std::min(hv * kInvActivationScale, 127.0f);
                hq[j] = uint8_t(std::lround(hv));
            }

            const __m256i ones = _mm256_set1_epi16(1);
            for (int k = 0; k < DENSE1; ++k) {
                const int8_t* row =
                    net.fc1W8.data() + size_t(k) * size_t(hidden);
                __m256i sum = _mm256_setzero_si256();
                for (int i = 0; i < hidden; i += 32) {
                    const __m256i h8 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(hq + i));
                    const __m256i w8 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row + i));
                    const __m256i p16 = _mm256_maddubs_epi16(h8, w8);
                    const __m256i p32 = _mm256_madd_epi16(p16, ones);
                    sum = _mm256_add_epi32(sum, p32);
                }
                int32_t lanes[8];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), sum);
                int total = 0;
                for (int lane = 0; lane < 8; ++lane) total += lanes[lane];
                const float v =
                    float(total) * net.fc1Scale8 + net.fc1B[k];
                l1[k] = v > 0.0f ? v : 0.0f;
            }
        }

        // 32x32 head: four independent output vectors.  Start from zero and
        // add the bias after the matrix-vector product.
        __m256 b0 = _mm256_setzero_ps();
        __m256 b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps();
        __m256 b3 = _mm256_setzero_ps();
        for (int i = 0; i < DENSE1; ++i) {
            const __m256 hv = _mm256_set1_ps(l1[i]);
            const float* row = net.fc2W.data() + size_t(i) * DENSE2;
            b0 = _mm256_add_ps(b0, _mm256_mul_ps(hv, _mm256_loadu_ps(row + 0)));
            b1 = _mm256_add_ps(b1, _mm256_mul_ps(hv, _mm256_loadu_ps(row + 8)));
            b2 = _mm256_add_ps(b2, _mm256_mul_ps(hv, _mm256_loadu_ps(row + 16)));
            b3 = _mm256_add_ps(b3, _mm256_mul_ps(hv, _mm256_loadu_ps(row + 24)));
        }
        _mm256_storeu_ps(l2 + 0, b0);
        _mm256_storeu_ps(l2 + 8, b1);
        _mm256_storeu_ps(l2 + 16, b2);
        _mm256_storeu_ps(l2 + 24, b3);
        for (int k = 0; k < DENSE2; ++k) {
            l2[k] += net.fc2B[k];
            if (l2[k] < 0.0f) l2[k] = 0.0f;
        }

        float out = net.outB;
        for (int k = 0; k < DENSE2; ++k) out += net.outW[k] * l2[k];
        if (!std::isfinite(out)) return 0;
        const double cp = double(out);
        if (cp <= -30000.0) return -30000;
        if (cp >= 30000.0) return 30000;
        return int(std::lround(cp));
    }
#endif

    float h[MAX_HIDDEN];
    int j = 0;
    for (; j < hidden; ++j) {
        const float v = float(acc[j]) * ftScale + net.ftBias[j];
        h[j] = v > 0.0f ? v : 0.0f;
    }

    for (int k = 0; k < DENSE1; ++k) l1[k] = net.fc1B[k];
    for (int i = 0; i < hidden; ++i) {
        const float hv = h[i];
        const float* row = net.fc1W.data() + size_t(i) * DENSE1;
        for (int k = 0; k < DENSE1; ++k) l1[k] += hv * row[k];
    }
    for (int k = 0; k < DENSE1; ++k)
        if (l1[k] < 0.0f) l1[k] = 0.0f;

    for (int k = 0; k < DENSE2; ++k) l2[k] = net.fc2B[k];
    for (int i = 0; i < DENSE1; ++i) {
        const float hv = l1[i];
        const float* row = net.fc2W.data() + size_t(i) * DENSE2;
        for (int k = 0; k < DENSE2; ++k) l2[k] += hv * row[k];
    }
    for (int k = 0; k < DENSE2; ++k)
        if (l2[k] < 0.0f) l2[k] = 0.0f;

    float out = net.outB;
    for (int k = 0; k < DENSE2; ++k) out += net.outW[k] * l2[k];
    if (!std::isfinite(out)) return 0;
    const double cp = double(out);
    if (cp <= -30000.0) return -30000;
    if (cp >= 30000.0) return 30000;
    return int(std::lround(cp));
}

inline int forward(const Network& net, const int32_t* acc) {
    return forwardT<int32_t>(net, acc);
}
inline int forward16(const Network& net, const int16_t* acc) {
    return forwardT<int16_t>(net, acc);
}

bool allFinite(const std::vector<float>& v) {
    for (float f : v)
        if (!std::isfinite(f)) return false;
    return true;
}

}  // namespace

int halfkpIndex(Color perspective, Piece pc, int sq, int kingSq) {
    const int bucket = bucketOf(perspective, kingSq);
    const int rel = (pieceColor(pc) == perspective) ? 0 : 1;
    const int osq = orientedSquare(perspective, sq);
    return (bucket * 12 + rel * 6 + int(pieceType(pc))) * 64 + osq;
}

int halfkpFeatures(const Position& pos, Color perspective, int* out,
                   int max) {
    if (!out || max <= 0) return 0;
    const int king = pos.kingSq[perspective];
    if (!validSq(king)) return 0;
    int n = 0;
    for (int p = 0; p < 12; ++p) {
        const Color c = pieceColor(Piece(p));
        const PieceType pt = pieceType(Piece(p));
        if (c == perspective && pt == KING) continue;  // bucket represents it
        U64 bb = pos.pieces(c, pt);
        while (bb && n < max) {
            const int sq = popLsb(bb);
            out[n++] = halfkpIndex(perspective, Piece(p), sq, king);
        }
    }
    return n;
}

bool loadFile(const std::string& path, std::string& error) {
    error.clear();
    if (path.empty()) {
        error = "empty EvalFile path";
        return false;
    }

    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            error = "cannot open \"" + path + "\"";
            return false;
        }
        in.seekg(0, std::ios::end);
        const std::streamoff fileSize = in.tellg();
        in.seekg(0, std::ios::beg);
        if (fileSize < 64) {
            error = "file shorter than the 64-byte header";
            return false;
        }

        uint8_t hdr[64];
        in.read(reinterpret_cast<char*>(hdr), 64);
        if (in.gcount() != 64) {
            error = "truncated header";
            return false;
        }

        if (readU32(hdr + 0) != kMagic) {
            error = "bad magic (expected LCNN)";
            return false;
        }
        const uint32_t version = readU32(hdr + 4);
        const uint32_t arch = readU32(hdr + 8);
        const uint32_t inputs = readU32(hdr + 12);
        const uint32_t hidden = readU32(hdr + 16);
        const uint32_t buckets = readU32(hdr + 20);
        const uint32_t quant = readU32(hdr + 24);
        const uint32_t reserved0 = readU32(hdr + 28);
        const float ftScale = readF32(hdr + 32);
        const uint32_t policySize = readU32(hdr + 36);
        const uint64_t checksum = readU64(hdr + 40);
        bool reservedZero = true;
        for (int i = 48; i < 64; ++i)
            if (hdr[i] != 0) reservedZero = false;

        if (version != kVersion) {
            error = "unsupported version " + std::to_string(version);
            return false;
        }
        if (arch != kArchHalfKP) {
            error = "unsupported architecture " + std::to_string(arch);
            return false;
        }
        if (inputs != uint32_t(HALFKP_INPUTS)) {
            error = "inputs " + std::to_string(inputs) + " != 49152";
            return false;
        }
        if (hidden == 0 || hidden > uint32_t(MAX_HIDDEN)) {
            error = "hidden " + std::to_string(hidden) +
                    " outside 1..512";
            return false;
        }
        if (buckets != uint32_t(HALFKP_BUCKETS)) {
            error = "buckets " + std::to_string(buckets) + " != 64";
            return false;
        }
        if (quant != kQuantInt16Ft) {
            error = "quant " + std::to_string(quant) + " != 1";
            return false;
        }
        if (reserved0 != 0 || !reservedZero) {
            error = "reserved header fields are not zero";
            return false;
        }
        if (!std::isfinite(ftScale) || ftScale <= 0.0f) {
            error = "ft_scale must be finite and positive";
            return false;
        }
        if (policySize != 0 && policySize != uint32_t(POLICY_SIZE)) {
            error = "policy_size must be 0 or 4352";
            return false;
        }

        const uint64_t ftCount = uint64_t(inputs) * hidden;
        uint64_t expected = ftCount * 2;              // int16 ft
        expected += uint64_t(hidden) * 4;             // ft_bias
        expected += uint64_t(hidden) * DENSE1 * 4;    // fc1_w
        expected += uint64_t(DENSE1) * 4;             // fc1_b
        expected += uint64_t(DENSE1) * DENSE2 * 4;    // fc2_w
        expected += uint64_t(DENSE2) * 4;             // fc2_b
        expected += uint64_t(DENSE2) * 4;             // out_w
        expected += 4;                                // out_b
        if (policySize)
            expected += uint64_t(hidden) * policySize * 4 + uint64_t(policySize) * 4;

        if (uint64_t(fileSize) != 64ULL + expected) {
            error = "size mismatch: header+payload " +
                    std::to_string(64ULL + expected) + " bytes, file " +
                    std::to_string(fileSize);
            return false;
        }

        std::vector<uint8_t> payload;
        payload.resize(size_t(expected));
        in.read(reinterpret_cast<char*>(payload.data()),
                std::streamsize(payload.size()));
        if (size_t(in.gcount()) != payload.size()) {
            error = "truncated payload";
            return false;
        }
        if (in.peek() != std::char_traits<char>::eof()) {
            error = "trailing bytes after payload";
            return false;
        }
        if (fnv1a64(payload.data(), payload.size()) != checksum) {
            error = "checksum mismatch";
            return false;
        }

        auto net = std::make_shared<Network>();
        net->version = int(version);
        net->arch = int(arch);
        net->inputs = int(inputs);
        net->hidden = int(hidden);
        net->buckets = int(buckets);
        net->quant = int(quant);
        net->policySize = int(policySize);
        net->ftScale = ftScale;

        const uint8_t* p = payload.data();
        auto take = [&](size_t bytes) -> const uint8_t* {
            const uint8_t* q = p;
            p += bytes;
            return q;
        };

        net->ft.resize(size_t(ftCount));
        std::memcpy(net->ft.data(), take(size_t(ftCount) * 2),
                    size_t(ftCount) * 2);

        const bool wantFt8 =
            gFtQuantPolicy == FTQ_INT8 ||
            (gFtQuantPolicy == FTQ_AUTO && net->hidden >= 128);
        if (wantFt8) {
            int maxAbs = 0;
            for (int16_t v : net->ft)
                maxAbs = std::max(maxAbs, v < 0 ? -int(v) : int(v));
            net->ft8.resize(size_t(ftCount));
            if (maxAbs > 0) {
                const double qscale = 127.0 / double(maxAbs);
                for (size_t i = 0; i < size_t(ftCount); ++i) {
                    long q = std::lround(double(net->ft[i]) * qscale);
                    q = std::clamp(q, -127L, 127L);
                    net->ft8[i] = int8_t(q);
                }
                net->ft8Scale = float(double(ftScale) * double(maxAbs) / 127.0);
            } else {
                std::memset(net->ft8.data(), 0, size_t(ftCount));
                net->ft8Scale = ftScale;
            }
            net->useFt8 = true;
        }

        net->ftBias.resize(hidden);
        std::memcpy(net->ftBias.data(), take(size_t(hidden) * 4),
                    size_t(hidden) * 4);

        net->fc1W.resize(size_t(hidden) * DENSE1);
        std::memcpy(net->fc1W.data(), take(size_t(hidden) * DENSE1 * 4),
                    size_t(hidden) * DENSE1 * 4);

        const bool wantFc1Int8 =
            (gFtQuantPolicy == FTQ_INT8 ||
             (gFtQuantPolicy == FTQ_AUTO && net->hidden >= 128)) &&
            (net->hidden % 32 == 0);
        if (wantFc1Int8) {
            float maxAbs = 0.0f;
            for (float v : net->fc1W)
                maxAbs = std::max(maxAbs, std::fabs(v));
            // Very large FC1 weights amplify the activation-quantization
            // error above the 2 cp contract; keep the exact float path there.
            if (maxAbs > 0.0f && maxAbs <= 0.12f) {
                const float wScale = maxAbs / 127.0f;
                net->fc1W8.resize(size_t(DENSE1) * hidden);
                for (int k = 0; k < DENSE1; ++k) {
                    for (int j = 0; j < net->hidden; ++j) {
                        const float w = net->fc1W[size_t(j) * DENSE1 + k];
                        long q = std::lround(double(w / wScale));
                        q = std::clamp(q, -127L, 127L);
                        net->fc1W8[size_t(k) * hidden + j] = int8_t(q);
                    }
                }
                net->fc1Scale8 = kActivationScale * wScale;
                net->useFc1Int8 = true;
            }
        }

        net->fc1B.resize(DENSE1);
        std::memcpy(net->fc1B.data(), take(size_t(DENSE1) * 4),
                    size_t(DENSE1) * 4);

        net->fc2W.resize(size_t(DENSE1) * DENSE2);
        std::memcpy(net->fc2W.data(), take(size_t(DENSE1) * DENSE2 * 4),
                    size_t(DENSE1) * DENSE2 * 4);

        net->fc2B.resize(DENSE2);
        std::memcpy(net->fc2B.data(), take(size_t(DENSE2) * 4),
                    size_t(DENSE2) * 4);

        net->outW.resize(DENSE2);
        std::memcpy(net->outW.data(), take(size_t(DENSE2) * 4),
                    size_t(DENSE2) * 4);

        std::memcpy(&net->outB, take(4), 4);

        if (policySize) {
            net->policyW.resize(size_t(hidden) * policySize);
            std::memcpy(net->policyW.data(),
                        take(size_t(hidden) * policySize * 4),
                        size_t(hidden) * policySize * 4);
            net->policyB.resize(policySize);
            std::memcpy(net->policyB.data(), take(size_t(policySize) * 4),
                        size_t(policySize) * 4);
        }

        if (!allFinite(net->ftBias) || !allFinite(net->fc1W) ||
            !allFinite(net->fc1B) || !allFinite(net->fc2W) ||
            !allFinite(net->fc2B) || !allFinite(net->outW) ||
            !std::isfinite(net->outB) || !allFinite(net->policyW) ||
            !allFinite(net->policyB)) {
            error = "non-finite weight in payload";
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(gLock);
            gInfo.version = net->version;
            gInfo.arch = net->arch;
            gInfo.inputs = net->inputs;
            gInfo.hidden = net->hidden;
            gInfo.buckets = net->buckets;
            gInfo.quant = net->quant;
            gInfo.policySize = net->policySize;
            gInfo.ftScale = net->ftScale;
            gInfo.ftQuant = net->useFt8 ? 8 : 16;
            gInfo.fc1Quant = net->useFc1Int8 ? 8 : 32;
            gRetained.push_back(net);
        }
        // Publish only after every field is populated.  Old networks stay in
        // gRetained so an in-flight reader can never observe a freed object.
        gNet.store(net.get(), std::memory_order_release);
        gGeneration.fetch_add(1, std::memory_order_release);
        return true;
    } catch (const std::exception& e) {
        error = std::string("exception: ") + e.what();
        return false;
    } catch (...) {
        error = "unknown exception while loading";
        return false;
    }
}

void unloadFile() {
    gNet.store(nullptr, std::memory_order_release);
    gGeneration.fetch_add(1, std::memory_order_release);
    std::lock_guard<std::mutex> lock(gLock);
    gInfo = NetInfo{};
}

bool fileLoaded() { return current() != nullptr; }

NetInfo info() {
    std::lock_guard<std::mutex> lock(gLock);
    return gInfo;
}

const char* simdName() {
#if CHESSBOT_NNUE_HAVE_AVX2
    return gUseAvx2 ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

uint32_t generation() { return gGeneration.load(std::memory_order_acquire); }

void setBackend(const std::string& name) {
    gBackendCpu.store(name != "none", std::memory_order_relaxed);
}
const char* backendName() { return gBackendCpu.load(std::memory_order_relaxed) ? "cpu" : "none"; }
bool backendIsCpu() { return gBackendCpu.load(std::memory_order_relaxed); }
bool legacyEnabled() { return gLegacyEnabled.load(std::memory_order_relaxed); }
void disableLegacy() { gLegacyEnabled.store(false, std::memory_order_relaxed); }

void setFtQuantMode(const std::string& mode) {
    if (mode == "int8") gFtQuantPolicy = FTQ_INT8;
    else if (mode == "int16") gFtQuantPolicy = FTQ_INT16;
    else gFtQuantPolicy = FTQ_AUTO;
}
const char* ftQuantMode() {
    return gFtQuantPolicy == FTQ_INT8 ? "int8"
           : gFtQuantPolicy == FTQ_INT16 ? "int16"
                                         : "auto";
}

void setNetWeight(int w) { gNetWeight = std::clamp(w, 0, 100); }
int getNetWeight() { return gNetWeight; }
void setNetClip(int c) { gNetClip = std::clamp(c, 0, 1500); }
int getNetClip() { return gNetClip; }
void setNetMaterial(int w) { gNetMaterial = std::clamp(w, 0, 100); }
int getNetMaterial() { return gNetMaterial; }

int materialCp(const Position& pos) {
    static constexpr int Value[5] = {100, 320, 330, 500, 900};
    int white = 0;
    int black = 0;
    for (int pt = PAWN; pt < KING; ++pt) {
        white += popcount(pos.pieces(WHITE, PieceType(pt))) * Value[pt];
        black += popcount(pos.pieces(BLACK, PieceType(pt))) * Value[pt];
    }
    const int diff = white - black;
    return pos.side == WHITE ? diff : -diff;
}

void refreshAccumulator(const Position& pos) {
    Position& p = const_cast<Position&>(pos);
    const Network* net = current();
    p.nnueAccGeneration = generation();
    p.nnueAccValid = false;
    p.nnueAccQuant = 0;
    if (!net || net->hidden <= 0) return;
    if (!validSq(pos.kingSq[WHITE]) || !validSq(pos.kingSq[BLACK])) return;
    bool ok;
    if (net->useFt8) {
        ok = recomputeOne16(*net, pos, WHITE, p.nnueAccumulator16[WHITE]) &&
             recomputeOne16(*net, pos, BLACK, p.nnueAccumulator16[BLACK]);
        p.nnueAccQuant = 8;
    } else {
        ok = recomputeOne(*net, pos, WHITE, p.nnueAccumulator[WHITE]) &&
             recomputeOne(*net, pos, BLACK, p.nnueAccumulator[BLACK]);
        p.nnueAccQuant = 16;
    }
    p.nnueAccValid = ok;
}

bool accumulatorValid(const Position& pos) {
    const Network* net = current();
    return net != nullptr && pos.nnueAccValid &&
           pos.nnueAccGeneration == generation() &&
           pos.nnueAccQuant == (net->useFt8 ? 8 : 16) &&
           validSq(pos.kingSq[WHITE]) && validSq(pos.kingSq[BLACK]);
}

void invalidateAccumulator(const Position& pos) {
    Position& p = const_cast<Position&>(pos);
    p.nnueAccValid = false;
    p.nnueAccQuant = 0;
}

bool accumulatorAfterMake(Position& pos, Move m, Undo& u, Color us) {
    const Network* net = current();
    if (!net || !accumulatorValid(pos)) return false;

    const int from = m.from();
    const int to = m.to();
    const Piece placed = pos.board[to];
    const Piece original =
        (m.type() == Move::PROMOTION) ? makePiece(us, PAWN) : placed;
    const int movedKing = (pieceType(original) == KING) ? int(us) : -1;

    int rookFrom = -1, rookTo = -1;
    if (m.type() == Move::CASTLE) {
        if (to > from) {
            rookFrom = us == WHITE ? 7 : 63;
            rookTo = rookFrom - 2;
        } else {
            rookFrom = us == WHITE ? 0 : 56;
            rookTo = rookFrom + 3;
        }
    }

    if (net->useFt8) {
        for (int c = 0; c < COLOR_NB; ++c) {
            const Color perspective = Color(c);
            int16_t* acc = pos.nnueAccumulator16[c];
            if (c == movedKing) {
                recomputeOne16(*net, pos, perspective, acc);
                continue;
            }
            const int king = pos.kingSq[c];
            if (u.captured != NO_PIECE)
                applyPiece16(*net, acc, perspective, u.captured, u.capturedSq,
                             king, false);
            applyPiece16(*net, acc, perspective, original, from, king, false);
            applyPiece16(*net, acc, perspective, placed, to, king, true);
            if (rookFrom >= 0) {
                const Piece rook = pos.board[rookTo];
                applyPiece16(*net, acc, perspective, rook, rookFrom, king,
                             false);
                applyPiece16(*net, acc, perspective, rook, rookTo, king, true);
            }
        }
    } else {
        for (int c = 0; c < COLOR_NB; ++c) {
            const Color perspective = Color(c);
            int32_t* acc = pos.nnueAccumulator[c];
            if (c == movedKing) {
                recomputeOne(*net, pos, perspective, acc);
                continue;
            }
            const int king = pos.kingSq[c];
            if (u.captured != NO_PIECE)
                applyPiece(*net, acc, perspective, u.captured, u.capturedSq,
                           king, false);
            applyPiece(*net, acc, perspective, original, from, king, false);
            applyPiece(*net, acc, perspective, placed, to, king, true);
            if (rookFrom >= 0) {
                const Piece rook = pos.board[rookTo];
                applyPiece(*net, acc, perspective, rook, rookFrom, king,
                           false);
                applyPiece(*net, acc, perspective, rook, rookTo, king, true);
            }
        }
    }
    u.accUpdated = true;
    return true;
}

void accumulatorAfterUnmake(Position& pos, Move m, const Undo& u, Color us) {
    if (!u.accUpdated) return;
    const Network* net = current();
    if (!net || !pos.nnueAccValid ||
        pos.nnueAccGeneration != generation()) {
        // Stale accumulator: force a full recompute on the next eval.
        pos.nnueAccValid = false;
        return;
    }

    const int from = m.from();
    const int to = m.to();
    const Piece original = pos.board[from];
    const Piece placed =
        (m.type() == Move::PROMOTION) ? makePiece(us, PieceType(m.promo()))
                                      : original;
    const int movedKing = (pieceType(original) == KING) ? int(us) : -1;

    int rookFrom = -1, rookTo = -1;
    if (m.type() == Move::CASTLE) {
        if (to > from) {
            rookFrom = us == WHITE ? 7 : 63;
            rookTo = rookFrom - 2;
        } else {
            rookFrom = us == WHITE ? 0 : 56;
            rookTo = rookFrom + 3;
        }
    }

    if (net->useFt8) {
        for (int c = 0; c < COLOR_NB; ++c) {
            const Color perspective = Color(c);
            int16_t* acc = pos.nnueAccumulator16[c];
            if (c == movedKing) {
                recomputeOne16(*net, pos, perspective, acc);
                continue;
            }
            const int king = pos.kingSq[c];
            applyPiece16(*net, acc, perspective, placed, to, king, false);
            applyPiece16(*net, acc, perspective, original, from, king, true);
            if (u.captured != NO_PIECE)
                applyPiece16(*net, acc, perspective, u.captured, u.capturedSq,
                             king, true);
            if (rookFrom >= 0) {
                const Piece rook = pos.board[rookFrom];
                applyPiece16(*net, acc, perspective, rook, rookTo, king,
                             false);
                applyPiece16(*net, acc, perspective, rook, rookFrom, king,
                             true);
            }
        }
    } else {
        for (int c = 0; c < COLOR_NB; ++c) {
            const Color perspective = Color(c);
            int32_t* acc = pos.nnueAccumulator[c];
            if (c == movedKing) {
                recomputeOne(*net, pos, perspective, acc);
                continue;
            }
            const int king = pos.kingSq[c];
            applyPiece(*net, acc, perspective, placed, to, king, false);
            applyPiece(*net, acc, perspective, original, from, king, true);
            if (u.captured != NO_PIECE)
                applyPiece(*net, acc, perspective, u.captured, u.capturedSq,
                           king, true);
            if (rookFrom >= 0) {
                const Piece rook = pos.board[rookFrom];
                applyPiece(*net, acc, perspective, rook, rookTo, king, false);
                applyPiece(*net, acc, perspective, rook, rookFrom, king, true);
            }
        }
    }
}

int evaluateNetwork(const Position& pos) {
    const Network* net = current();
    if (!net) return 0;
    if (!accumulatorValid(pos)) refreshAccumulator(pos);
    if (!pos.nnueAccValid) return 0;
    if (net->useFt8) return forward16(*net, pos.nnueAccumulator16[pos.side]);
    return forward(*net, pos.nnueAccumulator[pos.side]);
}

}  // namespace NNUE
}  // namespace chess


