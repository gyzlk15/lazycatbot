// ChessBot - a self-contained UCI chess engine in C++.
// Board: bitboards, magic bitboards for sliders.
// Search: iterative deepening PVS, transposition table, quiescence,
//         null-move pruning, LMR, futility pruning, SEE.
// Evaluation: tapered material/PST + mobility + pawn structure + king safety.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#if defined(__BMI2__)
#include <immintrin.h>
#define CHESSBOT_PEXT 1
#else
#define CHESSBOT_PEXT 0
#endif
#include <vector>

namespace chess {

using U64 = uint64_t;

constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;
constexpr int INF = 32001;
constexpr int MATE = 32000;
constexpr int MATE_IN_MAX = MATE - MAX_PLY;
constexpr int MAX_HIST = 4096;
// HalfKP v1 accumulator width (docs/NNUE_PILOT.md 3.3 allows hidden <= 512).
constexpr int MAX_NNUE_HIDDEN = 512;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5,
    PIECE_TYPE_NB = 6
};

inline constexpr int PieceValue[PIECE_TYPE_NB] = {
    100, 320, 330, 500, 950, 20000
};

// 0..5 white, 6..11 black, 12 empty
enum Piece : int {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE = 12, PIECE_NB = 13
};

constexpr Piece makePiece(Color c, PieceType pt) {
    return Piece(int(c) * 6 + int(pt));
}
constexpr PieceType pieceType(Piece p) {
    return p < NO_PIECE ? PieceType(int(p) % 6) : PIECE_TYPE_NB;
}
constexpr Color pieceColor(Piece p) {
    return Color(int(p) / 6);
}
constexpr bool isPiece(Piece p) { return p >= W_PAWN && p < NO_PIECE; }

inline int fileOf(int sq) { return sq & 7; }
inline int rankOf(int sq) { return sq >> 3; }
inline bool validSq(int sq) { return sq >= 0 && sq < 64; }

inline U64 bit(int sq) { return U64(1) << sq; }
inline bool testBit(U64 b, int sq) { return (b >> sq) & 1ULL; }
inline int popcount(U64 b) { return std::popcount(b); }
inline int lsb(U64 b) { return std::countr_zero(b); }
inline int popLsb(U64& b) { int s = lsb(b); b &= b - 1; return s; }
inline bool moreThanOne(U64 b) { return (b & (b - 1)) != 0; }

// ---------------------------------------------------------------------------
// Move encoding:
//   bits 0-5   from
//   bits 6-11  to
//   bits 12-14 promotion piece (0 none, otherwise KNIGHT..QUEEN)
//   bits 15-17 move type
//   bit  18    capture flag
// ---------------------------------------------------------------------------
class Move {
public:
    enum Type : uint32_t {
        NORMAL = 0, DOUBLE_PUSH = 1, EN_PASSANT = 2, CASTLE = 3, PROMOTION = 4
    };

private:
    uint32_t d_;

public:
    Move() : d_(0) {}
    Move(int from, int to, int promo = 0, int type = NORMAL, bool capture = false)
        : d_(uint32_t(from) | (uint32_t(to) << 6) | (uint32_t(promo) << 12) |
             (uint32_t(type) << 15) | (capture ? (1u << 18) : 0)) {}
    explicit Move(uint32_t d) : d_(d) {}

    int from() const { return int(d_ & 63); }
    int to() const { return int((d_ >> 6) & 63); }
    int promo() const { return int((d_ >> 12) & 7); }
    int type() const { return int((d_ >> 15) & 7); }
    bool isCapture() const { return (d_ & (1u << 18)) != 0; }
    bool isNone() const { return d_ == 0; }
    uint32_t raw() const { return d_; }

    bool operator==(Move o) const { return d_ == o.d_; }
    bool operator!=(Move o) const { return d_ != o.d_; }
};

inline std::string moveToUci(Move m) {
    if (m.isNone()) return "0000";
    std::string s;
    s += char('a' + fileOf(m.from()));
    s += char('1' + rankOf(m.from()));
    s += char('a' + fileOf(m.to()));
    s += char('1' + rankOf(m.to()));
    if (m.type() == Move::PROMOTION) s += " nbrq"[m.promo()];
    return s;
}

struct MoveList {
    Move moves[MAX_MOVES];
    int count = 0;
    void add(Move m) { if (count < MAX_MOVES) moves[count++] = m; }
    void clear() { count = 0; }
};

// ---------------------------------------------------------------------------
// Bitboard attack tables (initialized once at startup)
// ---------------------------------------------------------------------------
namespace Bitboards {
extern U64 pawnAttacks[2][64];
extern U64 knightAttacks[64];
extern U64 kingAttacks[64];
extern U64 between[64][64];
extern U64 line[64][64];

void init();
} // namespace Bitboards

struct Magic {
    U64 mask = 0;
    U64 magic = 0;
    int shift = 0;
    U64* attacks = nullptr;
};

extern Magic rookMagics[64];
extern Magic bishopMagics[64];

inline unsigned sliderIndex(const Magic& m, U64 occ) {
#if CHESSBOT_PEXT
    return unsigned(_pext_u64(occ, m.mask));
#else
    return unsigned(((occ & m.mask) * m.magic) >> m.shift);
#endif
}

inline U64 bishopAttacks(int sq, U64 occ) {
    const Magic& m = bishopMagics[sq];
    return m.attacks[sliderIndex(m, occ)];
}
inline U64 rookAttacks(int sq, U64 occ) {
    const Magic& m = rookMagics[sq];
    return m.attacks[sliderIndex(m, occ)];
}

inline U64 pawnAttacks(Color c, int sq) { return Bitboards::pawnAttacks[c][sq]; }
inline U64 knightAttacks(int sq) { return Bitboards::knightAttacks[sq]; }
inline U64 kingAttacks(int sq) { return Bitboards::kingAttacks[sq]; }
inline U64 betweenSquares(int a, int b) { return Bitboards::between[a][b]; }
inline U64 lineSquares(int a, int b) { return Bitboards::line[a][b]; }

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------
namespace Zobrist {
extern U64 psq[12][64];
extern U64 castling[16];
extern U64 epFile[8];
extern U64 rule50[128]; // halfmove-clock tweak for the transposition table
extern U64 side;
void init();
} // namespace Zobrist

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------
enum CastlingRight : int {
    WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8
};

struct Undo {
    U64 key = 0;
    int castling = 0;
    int ep = -1;
    int halfMove = 0;
    int fullMove = 1;
    Piece captured = NO_PIECE;
    int capturedSq = -1;
    // True when the move successfully updated the NNUE accumulators, so
    // unmakeMove knows whether it has to reverse the deltas.
    bool accUpdated = false;
};

class Position {
public:
    U64 byType[PIECE_TYPE_NB] = {};
    U64 byColor[COLOR_NB] = {};
    U64 occ = 0;
    Piece board[64];
    Color side = WHITE;
    int castling = 0;
    int ep = -1;
    int halfMove = 0;
    int fullMove = 1;
    U64 key = 0;
    int kingSq[COLOR_NB] = {-1, -1};
    int histPly = 0;
    U64 keyHist[MAX_HIST] = {};
    bool isNullHist_[MAX_HIST] = {};
    Undo undoStack[MAX_HIST] = {};

    // HalfKP v1 accumulators (one per perspective).  `mutable` lets
    // evaluateNetwork() refresh them lazily from a const Position; each
    // search thread owns its Position copy, so this never races with another
    // thread.  The generation tag invalidates them when a new net is loaded.
    // The int16 array is used only by the optional int8-FT fast path, where
    // the maximum sum (31 * 127) safely fits in int16.
    mutable int32_t nnueAccumulator[COLOR_NB][MAX_NNUE_HIDDEN] = {};
    mutable int16_t nnueAccumulator16[COLOR_NB][MAX_NNUE_HIDDEN] = {};
    mutable bool nnueAccValid = false;
    mutable uint32_t nnueAccGeneration = 0;
    mutable int nnueAccQuant = 0;  // 0 invalid, 16 = int16 FT, 8 = int8 FT

    Position();

    void clear();
    void setStartpos();
    bool setFen(const std::string& fen);
    std::string fen() const;

    U64 pieces(Color c) const { return byColor[c]; }
    U64 pieces(Color c, PieceType pt) const { return byColor[c] & byType[pt]; }
    U64 pieces(PieceType pt) const { return byType[pt]; }
    U64 occupied() const { return occ; }

    bool isSquareAttacked(int sq, Color by) const;
    U64 attackersTo(int sq, U64 occ) const;
    bool inCheck() const {
        return kingSq[side] >= 0 && isSquareAttacked(kingSq[side], ~side);
    }
    bool hasNonPawnMaterial(Color c) const {
        return (byColor[c] & ~(byType[PAWN] | byType[KING])) != 0;
    }

    bool makeMove(Move m);
    void unmakeMove(Move m);
    bool makeNullMove();
    void unmakeNullMove();

    bool isRepetition() const;
    bool isInsufficientMaterial() const;
    bool hasLegalMove() const;
    bool isDraw() const {
        // Common case: neither the 50-move rule nor repetition applies.
        if (halfMove < 100 && !isRepetition()) return isInsufficientMaterial();
        // Checkmate takes precedence over rule-50/repetition claims. If the
        // side to move is not in check, 0 is correct for both a claimable draw
        // and stalemate. Only an in-check position needs legal-move generation
        // to distinguish checkmate from a claimable draw.
        return !inCheck() || hasLegalMove();
    }
    bool epHashable() const;

    // Zobrist key extended with the halfmove clock so that TT scores which
    // depend on the 50-move rule are not reused across different clocks.
    U64 transpositionKey() const {
        return key ^ Zobrist::rule50[std::min(halfMove, 127)];
    }

    void generateMoves(MoveList& list) const;
    void generateCaptures(MoveList& list) const;
    Move parseMove(const std::string& str) const;
    int see(Move m) const;

    std::string ascii() const;

private:
    bool parseFenInto(const std::string& fen);
    void generateMovesImpl(MoveList& list, bool capturesOnly) const;
    void putPiece(Piece p, int sq);
    void removePiece(int sq);
    void movePiece(int from, int to);
    int seeRec(int to, Color stm, int targetValue, U64 occ) const;
};

// ---------------------------------------------------------------------------
// Syzygy tablebase probing (optional, Fathom backend)
// ---------------------------------------------------------------------------
namespace Syzygy {

enum class Wdl : int {
    Loss = -2,
    BlessedLoss = -1,
    Draw = 0,
    CursedWin = 1,
    Win = 2,
    Fail = 99
};

struct RootResult {
    bool ok = false;
    bool checkmate = false;
    bool stalemate = false;
    int from = -1;
    int to = -1;
    int promo = 0;  // Fathom TB_PROMOTES_* (0 none, 1 queen, 2 rook, 3 bishop, 4 knight)
    Wdl wdl = Wdl::Fail;
    int dtz = 0;
};

bool init(const std::string& path);
void shutdown();
bool enabled();
int tablebaseMaxPieces();
int maxPieces();
void setProbeDepth(int depth);
int probeDepth();
void setRule50(bool useRule50);
bool rule50();
void setProbeLimit(int limit);
int probeLimit();

RootResult probeRoot(const Position& pos);
Wdl probeWDL(const Position& pos);

} // namespace Syzygy

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
void initEvaluation();
int classicalEvaluate(const Position& pos);
int evaluate(const Position& pos);

// Learned evaluation: HalfKP-like 768 -> 64 -> 1 network. It predicts a
// correction to the classical evaluation and is blended in by weight.
namespace NNUE {
constexpr int INPUTS = 768;
constexpr int HIDDEN = 64;

extern float W1[INPUTS][HIDDEN];
extern float B1[HIDDEN];
extern float W2[HIDDEN];
extern float B2;
extern bool trained;
extern bool enabled;
extern bool direct;
extern int weight;
extern int clip;

void init();
int featureIndices(const Position& pos, int* out);
int correction(const Position& pos);
void setEnabled(bool v);
bool isEnabled();
void setWeight(int w);
int getWeight();
bool isAvailable();
} // namespace NNUE

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------
enum Bound : uint8_t {
    BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3
};

struct TTEntry {
    uint32_t key = 0;
    uint32_t move = 0;
    int16_t score = 0;
    int16_t eval = 0;
    uint8_t depth = 0;
    uint8_t bound = BOUND_NONE;
    uint8_t gen = 0;
    uint8_t pad = 0;
};
static_assert(sizeof(TTEntry) == 16, "TTEntry must be 16 bytes");

// Lockless, 16-byte-per-slot shared transposition table.
//
// Each slot is two 64-bit atomics.  The tag word holds the 32-bit key
// signature, an 8-bit generation, a 23-bit checksum of the data word and a
// busy bit:
//
//   tag  : signature(32) | checksum(23) | generation(8) | busy(1)
//   data : move(19) | score(16) | eval(16) | depth(8) | bound(2) | pad(3)
//
// Writer: tag=busy (relaxed), seq_cst fence, data (relaxed), tag=valid
// (release).  Reader: tag1 (acquire), reject busy, data (acquire), tag2
// (acquire), require tag1==tag2 and not busy, then verify the checksum.
//
// The busy marker plus the tag1==tag2 check means a reader can never combine
// the tag of one store with the data of another store: any interleaved store
// either changes the tag or leaves the busy bit set.  The checksum is a
// second line of defence for the (same-signature, same-generation) ABA case,
// where tag1==tag2 can still happen; accepting wrong data then requires a
// 23-bit checksum collision on top of the 32-bit signature match.  A torn
// read is therefore never silently used; there are no locks, no extra seq
// word and no extra bytes per slot.  The fence costs one mfence per TT store
// on x86 (stores are far rarer than probes; measured cost is in REPORT.md).
//
// resize() and clear() are only called while no search is running (the UCI
// layer joins every worker first).  newSearch()/generation and hashfull()
// are safe during a search.
class TranspositionTable {
public:
    TranspositionTable() { resize(64); }

    void resize(size_t megabytes);
    void clear();
    void newSearch() {
        generation_.fetch_add(1, std::memory_order_relaxed);
    }

    bool probe(U64 key, TTEntry& out) const {
        size_t idx = key & mask_;
        TTEntry e;
        if (!loadEntry(idx, e)) return false;
        if (e.key != signature(key) || e.bound == BOUND_NONE) return false;
        out = e;
        return true;
    }

    void store(U64 key, int depth, int score, int eval, Bound bound, Move m) {
        size_t idx = key & mask_;
        const uint8_t gen = generation_.load(std::memory_order_relaxed);
        const uint32_t sig = signature(key);

        TTEntry old;
        if (loadEntry(idx, old) && old.bound != BOUND_NONE) {
            const bool same = (old.key == sig);
            if (same && depth + 3 < old.depth) return;
            if (!same && old.gen == gen && depth < old.depth) return;
        }

        const uint32_t move = m.raw() & 0x7FFFFu;
        const uint8_t d = uint8_t(std::min(depth, 255));
        const uint8_t b = uint8_t(bound) & 3u;
        uint64_t data = uint64_t(move) |
                        (uint64_t(uint16_t(int16_t(score))) << 19) |
                        (uint64_t(uint16_t(int16_t(eval))) << 35) |
                        (uint64_t(d) << 51) | (uint64_t(b) << 59);
        uint64_t tag = uint64_t(sig) | (uint64_t(checksum(data, gen)) << 32) |
                       (uint64_t(gen) << 55);

        Slot& s = table_[idx];
        s.tag.store(tag | BUSY_BIT, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        s.data.store(data, std::memory_order_relaxed);
        s.tag.store(tag, std::memory_order_release);
    }

    int hashfull() const {
        const size_t n = std::min<size_t>(count_, 1000);
        const uint8_t gen = generation_.load(std::memory_order_relaxed);
        int used = 0;
        for (size_t i = 0; i < n; ++i) {
            TTEntry e;
            if (loadEntry(i, e) && e.gen == gen && e.bound != BOUND_NONE)
                ++used;
        }
        return int(used * 1000 / std::max<size_t>(n, 1));
    }

    size_t size() const { return count_; }

private:
    struct alignas(16) Slot {
        std::atomic<uint64_t> tag{0};
        std::atomic<uint64_t> data{0};
    };
    static_assert(sizeof(Slot) == 16, "TT slot must stay 16 bytes");

    static uint32_t signature(U64 key) {
        return uint32_t(key) ^ uint32_t(key >> 32);
    }

    static constexpr uint64_t BUSY_BIT = 1ULL << 63;

    // 23-bit payload checksum.  One multiply spreads every input bit into the
    // top 23 bits; the generation is folded into the payload.
    static uint32_t checksum(uint64_t data, uint8_t gen) {
        uint64_t z = data ^ (uint64_t(gen) << 56);
        z *= 0x9E3779B97F4A7C15ULL;
        return uint32_t(z >> 41);
    }

    bool loadEntry(size_t idx, TTEntry& out) const {
        const Slot& s = table_[idx];
        const uint64_t tag1 = s.tag.load(std::memory_order_acquire);
        if (tag1 & BUSY_BIT) return false;
        const uint64_t data = s.data.load(std::memory_order_acquire);
        const uint64_t tag2 = s.tag.load(std::memory_order_acquire);
        if (tag1 != tag2 || (tag2 & BUSY_BIT)) return false;

        const uint8_t gen = uint8_t((tag2 >> 55) & 0xFFu);
        const uint32_t check = uint32_t((tag2 >> 32) & 0x7FFFFFu);
        if (checksum(data, gen) != check) return false;

        out.key = uint32_t(tag2);
        out.move = uint32_t(data & 0x7FFFFu);
        out.score = int16_t(uint16_t((data >> 19) & 0xFFFFu));
        out.eval = int16_t(uint16_t((data >> 35) & 0xFFFFu));
        out.depth = uint8_t((data >> 51) & 0xFFu);
        out.bound = uint8_t((data >> 59) & 3u);
        out.gen = gen;
        out.pad = 0;
        return true;
    }

    std::unique_ptr<Slot[]> table_;
    size_t mask_ = 0;
    size_t count_ = 0;
    std::atomic<uint8_t> generation_{1};
};

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
struct SearchLimits {
    int64_t wtime = -1;
    int64_t btime = -1;
    int64_t winc = 0;
    int64_t binc = 0;
    int movestogo = 0;
    int depth = MAX_PLY;
    int movetime = -1;
    uint64_t nodes = 0;
    bool infinite = false;
    // True for a real UCI `go infinite` (not bench/trainer pseudo-infinite):
    // the worker must not publish bestmove until the controller stops it.
    bool waitForStop = false;
    int moveOverhead = 30;
};

// Shared by all searches of one `go`.  `stop` is the group's single stop
// flag; `nodes` is a coarse global counter (each thread publishes in blocks
// of 1024) used for `go nodes` and for the reported NPS.
struct SharedSearchState {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> nodes{0};
    std::atomic<uint64_t> nodeLimit{0};
};

class Search {
public:
    // Legacy/external position (trainer, bench, single-threaded callers).
    Search(Position& pos, TranspositionTable& tt) : pos_(pos), tt_(tt) {
        clearState();
    }

    // Worker with a private Position copy.  loadPosition() is called once per
    // `go`, before the thread is started.
    explicit Search(TranspositionTable& tt)
        : owned_(std::make_unique<Position>()), pos_(*owned_), tt_(tt) {
        clearState();
    }

    void run(const SearchLimits& limits);
    void stop() {
        stopFlag.store(true, std::memory_order_relaxed);
        if (SharedSearchState* s = shared_.load(std::memory_order_relaxed))
            s->stop.store(true, std::memory_order_relaxed);
    }
    bool stopped() const {
        if (stopFlag.load(std::memory_order_relaxed)) return true;
        SharedSearchState* s = shared_.load(std::memory_order_relaxed);
        return s && s->stop.load(std::memory_order_relaxed);
    }
    void resetStop() { stopFlag.store(false, std::memory_order_relaxed); }
    void clearState();
    void loadPosition(const Position& p) { pos_ = p; }
    Position& position() { return pos_; }
    void setShared(SharedSearchState* s) {
        shared_.store(s, std::memory_order_relaxed);
    }
    void setThreadId(int id) { threadId_ = id; }
    void setMainThread(bool m) { mainThread_ = m; }
    uint64_t nodes() const { return nodeCount_; }
    uint64_t reportedNodes() const;
    void setSilent(bool s) { silent_ = s; }
    int lastScore() const { return lastScore_; }
    Move lastBest() const { return lastBest_; }

    std::atomic<bool> stopFlag{false};

private:
    struct RootMove {
        Move move;
        int score = -INF;
        int prevScore = -INF;
    };

    std::unique_ptr<Position> owned_;
    Position& pos_;
    TranspositionTable& tt_;
    std::atomic<SharedSearchState*> shared_{nullptr};
    int threadId_ = 0;
    bool mainThread_ = true;
    bool tbEnabled_ = false;
    int tbProbeDepth_ = 1;
    int tbMaxPieces_ = 0;
    bool tbRule50_ = true;

    uint64_t nodeCount_ = 0;
    uint64_t nodeLimit_ = 0;
    int selDepth_ = 0;
    int searchDepth_ = 0;
    bool timeLimited_ = false;
    bool silent_ = false;
    int lastScore_ = 0;
    Move lastBest_ = Move();
    std::chrono::steady_clock::time_point startTime_;
    int64_t softLimitMs_ = 0;
    int64_t hardLimitMs_ = 0;

    RootMove rootMoves_[MAX_MOVES];
    int rootMoveCount_ = 0;

    Move killers_[MAX_PLY][2];
    Move counterMoves_[COLOR_NB][64][64];
    Move moveStack_[MAX_PLY + 2];
    int history_[COLOR_NB][64][64] = {};
    int captureHistory_[COLOR_NB][PIECE_TYPE_NB][64][PIECE_TYPE_NB] = {};
    int continuationHistory_[PIECE_NB][64][PIECE_NB][64] = {};
    int evalStack_[MAX_PLY] = {};
    Move pvTable_[MAX_PLY][MAX_PLY];
    int pvLen_[MAX_PLY] = {};
    Move rootBest_ = Move();
    Move rootFallback_ = Move();

    int searchRoot(int depth, int alpha, int beta);
    int negamax(int depth, int alpha, int beta, int ply, bool cutNode,
                bool nullAllowed);
    int quiescence(int alpha, int beta, int ply);
    void scoreMoves(Move* moves, int count, int* scores, Move ttMove,
                    int ply, bool useCounter = true) const;
    bool nodeCheck();
    void flushNodes();
    bool timeUp();
    void updatePv(int ply, Move m);
    int moveOrderScore(Move m, Move ttMove, int ply,
                       bool useCounter = true) const;
    void printInfo(int depth, int score, int64_t elapsed) const;
    int64_t elapsedMs() const;
    void clearHistory();
    static void updateHistory(int& entry, int bonus);
};

// ---------------------------------------------------------------------------
// Serialized UCI output
//
// The UCI loop and the search thread both write to stdout. Build every line /
// block in a string and emit it while holding one process-wide mutex so lines
// can never interleave (e.g. "readyok" in the middle of an "info" line).
// ---------------------------------------------------------------------------
namespace Output {
inline std::mutex& mutex() {
    static std::mutex m;
    return m;
}
inline void raw(const std::string& s) {
    std::lock_guard<std::mutex> lock(mutex());
    std::cout << s;
    std::cout.flush();
}
inline void line(const std::string& s) {
    std::lock_guard<std::mutex> lock(mutex());
    std::cout << s << '\n';
    std::cout.flush();
}
} // namespace Output

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline const char* colorName(Color c) { return c == WHITE ? "white" : "black"; }
inline const char* pieceName(Piece p) {
    static const char* names[] = {
        "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "."
    };
    return names[int(p)];
}

void initAll();
uint64_t perft(Position& pos, int depth);

} // namespace chess
