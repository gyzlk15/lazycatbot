// HalfKP feature extractor for the training pipeline.
//
// Supported input lines (one position per line):
//
//   FEN|wdl|cp|bestmove|pv     halfkp-text-v2 (see training/data_contract.md)
//   FEN|target                 legacy: side-to-move win probability in [0,1]
//   FEN|cp                     legacy with --cp: side-to-move centipawns
//
// wdl    side-to-move win probability in [0,1]
// cp     optional side-to-move centipawns ("nan"/"" when unknown)
// bestmove  optional first move, standard UCI
// pv     optional space separated UCI moves (UCI_Chess960 castling accepted)
//
// Output shards (little-endian, all in one directory):
//   features.u16  [N,32]   HalfKP indices, zero padded
//   counts.u8     [N]      active feature count (<= 32)
//   targets.f32   [N]      side-to-move win probability
//   cp.f32        [N]      side-to-move centipawns, NaN when unknown
//   moves.u16     [N]      policy index of bestmove, 0xFFFF when absent
//   pv.u16        [N,8]    policy indices of the PV, 0xFFFF padded
//   groups.u32    [N]      FNV-1a 32 of FEN (board/stm/castling/ep)
//   meta.json              filters, counts, feature version, POV statement
//
// Feature index (contract docs/NNUE_PILOT.md 3.1):
//   bucket = oriented own-king square (0..63)
//   index  = (bucket * 12 + relativeColor * 6 + pieceType) * 64 + orientedSq
// Inputs: 64 * 12 * 64 = 49,152. Own king is excluded from the piece list.
// Features are for the side to move.
//
// Policy index (contract 3.2, 4352 classes):
//   normal:    from * 64 + to
//   promotion: 4096 + (promo - 2) * 64 + from   (promo 2=N 3=B 4=R 5=Q)
//   absent:    0xFFFF
// Castling is normalized to standard UCI (king from -> king to) before
// encoding, whatever the source used.
//
// Usage:
//   extract_halfkp <input.txt> <outdir> [--max N] [--cp] [--source-stats FILE]
//   extract_halfkp --selftest
#include "engine.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace chess;

namespace {

constexpr int kInputs = 64 * 12 * 64;
constexpr int kMaxActive = 32;
constexpr int kPolicySize = 4352;
constexpr uint16_t kPolicyAbsent = 0xFFFF;
constexpr int kPvLen = 8;

// --------------------------------------------------------------------- output

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

template <typename T>
class ShardWriter {
public:
    bool open(const std::string& path, size_t rows_per_flush = 1 << 16) {
        file_.open(path, std::ios::binary | std::ios::trunc);
        if (!file_) return false;
        buf_.reserve((1u << 20) / sizeof(T) + 1);
        (void)rows_per_flush;
        return true;
    }

    void push(const T* row, size_t n) {
        buf_.insert(buf_.end(), row, row + n);
        if (buf_.size() >= (1u << 20) / sizeof(T)) flush();
    }

    void push(const T& value) { push(&value, 1); }

    void flush() {
        if (!buf_.empty()) {
            file_.write(reinterpret_cast<const char*>(buf_.data()),
                        std::streamsize(buf_.size() * sizeof(T)));
            buf_.clear();
        }
        file_.flush();
    }

    void close() {
        flush();
        file_.close();
    }

private:
    std::ofstream file_;
    std::vector<T> buf_;
};

// ------------------------------------------------------------------- parsing

struct MoveToken {
    int from = -1;
    int to = -1;
    int promo = 0;  // PieceType or 0
};

bool squareFromChars(char file, char rank) {
    return file >= 'a' && file <= 'h' && rank >= '1' && rank <= '8';
}

bool parseMoveToken(const std::string& tok, MoveToken& out) {
    if (tok.size() != 4 && tok.size() != 5) return false;
    if (!squareFromChars(tok[0], tok[1]) || !squareFromChars(tok[2], tok[3]))
        return false;
    out.from = (tok[1] - '1') * 8 + (tok[0] - 'a');
    out.to = (tok[3] - '1') * 8 + (tok[2] - 'a');
    out.promo = 0;
    if (tok.size() == 5) {
        switch (tok[4]) {
            case 'n': out.promo = KNIGHT; break;
            case 'b': out.promo = BISHOP; break;
            case 'r': out.promo = ROOK; break;
            case 'q': out.promo = QUEEN; break;
            default: return false;
        }
    }
    return true;
}

uint16_t policyIndex(const MoveToken& m) {
    if (m.promo != 0) {
        // promo (contract) = pieceType + 1, so (promo - 2) == (pt - 1).
        return uint16_t(4096 + (m.promo - 1) * 64 + m.from);
    }
    return uint16_t(m.from * 64 + m.to);
}

// UCI_Chess960 castling is "king from -> own rook square". Rewrite the
// destination to the standard king target (g/c file). A PV can contain later
// castles (both colours) and king/rook moves onto their original squares, so
// track which king/rooks have already moved: only the first, genuine castle is
// normalized and continuation moves are left intact.
struct CastleTracker {
    int king[COLOR_NB] = {-1, -1};
    bool kingMoved[COLOR_NB] = {false, false};
    bool rookMoved[64] = {};

    explicit CastleTracker(const Position& pos) {
        king[WHITE] = pos.kingSq[WHITE];
        king[BLACK] = pos.kingSq[BLACK];
    }

    void noteMove(const Position& pos, const MoveToken& m, Color mover,
                  int origTo, bool castle) {
        if (m.from == king[mover]) kingMoved[mover] = true;
        if (pos.board[m.from] == makePiece(mover, ROOK)) rookMoved[m.from] = true;
        if (castle) rookMoved[origTo] = true;
        // A move onto an original rook square (capture by either side, or any
        // piece after the rook left) means that rook is no longer available
        // for a later castle. This is what distinguishes a king capturing a
        // rook from a real 960 castle.
        if (pos.board[origTo] == makePiece(WHITE, ROOK) ||
            pos.board[origTo] == makePiece(BLACK, ROOK)) {
            rookMoved[origTo] = true;
        }
    }
};

bool normalizeCastling(const Position& pos, MoveToken& m, Color mover,
                       const CastleTracker& tracker) {
    if (m.from < 0 || m.from >= 64 || m.to < 0 || m.to >= 64) return false;
    if (m.from != tracker.king[mover] || tracker.kingMoved[mover]) return false;
    if ((m.from >> 3) != (m.to >> 3)) return false;
    if (pos.board[m.to] != makePiece(mover, ROOK)) return false;
    if (tracker.rookMoved[m.to]) return false;
    const int rank = m.to >> 3;
    const int targetFile = ((m.to & 7) > (m.from & 7)) ? 6 : 2;
    m.to = rank * 8 + targetFile;
    return true;
}

bool isPseudoLegalFirstMove(const Position& pos, const MoveToken& m) {
    MoveList list;
    pos.generateMoves(list);
    for (int i = 0; i < list.count; ++i) {
        const Move mv = list.moves[i];
        if (mv.from() != m.from || mv.to() != m.to) continue;
        if (m.promo != 0 && mv.promo() != m.promo) continue;
        if (m.promo == 0 && mv.type() == Move::PROMOTION) continue;
        return true;
    }
    return false;
}

std::vector<std::string> splitChar(const std::string& s, char sep) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t p = s.find(sep, start);
        if (p == std::string::npos) {
            parts.push_back(s.substr(start));
            break;
        }
        parts.push_back(s.substr(start, p - start));
        start = p + 1;
    }
    return parts;
}

std::vector<std::string> splitWs(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

bool parseFloat(const std::string& s, float& out) {
    std::string t = s;
    // trim
    while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.erase(t.begin());
    while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
    if (t.empty()) return false;
    char* end = nullptr;
    float v = std::strtof(t.c_str(), &end);
    if (end == t.c_str() || *end != '\0') return false;
    out = v;
    return true;
}

std::string positionKey(const std::string& fen) {
    // board + side + castling + ep; ignore move counters so duplicates and
    // resumptions group identically.
    std::string key;
    int fields = 0;
    for (size_t i = 0; i < fen.size() && fields < 4; ++i) {
        if (fen[i] == ' ') {
            ++fields;
            if (fields == 4) break;
        }
        key += fen[i];
    }
    return key;
}

uint32_t fnv1a(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

float winProbFromCp(float cp) {
    return 1.0f / (1.0f + std::exp(-cp / 400.0f));
}

struct LineRecord {
    std::string fen;
    float wdl = 0.0f;
    float cp = std::numeric_limits<float>::quiet_NaN();
    std::string best;
    std::vector<std::string> pv;
    bool havePv = false;
};

enum class ParseKind { JsonV2, LegacyProb, LegacyCp, Malformed };

// Parse one input line into LineRecord. Returns Malformed on bad syntax.
ParseKind parseLine(const std::string& line, bool cpMode, LineRecord& rec) {
    const std::vector<std::string> parts = splitChar(line, '|');
    if (parts.empty() || parts[0].empty()) return ParseKind::Malformed;
    rec = LineRecord{};
    rec.fen = parts[0];

    if (parts.size() == 2) {
        float v = 0.0f;
        if (!parseFloat(parts[1], v)) return ParseKind::Malformed;
        if (cpMode) {
            rec.cp = v;
            rec.wdl = winProbFromCp(v);
            return ParseKind::LegacyCp;
        }
        if (!std::isfinite(v)) return ParseKind::Malformed;
        if (v < 0.0f || v > 1.0f) return ParseKind::Malformed;
        rec.wdl = v;
        return ParseKind::LegacyProb;
    }

    if (parts.size() < 3) return ParseKind::Malformed;
    float wdl = 0.0f;
    if (!parseFloat(parts[1], wdl) || !std::isfinite(wdl)) {
        return ParseKind::Malformed;
    }
    if (wdl < 0.0f || wdl > 1.0f) return ParseKind::Malformed;
    rec.wdl = wdl;

    std::string cpText = parts[2];
    while (!cpText.empty() && std::isspace(static_cast<unsigned char>(cpText.front())))
        cpText.erase(cpText.begin());
    if (!cpText.empty() && cpText != "nan" && cpText != "NaN" &&
        cpText != "none") {
        float cp = 0.0f;
        if (parseFloat(cpText, cp)) rec.cp = cp;
    }
    if (parts.size() >= 4) rec.best = parts[3];
    if (parts.size() >= 5) {
        rec.pv = splitWs(parts[4]);
        rec.havePv = !rec.pv.empty();
    }
    return ParseKind::JsonV2;
}

// --------------------------------------------------------------------- stats

struct Stats {
    long long lines = 0;
    long long written = 0;
    long long skipped = 0;
    long long badFields = 0;
    long long badFen = 0;
    long long noKing = 0;
    long long noFeatures = 0;
    long long badTarget = 0;
    long long featureClip = 0;
    long long policyPresent = 0;
    long long policyAbsent = 0;
    long long pvPresent = 0;
    long long cpPresent = 0;
    long long castlingNormalized = 0;
    long long pvDupDropped = 0;
    long long pvBadToken = 0;
    long long legalCheckFailed = 0;
};

// ----------------------------------------------------------------- extraction

struct Row {
    std::array<uint16_t, kMaxActive> features{};
    uint8_t count = 0;
    bool clipped = false;
};

// Build HalfKP features for the side to move (contract 3.1).
Row extractFeatures(const Position& pos, bool& ok) {
    Row row;
    const Color us = pos.side;
    const Color them = ~us;
    const int ownKing = pos.kingSq[us];
    if (ownKing < 0) {
        ok = false;
        return row;
    }
    const bool black = (us == BLACK);
    const int bucket = black ? (ownKing ^ 56) : ownKing;
    int n = 0;
    for (int p = 0; p < 12; ++p) {
        const Color c = pieceColor(Piece(p));
        const PieceType pt = pieceType(Piece(p));
        if (c == us && pt == KING) continue;  // represented by the bucket
        U64 bb = pos.pieces(c, pt);
        while (bb) {
            const int sq = popLsb(bb);
            const int osq = black ? (sq ^ 56) : sq;
            const int rel = (c == us) ? 0 : 1;
            const int idx = (bucket * 12 + rel * 6 + int(pt)) * 64 + osq;
            if (idx >= 0 && idx < kInputs) {
                if (n < kMaxActive) {
                    row.features[size_t(n)] = uint16_t(idx);
                    ++n;
                } else {
                    row.clipped = true;
                }
            }
        }
    }
    row.count = uint8_t(n);
    ok = n > 0;
    return row;
}

// ------------------------------------------------------------------ selftest

int runSelfTest() {
    int failures = 0;
    auto check = [&](bool cond, const char* what) {
        if (!cond) {
            std::cerr << "FAIL: " << what << "\n";
            ++failures;
        }
    };

    initAll();

    MoveToken m;
    check(parseMoveToken("e2e4", m) && m.from == 12 && m.to == 28 && m.promo == 0,
          "parse e2e4");
    check(policyIndex(m) == 12 * 64 + 28, "policy e2e4");
    check(parseMoveToken("e7e8q", m) && m.promo == QUEEN, "parse promo");
    check(policyIndex(m) == 4096 + (QUEEN - 1) * 64 + 52, "policy promo q");
    check(parseMoveToken("a7a8n", m) &&
              policyIndex(m) == 4096 + (KNIGHT - 1) * 64 + 48,
          "policy promo n");
    check(!parseMoveToken("e2e9", m), "reject bad square");
    check(!parseMoveToken("e2e4k", m), "reject bad promo");
    check(!parseMoveToken("e2e4x", m), "reject trailing junk");

    // Standard castling: UCI_Chess960 encodes e1h1; normalize to e1g1.
    // Position is ~250 KB (history + NNUE accumulators), so keep it on the
    // heap: three of them on the stack overflow the default 1 MB stack.
    auto pos = std::make_unique<Position>();
    check(pos->setFen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "castle fen");
    CastleTracker tracker(*pos);
    check(parseMoveToken("e1h1", m), "parse e1h1");
    check(normalizeCastling(*pos, m, WHITE, tracker) && m.to == 6,
          "normalize e1h1 -> e1g1");
    check(policyIndex(m) == 4 * 64 + 6, "policy e1g1");
    check(parseMoveToken("e1a1", m), "parse e1a1");
    check(normalizeCastling(*pos, m, WHITE, tracker) && m.to == 2,
          "normalize e1a1 -> e1c1");
    check(policyIndex(m) == 4 * 64 + 2, "policy e1c1");
    check(isPseudoLegalFirstMove(*pos, m), "e1c1 is generated");

    // Continuation guards: a PV where the king or the rook already moved must
    // not turn a later king/rook move into a fake castle.
    CastleTracker kingMoved(*pos);
    kingMoved.kingMoved[WHITE] = true;
    check(parseMoveToken("e1h1", m), "parse e1h1 (king moved)");
    check(!normalizeCastling(*pos, m, WHITE, kingMoved),
          "no castle after the king moved");
    CastleTracker rookMoved(*pos);
    rookMoved.rookMoved[7] = true;
    check(!normalizeCastling(*pos, m, WHITE, rookMoved),
          "no castle after the h1 rook moved");

    // 960-shaped placement (rights dropped by setFen, board-based detection).
    check(pos->setFen("1k6/8/8/8/8/8/8/RK5R w KQ - 0 1"), "960 fen");
    CastleTracker tracker960(*pos);
    check(parseMoveToken("b1h1", m), "parse b1h1");
    check(normalizeCastling(*pos, m, WHITE, tracker960) && m.to == 6,
          "normalize b1h1 -> b1g1");
    check(policyIndex(m) == 1 * 64 + 6, "policy b1g1");
    check(parseMoveToken("b1a1", m), "parse b1a1");
    check(normalizeCastling(*pos, m, WHITE, tracker960) && m.to == 2,
          "normalize b1a1 -> b1c1");

    // Features: startpos has 31 pieces after removing the own king.
    check(pos->setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
          "startpos fen");
    bool ok = false;
    Row row = extractFeatures(*pos, ok);
    check(ok && row.count == 31, "startpos feature count 31");
    bool inRange = true;
    for (int i = 0; i < row.count; ++i)
        if (row.features[size_t(i)] >= kInputs) inRange = false;
    check(inRange, "feature indices in range");

    // Line parsing.
    LineRecord rec;
    check(parseLine("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|0.5320000|22|e2e4|e2e4 e7e5",
                    false, rec) == ParseKind::JsonV2,
          "parse v2 line");
    check(rec.wdl > 0.53f && rec.wdl < 0.54f && std::abs(rec.cp - 22.0f) < 1e-3,
          "v2 fields");
    check(rec.best == "e2e4" && rec.pv.size() == 2, "v2 best/pv");
    check(parseLine("8/8/8/8/8/8/8/K6k w - - 0 1|0.5", false, rec) ==
              ParseKind::LegacyProb,
          "parse legacy prob");
    check(std::isnan(rec.cp), "legacy cp is NaN");
    check(parseLine("8/8/8/8/8/8/8/K6k w - - 0 1|123", true, rec) ==
              ParseKind::LegacyCp,
          "parse legacy cp");
    check(rec.cp == 123.0f && rec.wdl > 0.5f && rec.wdl < 0.6f, "legacy cp wdl");
    check(parseLine("garbage", false, rec) == ParseKind::Malformed,
          "reject garbage");
    check(parseLine("fen|1.5", false, rec) == ParseKind::Malformed,
          "reject wdl > 1");
    check(parseLine("fen|0.5|nan||", false, rec) == ParseKind::JsonV2 &&
              std::isnan(rec.cp),
          "accept nan cp");

    if (failures == 0) {
        std::cout << "SELFTEST OK\n";
        return 0;
    }
    std::cerr << failures << " selftest failure(s)\n";
    return 1;
}

// ----------------------------------------------------------------------- meta

void writeMeta(const std::string& outDir, const Stats& st, long long lines,
               bool cpMode, const std::string& sourceStats,
               const std::string& sourceStatsPath) {
    std::ofstream meta(outDir + "/meta.json");
    if (!meta) return;
    const bool haveStats = !sourceStats.empty();
    meta << "{\n";
    meta << "  \"format\": \"halfkp-v1\",\n";
    meta << "  \"feature_version\": 1,\n";
    meta << "  \"inputs\": " << kInputs << ",\n";
    meta << "  \"max_active\": " << kMaxActive << ",\n";
    meta << "  \"policy_size\": " << kPolicySize << ",\n";
    meta << "  \"pv_len\": " << kPvLen << ",\n";
    meta << "  \"count\": " << st.written << ",\n";
    meta << "  \"lines\": " << lines << ",\n";
    meta << "  \"skipped\": " << st.skipped << ",\n";
    meta << "  \"skip_reasons\": {"
         << "\"bad_fields\": " << st.badFields
         << ", \"bad_fen\": " << st.badFen
         << ", \"no_king\": " << st.noKing
         << ", \"no_features\": " << st.noFeatures
         << ", \"bad_target\": " << st.badTarget << "},\n";
    meta << "  \"target\": \"win_probability_stm\",\n";
    meta << "  \"pov\": {\"source\": \"white\", \"output\": \"side_to_move\", "
            "\"conversion\": \"cp_stm = (stm==white ? cp : -cp); "
            "wdl = sigmoid(cp_stm/400)\"},\n";
    meta << "  \"cp_source\": \"raw_stm\",\n";
    meta << "  \"cp_present\": " << st.cpPresent << ",\n";
    meta << "  \"policy_present\": " << st.policyPresent << ",\n";
    meta << "  \"policy_absent\": " << st.policyAbsent << ",\n";
    meta << "  \"pv_present\": " << st.pvPresent << ",\n";
    meta << "  \"castling_960_normalized\": " << st.castlingNormalized << ",\n";
    meta << "  \"pv_duplicate_moves_dropped\": " << st.pvDupDropped << ",\n";
    meta << "  \"pv_bad_token\": " << st.pvBadToken << ",\n";
    meta << "  \"first_move_legality_failures\": " << st.legalCheckFailed << ",\n";
    meta << "  \"feature_clipped\": " << st.featureClip << ",\n";
    meta << "  \"policy_encoding\": \"normal from*64+to; promotion "
            "4096+(promo-2)*64+from (2=N,3=B,4=R,5=Q); absent 65535\",\n";
    meta << "  \"group_hash\": \"fnv1a32(board|stm|castling|ep)\",\n";
    meta << "  \"input_format\": \"FEN|wdl|cp|bestmove|pv (legacy FEN|target "
            "accepted)\",\n";
    meta << "  \"legacy_cp_mode\": " << (cpMode ? "true" : "false") << ",\n";
    if (haveStats) {
        meta << "  \"source_stats_file\": \""
             << jsonEscape(sourceStatsPath) << "\",\n";
        meta << "  \"source_stats\": " << sourceStats << ",\n";
    }
    meta << "  \"created_utc\": \""
         << [] {
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#ifdef _WIN32
                gmtime_s(&tm, &t);
#else
                gmtime_r(&t, &tm);
#endif
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
                return std::string(buf);
            }()
         << "\"\n";
    meta << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") return runSelfTest();
    if (argc < 3) {
        std::cerr << "usage: extract_halfkp <input.txt> <outdir> [--cp] "
                     "[--max N] [--source-stats FILE] [--quiet]\n"
                     "       extract_halfkp --selftest\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outDir = argv[2];
    bool cpMode = false;
    bool quiet = false;
    long long maxPositions = 0;
    std::string sourceStatsPath;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cp") cpMode = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--max" && i + 1 < argc) maxPositions = std::stoll(argv[++i]);
        else if (a == "--source-stats" && i + 1 < argc) sourceStatsPath = argv[++i];
    }

    // Avoid std::filesystem: the workspace path may be non-ASCII on Windows.
    std::ifstream in(inputPath, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << inputPath << "\n";
        return 1;
    }

    std::string sourceStats;
    if (!sourceStatsPath.empty()) {
        std::ifstream sf(sourceStatsPath, std::ios::binary);
        if (sf) {
            sourceStats.assign(std::istreambuf_iterator<char>(sf),
                               std::istreambuf_iterator<char>());
            size_t b = sourceStats.find_first_not_of(" \t\r\n");
            size_t e = sourceStats.find_last_not_of(" \t\r\n");
            if (b == std::string::npos || sourceStats[b] != '{' ||
                sourceStats[e] != '}') {
                std::cerr << "warning: " << sourceStatsPath
                          << " is not a JSON object; not embedding it\n";
                sourceStats.clear();
            } else {
                sourceStats = sourceStats.substr(b, e - b + 1);
            }
        } else {
            std::cerr << "warning: cannot read " << sourceStatsPath << "\n";
        }
    }

    ShardWriter<uint16_t> featOut;
    ShardWriter<uint8_t> countOut;
    ShardWriter<float> targetOut;
    ShardWriter<float> cpOut;
    ShardWriter<uint16_t> moveOut;
    ShardWriter<uint16_t> pvOut;
    ShardWriter<uint32_t> groupOut;
    if (!featOut.open(outDir + "/features.u16") ||
        !countOut.open(outDir + "/counts.u8") ||
        !targetOut.open(outDir + "/targets.f32") ||
        !cpOut.open(outDir + "/cp.f32") ||
        !moveOut.open(outDir + "/moves.u16") ||
        !pvOut.open(outDir + "/pv.u16") ||
        !groupOut.open(outDir + "/groups.u32")) {
        std::cerr << "cannot create output shards in " << outDir << "\n";
        return 1;
    }

    initAll();

    Stats st;
    std::string line;
    const auto t0 = std::chrono::steady_clock::now();
    std::array<uint16_t, kPvLen> pvRow{};
    // Position is large; allocate once and reuse (setFen commits atomically).
    auto posPtr = std::make_unique<Position>();

    while (std::getline(in, line)) {
        ++st.lines;
        if (maxPositions > 0 && st.written >= maxPositions) break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        LineRecord rec;
        const ParseKind kind = parseLine(line, cpMode, rec);
        if (kind == ParseKind::Malformed) {
            ++st.badFields;
            ++st.skipped;
            continue;
        }
        if (!std::isfinite(rec.wdl) || rec.wdl < 0.0f || rec.wdl > 1.0f) {
            ++st.badTarget;
            ++st.skipped;
            continue;
        }

        Position& pos = *posPtr;
        if (!pos.setFen(rec.fen)) {
            ++st.badFen;
            ++st.skipped;
            continue;
        }
        if (pos.kingSq[pos.side] < 0) {
            ++st.noKing;
            ++st.skipped;
            continue;
        }

        bool ok = false;
        Row row = extractFeatures(pos, ok);
        if (!ok) {
            ++st.noFeatures;
            ++st.skipped;
            continue;
        }
        if (row.clipped) ++st.featureClip;

        // ---- policy / PV
        pvRow.fill(kPolicyAbsent);
        int pvCount = 0;
        bool firstMoveUsable = false;
        std::vector<uint16_t> seen;
        CastleTracker tracker(pos);
        // The PV is authoritative; a lone bestmove is used only when the line
        // is absent. Our own downloader always writes bestmove == pv[0], so
        // this avoids encoding the same move twice.
        std::vector<std::string> tokens;
        if (!rec.pv.empty()) tokens = rec.pv;
        else if (!rec.best.empty()) tokens = splitWs(rec.best);

        for (size_t i = 0; i < tokens.size() && pvCount < kPvLen; ++i) {
            const Color mover = (i % 2 == 0) ? pos.side : ~pos.side;
            MoveToken m;
            if (!parseMoveToken(tokens[i], m)) {
                ++st.pvBadToken;
                break;
            }
            const int origTo = m.to;
            const bool castled = normalizeCastling(pos, m, mover, tracker);
            tracker.noteMove(pos, m, mover, origTo, castled);
            if (castled && m.to != origTo) ++st.castlingNormalized;
            const uint16_t idx = policyIndex(m);
            if (idx >= kPolicySize) {
                ++st.pvBadToken;
                break;
            }
            if (i == 0) {
                // Only the first move is checked for existence/legality; the
                // rest of the line is trusted (it comes from the engine).
                if (!isPseudoLegalFirstMove(pos, m)) {
                    ++st.legalCheckFailed;
                    break;
                }
                firstMoveUsable = true;
            } else {
                // Duplicate tokens indicate a broken line. Truncate instead of
                // dropping: dropping one move breaks the colour alternation
                // for every later castle/legality decision.
                if (std::find(seen.begin(), seen.end(), idx) != seen.end()) {
                    ++st.pvDupDropped;
                    break;
                }
            }
            seen.push_back(idx);
            pvRow[size_t(pvCount++)] = idx;
        }

        if (firstMoveUsable) {
            ++st.policyPresent;
            moveOut.push(pvRow[0]);
        } else {
            ++st.policyAbsent;
            pvRow.fill(kPolicyAbsent);
            pvCount = 0;
            moveOut.push(kPolicyAbsent);
        }
        if (pvCount > 0) ++st.pvPresent;
        pvOut.push(pvRow.data(), pvRow.size());
        if (std::isfinite(rec.cp)) ++st.cpPresent;

        featOut.push(row.features.data(), row.features.size());
        countOut.push(row.count);
        targetOut.push(rec.wdl);
        cpOut.push(rec.cp);
        groupOut.push(fnv1a(positionKey(rec.fen)));
        ++st.written;

        if (!quiet && st.written % 1000000 == 0) {
            const auto dt = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
            std::cerr << "extracted " << st.written << " positions ("
                      << (st.written / std::max(dt, 1e-9)) << " pos/s)\n";
        }
    }

    featOut.close();
    countOut.close();
    targetOut.close();
    cpOut.close();
    moveOut.close();
    pvOut.close();
    groupOut.close();

    writeMeta(outDir, st, st.lines, cpMode, sourceStats, sourceStatsPath);

    std::cout << "extracted " << st.written << " positions, skipped "
              << st.skipped << " -> " << outDir << "\n";
    if (st.skipped) {
        std::cout << "  reasons: bad_fields=" << st.badFields
                  << " bad_fen=" << st.badFen
                  << " no_king=" << st.noKing
                  << " no_features=" << st.noFeatures
                  << " bad_target=" << st.badTarget << "\n";
    }
    std::cout << "  policy_present=" << st.policyPresent
              << " pv_present=" << st.pvPresent
              << " cp_present=" << st.cpPresent
              << " castling_960=" << st.castlingNormalized << "\n";
    return st.written > 0 ? 0 : 1;
}
