#include "engine.hpp"
#include "nnue.hpp"

#include <cmath>
#include <cstring>

namespace chess {

int classicalEvaluate(const Position& pos);

namespace {

// Material values, tuned along PeSTO-like scales.
constexpr int MatMg[PIECE_TYPE_NB] = {82, 337, 365, 477, 1025, 0};
constexpr int MatEg[PIECE_TYPE_NB] = {94, 281, 297, 512, 936, 0};
constexpr int PhaseInc[PIECE_TYPE_NB] = {0, 1, 1, 2, 4, 0};
constexpr int MaxPhase = 24;

// Piece-square tables, a1-first, white point of view.
constexpr int PawnMg[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};
constexpr int PawnEg[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     5,  5,  5,  5,  5,  5,  5,  5,
    10, 10, 10, 10, 10, 10, 10, 10,
    25, 25, 25, 25, 25, 25, 25, 25,
    50, 50, 50, 50, 50, 50, 50, 50,
    80, 80, 80, 80, 80, 80, 80, 80,
     0,  0,  0,  0,  0,  0,  0,  0
};
constexpr int KnightMg[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
constexpr int KnightEg[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
constexpr int BishopMg[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
constexpr int BishopEg[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5, 10, 15, 15, 10,  5,-10,
   -10,  5, 10, 15, 15, 10,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
constexpr int RookMg[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
constexpr int RookEg[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};
constexpr int QueenMg[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
constexpr int QueenEg[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
    -5,  0, 10, 15, 15, 10,  0, -5,
    -5,  0, 10, 15, 15, 10,  0, -5,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
constexpr int KingMg[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -40,-40,-40,-50,-50,-40,-40,-40,
   -50,-50,-50,-60,-60,-50,-50,-50
};
constexpr int KingEg[64] = {
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50
};

const int* PstMg[PIECE_TYPE_NB] = {
    PawnMg, KnightMg, BishopMg, RookMg, QueenMg, KingMg
};
const int* PstEg[PIECE_TYPE_NB] = {
    PawnEg, KnightEg, BishopEg, RookEg, QueenEg, KingEg
};

int passedBonusMg[8] = {0, 5, 10, 20, 35, 60, 100, 0};
int passedBonusEg[8] = {0, 10, 20, 35, 60, 100, 160, 0};
U64 passedMask[2][64];
bool evalInitialized = false;

void initPassedMask() {
    for (int c = 0; c < 2; ++c) {
        for (int sq = 0; sq < 64; ++sq) {
            U64 m = 0;
            int f = fileOf(sq), r = rankOf(sq);
            int dr = c == WHITE ? 1 : -1;
            for (int rr = r + dr; rr >= 0 && rr < 8; rr += dr) {
                for (int df = -1; df <= 1; ++df) {
                    int nf = f + df;
                    if (nf >= 0 && nf < 8) m |= bit(rr * 8 + nf);
                }
            }
            passedMask[c][sq] = m;
        }
    }
}

inline int pstIndex(Color c, int sq) {
    return c == WHITE ? sq : (sq ^ 56);
}

inline int relRank(Color c, int sq) {
    return c == WHITE ? rankOf(sq) : 7 - rankOf(sq);
}

int pawnShield(const Position& pos, Color c) {
    int ks = pos.kingSq[c];
    if (ks < 0) return 0;
    int kf = fileOf(ks), kr = rankOf(ks);
    int dir = c == WHITE ? 1 : -1;
    int score = 0;
    for (int df = -1; df <= 1; ++df) {
        int f = kf + df;
        if (f < 0 || f >= 8) continue;
        bool shield1 = false, shield2 = false;
        int r1 = kr + dir;
        int r2 = kr + 2 * dir;
        if (r1 >= 0 && r1 < 8 && pos.board[r1 * 8 + f] == makePiece(c, PAWN))
            shield1 = true;
        if (r2 >= 0 && r2 < 8 && pos.board[r2 * 8 + f] == makePiece(c, PAWN))
            shield2 = true;
        if (shield1) score += 14;
        else if (shield2) score += 7;
        else score -= 10;
    }
    // Open file next to the king is dangerous.
    U64 ownPawns = pos.pieces(c, PAWN);
    U64 fileMask = 0x0101010101010101ULL << kf;
    if (!(ownPawns & fileMask)) score -= 18;
    return score;
}

} // namespace

void initEvaluation() {
    if (evalInitialized) return;
    initPassedMask();
    NNUE::init();
    evalInitialized = true;
}

int classicalEvaluate(const Position& pos) {
    int mg = 0, eg = 0;
    int phase = 0;
    int danger[COLOR_NB] = {0, 0};
    U64 pawns[COLOR_NB] = {pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN)};
    U64 allPawns = pawns[WHITE] | pawns[BLACK];
    U64 kingRingBB[COLOR_NB] = {0, 0};
    for (int c = 0; c < 2; ++c) {
        if (pos.kingSq[c] >= 0) {
            kingRingBB[c] = kingAttacks(pos.kingSq[c]) | bit(pos.kingSq[c]);
        }
    }

    for (int c = 0; c < 2; ++c) {
        Color color = Color(c);
        Color enemy = ~color;
        int sign = c == WHITE ? 1 : -1;
        U64 own = pos.byColor[c];

        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
            U64 bb = pos.byColor[c] & pos.byType[pt];
            while (bb) {
                int sq = popLsb(bb);
                int idx = pstIndex(color, sq);
                mg += sign * (MatMg[pt] + PstMg[pt][idx]);
                eg += sign * (MatEg[pt] + PstEg[pt][idx]);
                if (pt != PAWN && pt != KING) phase += PhaseInc[pt];

                if (pt != PAWN && pt != KING) {
                    U64 att = 0;
                    switch (pt) {
                        case KNIGHT: att = knightAttacks(sq); break;
                        case BISHOP: att = bishopAttacks(sq, pos.occ); break;
                        case ROOK: att = rookAttacks(sq, pos.occ); break;
                        case QUEEN:
                            att = bishopAttacks(sq, pos.occ) |
                                  rookAttacks(sq, pos.occ);
                            break;
                        default: break;
                    }
                    att &= ~own;
                    int cnt = popcount(att);
                    switch (pt) {
                        case KNIGHT:
                            mg += sign * 4 * cnt;
                            eg += sign * 3 * cnt;
                            break;
                        case BISHOP:
                            mg += sign * 3 * cnt;
                            eg += sign * 4 * cnt;
                            break;
                        case ROOK:
                            mg += sign * 2 * cnt;
                            eg += sign * 3 * cnt;
                            break;
                        case QUEEN:
                            mg += sign * 1 * cnt;
                            eg += sign * 1 * cnt;
                            break;
                        default: break;
                    }
                    if (pos.kingSq[enemy] >= 0 &&
                        (att & kingRingBB[enemy])) {
                        int w = (pt == QUEEN) ? 5 : (pt == ROOK) ? 3 : 2;
                        danger[enemy] += w;
                    }
                }
            }
        }

        // Bishop pair
        if (popcount(pos.pieces(color, BISHOP)) >= 2) {
            mg += sign * 30;
            eg += sign * 45;
        }

        // Pawn structure
        int fileCount[8] = {};
        U64 p = pawns[c];
        while (p) {
            int sq = popLsb(p);
            fileCount[fileOf(sq)]++;
        }
        for (int f = 0; f < 8; ++f) {
            if (!fileCount[f]) continue;
            if (fileCount[f] > 1) {
                mg -= sign * 12 * (fileCount[f] - 1);
                eg -= sign * 20 * (fileCount[f] - 1);
            }
            bool isolated = (f == 0 || fileCount[f - 1] == 0) &&
                            (f == 7 || fileCount[f + 1] == 0);
            if (isolated) {
                mg -= sign * 15 * fileCount[f];
                eg -= sign * 12 * fileCount[f];
            }
        }

        p = pawns[c];
        while (p) {
            int sq = popLsb(p);
            int rr = relRank(color, sq);
            U64 front = passedMask[c][sq];
            if (!(front & pawns[enemy])) {
                mg += sign * passedBonusMg[rr];
                eg += sign * passedBonusEg[rr];
            }
            // Protected passer / chain bonus
            U64 defenders = pawnAttacks(~color, sq) & pawns[c];
            if (defenders) {
                mg += sign * 5;
                eg += sign * 8;
            }
        }

        // Rooks
        U64 rooks = pos.pieces(color, ROOK);
        while (rooks) {
            int sq = popLsb(rooks);
            U64 fileMask = 0x0101010101010101ULL << fileOf(sq);
            if (!(allPawns & fileMask)) {
                mg += sign * 25;
                eg += sign * 15;
            } else if (!(pawns[c] & fileMask)) {
                mg += sign * 12;
                eg += sign * 8;
            }
            if (relRank(color, sq) == 6) {
                mg += sign * 15;
                eg += sign * 25;
            }
        }

        // Knight outposts
        U64 knights = pos.pieces(color, KNIGHT);
        while (knights) {
            int sq = popLsb(knights);
            int rr = relRank(color, sq);
            if (rr >= 3 && rr <= 5) {
                U64 ownSupport = pawnAttacks(~color, sq) & pawns[c];
                U64 enemyControl = pawnAttacks(color, sq) & pawns[enemy];
                if (ownSupport && !enemyControl) {
                    mg += sign * 18;
                    eg += sign * 12;
                }
            }
        }

        // King safety is applied after both colors have contributed attackers,
        // otherwise white would ignore black's attacking pieces (and vice versa).
    }

    for (int c = 0; c < 2; ++c) {
        Color color = Color(c);
        int sign = c == WHITE ? 1 : -1;
        if (pos.kingSq[c] >= 0) {
            int shield = pawnShield(pos, color);
            int safety = shield - danger[c] * 4;
            mg += sign * safety;
        }
    }

    // Space: count safe squares in the opponent's half.
    for (int c = 0; c < 2; ++c) {
        Color color = Color(c);
        int sign = c == WHITE ? 1 : -1;
        U64 central = 0;
        int r0 = c == WHITE ? 2 : 5;
        int r1 = c == WHITE ? 4 : 3;
        for (int r = std::min(r0, r1); r <= std::max(r0, r1); ++r)
            for (int f = 2; f <= 5; ++f) central |= bit(r * 8 + f);
        U64 safe = central & ~pos.byColor[~color];
        mg += sign * 2 * popcount(safe & pos.byColor[c]);
    }

    if (phase > MaxPhase) phase = MaxPhase;
    int score = (mg * phase + eg * (MaxPhase - phase)) / MaxPhase;

    // Small mop-up term for simplified endgames.
    if (phase <= 6) {
        int material = 0;
        for (int pt = PAWN; pt < KING; ++pt)
            material += (popcount(pos.pieces(WHITE, PieceType(pt))) -
                         popcount(pos.pieces(BLACK, PieceType(pt)))) * MatEg[pt];
        if (std::abs(material) > 300) {
            Color strong = material > 0 ? WHITE : BLACK;
            int weakKing = pos.kingSq[~strong];
            int strongKing = pos.kingSq[strong];
            if (weakKing >= 0 && strongKing >= 0) {
                int edge = std::abs(fileOf(weakKing) * 2 - 7) +
                           std::abs(rankOf(weakKing) * 2 - 7);
                int kingDist = std::abs(fileOf(weakKing) - fileOf(strongKing)) +
                               std::abs(rankOf(weakKing) - rankOf(strongKing));
                int mop = edge * 6 + (14 - kingDist) * 3;
                score += (strong == WHITE ? 1 : -1) * mop;
            }
        }
    }

    int result = (pos.side == WHITE) ? score : -score;
    result += 10; // tempo
    result = std::max(-MATE_IN_MAX + 1, std::min(MATE_IN_MAX - 1, result));
    return result;
}

// ---------------------------------------------------------------------------
// Neural evaluation: HalfKP-like features, sparse 768 -> 64 -> 1 network.
// The network learns the residual between a shallow search score and the
// classical evaluation. Weights live in nnue_weights.cpp and can be
// regenerated by the included trainer.
// ---------------------------------------------------------------------------
namespace NNUE {

int featureIndices(const Position& pos, int* out) {
    int n = 0;
    const Color us = pos.side;
    for (int p = 0; p < 12; ++p) {
        U64 bb = pos.pieces(pieceColor(Piece(p)), pieceType(Piece(p)));
        while (bb) {
            int sq = popLsb(bb);
            int relColor = (pieceColor(Piece(p)) == us) ? 0 : 1;
            int osq = (us == WHITE) ? sq : (sq ^ 56);
            out[n++] = (relColor * 6 + int(pieceType(Piece(p)))) * 64 + osq;
        }
    }
    return n;
}

int correction(const Position& pos) {
    if (!trained || !enabled) return 0;
    int feats[64];
    int n = featureIndices(pos, feats);

    float h[HIDDEN];
    for (int j = 0; j < HIDDEN; ++j) h[j] = B1[j];
    for (int i = 0; i < n; ++i) {
        const float* row = W1[feats[i]];
        for (int j = 0; j < HIDDEN; ++j) h[j] += row[j];
    }
    for (int j = 0; j < HIDDEN; ++j)
        if (h[j] < 0.0f) h[j] = 0.0f;

    float out = B2;
    for (int j = 0; j < HIDDEN; ++j) out += W2[j] * h[j];
    int value = int(std::lround(out * 100.0f));
    return std::clamp(value, -1500, 1500);
}

void setEnabled(bool v) { enabled = v; }
bool isEnabled() { return enabled; }
void setWeight(int w) {
    weight = std::clamp(w, 0, 200);
    // Keep the v1 network's blend knob in the same 0..100 range.  The UCI
    // option still advertises 0..200 for compatibility with old scripts.
    NNUE::setNetWeight(std::min(weight, 100));
}
int getWeight() { return weight; }
bool isAvailable() { return trained; }

} // namespace NNUE

int evaluate(const Position& pos) {
    const auto clampScore = [](int v) {
        return std::clamp(v, -MATE_IN_MAX + 1, MATE_IN_MAX - 1);
    };

    if (NNUE::isEnabled() && NNUE::backendIsCpu()) {
        // Preferred path: a v1 .nnue file loaded through EvalFile.
        if (NNUE::fileLoaded()) {
            int net = NNUE::evaluateNetwork(pos);
            const int material = NNUE::materialCp(pos);
            const int materialWeight = NNUE::getNetMaterial();
            if (materialWeight > 0)
                net += material * materialWeight / 100;
            const int clip = NNUE::getNetClip();
            if (clip > 0) net = std::clamp(net, -clip, clip);
            const int w = NNUE::getNetWeight();
            if (w <= 0) return clampScore(classicalEvaluate(pos));
            if (w >= 100) return clampScore(net);  // direct eval by default
            const int base = classicalEvaluate(pos);
            const long long blended =
                (static_cast<long long>(net) * w +
                 static_cast<long long>(base) * (100 - w)) / 100;
            return clampScore(int(blended));
        }

        // Compatibility path: the old compiled-in 768->64 residual network.
        // It is only reachable when no v1 file is loaded, so the new path is
        // never silently blended with the old one.
        if (NNUE::legacyEnabled() && NNUE::trained && NNUE::weight > 0) {
            int base = classicalEvaluate(pos);
            int net = NNUE::correction(pos);
            if (NNUE::clip > 0)
                net = std::clamp(net, -NNUE::clip, NNUE::clip);
            if (NNUE::direct) {
                base = (net * NNUE::weight + base * (100 - NNUE::weight)) / 100;
            } else {
                base += net * NNUE::weight / 100;
            }
            return clampScore(base);
        }
    }
    return clampScore(classicalEvaluate(pos));
}

} // namespace chess
