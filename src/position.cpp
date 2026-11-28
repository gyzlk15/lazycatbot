#include "engine.hpp"
#include "nnue.hpp"

#include <cctype>
#include <sstream>

namespace chess {

Position::Position() { clear(); }

void Position::clear() {
    for (auto& p : board) p = NO_PIECE;
    for (auto& b : byType) b = 0;
    for (auto& b : byColor) b = 0;
    occ = 0;
    side = WHITE;
    castling = 0;
    ep = -1;
    halfMove = 0;
    fullMove = 1;
    key = 0;
    kingSq[WHITE] = kingSq[BLACK] = -1;
    histPly = 0;
    std::memset(keyHist, 0, sizeof(keyHist));
    std::memset(isNullHist_, 0, sizeof(isNullHist_));
    nnueAccValid = false;
    nnueAccGeneration = 0;
    nnueAccQuant = 0;
}

void Position::putPiece(Piece p, int sq) {
    board[sq] = p;
    PieceType pt = pieceType(p);
    Color c = pieceColor(p);
    byType[pt] |= bit(sq);
    byColor[c] |= bit(sq);
    occ |= bit(sq);
    if (pt == KING) kingSq[c] = sq;
}

void Position::removePiece(int sq) {
    Piece p = board[sq];
    if (p == NO_PIECE) return;
    PieceType pt = pieceType(p);
    Color c = pieceColor(p);
    byType[pt] &= ~bit(sq);
    byColor[c] &= ~bit(sq);
    occ &= ~bit(sq);
    board[sq] = NO_PIECE;
    if (pt == KING) kingSq[c] = -1;
}

void Position::movePiece(int from, int to) {
    Piece p = board[from];
    removePiece(from);
    putPiece(p, to);
}

void Position::setStartpos() {
    setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// Parse into an already-cleared position (the temporary in setFen()). setFen()
// commits only after the whole FEN validated, so a malformed FEN can never
// leave a half-initialized position behind.
bool Position::parseFenInto(const std::string& fen) {
    std::istringstream ss(fen);
    std::string boardField, sideField, castleField, epField;
    if (!(ss >> boardField >> sideField)) return false;
    ss >> castleField >> epField;
    std::string halfStr, fullStr;
    ss >> halfStr >> fullStr;

    int rank = 7, file = 0;
    for (char ch : boardField) {
        if (ch == '/') {
            if (file != 8) return false;
            --rank;
            file = 0;
            if (rank < 0) return false;
            continue;
        }
        if (ch >= '1' && ch <= '8') {
            file += ch - '0';
            if (file > 8) return false;
            continue;
        }
        Piece p = NO_PIECE;
        switch (ch) {
            case 'P': p = W_PAWN; break;
            case 'N': p = W_KNIGHT; break;
            case 'B': p = W_BISHOP; break;
            case 'R': p = W_ROOK; break;
            case 'Q': p = W_QUEEN; break;
            case 'K': p = W_KING; break;
            case 'p': p = B_PAWN; break;
            case 'n': p = B_KNIGHT; break;
            case 'b': p = B_BISHOP; break;
            case 'r': p = B_ROOK; break;
            case 'q': p = B_QUEEN; break;
            case 'k': p = B_KING; break;
            default: return false;
        }
        if (rank < 0 || file >= 8) return false;
        putPiece(p, rank * 8 + file);
        ++file;
    }
    if (rank != 0 || file != 8) return false;

    // Structural validation.
    if (popcount(occ) > 32) return false;
    if (popcount(byColor[WHITE]) > 16 || popcount(byColor[BLACK]) > 16)
        return false;
    if (popcount(pieces(WHITE, KING)) != 1 ||
        popcount(pieces(BLACK, KING)) != 1)
        return false;
    if (popcount(pieces(WHITE, PAWN)) > 8 ||
        popcount(pieces(BLACK, PAWN)) > 8)
        return false;
    {
        // A pawn can never legally stand on its first or last rank.
        U64 pawns = pieces(PAWN);
        while (pawns) {
            const int sq = popLsb(pawns);
            if (rankOf(sq) == 0 || rankOf(sq) == 7) return false;
        }
    }
    if (kingAttacks(kingSq[WHITE]) & bit(kingSq[BLACK])) return false;

    if (sideField != "w" && sideField != "b") return false;
    side = (sideField == "b") ? BLACK : WHITE;

    castling = 0;
    if (castleField != "-") {
        for (char ch : castleField) {
            switch (ch) {
                case 'K': castling |= WHITE_OO; break;
                case 'Q': castling |= WHITE_OOO; break;
                case 'k': castling |= BLACK_OO; break;
                case 'q': castling |= BLACK_OOO; break;
                default: return false;
            }
        }
    }
    // Drop rights that are inconsistent with the actual king/rook placement.
    if (board[4] != W_KING || board[7] != W_ROOK) castling &= ~WHITE_OO;
    if (board[4] != W_KING || board[0] != W_ROOK) castling &= ~WHITE_OOO;
    if (board[60] != B_KING || board[63] != B_ROOK) castling &= ~BLACK_OO;
    if (board[60] != B_KING || board[56] != B_ROOK) castling &= ~BLACK_OOO;

    // En passant must be a real, capturable ep target, otherwise ignore it.
    ep = -1;
    if (epField.size() >= 2 && epField[0] >= 'a' && epField[0] <= 'h' &&
        epField[1] >= '1' && epField[1] <= '8') {
        int candidate = (epField[1] - '1') * 8 + (epField[0] - 'a');
        int capSq = -1;
        bool rankOk = false;
        if (side == BLACK && rankOf(candidate) == 2) {
            capSq = candidate + 8;
            rankOk = true;
        } else if (side == WHITE && rankOf(candidate) == 5) {
            capSq = candidate - 8;
            rankOk = true;
        }
        if (rankOk && board[candidate] == NO_PIECE && capSq >= 0 &&
            board[capSq] == makePiece(~side, PAWN) &&
            (pawnAttacks(~side, candidate) & pieces(side, PAWN)))
            ep = candidate;
    }

    try {
        halfMove = halfStr.empty() ? 0 : std::max(0, std::stoi(halfStr));
    } catch (...) {
        halfMove = 0;
    }
    try {
        fullMove = fullStr.empty() ? 1 : std::max(1, std::stoi(fullStr));
    } catch (...) {
        fullMove = 1;
    }
    halfMove = std::min(halfMove, 1000);
    fullMove = std::min(fullMove, 1000000);

    // The side that just moved cannot have left its own king in check.
    if (isSquareAttacked(kingSq[~side], side)) return false;

    key = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (board[sq] != NO_PIECE)
            key ^= Zobrist::psq[board[sq]][sq];
    }
    key ^= Zobrist::castling[castling];
    if (epHashable()) key ^= Zobrist::epFile[fileOf(ep)];
    if (side == BLACK) key ^= Zobrist::side;

    histPly = 0;
    keyHist[0] = key;
    isNullHist_[0] = false;
    return true;
}

bool Position::setFen(const std::string& fen) {
    Position tmp;
    if (!tmp.parseFenInto(fen)) return false;

    // Commit only after the whole FEN has been validated. Copy the game state,
    // not the (large) move/undo history: a new FEN starts a fresh game.
    std::memcpy(board, tmp.board, sizeof(board));
    std::memcpy(byType, tmp.byType, sizeof(byType));
    std::memcpy(byColor, tmp.byColor, sizeof(byColor));
    occ = tmp.occ;
    side = tmp.side;
    castling = tmp.castling;
    ep = tmp.ep;
    halfMove = tmp.halfMove;
    fullMove = tmp.fullMove;
    key = tmp.key;
    kingSq[WHITE] = tmp.kingSq[WHITE];
    kingSq[BLACK] = tmp.kingSq[BLACK];
    histPly = 0;
    keyHist[0] = key;
    isNullHist_[0] = false;
    NNUE::refreshAccumulator(*this);
    return true;
}

std::string Position::fen() const {
    std::string out;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[r * 8 + f];
            if (p == NO_PIECE) {
                ++empty;
            } else {
                if (empty) { out += char('0' + empty); empty = 0; }
                out += pieceName(p);
            }
        }
        if (empty) out += char('0' + empty);
        if (r) out += '/';
    }
    out += side == WHITE ? " w " : " b ";
    if (!castling) out += "-";
    if (castling & WHITE_OO) out += 'K';
    if (castling & WHITE_OOO) out += 'Q';
    if (castling & BLACK_OO) out += 'k';
    if (castling & BLACK_OOO) out += 'q';
    out += ' ';
    if (ep >= 0 && epHashable()) {
        out += char('a' + fileOf(ep));
        out += char('1' + rankOf(ep));
    } else {
        out += '-';
    }
    out += ' ' + std::to_string(halfMove) + ' ' + std::to_string(fullMove);
    return out;
}

std::string Position::ascii() const {
    std::string out;
    for (int r = 7; r >= 0; --r) {
        out += std::to_string(r + 1);
        out += " ";
        for (int f = 0; f < 8; ++f)
            out += std::string(pieceName(board[r * 8 + f])) + " ";
        out += '\n';
    }
    out += "  a b c d e f g h\n";
    out += "FEN: " + fen() + "\n";
    return out;
}

// An en-passant right is only part of the position (and therefore only
// belongs in the Zobrist key) if at least one *legal* ep capture exists.
// Pseudo-legal pawn attacks are not enough: a pinned ep pawn would make the
// same board hash differently on otherwise identical repetitions.
bool Position::epHashable() const {
    if (ep < 0 || ep >= 64) return false;
    const Color us = side;
    const Color them = ~us;
    if (rankOf(ep) != (us == WHITE ? 5 : 2)) return false;
    if (board[ep] != NO_PIECE) return false;
    const int capSq = ep + (us == WHITE ? -8 : 8);
    if (!validSq(capSq) || board[capSq] != makePiece(them, PAWN)) return false;

    U64 candidates = pawnAttacks(them, ep) & pieces(us, PAWN);
    if (!candidates) return false;

    const int ksq = kingSq[us];
    if (ksq < 0) return false;

    // Pieces of the side not to move, with the captured pawn removed.
    const U64 enemyPawns = pieces(them, PAWN) & ~bit(capSq);
    const U64 enemyKnights = pieces(them, KNIGHT);
    const U64 enemyBishopsQueens = pieces(them, BISHOP) | pieces(them, QUEEN);
    const U64 enemyRooksQueens = pieces(them, ROOK) | pieces(them, QUEEN);
    const U64 enemyKing = pieces(them, KING);

    while (candidates) {
        const int from = popLsb(candidates);
        // Simulate the ep capture: move our pawn, remove the captured pawn.
        const U64 occAfter = (occ & ~bit(from) & ~bit(capSq)) | bit(ep);
        const bool kingAttacked =
            (pawnAttacks(us, ksq) & enemyPawns) ||
            (knightAttacks(ksq) & enemyKnights) ||
            (kingAttacks(ksq) & enemyKing) ||
            (bishopAttacks(ksq, occAfter) & enemyBishopsQueens) ||
            (rookAttacks(ksq, occAfter) & enemyRooksQueens);
        if (!kingAttacked) return true;
    }
    return false;
}

bool Position::isSquareAttacked(int sq, Color by) const {
    if (!validSq(sq)) return false;
    if (pawnAttacks(~by, sq) & pieces(by, PAWN)) return true;
    if (knightAttacks(sq) & pieces(by, KNIGHT)) return true;
    if (kingAttacks(sq) & pieces(by, KING)) return true;
    U64 bishopsQueens = pieces(by, BISHOP) | pieces(by, QUEEN);
    if (bishopAttacks(sq, occ) & bishopsQueens) return true;
    U64 rooksQueens = pieces(by, ROOK) | pieces(by, QUEEN);
    if (rookAttacks(sq, occ) & rooksQueens) return true;
    return false;
}

U64 Position::attackersTo(int sq, U64 occLocal) const {
    U64 res = 0;
    res |= pawnAttacks(BLACK, sq) & pieces(WHITE, PAWN);
    res |= pawnAttacks(WHITE, sq) & pieces(BLACK, PAWN);
    res |= knightAttacks(sq) & pieces(KNIGHT);
    res |= kingAttacks(sq) & pieces(KING);
    res |= bishopAttacks(sq, occLocal) &
           (pieces(BISHOP) | pieces(QUEEN));
    res |= rookAttacks(sq, occLocal) & (pieces(ROOK) | pieces(QUEEN));
    return res;
}

bool Position::makeMove(Move m) {
    if (histPly >= MAX_HIST - 1) return false;
    const int from = m.from();
    const int to = m.to();
    if (!validSq(from) || !validSq(to)) return false;
    const Piece pc = board[from];
    if (pc == NO_PIECE) return false;
    const Color us = side;
    const Color them = ~us;
    const PieceType pt = pieceType(pc);
    if (pieceColor(pc) != us) return false;
    if (board[to] != NO_PIECE && pieceColor(board[to]) == us) return false;

    if (m.type() == Move::EN_PASSANT) {
        int capSq = to + (us == WHITE ? -8 : 8);
        if (pt != PAWN || board[to] != NO_PIECE || !validSq(capSq) ||
            board[capSq] != makePiece(them, PAWN))
            return false;
        if (rankOf(to) != (us == WHITE ? 5 : 2)) return false;
    }
    if (m.type() == Move::PROMOTION) {
        if (pt != PAWN || m.promo() < KNIGHT || m.promo() > QUEEN)
            return false;
        if (rankOf(to) != (us == WHITE ? 7 : 0)) return false;
    }
    if (m.type() == Move::DOUBLE_PUSH) {
        if (pt != PAWN || to - from != (us == WHITE ? 16 : -16))
            return false;
    }
    if (m.type() == Move::CASTLE) {
        if (pt != KING || (to != from + 2 && to != from - 2)) return false;
        if (board[to] != NO_PIECE) return false;
        const int rookFrom =
            us == WHITE ? (to > from ? 7 : 0) : (to > from ? 63 : 56);
        if (board[rookFrom] != makePiece(us, ROOK)) return false;
        int right = us == WHITE ? (to > from ? WHITE_OO : WHITE_OOO)
                                : (to > from ? BLACK_OO : BLACK_OOO);
        if (!(castling & right)) return false;
    }

    Undo& u = undoStack[histPly];
    u.key = key;
    u.castling = castling;
    u.ep = ep;
    u.halfMove = halfMove;
    u.fullMove = fullMove;
    u.captured = NO_PIECE;
    u.capturedSq = -1;
    u.accUpdated = false;

    key ^= Zobrist::castling[castling];
    if (epHashable()) key ^= Zobrist::epFile[fileOf(ep)];

    // Captures
    if (m.type() == Move::EN_PASSANT) {
        int capSq = to + (us == WHITE ? -8 : 8);
        Piece cap = board[capSq];
        u.captured = cap;
        u.capturedSq = capSq;
        key ^= Zobrist::psq[cap][capSq];
        removePiece(capSq);
    } else if (board[to] != NO_PIECE) {
        Piece cap = board[to];
        u.captured = cap;
        u.capturedSq = to;
        key ^= Zobrist::psq[cap][to];
        removePiece(to);
    }

    // Moving piece, with promotion
    key ^= Zobrist::psq[pc][from];
    removePiece(from);
    Piece placed = pc;
    if (m.type() == Move::PROMOTION) {
        placed = makePiece(us, PieceType(m.promo()));
    }
    putPiece(placed, to);
    key ^= Zobrist::psq[placed][to];

    // Castling rook
    if (m.type() == Move::CASTLE) {
        int rf = 0, rt = 0;
        if (to > from) {
            rf = us == WHITE ? 7 : 63;
            rt = rf - 2;
        } else {
            rf = us == WHITE ? 0 : 56;
            rt = rf + 3;
        }
        Piece rook = board[rf];
        key ^= Zobrist::psq[rook][rf];
        removePiece(rf);
        putPiece(rook, rt);
        key ^= Zobrist::psq[rook][rt];
    }

    // Castling rights
    if (pt == KING) {
        castling &= us == WHITE ? ~(WHITE_OO | WHITE_OOO)
                                : ~(BLACK_OO | BLACK_OOO);
    }
    auto clearRookRight = [&](int sq) {
        if (sq == 7) castling &= ~WHITE_OO;
        else if (sq == 0) castling &= ~WHITE_OOO;
        else if (sq == 63) castling &= ~BLACK_OO;
        else if (sq == 56) castling &= ~BLACK_OOO;
    };
    clearRookRight(from);
    clearRookRight(to);

    // En passant square
    ep = -1;
    if (pt == PAWN && m.type() == Move::DOUBLE_PUSH)
        ep = (from + to) / 2;

    // Clocks
    if (pt == PAWN || u.captured != NO_PIECE) halfMove = 0;
    else ++halfMove;
    if (us == BLACK) ++fullMove;

    side = them;
    key ^= Zobrist::side;
    key ^= Zobrist::castling[castling];
    if (epHashable()) key ^= Zobrist::epFile[fileOf(ep)];

    ++histPly;
    keyHist[histPly] = key;
    isNullHist_[histPly] = false;

    // Legality: did the move leave our king in check?
    if (isSquareAttacked(kingSq[us], them)) {
        unmakeMove(m);
        return false;
    }

    // Board is final and the move is legal: update both accumulators.
    u.accUpdated = NNUE::accumulatorAfterMake(*this, m, u, us);
    return true;
}

void Position::unmakeMove(Move m) {
    --histPly;
    Undo& u = undoStack[histPly];

    const Color us = ~side; // side is the side that was to move after the move
    const int from = m.from();
    const int to = m.to();

    Piece pc = board[to];
    removePiece(to);
    Piece original = (m.type() == Move::PROMOTION) ? makePiece(us, PAWN) : pc;
    putPiece(original, from);

    if (m.type() == Move::CASTLE) {
        int rf = 0, rt = 0;
        if (to > from) {
            rf = us == WHITE ? 7 : 63;
            rt = rf - 2;
        } else {
            rf = us == WHITE ? 0 : 56;
            rt = rf + 3;
        }
        Piece rook = board[rt];
        removePiece(rt);
        putPiece(rook, rf);
    }

    if (u.captured != NO_PIECE) putPiece(u.captured, u.capturedSq);

    side = us;
    castling = u.castling;
    ep = u.ep;
    halfMove = u.halfMove;
    fullMove = u.fullMove;
    key = u.key;

    if (u.accUpdated) NNUE::accumulatorAfterUnmake(*this, m, u, us);
}

bool Position::makeNullMove() {
    if (histPly >= MAX_HIST - 1) return false;
    Undo& u = undoStack[histPly];
    u.key = key;
    u.castling = castling;
    u.ep = ep;
    u.halfMove = halfMove;
    u.fullMove = fullMove;
    u.captured = NO_PIECE;
    u.capturedSq = -1;
    u.accUpdated = false;

    if (epHashable()) key ^= Zobrist::epFile[fileOf(ep)];
    ep = -1;
    key ^= Zobrist::side;
    side = ~side;
    // A null move is not a real game move.  Reset the halfmove clock so the
    // repetition scan cannot cross the null node and treat a null-move line
    // as a real repetition.  unmakeNullMove() restores the real clock.
    halfMove = 0;

    ++histPly;
    keyHist[histPly] = key;
    isNullHist_[histPly] = true;
    return true;
}

void Position::unmakeNullMove() {
    --histPly;
    Undo& u = undoStack[histPly];
    side = ~side;
    key = u.key;
    castling = u.castling;
    ep = u.ep;
    halfMove = u.halfMove;
    fullMove = u.fullMove;
}

bool Position::isRepetition() const {
    int limit = std::min(halfMove, histPly);
    for (int i = 2; i <= limit; i += 2) {
        if (isNullHist_[histPly - i]) continue;
        if (keyHist[histPly - i] == key) return true;
    }
    return false;
}

bool Position::isInsufficientMaterial() const {
    if (byType[PAWN] || byType[ROOK] || byType[QUEEN]) return false;
    int knights = popcount(byType[KNIGHT]);
    int bishops = popcount(byType[BISHOP]);
    if (knights + bishops <= 1) return true;

    // With only kings and bishops, if every bishop is on the same square
    // color, checkmate cannot be constructed: dead position.
    if (knights == 0 && bishops >= 2) {
        U64 bb = byType[BISHOP];
        int color = -1;
        bool sameColor = true;
        while (bb) {
            int sq = popLsb(bb);
            int c = (fileOf(sq) + rankOf(sq)) & 1;
            if (color < 0)
                color = c;
            else if (color != c)
                sameColor = false;
        }
        if (sameColor) return true;
    }
    return false;
}

bool Position::hasLegalMove() const {
    MoveList list;
    generateMoves(list);
    Position* self = const_cast<Position*>(this);
    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        if (self->makeMove(m)) {
            self->unmakeMove(m);
            return true;
        }
    }
    return false;
}

void Position::generateMoves(MoveList& list) const {
    generateMovesImpl(list, false);
}

void Position::generateCaptures(MoveList& list) const {
    generateMovesImpl(list, true);
}

void Position::generateMovesImpl(MoveList& list, bool capturesOnly) const {
    list.clear();
    const Color us = side;
    const Color them = ~us;
    const U64 own = byColor[us];
    const U64 enemy = byColor[them];
    const U64 empty = ~occ;
    const int up = us == WHITE ? 8 : -8;
    const int startRank = us == WHITE ? 1 : 6;
    const int promoRank = us == WHITE ? 7 : 0;

    // Pawns
    U64 pawns = pieces(us, PAWN);
    while (pawns) {
        const int from = popLsb(pawns);
        const int r = rankOf(from);
        const int to = from + up;

        if (validSq(to) && (empty & bit(to))) {
            if (rankOf(to) == promoRank) {
                for (int p = KNIGHT; p <= QUEEN; ++p)
                    list.add(Move(from, to, p, Move::PROMOTION));
            } else if (!capturesOnly) {
                list.add(Move(from, to));
                if (r == startRank) {
                    int to2 = to + up;
                    if (validSq(to2) && (empty & bit(to2)))
                        list.add(Move(from, to2, 0, Move::DOUBLE_PUSH));
                }
            }
        }

        U64 caps = pawnAttacks(us, from) & enemy;
        while (caps) {
            int t = popLsb(caps);
            if (rankOf(t) == promoRank) {
                for (int p = KNIGHT; p <= QUEEN; ++p)
                    list.add(Move(from, t, p, Move::PROMOTION, true));
            } else {
                list.add(Move(from, t, 0, Move::NORMAL, true));
            }
        }

        if (ep >= 0 && board[ep] == NO_PIECE &&
            (pawnAttacks(us, from) & bit(ep))) {
            int capSq = ep + (us == WHITE ? -8 : 8);
            if (board[capSq] == makePiece(them, PAWN))
                list.add(Move(from, ep, 0, Move::EN_PASSANT, true));
        }
    }

    // Knights
    U64 bb = pieces(us, KNIGHT);
    while (bb) {
        int from = popLsb(bb);
        U64 att = knightAttacks(from) & ~own;
        while (att) {
            int to = popLsb(att);
            if (!capturesOnly || !(empty & bit(to)))
                list.add(Move(from, to, 0, Move::NORMAL, !(empty & bit(to))));
        }
    }

    // Bishops
    bb = pieces(us, BISHOP);
    while (bb) {
        int from = popLsb(bb);
        U64 att = bishopAttacks(from, occ) & ~own;
        while (att) {
            int to = popLsb(att);
            if (!capturesOnly || !(empty & bit(to)))
                list.add(Move(from, to, 0, Move::NORMAL, !(empty & bit(to))));
        }
    }

    // Rooks
    bb = pieces(us, ROOK);
    while (bb) {
        int from = popLsb(bb);
        U64 att = rookAttacks(from, occ) & ~own;
        while (att) {
            int to = popLsb(att);
            if (!capturesOnly || !(empty & bit(to)))
                list.add(Move(from, to, 0, Move::NORMAL, !(empty & bit(to))));
        }
    }

    // Queens
    bb = pieces(us, QUEEN);
    while (bb) {
        int from = popLsb(bb);
        U64 att = (bishopAttacks(from, occ) | rookAttacks(from, occ)) & ~own;
        while (att) {
            int to = popLsb(att);
            if (!capturesOnly || !(empty & bit(to)))
                list.add(Move(from, to, 0, Move::NORMAL, !(empty & bit(to))));
        }
    }

    // King
    const int ks = kingSq[us];
    if (ks >= 0) {
        U64 att = kingAttacks(ks) & ~own;
        while (att) {
            int to = popLsb(att);
            if (!capturesOnly || !(empty & bit(to)))
                list.add(Move(ks, to, 0, Move::NORMAL, !(empty & bit(to))));
        }

        if (!capturesOnly) {
            if (us == WHITE) {
                if ((castling & WHITE_OO) && board[7] == W_ROOK &&
                    !(occ & (bit(5) | bit(6))) &&
                    !isSquareAttacked(4, BLACK) &&
                    !isSquareAttacked(5, BLACK) &&
                    !isSquareAttacked(6, BLACK))
                    list.add(Move(4, 6, 0, Move::CASTLE));
                if ((castling & WHITE_OOO) && board[0] == W_ROOK &&
                    !(occ & (bit(1) | bit(2) | bit(3))) &&
                    !isSquareAttacked(4, BLACK) &&
                    !isSquareAttacked(3, BLACK) &&
                    !isSquareAttacked(2, BLACK))
                    list.add(Move(4, 2, 0, Move::CASTLE));
            } else {
                if ((castling & BLACK_OO) && board[63] == B_ROOK &&
                    !(occ & (bit(61) | bit(62))) &&
                    !isSquareAttacked(60, WHITE) &&
                    !isSquareAttacked(61, WHITE) &&
                    !isSquareAttacked(62, WHITE))
                    list.add(Move(60, 62, 0, Move::CASTLE));
                if ((castling & BLACK_OOO) && board[56] == B_ROOK &&
                    !(occ & (bit(57) | bit(58) | bit(59))) &&
                    !isSquareAttacked(60, WHITE) &&
                    !isSquareAttacked(59, WHITE) &&
                    !isSquareAttacked(58, WHITE))
                    list.add(Move(60, 58, 0, Move::CASTLE));
            }
        }
    }
}

Move Position::parseMove(const std::string& str) const {
    if (str.size() < 4) return Move();
    int from = (str[1] - '1') * 8 + (str[0] - 'a');
    int to = (str[3] - '1') * 8 + (str[2] - 'a');
    if (!validSq(from) || !validSq(to)) return Move();

    int promo = 0;
    if (str.size() >= 5) {
        switch (char(std::tolower(static_cast<unsigned char>(str[4])))) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK; break;
            case 'q': promo = QUEEN; break;
            default: return Move();
        }
    }

    MoveList list;
    generateMoves(list);
    Position* self = const_cast<Position*>(this);
    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        if (m.from() != from || m.to() != to) continue;
        if (promo == 0) {
            if (m.type() == Move::PROMOTION) continue;
        } else {
            if (m.type() != Move::PROMOTION || m.promo() != promo) continue;
        }
        if (self->makeMove(m)) {
            self->unmakeMove(m);
            return m;
        }
    }
    return Move();
}

int Position::seeRec(int to, Color stm, int targetValue, U64 occLocal) const {
    U64 att = attackersTo(to, occLocal) & byColor[stm] & occLocal;
    if (!att) return 0;

    int bestSq = -1;
    int bestVal = 1000000;
    U64 a = att;
    while (a) {
        int sq = popLsb(a);
        int v = PieceValue[pieceType(board[sq])];
        if (v < bestVal) {
            bestVal = v;
            bestSq = sq;
        }
    }
    if (bestSq < 0) return 0;

    occLocal &= ~bit(bestSq);
    int value = targetValue - seeRec(to, ~stm, bestVal, occLocal);
    return std::max(0, value);
}

int Position::see(Move m) const {
    if (!m.isCapture() && m.type() != Move::PROMOTION) return 0;

    const int from = m.from();
    const int to = m.to();
    U64 occLocal = occ & ~bit(from);
    int targetValue = 0;

    if (m.type() == Move::EN_PASSANT) {
        int capSq = to + (side == WHITE ? -8 : 8);
        occLocal &= ~bit(capSq);
        targetValue = PieceValue[PAWN];
    } else if (board[to] != NO_PIECE) {
        targetValue = PieceValue[pieceType(board[to])];
    }

    if (m.type() == Move::PROMOTION)
        targetValue += PieceValue[PieceType(m.promo())] - PieceValue[PAWN];

    int moverValue = (m.type() == Move::PROMOTION)
                         ? PieceValue[PieceType(m.promo())]
                         : PieceValue[pieceType(board[from])];

    return targetValue - seeRec(to, ~side, moverValue, occLocal);
}

uint64_t perft(Position& pos, int depth) {
    if (depth <= 0) return 1;
    MoveList list;
    pos.generateMoves(list);
    uint64_t nodes = 0;
    for (int i = 0; i < list.count; ++i) {
        if (pos.makeMove(list.moves[i])) {
            if (depth == 1)
                ++nodes;
            else
                nodes += perft(pos, depth - 1);
            pos.unmakeMove(list.moves[i]);
        }
    }
    return nodes;
}

} // namespace chess
