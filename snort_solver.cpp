/* Snort is a two player game played on a graph. Players alternate coloring vertices of the graph, one player colors black and the other colors white. A player loses if they color a vertex that is adjacent to a vertex of the opposite color.
 * This program strongly solves the game of Snort on a grid.
 */

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cstdint>
#include <string>

struct GameState {
    uint64_t black;
    uint64_t white;
    bool operator==(const GameState& other) const {
        return black == other.black && white == other.white;
    }
};

// Hash function for GameState to be used in unordered_map
namespace std {
    template <>
    struct hash<GameState> {
        size_t operator()(const GameState& state) const {
            return hash<uint64_t>()(state.black) ^ (hash<uint64_t>()(state.white) << 1);
        }
    };
}

class SnortSolver {
public:
    SnortSolver(int width, int height) : GRID_WIDTH(width), GRID_HEIGHT(height), board_mask(initialize_board_mask()) {}

    const int GRID_WIDTH;
    const int GRID_HEIGHT;
    const uint64_t board_mask;

    std::unordered_map<GameState, int> memo;

    // Make a column mask, and swap each pair of columns to reflect the board horizontally.
    uint64_t reflect_horizontal(uint64_t board) {
        uint64_t column_mask = 0x0101010101010101ULL;
        for (int i = 0; i < GRID_WIDTH / 2; ++i) {
            uint64_t left_column = (board & (column_mask << i)) >> i;
            uint64_t right_column = (board & (column_mask << (GRID_WIDTH - 1 - i))) >> (GRID_WIDTH - 1 - i);
            board &= ~(column_mask << i);
            board &= ~(column_mask << (GRID_WIDTH - 1 - i));
            board |= (left_column << (GRID_WIDTH - 1 - i));
            board |= (right_column << i);
        }
        return board;
    }

    // Make a row mask, and swap each pair of rows to reflect the board vertically.
    uint64_t reflect_vertical(uint64_t board) {
        uint64_t row_mask = 0xFFULL;
        for (int i = 0; i < GRID_HEIGHT / 2; ++i) {
            uint64_t top_row = (board & (row_mask << (i * 8))) >> (i * 8);
            uint64_t bottom_row = (board & (row_mask << ((GRID_HEIGHT - 1 - i) * 8))) >> ((GRID_HEIGHT - 1 - i) * 8);
            board &= ~(row_mask << (i * 8));
            board &= ~(row_mask << ((GRID_HEIGHT - 1 - i) * 8));
            board |= (top_row << ((GRID_HEIGHT - 1 - i) * 8));
            board |= (bottom_row << (i * 8));
        }
        return board;
    }

    GameState canonicalize(const GameState& state) {
        // For a grid, we can consider the canonical form to be the one of the 4 rotations and reflections
        // such that the black uint64_t is the smallest possible, and to break ties, the white uint64_t is also the smallest possible.
        GameState transformed_state = state;
        GameState best_state = state;
        for (int i = 0; i < 4; ++i) {

            // Rotate the state
            if(i == 1 || i == 3) {
                transformed_state.black = reflect_vertical(transformed_state.black);
                transformed_state.white = reflect_vertical(transformed_state.white);
            }
            if(i == 2) {
                transformed_state.black = reflect_horizontal(transformed_state.black);
                transformed_state.white = reflect_horizontal(transformed_state.white);
            }

            if (transformed_state.black < best_state.black || (transformed_state.black == best_state.black && transformed_state.white < best_state.white)) {
                best_state = transformed_state;
            }
        }
        return best_state;
    }

    uint64_t legal_moves_for_player(const GameState& state, bool is_black) {
        uint64_t occupied = state.black | state.white;

        uint64_t opponent = is_black ? state.white : state.black;
        uint64_t adjacent = (opponent << 1) | (opponent >> 1) | (opponent << 8) | (opponent >> 8);

        return (~adjacent) & (~occupied) & board_mask;
    }

    bool is_finalized_get_score(const GameState& state, int& score, bool is_black_turn) {
        // The difference in the number of pieces, either played or playable, is the final score.
        uint64_t black_legal = legal_moves_for_player(state, true);
        uint64_t white_legal = legal_moves_for_player(state, false);
        uint64_t white_legal_expanded = white_legal | (white_legal << 1) | (white_legal >> 1) | (white_legal << 8) | (white_legal >> 8);
        uint64_t black_legal_expanded = black_legal | (black_legal << 1) | (black_legal >> 1) | (black_legal << 8) | (black_legal >> 8);
        uint64_t intersection = black_legal_expanded & white_legal_expanded;
        int popcnt = __builtin_popcountll(intersection);
        if(popcnt > 1) {
            return false;
        }
        int black_count = __builtin_popcountll(state.black | black_legal);
        int white_count = __builtin_popcountll(state.white | white_legal);
        score = black_count - white_count;
        if(popcnt == 1) {
            if (is_black_turn) {
                score += 1;
            } else {
                score -= 1;
            }
        }
        return true;
    }

    // Naive minimax
    int minimax(GameState state, bool is_black_turn) {
        int score;
        bool finalized = is_finalized_get_score(state, score, is_black_turn);
        if (finalized) {
            return score;
        }

        state = canonicalize(state);
        if (memo.find(state) != memo.end()) return memo[state];

        int value = is_black_turn ? -1000 : 1000;
        uint64_t legal_moves = legal_moves_for_player(state, is_black_turn);
        for (int x = 0; x < GRID_WIDTH; ++x) {
            for (int y = 0; y < GRID_HEIGHT; ++y) {
                uint64_t vertex = 1ULL << (y * 8 + x);
                if (!(legal_moves & vertex)) continue;
                GameState next_state = state;
                if (is_black_turn) {
                    next_state.black |= vertex;
                    value = std::max(value, minimax(next_state, !is_black_turn));
                } else {
                    next_state.white |= vertex;
                    value = std::min(value, minimax(next_state, !is_black_turn));
                }
            }
        }
        memo[state] = value;
        return value;
    }
    //your board states should be state.board[is_black_turn]=u64 to make everything easier here. u64 board[2]{};
    int negamax(GameState state, bool is_black_turn,int alph=-1000,int beta=1000) {
        int score;
        bool finalized = is_finalized_get_score(state, score, is_black_turn);
        int memo_val=1234; //sentinel val idk your system
        state=canonicalize(state);
        if (memo.contains(state)) memo_val=memo[state];
        if (is_black_turn) {
            if (finalized) return score;
            if (memo_val!=1234){return memo_val;}
        }
        if (!is_black_turn) {
            if (finalized) return -score;
            if (memo_val!=1234){return -memo_val;}

        }
        int value=-1000;
        uint64_t legal_moves = legal_moves_for_player(state, is_black_turn);
        std::vector<uint64_t> moves;
        for (int x = 0; x < GRID_WIDTH; ++x) {
            for (int y = 0; y < GRID_HEIGHT; ++y) {
                uint64_t vertex = 1ULL << (y * 8 + x);
                if (!(legal_moves & vertex)) continue;
                moves.push_back(vertex);
            }
        }
        //this is less efficient than o(n^2) sorting technically but i dont care to code it
        //o(n^2) sorting relies on beta cutoffs, if the first move provides a beta cutoff then sorting is done in o(n) time, if the second move is a betacutoff, o(n+n-1) etc...
        if (is_black_turn) {
            std::ranges::sort(moves,[](uint64_t a,uint64_t b){return __builtin_popcountll(a)>__builtin_popcountll(b);});
        }
        else {
            std::ranges::sort(moves,[](uint64_t a,uint64_t b){return __builtin_popcountll(a)>__builtin_popcountll(b);});
        }
        for (const uint64_t move:moves){
            GameState next_state = state;
            if (is_black_turn) {next_state.black |= move;}
            else {next_state.white |= move;}
            value = std::max(value, -negamax(next_state, !is_black_turn,-beta,-alph));
            if (value>=beta){return value;}
            if (value>alph) {
                alph=value;
            }
        }
        if (is_black_turn) {
            memo[state]=value;
        }
        if (!is_black_turn) {
            memo[state]=-value;
        }
        return value;
    }

    int solve(const GameState& state, bool is_black_turn) {
        return negamax(state, is_black_turn);
    }
//     1 2 3 4 5 6 7
//
// 1   1 2 3 1 2 1 2
// 2   2 0 2 0 2 0 2
// 3   3 2 2 1 2 1 2
// 4   1 0 1 0 1 0
// 5   2 2 2 1 2
// 6   1 0 1 0
// 7   2 2 2
// 8   1 0 1
    void print_all_move_scores(const GameState& state, bool is_black_turn) {
        uint64_t legal_moves = legal_moves_for_player(state, is_black_turn);
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                uint64_t vertex = 1ULL << (y * 8 + x);
                if (legal_moves & vertex) {
                    GameState next_state = state;
                    if (is_black_turn) {
                        next_state.black |= vertex;
                    } else {
                        next_state.white |= vertex;
                    }
                    int score = solve(next_state, !is_black_turn);
                    std::string score_str = std::to_string(score);
                    std::string padding(3 - score_str.size(), ' ');
                    std::cout << padding << score_str << std::flush;
                } else {
                    std::cout << "   ";
                }
            }
            std::cout << std::endl;
        }
    }

    uint64_t initialize_board_mask() {
        uint64_t mask = 0;
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                mask |= 1ULL << (y * 8 + x);
            }
        }
        return mask;
    }

    void print_state_with_coordinates(const GameState& state) {
        std::cout << "  ";
        for (int x = 0; x < GRID_WIDTH; ++x) {
            std::cout << (char)('A' + x) << " ";
        }
        std::cout << std::endl;
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            std::cout << y << " ";
            for (int x = 0; x < GRID_WIDTH; ++x) {
                uint64_t mask = 1ULL << (y * 8 + x);
                if(!(board_mask & mask)) {
                    std::cout << "  ";
                } else if (state.black & mask) {
                    std::cout << "X ";
                } else if (state.white & mask) {
                    std::cout << "O ";
                } else {
                    std::cout << ". ";
                }
            }
            std::cout << std::endl;
        }
    }

    void play_against_solver() {
        GameState state = {0, 0};
        bool is_black_turn = true;
        while (true) {
            std::cout << std::endl;
            print_state_with_coordinates(state);
            if (!legal_moves_for_player(state, is_black_turn)) {
                std::cout << (is_black_turn ? "Black" : "White") << " has no legal moves. Game over." << std::endl;
                int final_score;
                bool finalized = is_finalized_get_score(state, final_score, is_black_turn);
                if(!finalized) {
                    throw std::runtime_error("Game should be finalized if there are no legal moves for the current player.");
                }
                std::cout << "Final score (Black - White): " << final_score << std::endl;
                break;
            }
            if (is_black_turn) {
                std::cout << "Black's turn." << std::endl;
            } else {
                std::cout << "White's turn." << std::endl;
            }
            print_all_move_scores(state, is_black_turn);
            std::cout << "Enter your move (e.g. A0): ";
            char x, y;
            std::cin >> x >> y;
            x -= 'A';
            uint64_t vertex = 1ULL << (y * 8 + x);
            if (legal_moves_for_player(state, is_black_turn) & vertex) {
                if(is_black_turn) {
                    state.black |= vertex;
                } else {
                    state.white |= vertex;
                }
                is_black_turn = !is_black_turn;
            } else {
                std::cout << "Invalid move. Try again." << std::endl;
            }
        }
    }
};

void print_optimal_result_array() {
    GameState initial_state = {0, 0};

    std::cout << "    ";
    for (int width = 1; width <= 7; ++width) {
        std::cout << width << " ";
    }
    std::cout << std::endl << std::endl;

    for (int height = 1; height <= 8; ++height) {
        std::cout << height << "   ";
        for (int width = 1; width <= 7; ++width) {
            if(width * height > 27) {
                std::cout << "   ";
                continue;
            }
            SnortSolver solver(width, height);
            int result = solver.solve(initial_state, true);
            std::cout << result << " " << std::flush;
        }
        std::cout << std::endl;
    }
}

int main() {
    print_optimal_result_array();
    return 0;
}
