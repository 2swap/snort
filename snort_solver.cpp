/* Snort is a two player game played on a graph. Players alternate coloring vertices of the graph, one player colors black and the other colors white. A player loses if they color a vertex that is adjacent to a vertex of the opposite color.
 * This program strongly solves the game of Snort on a grid.
 */

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstdint>
#include <string>
#include <chrono>
#include <thread>

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

    uint64_t rotate_90_clockwise(uint64_t board) {
        uint64_t result = 0;
        for (int x = 0; x < GRID_WIDTH; ++x) {
            for (int y = 0; y < GRID_HEIGHT; ++y) {
                uint64_t vertex = 1ULL << (y * 8 + x);
                if (board & vertex) {
                    int new_x = GRID_HEIGHT - 1 - y;
                    int new_y = x;
                    result |= 1ULL << (new_y * 8 + new_x);
                }
            }
        }
        return result;
    }

    GameState canonicalize(const GameState& state) {
        // For a grid, we can consider the canonical form to be the one of the 4 rotations and reflections
        // such that the black uint64_t is the smallest possible, and to break ties, the white uint64_t is also the smallest possible.
        GameState transformed_state = state;
        GameState best_state = state;
        int num_symmetries = 4;
        if(GRID_WIDTH == GRID_HEIGHT) {
            num_symmetries = 8;
        }
        for (int i = 0; i < num_symmetries; ++i) {
            // Rotate the state
            if((i & 1) == 1) {
                transformed_state.black = reflect_vertical(transformed_state.black);
                transformed_state.white = reflect_vertical(transformed_state.white);
            }
            if((i & 3) == 2) {
                transformed_state.black = reflect_horizontal(transformed_state.black);
                transformed_state.white = reflect_horizontal(transformed_state.white);
            }
            if((i & 7) == 4) {
                transformed_state.black = rotate_90_clockwise(transformed_state.black);
                transformed_state.white = rotate_90_clockwise(transformed_state.white);
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

    bool is_finalized_get_score(const GameState& state, int& score, bool is_black_turn, GameState& finalized_state) {
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
        finalized_state = {state.black | black_legal, state.white | white_legal};
        int black_count = __builtin_popcountll(finalized_state.black);
        int white_count = __builtin_popcountll(finalized_state.white);
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
    int solve(GameState state, bool is_black_turn) {
        int score;
        GameState finalized_state;
        bool finalized = is_finalized_get_score(state, score, is_black_turn, finalized_state);
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
                    value = std::max(value, solve(next_state, !is_black_turn));
                } else {
                    next_state.white |= vertex;
                    value = std::min(value, solve(next_state, !is_black_turn));
                }
            }
        }
        memo[state] = value;
        return value;
    }

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

    uint64_t get_random_best_move(const GameState& state, bool is_black_turn) {
        uint64_t legal_moves = legal_moves_for_player(state, is_black_turn);
        // Best score is the score of the current state
        int best_score = solve(state, is_black_turn);
        // Choose a random move with score equal to best_score
        std::vector<uint64_t> best_moves;
        for (int x = 0; x < GRID_WIDTH; ++x) {
            for (int y = 0; y < GRID_HEIGHT; ++y) {
                uint64_t vertex = 1ULL << (y * 8 + x);
                if (!(legal_moves & vertex)) continue;
                GameState next_state = state;
                if (is_black_turn) {
                    next_state.black |= vertex;
                } else {
                    next_state.white |= vertex;
                }
                int score = solve(next_state, !is_black_turn);
                if(score == best_score) {
                    best_moves.push_back(vertex);
                }
            }
        }
        return best_moves[rand() % best_moves.size()];
    }

    GameState play_against_solver(bool black_is_human, bool white_is_human, bool black_cheat, bool white_cheat) {
        GameState state = {0, 0};
        bool is_black_turn = true;
        while (true) {
            std::cout << std::endl;
            print_state_with_coordinates(state);
            if (!legal_moves_for_player(state, is_black_turn)) {
                std::cout << (is_black_turn ? "Black" : "White") << " has no legal moves. Game over." << std::endl;
                int final_score;
                GameState finalized_state;
                bool finalized = is_finalized_get_score(state, final_score, is_black_turn, finalized_state);
                if(!finalized) {
                    throw std::runtime_error("Game should be finalized if there are no legal moves for the current player.");
                }
                std::cout << "Final score (Black - White): " << final_score << std::endl;
                return state;
            }
            if (is_black_turn) {
                std::cout << "Black's turn." << std::endl;
            } else {
                std::cout << "White's turn." << std::endl;
            }
            if(is_black_turn ? black_cheat : white_cheat) print_all_move_scores(state, is_black_turn);
            char x, y;
            if(is_black_turn ? black_is_human : white_is_human) {
                std::cout << "Enter your move (e.g. A0): ";
                std::cin >> x >> y;
                x -= 'A';
            } else {
                uint64_t best_move = get_random_best_move(state, is_black_turn);
                x = (best_move & -best_move) ? __builtin_ctzll(best_move) % 8 : 0;
                y = (best_move & -best_move) ? __builtin_ctzll(best_move) / 8 : 0;
                std::cout << "Computer plays: " << (char)('A' + x) << (int)y << std::endl;
            }
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

    void all_optimal_endgame_boards(std::unordered_set<GameState>& endgame_states, const GameState& state, bool is_black_turn) {
        int score;
        GameState finalized_state;
        bool finalized = is_finalized_get_score(state, score, is_black_turn, finalized_state);
        if (finalized) {
            endgame_states.insert(canonicalize(finalized_state));
            return;
        }
        // Call this function recursively on all optimal moves, which are the moves that lead to a score equal to the score of the current state.
        int best_score = solve(state, is_black_turn);
        uint64_t legal_moves = legal_moves_for_player(state, is_black_turn);
        for (int x = 0; x < GRID_WIDTH; ++x) {
            for (int y = 0; y < GRID_HEIGHT; ++y) {
                uint64_t vertex = 1ULL << (y * 8 + x);
                if (!(legal_moves & vertex)) continue;
                GameState next_state = state;
                if (is_black_turn) {
                    next_state.black |= vertex;
                } else {
                    next_state.white |= vertex;
                }
                if(solve(next_state, !is_black_turn) == best_score) {
                    all_optimal_endgame_boards(endgame_states, next_state, !is_black_turn);
                }
            }
        }
    }
};

void print_optimal_result_array() {
    GameState initial_state = {0, 0};

    std::vector<std::vector<int>> results(8, std::vector<int>(8, -1000));
    // Known results

    /*
   1 2 3 1 2 1 2
   2 0 2 0 2 0 2
   3 2 2 1 2 1 2
   1 0 1 0 1 0 2
   2 2 2 1 2
   1 0 1 0
   2 2 2 2
   1 0 1
   */


    results[0][0] = 1;
    results[0][1] = 2;
    results[0][2] = 3;
    results[0][3] = 1;
    results[0][4] = 2;
    results[0][5] = 1;
    results[0][6] = 2;
    results[0][7] = 1;

    results[1][1] = 0;
    results[1][2] = 2;
    results[1][3] = 0;
    results[1][4] = 2;
    results[1][5] = 0;
    results[1][6] = 2;
    results[1][7] = 0;

    results[2][2] = 2;
    results[2][3] = 1;
    results[2][4] = 2;
    results[2][5] = 1;
    results[2][6] = 2;
    results[2][7] = 1;

    results[3][3] = 0;
    results[3][4] = 1;
    results[3][5] = 0;
    results[3][6] = 2;

    results[4][4] = 2;

    results[4][5] = 2;

    std::cout << "    ";
    for (int width = 1; width <= 7; ++width) {
        std::cout << width << " ";
    }
    std::cout << std::endl << std::endl;

    for (int height = 1; height <= 8; ++height) {
        std::cout << height << "   ";
        for (int width = 1; width <= 7; ++width) {
            if(width * height > 32) {
                std::cout << "   ";
                continue;
            }
            int& result = results[height - 1][width - 1];
            if(result == -1000) {
                result = results[width - 1][height - 1];
                if(result == -1000) {
                    SnortSolver solver(width, height);
                    result = solver.solve(initial_state, true);
                    results[height - 1][width - 1] = result;
                }
            }
            std::cout << result << " " << std::flush;
        }
        std::cout << std::endl;
    }
}

void print_optimal_move_scores_array() {
    std::cout << std::endl << std::endl;

    for (int height = 1; height <= 8; ++height) {
        for (int width = 1; width <= 7; ++width) {
            if(width * height > 27) {
                std::cout << "   ";
                continue;
            }

            std::cout << "Optimal move scores for " << width << "x" << height << ":" << std::endl;
            SnortSolver solver(width, height);
            solver.print_all_move_scores({0, 0}, true);
        }
    }
}

void print_number_of_optimal_endgame_boards_array() {
    GameState initial_state = {0, 0};

    std::cout << "      ";
    for (int width = 1; width <= 7; ++width) {
        std::cout << width << "   ";
    }
    std::cout << std::endl << std::endl;

    for (int height = 1; height <= 8; ++height) {
        std::cout << height << "   ";
        for (int width = 1; width <= 7; ++width) {
            if(width * height > 27) {
                continue;
            }
            SnortSolver solver(width, height);
            std::unordered_set<GameState> endgame_states;
            solver.all_optimal_endgame_boards(endgame_states, {0, 0}, true);
            std::string num_str = std::to_string(endgame_states.size());
            std::string padding(3 - num_str.size(), ' ');
            std::cout << padding << num_str << " " << std::flush;
        }
        std::cout << std::endl;
    }
}

void get_stats(int w, int h) {
    SnortSolver solver(w, h);
    solver.print_all_move_scores({0, 0}, true);

    return;
    std::unordered_set<GameState> endgame_states;
    solver.all_optimal_endgame_boards(endgame_states, {0, 0}, true);
    for(const auto& state : endgame_states) {
        std::cout << "Optimal endgame board:" << std::endl;
        solver.print_state_with_coordinates(state);
        std::cout << std::endl;
    }
    std::cout << "Number of optimal endgame boards: " << endgame_states.size() << std::endl;
}

int main(int argc, char* argv[]) {
    get_stats(8,4);
    print_optimal_result_array();
    /*
    print_number_of_optimal_endgame_boards_array();
    return 0;
    // Parse args
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height>" << std::endl;
        return 1;
    }
    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    return 0;
    */
}
