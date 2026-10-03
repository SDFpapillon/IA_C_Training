/*
 * Hybrid training demo on a real (tiny) game, as promised back in
 * TODO.md's Phase 5 notes. Tic-tac-toe game logic (board, win detection,
 * a minimax solver) lives entirely in this file — the library stays
 * game-agnostic, per the project's own design principle.
 *
 * Setup:
 *   - A network looks at a board (from the current player's perspective:
 *     +1 own mark, -1 opponent's, 0 empty) and outputs 9 desirability
 *     scores, one per cell; the move played is the legal cell with the
 *     highest score.
 *   - "Known cases" (TODO's "mate in 1 is always the best move") are a
 *     small set of positions where an immediate win or a forced block
 *     exists, labeled with the minimax-best move.
 *   - Full-task fitness: fraction of self-play games not lost against a
 *     perfect minimax opponent (wins are impossible against perfect play,
 *     so this is effectively the draw rate — "did it learn not to blunder").
 *
 * Compares pure genetic training against Strategy 2 (supervised pretrain
 * on the known cases, then seed the genetic population from it), the
 * cleanest of Phase 5's three strategies to demonstrate on a real game.
 */
#include <stdio.h>
#include <string.h>

#include "nn.h"

/* --------------------------------------------------------------------- */
/* Tic-tac-toe                                                           */
/* --------------------------------------------------------------------- */

typedef int ttt_board[9]; /* 0 = empty, 1 = X, -1 = O */

static const int WIN_LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, /* rows */
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, /* columns */
    {0, 4, 8}, {2, 4, 6},           /* diagonals */
};

static int ttt_winner(const ttt_board board)
{
    for (int l = 0; l < 8; l++) {
        int a = board[WIN_LINES[l][0]];
        int b = board[WIN_LINES[l][1]];
        int c = board[WIN_LINES[l][2]];
        if (a != 0 && a == b && b == c) return a;
    }
    return 0;
}

static int ttt_is_full(const ttt_board board)
{
    for (int i = 0; i < 9; i++) {
        if (board[i] == 0) return 0;
    }
    return 1;
}

/* Negamax with alpha-beta pruning. Returns the score for `player` to move
 * (+1 win, 0 draw, -1 loss, assuming perfect play on both sides); writes
 * the first best move found to *best_move if non-NULL. */
static int ttt_minimax(ttt_board board, int player, int alpha, int beta, int *best_move)
{
    int winner = ttt_winner(board);
    if (winner != 0) return winner * player;
    if (ttt_is_full(board)) return 0;

    int best_score = -2;
    int best_mv = -1;
    for (int i = 0; i < 9; i++) {
        if (board[i] != 0) continue;
        board[i] = player;
        int score = -ttt_minimax(board, -player, -beta, -alpha, NULL);
        board[i] = 0;
        if (score > best_score) {
            best_score = score;
            best_mv = i;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    if (best_move != NULL) *best_move = best_mv;
    return best_score;
}

/* --------------------------------------------------------------------- */
/* Network <-> board                                                     */
/* --------------------------------------------------------------------- */

static void ttt_encode(const ttt_board board, int player, nn_real *input)
{
    for (int i = 0; i < 9; i++) input[i] = (nn_real)(board[i] * player);
}

/* Highest-scoring legal move. */
static int ttt_best_legal_move(const ttt_board board, const nn_real *output)
{
    int best = -1;
    for (int i = 0; i < 9; i++) {
        if (board[i] != 0) continue;
        if (best == -1 || output[i] > output[best]) best = i;
    }
    return best;
}

static int ttt_network_move(const ttt_board board, int player, const nn_network *net,
                             nn_forward_buffer *buf)
{
    nn_real input[9];
    nn_real output[9];
    ttt_encode(board, player, input);
    nn_forward(net, buf, input, output);
    return ttt_best_legal_move(board, output);
}

/* --------------------------------------------------------------------- */
/* Known cases: immediate win or forced block                           */
/* --------------------------------------------------------------------- */

/* A position is "tactical" if the player to move has an immediate win, or
 * must block one (i.e. minimax's score swings sharply on this move). */
static int ttt_is_tactical(ttt_board board, int player, int *best_move)
{
    int score = ttt_minimax(board, player, -2, 2, best_move);
    if (score <= 0) return 0; /* can't force a win here; not the clean-cut case we want */
    /* Require a single ply to decide it: taking the suggested move must
     * already win or must be forced (removing it loses). */
    board[*best_move] = player;
    int immediate = ttt_winner(board) == player;
    board[*best_move] = 0;
    return immediate;
}

static void build_known_cases(nn_dataset **out, nn_rng *rng)
{
    #define MAX_KNOWN 40
    nn_real inputs[MAX_KNOWN][9];
    nn_real targets[MAX_KNOWN][9];
    size_t n = 0;

    while (n < MAX_KNOWN) {
        ttt_board board = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        int player = 1;
        int plies = 2 + (int)(nn_rng_next_u64(rng) % 4); /* 2..5 random opening plies */
        int valid = 1;
        for (int p = 0; p < plies && valid; p++) {
            if (ttt_winner(board) != 0 || ttt_is_full(board)) {
                valid = 0;
                break;
            }
            int empties[9], n_empty = 0;
            for (int i = 0; i < 9; i++) if (board[i] == 0) empties[n_empty++] = i;
            int choice = empties[nn_rng_next_u64(rng) % (size_t)n_empty];
            board[choice] = player;
            player = -player;
        }
        if (!valid || ttt_winner(board) != 0 || ttt_is_full(board)) continue;

        int best_move = -1;
        if (!ttt_is_tactical(board, player, &best_move)) continue;

        ttt_encode(board, player, inputs[n]);
        for (int i = 0; i < 9; i++) targets[n][i] = (i == best_move) ? 1.0 : -1.0;
        n++;
    }

    nn_dataset_create(n, 9, 9, out);
    for (size_t s = 0; s < n; s++) {
        memcpy(&(*out)->inputs[s * 9], inputs[s], 9 * sizeof(nn_real));
        memcpy(&(*out)->targets[s * 9], targets[s], 9 * sizeof(nn_real));
    }
    #undef MAX_KNOWN
}

/* --------------------------------------------------------------------- */
/* Fitness: non-loss rate in self-play against perfect minimax           */
/* --------------------------------------------------------------------- */

#define GAMES_PER_EVAL 6

static nn_real play_one_game(const nn_network *net, nn_forward_buffer *buf, int network_plays_first)
{
    ttt_board board = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int player = 1; /* X always moves first; network_plays_first picks which mark it uses */
    int network_mark = network_plays_first ? 1 : -1;

    while (ttt_winner(board) == 0 && !ttt_is_full(board)) {
        int move;
        if (player == network_mark) {
            move = ttt_network_move(board, player, net, buf);
        } else {
            ttt_minimax(board, player, -2, 2, &move);
        }
        board[move] = player;
        player = -player;
    }

    int winner = ttt_winner(board);
    if (winner == network_mark) return 1.0;  /* shouldn't happen vs perfect play, but handle it */
    if (winner == 0) return 0.5;             /* draw */
    return 0.0;                              /* loss */
}

static nn_real fitness(const nn_network *net, void *ctx_)
{
    nn_forward_buffer *buf = ctx_;
    nn_real total = 0.0;
    for (int g = 0; g < GAMES_PER_EVAL; g++) {
        total += play_one_game(net, buf, g % 2);
    }
    return total / (nn_real)GAMES_PER_EVAL;
}

/* --------------------------------------------------------------------- */

static size_t best_individual(nn_population *pop, nn_forward_buffer *buf)
{
    size_t best = 0;
    nn_real best_f = fitness(pop->networks[0], buf);
    for (size_t i = 1; i < pop->size; i++) {
        nn_real f = fitness(pop->networks[i], buf);
        if (f > best_f) {
            best_f = f;
            best = i;
        }
    }
    return best;
}

typedef struct {
    int wins, draws, losses;
} game_record;

/* A larger, separate sample than GAMES_PER_EVAL (used internally by the GA
 * for speed) for a more reliable final readout. */
static game_record record_games(const nn_network *net, nn_forward_buffer *buf, int n_games)
{
    game_record rec = {0, 0, 0};
    for (int g = 0; g < n_games; g++) {
        nn_real r = play_one_game(net, buf, g % 2);
        if (r > 0.75) rec.wins++;
        else if (r > 0.25) rec.draws++;
        else rec.losses++;
    }
    return rec;
}

int main(void)
{
    size_t sizes[] = {9, 27, 9};
    nn_activation acts[] = {NN_ACT_TANH, NN_ACT_TANH};

    nn_rng rng;
    nn_rng_seed(&rng, 1);
    nn_dataset *known = NULL;
    build_known_cases(&known, &rng);
    printf("built %zu known tactical positions (immediate win / forced block)\n", known->n_samples);

    nn_network *shape_net = NULL;
    nn_create(sizes, 3, acts, 2, NN_INIT_UNIFORM, 0, &shape_net);
    nn_forward_buffer *buf = NULL;
    nn_forward_buffer_create(shape_net, &buf);

    nn_genetic_params params = {0};
    params.generations = 40;
    params.elitism = 2;
    params.selection = NN_SELECT_TOURNAMENT;
    params.tournament_size = 4;
    params.crossover = NN_CROSSOVER_UNIFORM;
    params.mutation_rate = 0.15;
    params.mutation_stddev = 0.4;
    params.seed = 123;

    /* --- Pure genetic: random init. --- */
    nn_population *pop_pure = NULL;
    nn_population_create_random(sizes, 3, acts, 2, NN_INIT_XAVIER, 30, 777, &pop_pure);
    nn_train_genetic(pop_pure, fitness, buf, &params);
    game_record rec_pure = record_games(pop_pure->networks[best_individual(pop_pure, buf)], buf, 20);

    /* --- Strategy 2: pretrain on known cases, then seed the population. --- */
    nn_network *pretrained = NULL;
    nn_create(sizes, 3, acts, 2, NN_INIT_XAVIER, 777, &pretrained);
    nn_train_params sup_params = {0};
    sup_params.learning_rate = 0.3;
    sup_params.epochs = 400;
    sup_params.batch_size = 0;
    sup_params.loss = NN_LOSS_MSE;
    sup_params.shuffle_seed = 1;
    nn_train_supervised(pretrained, known, &sup_params);

    nn_population *pop_hybrid = NULL;
    nn_population_create_seeded(pretrained, 30, 0.15, 0.4, 777, &pop_hybrid);
    nn_train_genetic(pop_hybrid, fitness, buf, &params);
    game_record rec_hybrid = record_games(pop_hybrid->networks[best_individual(pop_hybrid, buf)], buf, 20);

    printf("\nbest individual's record over 20 games against perfect play\n");
    printf("(a loss is a blunder; a draw is the best possible outcome here, since winning\n");
    printf(" against perfect play is impossible by definition):\n");
    printf("  pure genetic:               %2d wins, %2d draws, %2d losses\n", rec_pure.wins,
           rec_pure.draws, rec_pure.losses);
    printf("  Strategy 2 (pretrain+seed): %2d wins, %2d draws, %2d losses\n", rec_hybrid.wins,
           rec_hybrid.draws, rec_hybrid.losses);

    nn_forward_buffer_free(buf);
    nn_population_free(pop_pure);
    nn_population_free(pop_hybrid);
    nn_free(pretrained);
    nn_free(shape_net);
    nn_dataset_free(known);
    return 0;
}
