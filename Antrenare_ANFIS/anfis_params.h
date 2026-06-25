// ================================================================
// anfis_params.h
// Parametrii ANFIS exportati automat din anfis_train.py
// Arhitectura: 3 intrari | 3 MF gaussiene | 27 reguli
// Algoritm: GD + LSE (hibrid, GD aplicat inaintea LSE)
// Intrari: IR_raw, Tmax_kalman, Tmean_kalman
// p_i: min=-15400.3226, max=549149.8488
// ================================================================

#pragma once

#define ANFIS_N_INPUTS  3
#define ANFIS_N_MF      3
#define ANFIS_N_RULES   27
#define ANFIS_WSUM_GUARD 1e-6f

// -- Parametrii de normalizare --------------------------------
const float ANFIS_MINS[ANFIS_N_INPUTS] = {0.0000f, 20.0000f, 20.0000f};
const float ANFIS_MAXS[ANFIS_N_INPUTS] = {4095.0000f, 300.0000f, 65.0000f};

// -- Centre gaussiene [n_inputs][n_mf] -----------------------
// IR:    LOW=0.249, MED=0.517, HIGH=0.705
// Tmax:  LOW=0.260, MED=0.501, HIGH=0.750
// Tmean: LOW=0.248, MED=0.494, HIGH=0.750
const float ANFIS_C[ANFIS_N_INPUTS][ANFIS_N_MF] = {
  {0.249140f, 0.516749f, 0.705410f},
  {0.260168f, 0.501109f, 0.750039f},
  {0.247553f, 0.493749f, 0.749507f}
};

// -- Latimi gaussiene [n_inputs][n_mf] -----------------------
const float ANFIS_SIGMA[ANFIS_N_INPUTS][ANFIS_N_MF] = {
  {0.099128f, 0.094897f, 0.066378f},
  {0.032278f, 0.070111f, 0.099517f},
  {0.099353f, 0.124520f, 0.104317f}
};

// -- Parametrii consecinta [n_rules] -------------------------
const float ANFIS_P[ANFIS_N_RULES] = {
    +7.09043106e-01f,   -3.42397955e-01f,   +9.99313989e-01f,
    +9.18193951e-01f,   -7.89789633e-01f,   +9.91515502e-01f,
    +1.00114708e+00f,   +1.04905811e+00f,   +9.75007980e-01f,
    -2.59118211e+01f,   +1.25764158e+01f,   +1.00160088e+00f,
    +2.93984790e+00f,   +9.71717824e-01f,   +1.00517500e+00f,
    +9.56255780e-01f,   +1.00731146e+00f,   +1.00243548e+00f,
    +5.49149849e+05f,   -3.56712017e+00f,   +6.50528106e-01f,
    -1.54003226e+04f,   +1.03541489e+00f,   +9.80715441e-01f,
    +3.83708925e+02f,   +9.90247092e-01f,   +1.00368320e+00f
};

// -- Indecsi reguli [n_rules][n_inputs] ----------------------
const int ANFIS_RULES[ANFIS_N_RULES][ANFIS_N_INPUTS] = {
  {0, 0, 0},
  {0, 1, 0},
  {0, 2, 0},
  {1, 0, 0},
  {1, 1, 0},
  {1, 2, 0},
  {2, 0, 0},
  {2, 1, 0},
  {2, 2, 0},
  {0, 0, 1},
  {0, 1, 1},
  {0, 2, 1},
  {1, 0, 1},
  {1, 1, 1},
  {1, 2, 1},
  {2, 0, 1},
  {2, 1, 1},
  {2, 2, 1},
  {0, 0, 2},
  {0, 1, 2},
  {0, 2, 2},
  {1, 0, 2},
  {1, 1, 2},
  {1, 2, 2},
  {2, 0, 2},
  {2, 1, 2},
  {2, 2, 2}
};
