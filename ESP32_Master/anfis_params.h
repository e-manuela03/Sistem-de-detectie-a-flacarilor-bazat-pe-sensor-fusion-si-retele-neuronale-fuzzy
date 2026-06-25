// ================================================================
// anfis_params.h
// Parametrii ANFIS exportati automat din anfis_train.py
// Arhitectura: 3 intrari | 3 MF gaussiene | 27 reguli
// Algoritm: GD + LSE (hibrid, GD aplicat inaintea LSE)
// Intrari: IR_raw, Tmax_kalman, Tmean_kalman
// p_i: min=-12019.9203, max=523146.2859
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
// IR:    LOW=0.249, MED=0.517, HIGH=0.706
// Tmax:  LOW=0.261, MED=0.501, HIGH=0.750
// Tmean: LOW=0.248, MED=0.494, HIGH=0.750
const float ANFIS_C[ANFIS_N_INPUTS][ANFIS_N_MF] = {
  {0.249203f, 0.516829f, 0.705696f},
  {0.260952f, 0.500999f, 0.750055f},
  {0.247689f, 0.493759f, 0.749505f}
};

// -- Latimi gaussiene [n_inputs][n_mf] -----------------------
const float ANFIS_SIGMA[ANFIS_N_INPUTS][ANFIS_N_MF] = {
  {0.097824f, 0.094516f, 0.067992f},
  {0.032421f, 0.069987f, 0.099566f},
  {0.099179f, 0.124494f, 0.104360f}
};

// -- Parametrii consecinta [n_rules] -------------------------
const float ANFIS_P[ANFIS_N_RULES] = {
    +6.83687062e-01f,   -3.47605100e-01f,   +9.99446691e-01f,
    +9.28649971e-01f,   -8.67003067e-01f,   +9.93582063e-01f,
    +1.00099664e+00f,   +1.03276234e+00f,   +9.82479200e-01f,
    -2.46019594e+01f,   +1.26396738e+01f,   +1.00127333e+00f,
    +2.68007133e+00f,   +9.71586574e-01f,   +1.00393027e+00f,
    +9.66524201e-01f,   +1.00639844e+00f,   +1.00172735e+00f,
    +5.23146286e+05f,   -1.81720323e+00f,   +7.23848908e-01f,
    -1.20199203e+04f,   +1.03901017e+00f,   +9.78107889e-01f,
    +2.68829081e+02f,   +9.91457095e-01f,   +1.00343322e+00f
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
