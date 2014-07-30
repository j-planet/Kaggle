ORDINAL_CONT_COLS = ['var1', 'var3', 'var7', 'var8']
DISCRETE_COLS = ['var2', 'var4', 'var5', 'var6', 'var9']

# ----- coding for discrete columns ------------
VAR2_COLS_LOOKUP = dict(zip('ABC', range(1, 4)))

temp = ['A1', 'B1', 'C1', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'F1', 'G1', 'G2', 'H1', 'H2', 'H3', 'I1', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'K1', 'L1', 'M1', 'N1', 'O1', 'O2', 'P1', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'S1']
VAR4_COLS_LOOKUP = dict(zip(temp, range(1, len(temp)+1)))

VAR5_COLS_LOOKUP = dict(zip('ABCDEF', range(1, 7)))

VAR6_COLS_LOOKUP = VAR2_COLS_LOOKUP

VAR9_COLS_LOOKUP = dict(zip('AB', [1, 2]))

DISCRETE_COLS_LOOKUP = {'var2': VAR2_COLS_LOOKUP, 'var4': VAR4_COLS_LOOKUP, 'var5': VAR5_COLS_LOOKUP,
                        'var6': VAR6_COLS_LOOKUP, 'var9': VAR9_COLS_LOOKUP}