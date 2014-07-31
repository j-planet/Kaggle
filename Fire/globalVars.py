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

# ----- selected fields ------------
FIELDS_51 = [c[0] for c in [['crimeVar5', 0.08090102112971892],
                           ['var8', 0.062468280647729225],
                           ['var13', 0.062340567753313501],
                           ['weatherVar99', 0.057465905844226055],
                           ['geodemVar7', 0.036199291980605487],
                           ['weatherVar70', 0.028436213714319127],
                           ['geodemVar2', 0.027258248941988351],
                           ['var15', 0.024292331587827409],
                           ['weatherVar33', 0.022032658530686491],
                           ['var4', 0.021647271069815059],
                           ['weatherVar182', 0.020482723777822787],
                           ['weatherVar140', 0.019937621454365596],
                           ['weatherVar6', 0.019864246595856744],
                           ['weatherVar222', 0.018434059901062158],
                           ['weatherVar201', 0.017146237393160981],
                           ['var10', 0.017016412346498658],
                           ['weatherVar116', 0.014792303599264436],
                           ['weatherVar225', 0.014468701886785104],
                           ['weatherVar216', 0.01405570790878031],
                           ['crimeVar8', 0.013472009018447719],
                           ['var6', 0.013119844352716426],
                           ['weatherVar118', 0.011985605453093868],
                           ['weatherVar209', 0.011979176447360801],
                           ['geodemVar12', 0.011753270584117041],
                           ['geodemVar20', 0.010921834414345057],
                           ['weatherVar166', 0.010865122671656123],
                           ['crimeVar9', 0.010613888360515128],
                           ['weatherVar16', 0.0094339779541444783],
                           ['weatherVar122', 0.009085865267028875],
                           ['weatherVar112', 0.009079212646841011],
                           ['crimeVar6', 0.0090587539306257389],
                           ['weatherVar190', 0.0090174452360062135],
                           ['weatherVar121', 0.0089537905514306421],
                           ['var7', 0.0087087681253227543],
                           ['weatherVar10', 0.0086441481574310187],
                           ['weatherVar59', 0.0084657467399718803],
                           ['weatherVar170', 0.0081025206271305526],
                           ['geodemVar5', 0.008086397478949776],
                           ['weatherVar213', 0.0077128162207735582],
                           ['weatherVar63', 0.0072166394747977267],
                           ['crimeVar2', 0.0071873549847505302],
                           ['weatherVar221', 0.0065671248101127596],
                           ['weatherVar27', 0.0064855076934583079],
                           ['var9', 0.0064388095151720328],
                           ['weatherVar164', 0.0062541149896197436],
                           ['weatherVar153', 0.0060500888151908483],
                           ['weatherVar36', 0.0059398358459206643],
                           ['weatherVar198', 0.0057675510339364414],
                           ['weatherVar230', 0.0056191735284689534],
                           ['weatherVar139', 0.0055765416140594552],
                           ['weatherVar68', 0.005080393733933021]]]

FIELDS_20 = FIELDS_51[:20]