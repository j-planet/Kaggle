NUM_TRAIN_SAMPLES = 452061

# ----------- column stuff -------------
NON_PREDICTOR_COLS = ['id', 'dummy', 'var11']
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

# ------ overall feature selection --------------
FIELDS_20 = FIELDS_51[:20]

FIELDS_10 = FIELDS_51[:10]

FIELDS_5 = FIELDS_51[:5]

FIELDS_RFE = ['crimeVar5', 'crimeVar6', 'crimeVar7', 'crimeVar9', 'geodemVar1', 'geodemVar10',
              'geodemVar11', 'geodemVar12', 'geodemVar13', 'geodemVar15', 'geodemVar16',
              'geodemVar17', 'geodemVar2', 'geodemVar3', 'geodemVar4', 'geodemVar5',
              'geodemVar7', 'geodemVar8', 'geodemVar9']

FIELDS_CORR_ORDERED_TOP99 = [u'var13', u'var10', u'var8', u'var15', u'weatherVar118', u'weatherVar102', u'weatherVar103', u'var14', u'weatherVar227', u'geodemVar37', u'weatherVar235', u'weatherVar47', u'geodemVar24', u'geodemVar20', u'geodemVar13', u'geodemVar17', u'geodemVar8', u'geodemVar26', u'geodemVar11', u'geodemVar10', u'weatherVar104', u'geodemVar15', u'weatherVar61', u'geodemVar21', u'geodemVar33', u'geodemVar28', u'geodemVar35', u'geodemVar27', u'weatherVar72', u'geodemVar34', u'weatherVar220', u'weatherVar31', u'weatherVar137', u'geodemVar36', u'weatherVar153', u'geodemVar6', u'geodemVar29', u'weatherVar236', u'geodemVar4', u'weatherVar10', u'weatherVar108', u'weatherVar48', u'weatherVar81', u'weatherVar203', u'weatherVar173', u'weatherVar194', u'geodemVar16', u'crimeVar4', u'weatherVar24', u'geodemVar30', u'var4', u'weatherVar217', u'weatherVar38', u'weatherVar1', u'weatherVar172', u'weatherVar79', u'weatherVar35', u'crimeVar5', u'weatherVar211', u'weatherVar135', u'weatherVar70', u'weatherVar95', u'weatherVar130', u'geodemVar1', u'weatherVar44', u'weatherVar6', u'geodemVar19', u'weatherVar99', u'weatherVar51', u'weatherVar191', u'weatherVar69', u'weatherVar150', u'weatherVar178', u'weatherVar11', u'weatherVar80', u'weatherVar133', u'weatherVar7', u'weatherVar110', u'weatherVar98', u'weatherVar86', u'weatherVar68', u'weatherVar87', u'weatherVar229', u'weatherVar189', u'weatherVar151', u'weatherVar53', u'weatherVar138', u'weatherVar140', u'weatherVar224', u'weatherVar59', u'weatherVar32', u'weatherVar105', u'weatherVar175', u'weatherVar77', u'weatherVar117', u'weatherVar193', u'crimeVar7', u'weatherVar197', u'weatherVar132']

FIELDS_CDF_CORR_TOP99 = [u'var13', u'var10', u'var8', u'var15', u'var12', u'weatherVar44', u'weatherVar47', u'weatherVar69', u'weatherVar7', u'weatherVar227', u'weatherVar104', u'weatherVar150', u'var9', u'geodemVar13', u'weatherVar135', u'weatherVar130', u'weatherVar190', u'weatherVar217', u'weatherVar32', u'weatherVar51', u'crimeVar4', u'weatherVar193', u'geodemVar37', u'weatherVar194', u'weatherVar31', u'geodemVar8', u'geodemVar17', u'weatherVar182', u'weatherVar235', u'weatherVar23', u'weatherVar28', u'weatherVar175', u'var4', u'geodemVar26', u'weatherVar198', u'geodemVar15', u'weatherVar80', u'geodemVar11', u'weatherVar40', u'geodemVar20', u'weatherVar191', u'geodemVar10', u'weatherVar192', u'weatherVar196', u'weatherVar79', u'weatherVar189', u'geodemVar28', u'geodemVar33', u'weatherVar110', u'weatherVar98', u'geodemVar27', u'weatherVar118', u'weatherVar68', u'geodemVar35', u'crimeVar7', u'geodemVar21', u'weatherVar153', u'weatherVar105', u'weatherVar14', u'weatherVar87', u'weatherVar103', u'weatherVar236', u'geodemVar6', u'weatherVar102', u'weatherVar52', u'weatherVar116', u'weatherVar197', u'weatherVar61', u'weatherVar157', u'weatherVar3', u'weatherVar10', u'weatherVar183', u'weatherVar211', u'geodemVar36', u'weatherVar6', u'weatherVar139', u'geodemVar1', u'weatherVar203', u'weatherVar37', u'weatherVar25', u'weatherVar76', u'geodemVar2', u'weatherVar19', u'weatherVar53', u'weatherVar146', u'geodemVar29', u'weatherVar101', u'weatherVar128', u'weatherVar15', u'var14', u'weatherVar117', u'weatherVar200', u'geodemVar24', u'geodemVar31', u'weatherVar48', u'weatherVar13', u'weatherVar86', u'weatherVar195', u'weatherVar96']

FIELDS_RIDGE = ['weatherVar70', 'var10', 'geodemVar24', 'weatherVar195', 'var12', 'var2', 'var13', 'weatherVar140', 'weatherVar27', 'var14', 'weatherVar170', 'weatherVar186', 'weatherVar134', 'geodemVar37', 'weatherVar47', 'weatherVar51', 'weatherVar64', 'weatherVar10', 'weatherVar190', 'var8', 'geodemVar19', 'weatherVar192', 'geodemVar2', 'geodemVar9', 'geodemVar16', 'weatherVar36', 'crimeVar4', 'weatherVar153', 'weatherVar83', 'weatherVar176', 'weatherVar100', 'crimeVar5', 'geodemVar4', 'geodemVar30', 'weatherVar187', 'weatherVar157', 'weatherVar196', 'geodemVar34', 'weatherVar198', 'weatherVar143', 'weatherVar183', 'weatherVar33', 'crimeVar3', 'weatherVar14', 'weatherVar123', 'weatherVar106', 'weatherVar185', 'weatherVar107', 'weatherVar159', 'weatherVar21', 'weatherVar119', 'weatherVar40', 'weatherVar128', 'var5', 'weatherVar104', 'weatherVar227', 'crimeVar7', 'weatherVar175', 'weatherVar127', 'weatherVar58', 'geodemVar3', 'weatherVar161', 'var1', 'weatherVar188', 'geodemVar22', 'weatherVar109', 'weatherVar148', 'weatherVar179', 'weatherVar16', 'var3', 'weatherVar146', 'weatherVar199', 'weatherVar99', 'weatherVar164', 'weatherVar194', 'weatherVar139', 'var6', 'weatherVar18', 'weatherVar182', 'var9', 'weatherVar69', 'weatherVar5', 'weatherVar31', 'weatherVar112', 'geodemVar23', 'weatherVar150', 'weatherVar133', 'weatherVar15', 'weatherVar232', 'weatherVar216', 'weatherVar89', 'weatherVar42', 'weatherVar124', 'weatherVar55', 'weatherVar189', 'weatherVar193', 'weatherVar204', 'geodemVar7', 'weatherVar73', 'weatherVar135']
# ------ classification-only feature selection --------------
FIELDS_CLASS_GBC_TOP100 = [u'var4', u'var17', u'var15', u'var12', u'var13', u'weatherVar196', u'weatherVar197', u'weatherVar103', u'weatherVar102', u'weatherVar91', u'var1', u'var16', u'weatherVar142', u'var3', u'weatherVar36', u'weatherVar113', u'weatherVar128', u'weatherVar77', u'weatherVar186', u'weatherVar6', u'weatherVar187', u'weatherVar178', u'weatherVar27', u'weatherVar78', u'weatherVar85', u'weatherVar159', u'weatherVar163', u'weatherVar182', u'var10', u'weatherVar84', u'weatherVar119', u'weatherVar58', u'weatherVar86', u'weatherVar168', u'var8', u'weatherVar227', u'weatherVar129', u'geodemVar37', u'weatherVar141', u'var9', u'weatherVar15', u'weatherVar126', u'weatherVar212', u'weatherVar206', u'crimeVar3', u'geodemVar36', u'weatherVar22', u'weatherVar156', u'weatherVar13', u'weatherVar185', u'weatherVar179', u'weatherVar234', u'weatherVar123', u'weatherVar60', u'geodemVar3', u'weatherVar31', u'weatherVar70', u'weatherVar155', u'weatherVar161', u'weatherVar73', u'weatherVar104', u'geodemVar29', u'weatherVar12', u'geodemVar8', u'weatherVar44', u'weatherVar143', u'weatherVar194', u'weatherVar74', u'crimeVar1', u'weatherVar110', u'weatherVar20', u'weatherVar33', u'weatherVar53', u'weatherVar87', u'weatherVar68', u'weatherVar191', u'geodemVar1', u'weatherVar193', u'weatherVar131', u'weatherVar190', u'weatherVar118', u'weatherVar218', u'weatherVar171', u'weatherVar184', u'weatherVar133', u'geodemVar27', u'geodemVar22', u'weatherVar88', u'weatherVar188', u'weatherVar45', u'weatherVar154', u'geodemVar19', u'weatherVar192', u'weatherVar16', u'weatherVar109', u'weatherVar96', u'weatherVar48', u'weatherVar106', u'weatherVar10', u'weatherVar160']

# ------ regression-only feature selection --------------