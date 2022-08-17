category_features = ['PRACT_COLL_MODE', 'HEAT_NO', 'STATION_ID', 'STATION_NO',
                     'PROD_SHIFT_NO', 'PROD_SHIFT_GROUP', 'ST_NO', 'FURNACE_NUM', 'FURNACE_AGE', 'STEEL_HOLE_AGE',
                     'MAIL_LANCE_NO', 'MAIL_LANCE_AGE', 'FIT_LANCE_NO', 'FIT_LANCE_AGE']

integer_features = ['TOTAL_COST_SEC', 'IRON_BLOW_COST_SEC', 'BLOW_SCRAP_COST_SEC', 'BLOW_COST_SEC', 'SEC_BLOW_SEC',
                    'SEC_BLOW_CATCH_COST_SEC', 'TAP_COST_SEC', 'IRON_TEMP', 'BLOW_DURATION', 'REBLOW_NUM',
                    'REBLOW_DURATION', 'TAP_DURATION', 'OUT_STEEL_PRE_TEMP', 'OUT_STEEL_TEMP', 'LADLE_AGE',
                    'AR_SUM_COMSUME', 'SPATTER_SLAG_AR_COMSUME', '5G11', '8ZCST'
                    ]

double_features = ['START_LANCE_POS', 'MID_LANCE_POS', 'END_LANCE_POS', 'START_MID_LANCE_POS_DIFF',
                   'MID_END_LANCE_POS_DIFF', 'START_END_LANCE_POS_DIFF', 'N_SUM_COMSUME',
                   'CO_RETURN_WT', 'MOLTIRON_WT', 'IRON_C', 'IRON_SI', 'IRON_MN', 'IRON_P', 'IRON_S', '8ZCSSH',
                   '4MBY_N', '6QSBYS', '8ZCQSBYS', '6YS', '6SZYL', '91LTFK', '6ZTJ_N', '3Al_D', '3FeV', '3FeTi',
                   '3GNH', '3SiMn', '3FeCr_GC', '3FeSi_K', '3SiMn_II', 'ST_C', 'ST_SI', 'ST_MN', 'ST_P', 'ST_S',
                   'ST_NI', 'ST_CR', 'ST_CU', 'ST_MO', 'ST_V', 'ST_TI', 'ST_CEQ', 'FD_C_VALUE', 'FD_SI_VALUE',
                   'FD_MN_VALUE', 'FD_P_VALUE', 'FD_S_VALUE', 'FD_NI_VALUE', 'FD_CR_VALUE', 'FD_CU_VALUE',
                   'FD_MO_VALUE', 'FD_V_VALUE', 'FD_TI_VALUE', 'FD_CEQ_VALUE', 'LADLE_ARRIVE_WT', 'LADLE_LEAVE_WT',
                   'TOTAL_LOSS_WT', 'CAST_YIELD', 'Al2O3', 'CaO', 'MgO', 'MnO', 'S', 'SiO2', 'TFe', 'TiO2',
                   'CaO_SiO2_R', 'P2O5'
                   ]

time_diff_features = ['TOTAL_COST_SEC', 'IRON_BLOW_COST_SEC', 'BLOW_SCRAP_COST_SEC', 'BLOW_COST_SEC', 'SEC_BLOW_SEC',
                      'SEC_BLOW_CATCH_COST_SEC', 'TAP_COST_SEC']

model_confs = {
    'model_conf':
        {
            'model_parameter':
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    'n_jobs': -1,
                    'num_threads': 4,
                    "num_leaves": 2 ** 6,
                    "max_depth": 6,
                    "min_child_samples": 2,
                    'objective': 'regression',
                    'reg_lambda': 0.3,
                }
        }
}
