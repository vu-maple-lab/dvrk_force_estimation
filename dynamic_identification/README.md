# da Vinci Research Kit (dVRK) Dynamic Identification

*   Replicate the 2k traj for 3 times. (✓)
*   Use psm_simp. (✓)
*   Automate column merge (jaw insert to the others). (✓)
*   The insertion joint is quite shaky when rises back. No idea.
**  - From observation this does bring in jerky prediction
*   Probably insertion joint calibration?? 

## Trocar Correction:

Direct plug into 2nd MLP network in Jie Ying's code (✓)

## Standardized workflow:

1, Follow the screenshot below to collect data and preprocess.

![workflow](https://github.com/JackHaoyingZhou/daVinci_dynamic_identification/assets/33953293/645f6a55-e565-425e-b553-48a459b91f15)

2, Run psm_ID_run_w_FE.ipynb. Use simplified model and modify test set related args accordingly.

3, Run train_trocar.py and test_trocar.py. I use python train_trocar.py 1 1 and python test_trocar.py net no_seal but args always changeable. Note you need to play with window size.

4, Run trocar_tq_pred.mlx and trocar_fc_est.mlx to evaluate.
