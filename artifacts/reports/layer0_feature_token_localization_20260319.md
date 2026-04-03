# Layer 0 Feature Token Localization

Candidate checkpoint: `artifacts/runs/sae/v2_tanh_n2048_l0_5e2/probe/checkpoints/layer_0/best.pt`
Layer: `0`

## bace_classification feature 1419

- coef_mean: `1.4908`
- single_feature_roc_auc: `0.4756`

Top localized tokens per example:
- `ClC1=CN=C(NC1=O)C(NC1=NC(Cc2c1ccc(Cl)c2)(C)C)Cc1ccccc1`
  top tokens: `O` (2.702), `NC` (2.445), `CN` (2.019), `C` (1.342)
- `IC1=CN=C(NC1=O)C(NC1=NC(Cc2c1ccc(Cl)c2)(C)C)Cc1ccccc1`
  top tokens: `O` (2.691), `NC` (2.405), `CN` (1.979), `C` (1.303)
- `[NH+]=1C(C=2N(CCCN=2)C=1N)(c1ccccc1)c1ccccc1`
  top tokens: `C` (2.683), `CCCN` (1.324), `C` (0.686), `C` (0.517)
- `O=C1NC(=NC(=C1)CCC)N`
  top tokens: `C` (2.958), `O` (0.190), `NC` (-0.000), `NC` (-0.000)
- `O=C1NC(=NC(=C1)CCc1cc2[nH]ccc2cc1)N`
  top tokens: `C` (3.070), `ccc` (0.070), `O` (-0.000), `NC` (-0.000)
- `O=C(NCC1CCCCC1)CCc1cc2c(nc1N)cccc2`
  top tokens: `C` (2.989), `cccc` (0.073), `O` (-0.000), `NCC` (-0.000)

Aggregated frequent high-activation substrings:
- `C`: count=5, mean_top_activation=2.477
- `O`: count=5, mean_top_activation=1.117
- `NC`: count=3, mean_top_activation=1.617
- `CN`: count=2, mean_top_activation=1.999
- `CCCN`: count=1, mean_top_activation=1.324
- `cccc`: count=1, mean_top_activation=0.073
- `ccc`: count=1, mean_top_activation=0.070

## bace_classification feature 1785

- coef_mean: `1.4368`
- single_feature_roc_auc: `0.5735`

Top localized tokens per example:
- `Clc1sc(cc1C1(N=C(N)N(C)C1=O)c1cc(ccc1)-c1cncnc1)C`
  top tokens: `Clc` (-0.000), `sc` (-0.000), `cc` (-0.000), `C` (-0.000)
- `FC(F)(F)Oc1ccc(cc1)C1(N=C(N)N(C)C1=O)C12CC3CC(C1)CC(C2)C3`
  top tokens: `FC` (-0.000), `F` (-0.000), `F` (-0.000), `Oc` (-0.000)
- `Fc1c2c(ccc1)C(N=C2N)(C=1C=C(C)C(=O)N(C=1)C)c1cc(ccc1)-c1cncnc1`
  top tokens: `Fc` (-0.000), `c` (-0.000), `c` (-0.000), `ccc` (-0.000)
- `s1cc(cc1C1(N=C(N)N(C)C(=O)C1)C)-c1cc(ccc1)C#N`
  top tokens: `s` (-0.000), `cc` (-0.000), `cc` (-0.000), `C` (-0.000)
- `O=C1N(C)C(N[C@@]1(CC1CCCCC1)CCC1CCCCC1)=N`
  top tokens: `O` (-0.000), `C` (-0.000), `N` (-0.000), `C` (-0.000)
- `O=C1N(C)C(=N[C@@]12CC(Cc1c2cc(cc1)-c1cncnc1)(C)C)N`
  top tokens: `O` (-0.000), `C` (-0.000), `N` (-0.000), `C` (-0.000)

Aggregated frequent high-activation substrings:
- `cc`: count=3, mean_top_activation=0.000
- `C`: count=2, mean_top_activation=0.000
- `F`: count=2, mean_top_activation=0.000
- `N`: count=2, mean_top_activation=0.000
- `O`: count=2, mean_top_activation=0.000
- `c`: count=2, mean_top_activation=0.000
- `Clc`: count=1, mean_top_activation=0.000
- `FC`: count=1, mean_top_activation=0.000

## bbbp feature 1011

- coef_mean: `-1.0199`
- single_feature_roc_auc: `0.5706`

Top localized tokens per example:
- `c1(ccccc1)CC`
  top tokens: `c` (-0.000), `ccccc` (-0.000), `CC` (-0.000)
- `CNCCCN1c2ccccc2Sc3ccccc13`
  top tokens: `CN` (-0.000), `CCCN` (-0.000), `c` (-0.000), `ccccc` (-0.000)
- `CNCCCN1c2ccccc2CCc3ccccc13`
  top tokens: `CN` (-0.000), `CCCN` (-0.000), `c` (-0.000), `ccccc` (-0.000)
- `Cc1nccc2c1[nH]c3ccccc23`
  top tokens: `Cc` (-0.000), `nccc` (-0.000), `c` (-0.000), `nH` (-0.000)
- `CC(COc1ccccc1)N(CCCl)Cc2ccccc2`
  top tokens: `CC` (-0.000), `COc` (-0.000), `ccccc` (-0.000), `N` (-0.000)
- `[H+].[Cl-].Clc1ccc2Sc3ccccc3N(CCCN4CCC5(CC4)NC(=O)CS5)c2c1`
  top tokens: `H` (-0.000), `Cl` (-0.000), `Clc` (-0.000), `ccc` (-0.000)

Aggregated frequent high-activation substrings:
- `c`: count=4, mean_top_activation=0.000
- `CC`: count=2, mean_top_activation=0.000
- `CCCN`: count=2, mean_top_activation=0.000
- `CN`: count=2, mean_top_activation=0.000
- `ccccc`: count=2, mean_top_activation=0.000
- `COc`: count=1, mean_top_activation=0.000
- `Cc`: count=1, mean_top_activation=0.000
- `Cl`: count=1, mean_top_activation=0.000

## bbbp feature 1237

- coef_mean: `2.0021`
- single_feature_roc_auc: `0.5611`

Top localized tokens per example:
- `C1=C(C=CC(=C1)NC(C)=O)OC(C)(C)C`
  top tokens: `C` (-0.000), `C` (-0.000), `C` (-0.000), `CC` (-0.000)
- `C(OC(NC(C(Cl)(Cl)Cl)O)=O)C`
  top tokens: `C` (-0.000), `OC` (-0.000), `NC` (-0.000), `C` (-0.000)
- `C(C(NC(C)=O)C(O)=O)CC(N)=O`
  top tokens: `C` (-0.000), `C` (-0.000), `NC` (-0.000), `C` (-0.000)
- `Nc1ncnc2n(cnc12)C3OC(CO)C(O)C3O`
  top tokens: `Nc` (-0.000), `ncnc` (-0.000), `n` (-0.000), `cnc` (-0.000)
- `CN1C(CNC(=O)c2cscc2)CN=C(c3ccccc3F)c4ccccc14`
  top tokens: `CN` (-0.000), `C` (-0.000), `CNC` (-0.000), `O` (-0.000)
- `CN1C(=O)N(C)c2nc[nH]c2C1=O.CN3C(=O)N(C)c4nc[nH]c4C3=O.NCCN`
  top tokens: `CN` (-0.000), `C` (-0.000), `O` (-0.000), `N` (-0.000)

Aggregated frequent high-activation substrings:
- `C`: count=8, mean_top_activation=0.000
- `CN`: count=2, mean_top_activation=0.000
- `NC`: count=2, mean_top_activation=0.000
- `CNC`: count=1, mean_top_activation=0.000
- `Nc`: count=1, mean_top_activation=0.000
- `O`: count=1, mean_top_activation=0.000
- `OC`: count=1, mean_top_activation=0.000
- `n`: count=1, mean_top_activation=0.000

## bbbp feature 1492

- coef_mean: `1.2254`
- single_feature_roc_auc: `0.5760`

Top localized tokens per example:
- `C1CC1`
  top tokens: `C` (-0.000), `CC` (-0.000)
- `Cc1ncsc1CCCl`
  top tokens: `Cc` (-0.000), `ncsc` (-0.000), `CCCl` (-0.000)
- `C1CCCCC1`
  top tokens: `C` (-0.000), `CCCCC` (-0.000)
- `CC1CCCC1`
  top tokens: `CC` (-0.000), `CCCC` (-0.000)
- `CCC1(C)CC(=O)NC1=O`
  top tokens: `CCC` (-0.000), `C` (-0.000), `CC` (-0.000), `O` (-0.000)
- `NC1CONC1=O`
  top tokens: `NC` (-0.000), `CONC` (-0.000), `O` (-0.000)

Aggregated frequent high-activation substrings:
- `C`: count=3, mean_top_activation=0.000
- `CC`: count=3, mean_top_activation=0.000
- `CCC`: count=1, mean_top_activation=0.000
- `CCCC`: count=1, mean_top_activation=0.000
- `CCCCC`: count=1, mean_top_activation=0.000
- `CCCl`: count=1, mean_top_activation=0.000
- `CONC`: count=1, mean_top_activation=0.000
- `Cc`: count=1, mean_top_activation=0.000

