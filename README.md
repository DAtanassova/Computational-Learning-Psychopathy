# Computational-Learning-Psychopathy

This repository contains computational models developed and adapted to investigate learning and decision-making in relation to psychopathic traits across different populations.

**Models**
- **ehgf_ar1_binary_mab**
An implementation of the ehgf_ar1_binary model from the HGF tapas toolbox (https://github.com/translationalneuromodeling/tapas) adapted for multi-armed bandit (MAB) configurations. This model was used in Atanassova, D.V., Oosterman, J.M., Diaconescu, A.O. et al. Exploring when to exploit: the cognitive underpinnings of foraging-type decisions in relation to psychopathy. Transl Psychiatry 15, 31 (2025). https://doi.org/10.1038/s41398-025-03245-2

- **softmax_binary_wld**
An adaptation of the softmax_wld model from the HGF tapas toolbox for binary outcomes. This implements a Win-Loss Distortion (WLD) model for modeling decision-making in binary-outcome environments.

- **softmax_binary_mu3**
An adaptation of the softmax_mu3 model from the HGF tapas toolbox for binary outcomes. Binary decisions are modelled based on the trial-wise mu3 (meta-volatility)
estimates as decision temperature. 

Additional models and adaptations will be added as they are developed and used across projects.

