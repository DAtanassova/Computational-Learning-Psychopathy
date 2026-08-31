function y = tapas_softmax_binary_mu3_sim(r, infStates, p)
% Simulates observations from a Boltzmann distribution with volatility as temperature
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2017-2019 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Predictions or posteriors?
pop = 1; % Default: predictions
if r.c_obs.predorpost == 2
    pop = 3; % Alternative: posteriors
end

% Log-volatility trajectory
mu3 = squeeze(infStates(:,3,3));

% Inverse decision temperature
be = exp(-mu3);

% Belief trajectories at 1st level
states = squeeze(infStates(:,1,pop));

% Apply the logistic sigmoid to the inferred states
prob = tapas_sgm(be.*(2.*states-1),1);

% Initialize random number generator
if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end

% Simulate
y = binornd(1, prob);

end
