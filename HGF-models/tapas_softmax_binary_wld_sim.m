function y = tapas_softmax_binary_wld_sim(r, infStates, p)
% Simulates observations from a Bernoulli distribution
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
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

% Inverse decision temperature beta
be = p;

% Win- and loss-distortion parameters
la_wd = p(1);
la_ld = p(2);

% Assumed structure of infStates:
% dim 1: time (ie, input sequence number)
% dim 2: HGF level
% dim 3: 1: muhat, 2: sahat, 3: mu, 4: sa

% Belief trajectories at 1st level
states = infStates(:,1,pop);

% true choces
y = r.u(:,2);

% input
u = r.u(:,1);

%% Distortion by previous loss/win
% Choice matrix Y corresponding to states
Y(:,1) = zeros(size(states));
Y(:,2) = zeros(size(states));

% Alter the mapping (binary to 1&2)
y_remapped = y + 1;
Y(sub2ind(size(Y), 1:size(Y, 1), y_remapped')) = 1;

% Drop the second column (since on paired tasks, outcome on option 1 = 1 -
% outcome on option 2)
Y = Y(:,1);

% Choice on previous trial
Yprev = Y;
Yprev = [zeros(1,size(Yprev,2)); Yprev];
Yprev(end,:) = [];

% Wins on previous trial
wprev = u;
wprev = [0; wprev];
wprev(end) = [];

% Losses on previous trial
lprev = 1 - wprev;
lprev(1) = 0;

% In matrix form corresponding to states
Wprev = Yprev;
Wprev(find(lprev),:) = 0;
Lprev = Yprev;
Lprev(find(wprev),:) = 0;

% Win- and loss-distortion
states = states + la_wd*Wprev + la_ld*Lprev;

%%

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
