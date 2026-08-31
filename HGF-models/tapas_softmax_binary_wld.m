function [logp, yhat, res] = tapas_softmax_binary_wld(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the binary softmax model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2016 Christoph Mathys, TNU, UZH & ETHZ

% Adapted by D.V. Atanassova 2025: softmax function with win/loss distortion for binary outcomes
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

% Transform zeta to its native space
be = exp(ptrans(1));

% Win- and loss-distortion parameters
la_wd = ptrans(2);
la_ld = ptrans(3);

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Load the input and choice
u = r.u;
y = r.y;

% Check input format
if size(r.u,2) ~= 1 && size(r.u,2) ~= 3
    error('tapas:hgf:SoftMaxBinary:InputsIncompatible', 'Inputs incompatible with tapas_softmax_binary observation model. See tapas_softmax_binary_config.m.')
end

% Belief trajectories at 1st level
states = infStates(:,1,pop);

% Weed irregular trials out from inferred states, inputs, and responses
states(r.irr,:) = [];
u(r.irr) = [];
y(r.irr) = [];

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

% Make states in the format used below
x = states;

%%
if size(r.u,2) == 3
    r0 = r.u(:,2);
    r0(r.irr) = [];
    r1 = r.u(:,3);
    r1(r.irr) = [];
end

% Calculate log-probabilities for non-irregular trials
% If input matrix has only one column, assume the weight (reward value)
% of both options is equal to 1
if size(r.u,2) == 1
    % Probability of observed choice
    probc = 1./(1+exp(-be.*(2.*x-1).*(2.*y-1)));
end
% If input matrix has three columns, the second contains the weights of
% outcome 0 and the third contains the weights of outcome 1
if size(r.u,2) == 3
    % Probability of observed choice
    probc = 1./(1+exp(-be.*(r1.*x-r0.*(1-x)).*(2.*y-1)));
end

reg = ~ismember(1:n,r.irr);
logp(reg) = log(probc);
yh = y.*probc +(1-y).*(1-probc);
yhat(reg) = yh;
res(reg) = (y -yh)./sqrt(yh.*(1 -yh));

return;
