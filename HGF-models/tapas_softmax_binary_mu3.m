function [logp, yhat, res] = tapas_softmax_binary_mu3(r, infStates, ptrans)
% Calculates the log-probability of responses under the softmax model
%
% Adapted by D.V. Atanassova (2024): models decisions under the softmax model for binary choices, 
% using trial-wise mu3 as decision temperature 
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

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Assumed structure of infStates:
% dim 1: time (ie, input sequence number)
% dim 2: HGF level
% dim 3: choice number
% dim 4: 1: muhat, 2: sahat, 3: mu, 4: sa

% Check input format
if size(r.u,2) ~= 1 && size(r.u,2) ~= 3
    error('tapas:hgf:SoftMaxBinary:InputsIncompatible', 'Inputs incompatible with tapas_softmax_binary observation model. See tapas_softmax_binary_config.m.')
end

% Log-volatility trajectory
mu3 = squeeze(infStates(:,3,3));

% Inverse decision temperature
be = exp(-mu3);

% Weed irregular trials out from inferred states, responses, and inputs
x = squeeze(infStates(:,1,pop));
x(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];
be(r.irr) = [];

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
