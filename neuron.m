
% Project 2
% Part iii: A NEURON

%INPUT:
  % 1. o1,...,oi,...,on where each oi is either a data point or input from
  % other neurons.
  % 2. w1,...,wi,...,wn where each wi is the input weights for each
  % corresponding data point.

% OUTPUT: 
  % OUT = F(NET) = 1/(1+e^(-NET)) where NET = sum of the weights multiplied
  % by their corresponding data inputs.
function [ OUT ] = neuron(oi,wi)
NET = oi*wi;

OUT = 1./(1+exp(-1*NET));
end
