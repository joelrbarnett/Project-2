%
% This function just initializes an array of random weight matrices 
% Input: m, the number of layers in the network
%       n, the number of neurons in each layer (same dimension as input)
% Output: a cell array with randomly weighted matrices of given dimensions.

function W=weightInit(m,n)
    W=cell(m,1);
    for i=1:m
        %creates random nxn matrix where values are normally distributed 
        % with mean 0 and stdDev=1
        W{i}=random('norm', 0, 1, n);
    end
end