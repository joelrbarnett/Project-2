%%
% function Network creates a network of given an intial
% input and intial Weights
% Input: W, cell array of weight matrices, dimensions [m,1], m= # layers
%         To access each weight matrix i, we use W{i}. 
%        I, Input vector;
% Output: the final output vector

function Out=network(I, W)
    m=length(W);
    Out=layer(I,W{1});      %Note: we may not need different weights for each step
                            %depending on how backpropogation works. 
                            %For now, there is a different weighting matrix between each layer.                           
    for i=2:m
        Out=layer(Out,W{i});
    end
end
        