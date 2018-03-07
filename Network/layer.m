%%
% function layer.m creates a layer in a neural network
% Input:I, the input vector, this will determine the # of neurons/layer
%       W, weight matrix, must be same dimension as I
% Output: A column vector Out holding neurons in each element. (will be the input for
% a neuron in the next layer).
%
function Out = layer(I,W)
    n=length(I);
    L=W*I; %This sets the weighted sum to each corresponding intry in L
    %take the sigmoidal of each entry in Layer
    for i=1:n
        L(i,1)=1/(1+exp(-L(i,1))); 
    end
    Out=L;
end
