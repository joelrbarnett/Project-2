%%
% Here we initialize a sample Network

%First, we set up our random weights between layers. If we want a 3 layer
%system, with 5 neurons in each layer, we do the following:
W=weightInit(3,5);

%Then we create a 5-vector for the initial input.
I= [1 2 3 4 5]'; %Notice, we make it a column vector

%Running the Network a single pass gives:
network(I,W)
