% AUTHOR: DOUGLAS KUBOTA
% PROJECT 2 FINAL
clear,clc
% Part i
% Loads data
load mnist_all.mat;

%{
% Part ii
% Plot examples from database
showdigit(train1(1101,:));
figure
showdigit(train3(4725,:));
figure
showdigit(train8(381,:));

% Calculate the average digit
avgDigit = zeros(10,784);
avgDigit(1,:)  = getAvg(train1);
avgDigit(2,:)  = getAvg(train2);
avgDigit(3,:)  = getAvg(train3);
avgDigit(4,:)  = getAvg(train4);
avgDigit(5,:)  = getAvg(train5);
avgDigit(6,:)  = getAvg(train6);
avgDigit(7,:)  = getAvg(train7);
avgDigit(8,:)  = getAvg(train8);
avgDigit(9,:)  = getAvg(train9);
avgDigit(10,:) = getAvg(train0);
%}

% PART iii
% See neuron.m

% PART iv
% Set up multilayer network
tR = 0.05;                  % Value between [0.01,1]
layers = [784 50 50 50 50 10];   % 1st entry is input layer size, middle are 
                                 % are hidden layer size, last is output size.
n = length(layers);
  % Set up input matrices
  for i=1:n
    inC{i} = zeros(1,layers(1,i));
  end

  % Set up delta matrices
  for i=1:n-1
    deltaC{i} = zeros(1,layers(i));
  end
  
  % Set up weight matrices
  for i=1:n-1
    wiC{i} = rand(layers(1,i),layers(1,i+1));
  end
  
  % Set up output matrices
  for i=1:n-1
    outC{i} = zeros(1,layers(1,i+1));
  end
  
TARGET = eye(10);
% Set up training data  
% Creates a cell filled with all of the training data from MNIST database.
trainData = {train0,train1,train2,train3,train4,train5,train6,train7,train8,train9};
% Creates a cell filled with all of the testing data from the MNIST database.
testData = {test0,test1,test2,test3,test4,test5,test6,test7,test8,test9};

saveOut = zeros(10,10); % Creates a matrix of zeros to store the last entry from each set of training data.


% TRAIN THE NETWORK

for numI=1:10 %Iterate through the sets of training data 0-9
  [numD, numPix] = size(trainData{numI}); % numD is the total number of data to train for each cell of trainData.
  [numT, numPix] = size(testData{numI});  % numT is the total number of data to test for each cell of testData.
  data = trainData{numI};     % Counts through training data numbers
  dataT = testData{numI};     % Counts through test data numbers
  %TRAIN LOOP 
  for countD=1:numD           % Counts throught total number of data points.
    inC{1} = im2double(data(countD,:));  % Counts through train_n entries.
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Forward
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Iterates through the layers storing the output as the next input.
    for countLayer=1:n-1  
      inC{countLayer+1} = neuron(inC{countLayer},wiC{countLayer});
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BACKWARD PROPAGATION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % OUTPUT LAYER
    % Calculates the change in weight for the output layer.
    ERROR = TARGET(numI,:)-inC{n};
    deltaC{n-1} = inC{n}.*(1-inC{n}).*ERROR;
    dw{n-1} = tR.*(deltaC{n-1}'*inC{n-1})';
    wiC{n-1} = wiC{n-1} + dw{n-1};
    
    % HIDDEN LAYERS
    % Calculates the change in weight for the hidden layers.
    for i=n-2:-1:1
      deltaC{i} = (deltaC{i+1}*wiC{i+1}').*(inC{i+1}.*(1-inC{i+1}));
      dw{i} = tR.*(deltaC{i}'*inC{i})';
      wiC{i} = wiC{i}+dw{i};
    end
  
    % Saves the last entry of the training data per digit.
    if countD==numD
      saveOut(numI,:) = inC{n}; % Save output for last entry of each didgit
    end
    
  end

end


disp('done')
