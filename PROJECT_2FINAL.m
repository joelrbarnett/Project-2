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
trainData = {train0,train1,train2,train3,train4,train5,train6,train7,train8,train9};
testData = {test0,test1,test2,test3,test4,test5,test6,test7,test8,test9};
perRight = zeros(1,10);
score=0;

saveOut = zeros(10,10);


% TRAIN THE NETWORK

for numI=1:10 %Iterate through the sets of training data 0-9
  [numD, numPix] = size(trainData{numI});
  [numT, numPix] = size(testData{numI});
  data = trainData{numI};     % Counts through training data numbers
  dataT = testData{numI};     % Counts through test data numbers
  %TRAIN LOOP 
  for countD=1:1000%numD
    inC{1} = im2double(data(countD,:));  % Counts through train_n entries

    % Forward
    for countLayer=1:n-1
      inC{countLayer+1} = neuron(inC{countLayer},wiC{countLayer});
    end

    % Backward
    
    % OUTPUT LAYER
    ERROR = TARGET(numI,:)-inC{n};
    deltaC{n-1} = inC{n}.*(1-inC{n}).*ERROR;
    dw{n-1} = tR.*(deltaC{n-1}'*inC{n-1})';
    wiC{n-1} = wiC{n-1} + dw{n-1};
    % HIDDEN LAYERS
    for i=n-2:-1:1
      deltaC{i} = (deltaC{i+1}*wiC{i+1}').*(inC{i+1}.*(1-inC{i+1}));
      dw{i} = tR.*(deltaC{i}'*inC{i})';
      wiC{i} = wiC{i}+dw{i};
    end

    for i=1:n-1
      wiC{i} = wiC{i}+dw{i};
    end
    if countD==1000%numD
      saveOut(numI,:) = inC{n};
    end
    
  end
  score=0;
  for countT=1:numT
    inC{1} = im2double(dataT(countT,:));
    
    for countLayer=1:n-1
      inC{countLayer+1} = neuron(inC{countLayer},wiC{countLayer});
    end
    A = inC{n};
    [x, y]=max(A(1,:));
    if y==numI
      score = score+1;
    end
  end
  perRight(1,numI) = 1-(score/numT);
end


disp('done')



