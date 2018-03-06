% Playing around with the mnist database. This file prints the average
% training digit image, and also uses a crude method of comparing vector
% norms to see try to determine a digit (done in function testDigit).
%
clc
clear
load('mnist_all.mat');
%Sets up T to hold the digits used to train
T(1,:)=mean(train0); T(2,:)=mean(train1); T(3,:)= mean(train2); T(4,:)=mean(train3);
T(5,:)=mean(train4); T(6,:)=mean(train5); T(7,:)=mean(train6); T(8,:)=mean(train7); 
T(9,:)=mean(train8); T(10,:)=mean(train9);

%plot the 10 digits as is done in problem 7.17 in textbook 
for i=1:10
subplot(2,5,i);
digitImage = reshape(T(i,:),28,28);
image(rot90(flipud(digitImage),-1));
colormap(gray(256)), axis square tight off;
end

%Here, we try a couple different values from the test{_} vector to see if
%our testDigit function accurately determines the digit d. Change the
%test_(j,:) value to be any digit 0-9.
for j=1:12
    d=double(test0(j,:));
    testDigit(d,T)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function testDigit takes in a digit vector to test d, and compares it with a 
% known test reference T, which has a sample digit for each of 0,1, ... , 9
% Input: d (a 1x784 vector) and T (a 10x784 matrix)
% Out: A digit from 0-9, which is the best (smallest norm difference)
% approx of digit d. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function n=testDigit(d,T)
    error= zeros(10,1); % error vector
    m=0; n=0; 
    for i=1:10 % check the difference between d and test digits 0-9
        error(i,1)=norm(d-T(i,:),2);
    end
    [m ,n]=min(error); %return the position of the minimum norm-difference
    n=n-1;              % digit is most likely n-1.
    
end
        