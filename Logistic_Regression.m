clear all
clc

%PART 1

%question a
load('hw1data.mat')
X = hw1x;
y = hw1y;

% X = X(:, 4:end);
% X = [ones(size(X,1),1), X];

% [m,n] = size(X);
% 
% gscatter(X(:,100), X(:,101), y, 'rb', '*o', ...
%      10, 'on', 'X Feature 100', 'X Feature 105')
% grid on
% title('Data Set for Logistic Classification')
%  
% fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
%           'indicating (y = 0) examples.\n']);
%       
% plotData(X, y);



%Data visualisation in Matlab
% figure
% X = hw1x;
% X = X(:, 1:6);
% gplotmatrix(X,[],y,['c' 'b' 'm' 'g' 'r'],[],[],false);
% 


%Dimensionality Reduction technique : PCA
 X = pca(X');
 X = [ones(size(X,1),1), X];


%question b
% X = X(:, 14:16);
[m,n] = size(X);

[trainInd,valInd,testInd] = dividerand(size(X,1),0.6,0.2,0.2);

data = [X,y];

training = data(trainInd', :);
validation = data(valInd', :);
test = data(testInd', :);

train_validation_data = [training; validation];
test = [test; train_validation_data(end, :)];
train_validation_data = train_validation_data(1:100, :);

% 
% x_train = training(:, 1:end-1);
% y_train = training(:, end);
% 
% x_valid = validation(:, 1:end-1);
% y_valid = validation(:, end);
% 
x_test = test(:, 1:end-1);
y_test = test(:, end);


[crossValidation_Training, crossValidation_Validation] = cross_val(train_validation_data);

%question c
%objective function - including the regularizer

%question d

alpha = 0.00004;
iterations = 15000;
lambda = [0, 0.1, 1, 10, 100, 1000]';
[m,n] = size(X);


Train_Obj_Lambda = zeros(length(lambda), 1);
Valid_Obj_Lambda = zeros(length(lambda), 1);
Test_Obj_Lambda = zeros(length(lambda), 1);

Training_log_likelihood = zeros(iterations,length(lambda));
Testing_log_likelihood = zeros(iterations,length(lambda));
Validation_log_likelihood = zeros(iterations,length(lambda));

l2_norm = zeros(length(lambda), 1);
weight_plot = zeros(size(X,2), length(lambda));

val = 5;
Validation_Cost_Train = zeros(iterations, val);
Validation_Cost_Validation = zeros(iterations, val);
Validation_Cost_Test = zeros(iterations, val);

for j = 1:size(lambda,1)
    
    w = zeros(n, 1);

    %may need to perform k-fold cross validation here?
    for v = 1:5
        if v ==1
            training = crossValidation_Training(1:80, :);
            validation = crossValidation_Validation(1:20, :);
        elseif v==2
            training = crossValidation_Training(81:160, :);
            validation = crossValidation_Validation(21:40, :);
        elseif v==3
            training = crossValidation_Training(161:240, :);
            validation = crossValidation_Validation(41:60, :);
        elseif v==4
            training = crossValidation_Training(241:320, :);
            validation = crossValidation_Validation(61:80, :);
        elseif v==5
            training = crossValidation_Training(321:400, :);
            validation = crossValidation_Validation(81:100, :);
            
        end 
         
        x_train = training(:, 1:end-1);
        y_train = training(:, end);

        x_valid = validation(:, 1:end-1);
        y_valid = validation(:, end);


        for i = 1:iterations 

            sigmoid = (1 ./ (1 + exp(- (x_train * w))));

            %update rule of gradient descent
            grad_d = x_train' * (sigmoid - y_train);
            grad_w = lambda(j)*w;
            w = w - alpha*grad_d - alpha*grad_w;

            Cost_Train =   (-1/length(y_train)) *  sum(   (y_train .* log10(1 ./ (1 + exp(- x_train * w))))     +   (1-y_train) .* log10(1 - (1 ./ (1 + exp(- x_train * w)))))     ;
            Validation_Cost_Train(i, v) = Cost_Train;
            Training_log_likelihood(i,j) = mean(Validation_Cost_Train(i, :));   
            Train_Obj_Lambda(j,:) = mean(Validation_Cost_Train(i,:));

            Cost_Valid =   (-1/length(y_valid))  *    sum(   (y_valid .* log10(1 ./ (1 + exp(- x_valid * w))))     +   (1-y_valid) .* log10(1 - (1 ./ (1 + exp(- x_valid * w))))    );
            Validation_Cost_Validation(i, v) = Cost_Valid;
            Validation_log_likelihood(i,j) = mean(Validation_Cost_Validation(i, :));  
            Valid_Obj_Lambda(j,:) = mean(Validation_Cost_Validation(i,:));

            Cost_Test =   (-1/length(y_test)) *  sum(   (y_test .* log10(1 ./ (1 + exp(- x_test * w))))     +   (1-y_test) .* log10(1 - (1 ./ (1 + exp(- x_test * w)))))     ;
            Validation_Cost_Test(i, v) = Cost_Test;
            Testing_log_likelihood(i,j) = mean(Validation_Cost_Test(i, :));    
            Test_Obj_Lambda(j,:) = mean(Validation_Cost_Test(i,:));

        end

        %l2_norm(j, :) = lambda(j) * w' * w;
        l2_norm(j, :) = norm(w);
        weight_plot(:, j) = w;


    end
    
end


% log_l = log10(lambda(2:end));
log_lambda = log10(lambda);

%First Plot
% plot(log_lambda, Train_Obj_Lambda, 'r'); hold on;
% plot(log_lambda, Test_Obj_Lambda, 'k'); hold on;
plot(log_lambda, Train_Obj_Lambda, 'r'); hold on;
plot(log_lambda, Valid_Obj_Lambda, 'b'); 
plot(log_lambda, Test_Obj_Lambda, 'k'); 
xlabel('Log Lambda', 'FontSize', 16);
ylabel('Error Rate','FontSize', 16)
title('Training, Validation and  Test Error as Function of Lambda','FontSize',16)
legend('Training Error', 'Validation Error ','Test Error')

%Second Plot
plot(1:length(lambda), l2_norm, 'k');
xlabel('Lambda Values', 'FontSize', 16)
ylabel('L2 Norm of Weight Vecotr', 'FontSize', 16)
legend('L2 Norm as Function of Lambda', 'FontSize',16)
title('L2 Norm of Weight Vector', 'FontSize',16)

%Third Plot
plot([1:size(weight_plot, 1)], weight_plot(:, 1), 'k', [1:size(weight_plot, 1)], weight_plot(:, 2), 'r', [1:size(weight_plot, 1)], weight_plot(:, 3), 'b', [1:size(weight_plot, 1)], weight_plot(:, 4), 'green', [1:size(weight_plot, 1)], weight_plot(:, 5), 'yellow', [1:size(weight_plot, 1)], weight_plot(:, 6), 'cyan' )
legend('Lambda=0', 'Lambda=0.1','Lambda=1','Lambda=10','Lambda=100','Lambda=1000', 'FontSize',16)
xlabel('Number of Samples', 'FontSize', 16)
ylabel('Actual Weight Values', 'FontSize', 16)
title('Actual Weight Values per Regularisation Parameter Lambda', 'FontSize',16)


%intermediate
plot([1:iterations],Training_log_likelihood(:,1), 'red', [1:iterations],Training_log_likelihood(:,2), 'black', [1:iterations],Training_log_likelihood(:,3), 'blue', [1:iterations],Training_log_likelihood(:,4), 'green', [1:iterations],Training_log_likelihood(:,4), 'yellow', [1:iterations],Training_log_likelihood(:,6), 'cyan', [1:iterations],Validation_log_likelihood(:,1), 'red*', [1:iterations],Validation_log_likelihood(:,2), 'black*', [1:iterations],Validation_log_likelihood(:,3), 'blue*', [1:iterations],Validation_log_likelihood(:,4), 'green*', [1:iterations],Validation_log_likelihood(:,4), 'yellow*', [1:iterations],Validation_log_likelihood(:,6), 'cyan*'); 
legend('Training, Lambda=0', 'Training, Lambda=0.1','Training, Lambda=1','Training, Lambda=10','Training, Lambda=100','Training, Lambda=1000',  'Validation, Lambda=0', 'Validation, Lambda=0.1','Validation, Lambda=1','Validation, Lambda=10','Validation, Lambda=100','Validation, Lambda=1000')
xlabel('Number of Iterations', 'FontSize', 16)
ylabel('Training and Validation Error', 'FontSize', 16)
title('Accuracy on Training and Validation Set', 'FontSize', 16)


%Fourth Plot
plot([1:iterations],Training_log_likelihood(:,1), 'red', [1:iterations],Training_log_likelihood(:,2), 'black', [1:iterations],Training_log_likelihood(:,3), 'blue', [1:iterations],Training_log_likelihood(:,4), 'green', [1:iterations],Training_log_likelihood(:,4), 'yellow', [1:iterations],Training_log_likelihood(:,6), 'cyan', [1:iterations],Testing_log_likelihood(:,1), 'red*', [1:iterations],Testing_log_likelihood(:,2), 'black*', [1:iterations],Testing_log_likelihood(:,3), 'blue*', [1:iterations],Testing_log_likelihood(:,4), 'green*', [1:iterations],Testing_log_likelihood(:,4), 'yellow*', [1:iterations],Testing_log_likelihood(:,6), 'cyan*'); 
legend('Training, Lambda=0', 'Training, Lambda=0.1','Training, Lambda=1','Training, Lambda=10','Training, Lambda=100','Training, Lambda=1000',  'Testing, Lambda=0', 'Testing, Lambda=0.1','Testing, Lambda=1','Testing, Lambda=10','Testing, Lambda=100','Testing, Lambda=1000')
xlabel('Number of Iterations', 'FontSize', 16)
ylabel('Training and Test Error', 'FontSize', 16)
title('Accuracy on Training and Test Set', 'FontSize', 16)



TR = 1 - Training_log_likelihood(:, 3);
TE = 1 - Testing_log_likelihood(:, 3);


plot( [1:iterations],TR, 'blue', [1:iterations],TE, 'blue*'); 
legend('Training, Lambda=1','Testing, Lambda=0', 'Testing, Lambda=1')
xlabel('Number of Iterations', 'FontSize', 16)
ylabel('Training and Test Error', 'FontSize', 16)
title('Accuracy on Training and Test Set', 'FontSize', 16)

















