clear all
clc

load('hw1data.mat')
X = hw1x;
y = hw1y;
[m, n] = size(X);

[trainInd,valInd,testInd] = dividerand(size(X,1),0.6,0.2,0.2);

data = [X,y];

training = data(trainInd', :);
validation = data(valInd', :);
test = data(testInd', :);

x_train = training(:, 1:end-1);
y_train = training(:, end);

x_valid = validation(:, 1:end-1);
y_valid = validation(:, end);

x_test = test(:, 1:end-1);
y_test = test(:, end);

iterations = 15000;
alpha = 0.009;


kernel_dim = [1, 2, 3]';


Training_log_likelihood = zeros(iterations,length(kernel_dim));
Testing_log_likelihood = zeros(iterations,length(kernel_dim));
Validation_log_likelihood = zeros(iterations,length(kernel_dim));

val = 5;
Validation_Cost_Train = zeros(iterations, val);
Validation_Cost_Validation = zeros(iterations, val);
Validation_Cost_Test = zeros(iterations, val);


    for j = 1:size(kernel_dim,1)

        Kernel = zeros(size(X, 1), size(X,1));
        Kernel = (X * X' + 1).^kernel_dim(j);
                
        data = [Kernel, y];
        
        [m, n] = size(Kernel);
        w = zeros(size(Kernel,2), 1);
        
        training = data(trainInd', :);
        validation = data(valInd', :);
        test = data(testInd', :);
        
        train_validation_data = [training; validation];
        test = [test; train_validation_data(end, :)];
        train_validation_data = train_validation_data(1:100, :);

        x_test = test(:, 1:end-1);
        y_test = test(:, end);
        
        [crossValidation_Training, crossValidation_Validation] = cross_val(train_validation_data);

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
            w = w - alpha*grad_d;

            Cost_Train =   (-1/length(y_train)) *  sum(   (y_train .* log10(1 ./ (1 + exp(- x_train * w))))     +   (1-y_train) .* log10(1 - (1 ./ (1 + exp(- x_train * w)))))     ;
            Validation_Cost_Train(i, v) = Cost_Train;
            Training_log_likelihood(i,j) = mean(Validation_Cost_Train(i, :));   

            Cost_Valid =   (-1/length(y_valid))  *    sum(   (y_valid .* log10(1 ./ (1 + exp(- x_valid * w))))     +   (1-y_valid) .* log10(1 - (1 ./ (1 + exp(- x_valid * w))))    );
            Validation_Cost_Validation(i, v) = Cost_Valid;
            Validation_log_likelihood(i,j) = mean(Validation_Cost_Validation(i, :));  
            
            Cost_Test =   (-1/length(y_test)) *  sum(   (y_test .* log10(1 ./ (1 + exp(- x_test * w))))     +   (1-y_test) .* log10(1 - (1 ./ (1 + exp(- x_test * w)))))     ;
            Validation_Cost_Test(i, v) = Cost_Test;
            Testing_log_likelihood(i,j) = mean(Validation_Cost_Test(i, :));    

        end

        end
    end
   

 
    
    
