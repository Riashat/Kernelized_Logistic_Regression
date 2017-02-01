load('hw1data.mat')
X = hw1x;
y = hw1y;

[m,n] = size(X);
% X = pca(X');

[trainInd,valInd,testInd] = dividerand(size(X,1),0.6,0.2,0.2);

alpha = 0.0004;
iterations = 15000;


sigma = 1;

c1 = repmat(-10, size(X, 1), size(X, 2));
basis_function1 = exp(  - (  ( X-c1 ).^2 ./ (2 * sigma.^2)    )   );

c2 = repmat(-5, size(X, 1), size(X, 2));
basis_function2 = exp(  - (  ( X-c2 ).^2 ./ (2 * sigma.^2)    )   );

c3 = repmat(0, size(X, 1), size(X, 2));
basis_function3 = exp(  - (  ( X-c3 ).^2 ./ (2 * sigma.^2)    )   );

c4 = repmat(5, size(X, 1), size(X, 2));
basis_function4 = exp(  - (  ( X-c4 ).^2 ./ (2 * sigma.^2)    )   );

c5 = repmat(10, size(X, 1), size(X, 2));
basis_function5 = exp(  - (  ( X-c5 ).^2 ./ (2 * sigma.^2)    )   );


%adding all the basis functions together
Feature_Map_Input = basis_function1 + basis_function2 + basis_function3 + basis_function4 + basis_function5;

data = [Feature_Map_Input,y];
training = data(trainInd', :);
validation = data(valInd', :);
test = data(testInd', :);

x_train = training(:, 1:end-1);
y_train = training(:, end);
x_valid = validation(:, 1:end-1);
y_valid = validation(:, end);
x_test = test(:, 1:end-1);
y_test = test(:, end);

%CORRECT THE LAMBDA VALUES LATER
lambda = [0, 0.1, 1, 10, 100, 1000, 10000]';

Train_Obj_Lambda = zeros(length(lambda), 1);
Valid_Obj_Lambda = zeros(length(lambda), 1);
Test_Obj_Lambda = zeros(length(lambda), 1);
Training_log_likelihood = zeros(iterations,length(lambda));
Testing_log_likelihood = zeros(iterations,length(lambda));
Validation_log_likelihood = zeros(iterations,length(lambda));

l2_norm = zeros(length(lambda), 1);
weight_plot = zeros(size(X,2), length(lambda));

sigma = [0.1, 0.5, 1, 5, 10]';
l2_norm_sigma = zeros(length(lambda), length(sigma));

for k = 1:size(sigma);
    c1 = repmat(-10, size(X, 1), size(X, 2));
    basis_function1 = exp(  - (  ( X-c1 ).^2 ./ (2 * sigma(k).^2)    )   );

    c2 = repmat(-5, size(X, 1), size(X, 2));
    basis_function2 = exp(  - (  ( X-c2 ).^2 ./ (2 * sigma(k).^2)    )   );

    c3 = repmat(0, size(X, 1), size(X, 2));
    basis_function3 = exp(  - (  ( X-c3 ).^2 ./ (2 * sigma(k).^2)    )   );

    c4 = repmat(5, size(X, 1), size(X, 2));
    basis_function4 = exp(  - (  ( X-c4 ).^2 ./ (2 * sigma(k).^2)    )   );

    c5 = repmat(10, size(X, 1), size(X, 2));
    basis_function5 = exp(  - (  ( X-c5 ).^2 ./ (2 * sigma(k).^2)    )   );

    %adding all the basis functions together
    Feature_Map_Input = basis_function1 + basis_function2 + basis_function3 + basis_function4 + basis_function5;
        
    data = [Feature_Map_Input,y];
    training = data(trainInd', :);
    validation = data(valInd', :);
    test = data(testInd', :);

    x_train = training(:, 1:end-1);
    y_train = training(:, end);
    x_valid = validation(:, 1:end-1);
    y_valid = validation(:, end);
    x_test = test(:, 1:end-1);
    y_test = test(:, end);

    for j = 1:size(lambda,1)

        [m,n] = size(Feature_Map_Input);
        w = zeros(n, 1);

        for i = 1:iterations 

            sigmoid = (1 ./ (1 + exp(- (x_train * w))));

            %update rule of gradient descent
            grad_d = x_train' * (sigmoid - y_train);
            grad_w = lambda(j)*w;
            w = w - alpha*grad_d - alpha*grad_w;

            Cost_Train =   (-1/length(y_train))  *    sum(   (y_train .* log(1 ./ (1 + exp(- x_train * w))))     +   (1-y_train) .* log(1 - (1 ./ (1 + exp(- x_train * w))))  ) ;
            Training_log_likelihood(i,j) = Cost_Train;   
            Train_Obj_Lambda(j,:) = Cost_Train;


            Cost_Valid =   (-1/length(y_valid))  *    sum(   (y_valid .* log(1 ./ (1 + exp(- x_valid * w))))     +   (1-y_valid) .* log(1 - (1 ./ (1 + exp(- x_valid * w))))      );
            Validation_log_likelihood(i,j) = Cost_Valid; 
            Valid_Obj_Lambda(j,:) = Cost_Valid;

            Cost_Test =  (-1/length(y_test) ) *      sum(   (y_test .* log(1 ./ (1 + exp(- x_test * w))))     +   ((1-y_test) .* log(1 - (1 ./ (1 + exp(- x_test * w)))))   );
            Testing_log_likelihood(i,j) = Cost_Test;
            Test_Obj_Lambda(j,:) = Cost_Test;

        end

        l2_norm(j, :) = norm(w);
        l2_norm_sigma(j, k) = norm(w);
    end
end



log_lambda = log10(lambda);



% %First Plot
plot(log_lambda, Train_Obj_Lambda, 'r', log_lambda, Valid_Obj_Lambda, 'k', log_lambda, Test_Obj_Lambda, 'b' ); 
legend('Training Error', 'Validation Error', 'Test Error', 'FontSize', 16)
xlabel('Log Lambda Values', 'FontSize', 16)
ylabel('Error Rate', 'FontSize', 16)



plot([1:iterations], Training_log_likelihood(:,1), 'r', [1:iterations], Training_log_likelihood(:,2), 'k',[1:iterations], Training_log_likelihood(:,3), 'b',[1:iterations], Training_log_likelihood(:,4), 'green',[1:iterations], Training_log_likelihood(:,5), 'yellow',  [1:iterations], Training_log_likelihood(:,6), 'cyan',  [1:iterations], Validation_log_likelihood(:,1), 'r*', [1:iterations], Validation_log_likelihood(:,2), 'k*',[1:iterations], Validation_log_likelihood(:,3), 'b*',[1:iterations], Validation_log_likelihood(:,4), 'green*',[1:iterations], Validation_log_likelihood(:,5), 'yellow*', [1:iterations], Validation_log_likelihood(:,6), 'cyan*'  )
legend('Training, Lambda=0', 'Training, Lambda=0.1', 'Training, Lambda=1', 'Training, Lambda=10', 'Training, Lambda=100',  'Training, Lambda=10000',  'Validation, Lambda=0', 'Validation, Lambda=0.1', 'Validation, Lambda=1', 'Validation, Lambda=10', 'Validation, Lambda=100',  'Validation, Lambda=1000')
xlabel('Number of Iterations', 'FontSize', 16)
ylabel('Training and Validation Error', 'FontSize', 16)
title('Training and Validation Error as Function of Sigma', 'FontSize', 16)



% %Second Plot
plot(1:length(lambda), l2_norm);
xlabel('Lambda Values','FontSize', 16)
ylabel('L2 Norm of Weight Vector', 'FontSize', 16)
title('L2 Norm of Weight Vector')



plot([1:length(lambda)], l2_norm_sigma(:,1), 'r', [1:length(lambda)], l2_norm_sigma(:,2), 'k', [1:length(lambda)], l2_norm_sigma(:,3), 'b', [1:length(lambda)], l2_norm_sigma(:,4), 'green', [1:length(lambda)], l2_norm_sigma(:,5), 'magenta');
legend('Sigma=0.1', 'Sigma=0.5', 'Sigma=1', 'Sigma=5', 'Sigma=10' )
xlabel('Lambda Values', 'FontSize', 16)
ylabel('L2 Norm of Weight Vector', 'FontSize', 16)
title('L2 Norm as Function of Lambda, for corresponding basis function means')


% %Third Plot
% plot(weight_plot);
% 
% 
% %Fourth Plot
% plot([1:iterations],Training_log_likelihood,'color','r'); hold on;
% plot([1:iterations],Testing_log_likelihood,'color','b');
% 
% 
% 
% 
% 
% 
% %Question i
% 
% % Consider the position of the mean mu_i for the placement of the basis
% % function as a parameter itself which relates to the overall loss function
% 
% % then can take the derivative w.r.t to mu_i and find the best solution for
% % mu_i just like we did for w
% 
% %this will find the best placement of the basis function which will
% %optimize the overall loss function
% 
% 
% %Question j
% %to test for convergence - can monitor the objective function for the best
% %mu_i and best w? See if the graphs converges for training and testing?
% 
% % See what kind of plots were shown above - and relate!!!
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
