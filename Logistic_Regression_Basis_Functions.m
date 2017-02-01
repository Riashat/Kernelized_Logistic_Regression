
%question e and f
load('hw1data.mat')
X = hw1x;
y = hw1y;

[m,n] = size(X);
% X = pca(X');

[trainInd,valInd,testInd] = dividerand(size(X,1),0.6,0.2,0.2);


sigma = [0.1, 0.5, 1, 5, 10]';
basis_means = [-10, -5, 0, 5, 10]';
alpha = 0.0004;
iterations = 15000;

Train_Obj_Sigma_Basis_Means = zeros(length(sigma), length(basis_means));
Valid_Obj_Sigma_Basis_Means = zeros(length(sigma), length(basis_means));
Test_Obj_Sigma_Basis_Means = zeros(length(sigma), length(basis_means));
Training_log_likelihood = zeros(iterations,length(sigma));
Testing_log_likelihood = zeros(iterations,length(sigma));
Validation_log_likelihood = zeros(iterations,length(sigma));

val=5;
Validation_Cost_Train = zeros(iterations, val);
Validation_Cost_Valid = zeros(iterations, val);
Validation_Cost_Test = zeros(iterations, val);


for k = 1:size(basis_means,1)
    
    c = repmat(basis_means(k), size(X, 1), size(X, 2));
    
    for j = 1:size(sigma,1)

        basis_function = exp(  - (  (X - c).^2 ./ (2 * sigma(j).^2)    )   );
        
        [m, n] = size(basis_function);
        
        w_basis = zeros(n, 1);
        data = [basis_function,y];

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
        
        x_train = training(:, 1:end-1);
        y_train = training(:, end);

        x_valid = validation(:, 1:end-1);
        y_valid = validation(:, end);


        for i = 1:iterations
            sigmoid = (1 ./ (1 + exp(- x_train * w_basis)));
            grad_objective = x_train' * (sigmoid - y_train) ;
            w_basis = w_basis - alpha*grad_objective;

            Cost_Train =   (-1/length(y_train))  *    sum(   (y_train .* log(1 ./ (1 + exp(- x_train * w_basis))))     +   (1-y_train) .* log(1 - (1 ./ (1 + exp(- x_train* w_basis))))  ) ;
            Validation_Cost_Train(i, v) = Cost_Train; 
            Training_log_likelihood(i,j) = mean(Validation_Cost_Train(i, :));   
            Train_Obj_Sigma_Basis_Means(j,k) = mean(Validation_Cost_Train(i, :));

            Cost_Valid =   (-1/length(y_valid))  *    sum(   (y_valid .* log(1 ./ (1 + exp(- x_valid * w_basis))))     +   (1-y_valid) .* log(1 - (1 ./ (1 + exp(- x_valid * w_basis))))      );
            Validation_Cost_Valid(i, v) = Cost_Valid; 
            Validation_log_likelihood(i,j) = mean(Validation_Cost_Valid(i, :));   
            Valid_Obj_Sigma_Basis_Means(j,k) = mean(Validation_Cost_Valid(i, :));

            Cost_Test =  (-1/length(y_test) ) *      sum(   (y_test .* log(1 ./ (1 + exp(- x_test * w_basis))))     +   ((1-y_test) .* log(1 - (1 ./ (1 + exp(- x_test * w_basis)))))   );
            Validation_Cost_Test(i, v) = Cost_Valid; 
            Testing_log_likelihood(i,j) = mean(Validation_Cost_Test(i, :));   
            Test_Obj_Sigma_Basis_Means(j,k) = mean(Validation_Cost_Test(i, :));

        end

            end
        end
    end
end



%%%plot 1
plot([1:iterations], Training_log_likelihood(:,1), 'r', [1:iterations], Training_log_likelihood(:,2), 'k',[1:iterations], Training_log_likelihood(:,3), 'b',[1:iterations], Training_log_likelihood(:,4), 'green',[1:iterations], Training_log_likelihood(:,5), 'yellow', [1:iterations], Testing_log_likelihood(:,1), 'r*', [1:iterations], Testing_log_likelihood(:,2), 'k*',[1:iterations], Testing_log_likelihood(:,3), 'b*',[1:iterations], Testing_log_likelihood(:,4), 'green*',[1:iterations], Testing_log_likelihood(:,5), 'yellow*', [1:iterations],TR(:,3), 'cyan*',  [1:iterations],TE(:,3), 'magenta*' )
legend('Training, Sigma=0.1', 'Training, Sigma=0.5', 'Training, Sigma=1', 'Training, Sigma=5', 'Training, Sigma=10', 'Testing, Sigma=0.1', 'Testing, Sigma=0.5', 'Testing, Sigma=1', 'Testing, Sigma=5', 'Testing, Sigma=10', 'TRAINING Part C', 'TESTING Part C')
xlabel('Number of Iterations', 'FontSize', 16)
ylabel('Training and Test Error', 'FontSize', 16)
title('Training and Test Error as Function of Sigma', 'FontSize', 16)


plot([1:iterations], Training_log_likelihood(:,1), 'r', [1:iterations], Training_log_likelihood(:,2), 'k',[1:iterations], Training_log_likelihood(:,3), 'b',[1:iterations], Training_log_likelihood(:,4), 'green',[1:iterations], Training_log_likelihood(:,5), 'yellow', [1:iterations], Validation_log_likelihood(:,1), 'r*', [1:iterations], Validation_log_likelihood(:,2), 'k*',[1:iterations], Validation_log_likelihood(:,3), 'b*',[1:iterations], Validation_log_likelihood(:,4), 'green*',[1:iterations], Validation_log_likelihood(:,5), 'yellow*', [1:iterations],TR(:,3), 'cyan*',  [1:iterations],TE(:,3), 'magenta*' )
legend('Training, Sigma=0.1', 'Training, Sigma=0.5', 'Training, Sigma=1', 'Training, Sigma=5', 'Training, Sigma=10', 'Testing, Sigma=0.1', 'Testing, Sigma=0.5', 'Testing, Sigma=1', 'Testing, Sigma=5', 'Testing, Sigma=10', 'TRAINING Part C', 'TESTING Part C')
xlabel('Variance Sigma Values', 'FontSize', 16)
ylabel('Training and Validation Error', 'FontSize', 16)
title('Training and Validation Error as Function of Sigma', 'FontSize', 16)





plot([1:iterations], Training_log_likelihood(:,1), 'r*', [1:iterations], Training_log_likelihood(:,2), 'k*',[1:iterations], Training_log_likelihood(:,3), 'b*',[1:iterations], Training_log_likelihood(:,4), 'green*',[1:iterations], Training_log_likelihood(:,5), 'yellow*')
legend('Training, Sigma=0.1', 'Training, Sigma=0.5', 'Training, Sigma=1', 'Training, Sigma=5', 'Training, Sigma=10')
xlabel('Variance Sigma Values', 'FontSize', 16)
ylabel('Training Error', 'FontSize', 16)
title('Training Error as Function of Sigma', 'FontSize', 16)





% 
% plot([1:iterations], Training_log_likelihood(:,1), 'r', [1:iterations], Training_log_likelihood(:,2), 'k',[1:iterations], Training_log_likelihood(:,3), 'b',[1:iterations], Training_log_likelihood(:,4), 'green',[1:iterations], Training_log_likelihood(:,5), 'yellow')
% legend('Training, Sigma=0.1', 'Training, Sigma=0.5', 'Training, Sigma=1', 'Training, Sigma=5', 'Training, Sigma=10')
% xlabel('Variance Sigma Values', 'FontSize', 16)
% ylabel('TrainingError', 'FontSize', 16)
% title('Training Error as Function of Sigma', 'FontSize', 16)
% 


% 
%plotting train and test error as function of sigma 
%plotting for one of the basis_means here:
plot(1:length(sigma), Train_Obj_Sigma_Basis_Means(:,1), 'r', 1:length(sigma), Train_Obj_Sigma_Basis_Means(:,2), 'k',1:length(sigma), Train_Obj_Sigma_Basis_Means(:,3), 'b',1:length(sigma), Train_Obj_Sigma_Basis_Means(:,4), 'green',1:length(sigma), Train_Obj_Sigma_Basis_Means(:,5), 'magenta'); hold on;
plot(1:length(sigma), Test_Obj_Sigma_Basis_Means(:,1), 'r--o', 1:length(sigma), Test_Obj_Sigma_Basis_Means(:,2), 'k--o', 1:length(sigma), Test_Obj_Sigma_Basis_Means(:,3), 'b--o',1:length(sigma), Test_Obj_Sigma_Basis_Means(:,4), 'green--o', 1:length(sigma), Test_Obj_Sigma_Basis_Means(:,5), 'magenta--o')
legend('Train, sigma=0.1', 'Train, sigma=0.5', 'Train, sigma=1', 'Train, sigma=5', 'Train, sigma=10', 'Test, sigma=0.1', 'Test, sigma=0.5', 'Test, sigma=1', 'Test, sigma=5', 'Test, sigma=10' )
xlabel('Variance parameter Sigma Values','FontSize', 16)
ylabel('Final Error for each Basis Mean','FontSize', 16)
title('Train and Test Error with Sigma Values', 'FontSize', 16)












