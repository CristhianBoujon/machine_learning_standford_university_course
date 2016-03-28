clear ; close all; clc




num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
training_set_idx = rand_indices(1:2000);
training_set_x = X(training_set_idx, :);
training_set_y = y(training_set_idx);

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(training_set_x, training_set_y, num_labels, lambda);


for i = 1:10
	img = X(floor(rand() * 5000), :);
	imshow(reshape(img, 20, 20));
	printf('El número de la images es: %d \n', predictOneVsAll(all_theta, [img]));
	more off;
	pause(5);
endfor
