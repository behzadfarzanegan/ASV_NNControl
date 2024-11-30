clear all
clc
close all

% Define the number of neurons and input size
Neuron_Num_g = 50;  % Number of neurons in the hidden layer
n = 3;              % Input size (before adding the bias)

samples = 10000;
data = 4 * rand(n, samples) - 2;  % Input data

% Define constants
m11 = 189; m22 = 1036; m33 = 2411.1; 
m23 = -543.5; m32 = -543.5;
ay = 0.595; a_psi = 1.134;
T = 0.01;

% Initialize dPol for storing the target output
dPol = zeros(3, 2, samples);

% Generate target output dPol for each sample
for i = 1:samples
    % Compute matrix G with the given constants
    G = T * [1/m11 0; 0 ay/m33; 0 a_psi/m33];
    
    % Store the computed matrix in dPol for each sample
    dPol(:, :, i) = G;
end

% Reshape dPol to a 2D matrix [12 x samples] for network training
dPol_reshaped = reshape(dPol, [], samples);

% Create the network with one hidden layer (50 neurons)
net = fitnet(Neuron_Num_g);  % Create a network with 50 neurons in the hidden layer

% Disable normalization for all layers
net.inputs{1}.processFcns = {};   % Disable all input processing functions
net.outputs{2}.processFcns = {};  % Disable all output processing functions

% Turn off bias for both the hidden and output layers
net.biasConnect = [0; 0];  % No bias for the hidden and output layers

% Add a bias to the input by adding an extra input neuron with constant value
extended_data = [data; ones(1, samples)];  % Add a row of 1's to the input data

% Set the training parameters
net.trainParam.epochs = 200;
net.trainParam.max_fail = 100;  % Allow more validation failures
net.trainParam.lr = 0.01;  % Set learning rate

% Train the network on the extended data and reshaped dPol (target output)
net = train(net, extended_data, dPol_reshaped);

% Extract the final trained weights from the trained network (no biases)
V_trained = net.IW{1};       % Trained weights between input and hidden layer
W_trained = net.LW{2,1};     % Trained weights between hidden layer and output layer

% Manually compute the output using the final trained weights
X = data(:, 1);  % Use the first input data sample for testing
X_bias = [X; 1];  % Append the bias (1) to the input

% Step 1: Compute the hidden layer activation using the built-in tansig
H_trained = tansig(V_trained * X_bias);  % Use MATLAB's built-in tansig function

% Step 2: Compute the final output with the final trained weights W (no bias)
y_manual_trained = W_trained * H_trained;

% Display the manually calculated output with trained weights
disp('Manually calculated output (using final trained weights and tansig):');
disp(y_manual_trained);

% Simulate the network for comparison using the same input
y_net = net([X; 1]);
y_net1 = net(extended_data);

% Display the network-predicted output
disp('Network-predicted output:');
disp(y_net);

% Compare the manually calculated output with the network-predicted output
difference = abs(y_manual_trained - y_net);
disp('Difference between manually calculated output and network output:');
disp(difference);

% Plot the actual vs predicted data
figure
plot(dPol_reshaped(1, :), 'b')  % Plot actual target output (tau1)
hold on
plot(y_net1(1, :), '--r')  % Plot predicted output
legend('Actual tau1', 'Predicted tau1')
xlabel('Sample')
ylabel('Output')
title('Actual vs Predicted Output for tau1')

% Save the trained network
Vg = V_trained';
Wg = W_trained';

save('WgVgall.mat', "Wg", "Vg", "Neuron_Num_g")
