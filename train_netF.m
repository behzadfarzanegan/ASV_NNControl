clear all
clc
close all

% Define the number of neurons and input size
Neuron_Num_f = 50;  % Number of neurons in the hidden layer
% m = 2;              % Output size
n = 3;              % Input size (before adding the bias)

samples = 10000;
data = 2 * rand(n, samples) - 1;  % Input data

% Define the target output (dPol)
% dPol = zeros(1, samples);
u = data(1, :);
v = data(2, :);
r = data(3, :);
%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%
m11=189;m22=1036;m33=2411.1; 
m23=-543.5;m32=-543.5;
ay=0.595; a_psi=1.134;

T=0.01;
%%%%%%%%%%%%
% A = [0.8  0.1  0    0.1  0    0;
%      0.1  0.9  0    0    0.1  0;
%      0    0    0.7  0    0.1  0;
%      0.1  0    0    0.6  0.1  0;
%      0    0.1  0.1  0.1  0.5  0.1;
%      0    0    0.1  0    0.1  0.4];

% A= -eye(6);
%%%%

for i = 1:samples
d_x = 50*u(i)+70*u(i)*abs(u(i));
d_y = 948.2*v(i)+385.4*r(i);
d_psi = 385.4*v(i) +1926.9*r(i);

% Compute intermediate forces f_y' and f_psi'
f_y_prime = - (1/m22) * (m11 * u(i) * r(i) + d_y);
f_psi_prime = - (1/m33) * ((m22 - m11) * u(i) * v(i) + ((m23 + m32)/2) * u(i) * r(i) + d_psi);

% Compute the forces f_x, f_y, and f_psi
f_x = (1/m11) * (m22 * v(i) * r(i) + ((m23 + m32)/2) * r(i)^2 - d_x);
f_y = a_psi * (f_y_prime - (m23/m22) * f_psi_prime);
f_psi = a_psi * (f_psi_prime - (m32/m33) * f_y_prime);
X=[u(i) v(i) r(i)]';
    % dPol(:,i) = X + T*[f1;f2;f3;f_x;f_y;f_psi]-A*X;
    dPol(:,i) = T*[f_x;f_y;f_psi];

    % dPol(i) = tau1;
end

% Create the network with one hidden layer (50 neurons)
net = fitnet(Neuron_Num_f);  % Create a network with 50 neurons in the hidden layer

% Disable normalization for all layers
net.inputs{1}.processFcns = {};   % Disable all input processing functions (xoffset = 0, gain = 1, ymin = 0)
net.outputs{2}.processFcns = {};  % Disable all output processing functions

% Turn off bias for both the hidden and output layers
net.biasConnect = [0; 0];  % No bias for the hidden and output layers

% Add a bias to the input by adding an extra input neuron with constant value
extended_data = [data; ones(1, samples)];  % Add a row of 1's to the input data

% Set the number of epochs to 400
net.trainParam.epochs = 300;
net.trainParam.max_fail = 100;  % Increase the max_fail to allow more validation failures before stopping
net.trainParam.lr = 0.01;  % Reduce learning rate

% Train the network on the extended data (which includes the bias as an input)
net = train(net, extended_data, dPol);

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
plot(dPol(1, :), 'b')  % Plot actual target output (tau1)
hold on
plot(y_net1(1, :), '--r')  % Plot actual target output (tau3)
% plot(net(data), '--r')  % Plot network's predicted output
legend('Actual tau1', 'Actual tau3', 'Predicted')
xlabel('Sample')
ylabel('Output')
title('Actual vs Predicted Output')

% Save the trained network
Vf = V_trained';
Wf = W_trained';

save('WfVfall.mat', "Wf","Vf","Neuron_Num_f")


