%% Setup
clear all
close all
clc

Fs = 44100;
f0 = 100;
Ts=1/Fs;
T_sim = 1;
t = linspace(0,T_sim,Fs*T_sim);
N_sim = T_sim*Fs;

Vin_amp = 0.7;
Vin = Vin_amp * sin(2 * pi * t * f0);

%% Circuit parameters
R5 = 68e3;
C6 = 0.01e-6;
C7 = 4.7e-6;
R8 = 50e3;
R9 = 22e3;
V9 = 9;
L10 = 500e-3;
C11 = 0.22e-6;
R12 = 470e3;
R13 = 33e3;
R14 = 1.5e3;
R15 = 470e3;
C16 = 0.22e-6;
R17 = 50e3;
R18 = 510;
R19 = 100e3;
C20 = 0.01e-6;
R21 = 10e3;
R22 = 1e3;
V22 = 9;
R23 = 1e6;

%% Compute scattering matrix
Z_bjt = sym('Z', [4 4]);
Z = sym(diag([0,0,0,0,R5,Ts/C6,Ts/C7,R8,R9,L10/Ts,Ts/C11,R12,R13,R14,R15,Ts/C16,R17,R18,R19,Ts/C20,R21,R22,R23]));
Z(1:4,1:4) = Z_bjt;
%{
B = zeros(12,23);
B(1:12, 1:12) = eye(12);
B(1:12, 13:23) = [
    1 -1 0 0 0 1 -1 0 0 0 0;
    1 -1 0 -1 0 0 -1 0 0 0 1;
    0 0 -1 1 0 0 0 0 1 0 -1;
    0 0 -1 1 0 0 0 0 0 1 -1;
    1 -1 0 0 0 0 -1 1 0 0 0;
    -1 0 0 0 0 0 1 0 -1 0 0;
    0 0 0 0 0 0 -1 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0 -1;
    0 0 0 1 0 0 0 0 0 0 -1;
    -1 0 0 0 0 0 0 0 0 0 0;
    0 0 -1 1 -1 0 0 0 0 0 0;
    0 0 0 1 0 0 1 0 0 0 -1
    ];
S_sym = eye(23)-2*Z*B'*inv(B*Z*B')*B;
sol = solve(S_sym(1:4, 1:4)==0, Z_bjt);
S = double(subs(S_sym, sol));
Z = double(subs(Z, sol));
Z_bjt = double(subs(Z_bjt, sol));
save("data/vox_adapted_impedance.mat","sol","S","Z_bjt","Z");
%}
load('data/vox_adapted_impedance.mat');

%% Load NN parameters
%{
% Import weights and bias from onnx file
net = importNetworkFromONNX("data/pretrained.onnx")
net = addInputLayer(net,featureInputLayer(4, Normalization="none"), Initialize=true);

net_params = struct;
net_params.w.l0 = net.Layers(3).model_0_weight';
net_params.b.l0 = net.Layers(3).model_0_bias;
net_params.w.l2 = net.Layers(3).model_2_weight';
net_params.b.l2 = net.Layers(3).model_2_bias;
net_params.w.l4 = net.Layers(3).model_4_weight';
net_params.b.l4 = net.Layers(3).model_4_bias;
net_params.w.l6 = net.Layers(3).model_6_weight';
net_params.b.l6 = net.Layers(3).model_6_bias;
net_params.w.l8 = net.Layers(3).model_8_weight';
net_params.b.l8 = net.Layers(3).model_8_bias;

%save('data/pretrained_parameters.mat', 'net_params');
%}
load('data/pretrained_parameters.mat');

%% WDF
load('data/wave_domain_steady_state_vars.mat');
K = 10;
T = zeros(1,K);
for i=1:K
    b = b_steady;
    a = a_steady;
    Vout = zeros(1, length(t));
    tic
    k=1;
    while (k<N_sim)
        % Dynamic elements (BE)
        b(6) = (a(6)+b(6))/2; 
        b(7) = (a(7)+b(7))/2;
        b(11) = (a(11)+b(11))/2;
        b(16) = (a(16)+b(16))/2;
        b(20) = (a(20)+b(20))/2;
        b(10) = (b(10)-a(10))/2;
    
        % Voltage generators
        b(5) = Vin(k);
        b(9) = V9;
        b(22) = V22;
        
        % Nonlinear root scattering
        b(1:4) = predict(S(1:4, :)*b, net_params);
        
        % Scattering
        a = S*b;
    
        % Output signal
        Vout(k) = (a(23)+b(23))/2;
        k=k+1;
    end
    T(i) = toc;
end
T(1) = 0;
rtr = sum(T/T_sim)/K
%% Plot sin 100 vs groundtruth
groundtruth = readmatrix('data/groundtruth_ebersmoll_sin100_steady.csv')';
plot_length = min(length(Vout), length(groundtruth));

figure
plot(t(:,1:plot_length), Vout(:,1:plot_length), 'b', 'DisplayName', 'WD');
hold on
plot(t(:,1:plot_length), groundtruth(:,1:plot_length), 'r--', 'DisplayName', 'SSC');
xlim([0,t(end)])
xlabel('Time(seconds)');
ylabel('Voltage (V)');
legend('show')

mse = mean((Vout(:,1:plot_length)- groundtruth(:,1:plot_length)).^2);
sprintf('mean square error: %f', mse)

%% Func def
function out = elu(in)
    mask = in>=0;
    out = zeros(size(in));
    out(mask) = in(mask);
    out(~mask) = exp(in(~mask)) - 1;
end

function out = predict(in, net_params)
    layer_out = elu(net_params.w.l0 * in + net_params.b.l0);
    layer_out = elu(net_params.w.l2 * layer_out + net_params.b.l2);
    layer_out = elu(net_params.w.l4 * layer_out + net_params.b.l4);
    layer_out = elu(net_params.w.l6 * layer_out + net_params.b.l6);
    out = net_params.w.l8 * layer_out + net_params.b.l8;
end