close all; clear all; clc

% Test signal
fs = 44.1e3;
Ts = 1/fs;
stop_time = 1;
t = 0:Ts:stop_time;
f0 = 100;
Vin = [t' 0.7*sin(t*f0*2*pi)'];

% Ebers Moll model parameters
% MPSA18 .MODEL MPSA18 NPN(IS=33.58f ISE=166.7f ISC=0 XTI=3 BF=236 BR=5.774 IKF=0.1172 IKR=0 XTB=1.5 VAF=100 VAR=30 VJE=0.65 VJC=0.65 RE=0.1 RC=1 RB=10 CJE=7.547p CJC=4.948p XCJC=0.75 FC=0.5 NF=1 NR=1 NE=1.579 NC=2 MJE=0.3765 MJC=0.4109 TF=310.1p TR=800.3p ITF=0.6 VTF=6 XTF=35 EG=1.11 VCEO=45 ICRATING=200m MFG=NSC)
N_f=1; 
N_r=1;
Rs1 = 1e-5;
Rs2 = 1e-5;
Rp1 = 1e11;
Rp2 = 1e8;
Vt = 25.85e-3;
alpha_f = 236/(236+1); % alfa_f = BF/(1+BF)
alpha_r = 5.774/(5.774+1); %alfa_r = BR/(1+BR)
Is1 = 33.58e-15; % Is1=Is2=IS
Is2 = 33.58e-15;
%% Run simulation
tic
simOut = sim("vox_ebersmoll.slx");
toc
Vout = get(simOut, 'Vout').Data;
Vbe1 = get(simOut, 'Vbe1').Data;
Vbc1 = get(simOut, 'Vbc1').Data;
Vbe2 = get(simOut, 'Vbe2').Data;
Vbc2 = get(simOut, 'Vbc2').Data;
Ibe1 = get(simOut, 'Ibe1').Data;
Ibc1 = get(simOut, 'Ibc1').Data;
Ibe2 = get(simOut, 'Ibe2').Data;
Ibc2 = get(simOut, 'Ibc2').Data;
%% Plot
close all;

figure()
hold on
plot(t, Vin(:,2)')
plot(t, Vout)
legend("Vin","Vout")

figure()
hold on
plot(t, Vbe1)
plot(t, Vbc1)
plot(t, Vbe2)
plot(t, Vbc2)
legend("Vbe1","Vbc1","Vbe2","Vbc2")

figure()
hold on
plot(t, Ibe1)
plot(t, Ibc1)
plot(t, Ibe2)
plot(t, Ibc2)
legend("Ibe1","Ibc1","Ibe2","Ibc2")
%% Write to file
writematrix(Vout,'data/groundtruth_ebersmoll_sin100_steady.csv');
%% Write to file v2
data = [Vbc1, Ibc1, Vbe1, Ibe1, Vbc2, Ibc2, Vbe2, Ibe2];
labels = {'Vbc1', 'Ibc1', 'Vbe1', 'Ibe1', 'Vbc2', 'Ibc2', 'Vbe2', 'Ibe2'};
writetable(array2table(data, 'VariableNames', labels), 'data/groundtruth_vox_4port_data.csv')