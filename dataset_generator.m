close all; clear all; clc

%% Simulation setup
fs = 44.1e3;
Ts = 1/fs;
Vbe_n_samples = 1000;
Vbc_n_samples = 3000;
stop_time = Ts*Vbe_n_samples;
t = 0:Ts:stop_time;

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
%% Simulate BJT 1
output_signals_1 = [];

% Non uniform sampling of Vbe and Vbc, defined as piecewise functions with
% three equations
A_vbe = 0;
B_vbe = 0.57;
C_vbe = 0.6;
D_vbe = 0.7;
AB_vbe_n_samples = floor(length(t)*0.1);
BC_vbe_n_samples = length(t)-2*AB_vbe_n_samples;
Vbe = [t', [linspace(A_vbe, B_vbe, AB_vbe_n_samples), linspace(B_vbe, C_vbe, BC_vbe_n_samples), linspace(C_vbe, D_vbe, AB_vbe_n_samples)]'];

A_vbc = -9;
B_vbc = -5;
C_vbc = -2;
D_vbc = -1;
AB_vbc_n_samples = Vbc_n_samples*0.1;
BC_vbc_n_samples = Vbc_n_samples-2*AB_vbc_n_samples;
Vbc_ = [linspace(A_vbc, B_vbc, AB_vbc_n_samples), linspace(B_vbc, C_vbc, BC_vbc_n_samples), linspace(C_vbc, D_vbc, AB_vbc_n_samples)];

for i = 1:Vbc_n_samples
    sprintf('BJT 1, sample %.f',i)
    Vbc = [t', repelem(Vbc_(i), length(t))'];

    simOut = sim("BJTModel.slx"); 
    t_sim = simOut.tout;
    outputs = simOut.yout;
    Ibc = outputs.getElement('Ibc').Values.Data;
    Ibe = outputs.getElement('Ibe').Values.Data;
    output_signals_1 = [output_signals_1; Vbc(:,2), Ibc, Vbe(:,2), Ibe];
end
%% Simulate BJT 2
output_signals_2 = [];

A_vbe = 0;
B_vbe = 0.58;
C_vbe = 0.6;
D_vbe = 0.7;
AB_vbe_n_samples = floor(length(t)*0.1);
BC_vbe_n_samples = length(t)-2*AB_vbe_n_samples;
Vbe = [t', [linspace(A_vbe, B_vbe, AB_vbe_n_samples), linspace(B_vbe, C_vbe, BC_vbe_n_samples), linspace(C_vbe, D_vbe, AB_vbe_n_samples)]'];

A_vbc = -9;
B_vbc = -6;
C_vbc = -4;
D_vbc = -1;
AB_vbc_n_samples = Vbc_n_samples*0.1;
BC_vbc_n_samples = Vbc_n_samples-2*AB_vbc_n_samples;
Vbc_ = [linspace(A_vbc, B_vbc, AB_vbc_n_samples), linspace(B_vbc, C_vbc, BC_vbc_n_samples), linspace(C_vbc, D_vbc, AB_vbc_n_samples)];

for i = 1:Vbc_n_samples
    sprintf('BJT 2, sample %.f',i)
    Vbc = [t', repelem(Vbc_(i), length(t))'];

    simOut = sim("BJTModel.slx"); 
    t_sim = simOut.tout;
    outputs = simOut.yout;
    Ibc = outputs.getElement('Ibc').Values.Data;
    Ibe = outputs.getElement('Ibe').Values.Data;
    output_signals_2 = [output_signals_2; Vbc(:,2), Ibc, Vbe(:,2), Ibe];
end
%% Plot
close all;
t_plot = 0:Ts:Ts*(length(output_signals_1)-1);

figure()
hold on
plot(t_plot, output_signals_1(:, 1))
plot(t_plot, output_signals_1(:, 3))
plot(t_plot, output_signals_2(:, 1))
plot(t_plot, output_signals_2(:, 3))
legend("Vbc1","Vbe1","Vbc2","Vbe2")

figure()
hold on
plot(t_plot, output_signals_1(:, 2))
plot(t_plot, output_signals_1(:, 4))
plot(t_plot, output_signals_2(:, 2))
plot(t_plot, output_signals_2(:, 4))
legend("Ibc1","Ibe1","Ibc2","Ibe2")
%% Save dataset to file
data = [output_signals_1(:,1:4) output_signals_2(:,1:4)];
labels = {'Vbc1', 'Ibc1', 'Vbe1', 'Ibe1', 'Vbc2', 'Ibc2', 'Vbe2', 'Ibe2'};
writetable(array2table(data, 'VariableNames', labels), 'data/dataset.csv')