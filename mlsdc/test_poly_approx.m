p = [1 5 -3 1 6 -3 5 -7 1]; % higher degree polynomial p(x), which shall be approximated  by a lower degree polynomial q(x) 

m = 3;                 % prescribed degree of the approximating polynomial q(x) 

a = 0;              % domain x in [a,b]
b = 1; 

plot_flag = 0; % no plots
%plot_flag = 1;

%[q,qc] = poly_approx(p,m,a,b,plot_flag)


% A=importdata(filename);
% Real = A(1:6);
% Imag = A(7:end);
% 
% [qR, qcR]=poly_approx(Real, m, a, b, plot_flag)
%[qI, qcI]=poly_approx(Imag, m, a, b, plot_flag)
% data=[qR, qI]
% writematrix(data, 'order63.txt')

filecsv='coefficient_fine.csv';
C=csvread(filecsv);
u_pos=C(1:5);
u_vel=C(6:end);
addpath(fullfile(cd,'chebfun'))
savepath
[qR, qcR]=poly_approx(u_pos, m, a, b, plot_flag)
[qI, qcI]=poly_approx(u_vel, m, a, b, plot_flag)
data=[qcR, qcI]
disp('welcome')
writematrix(data, 'coefficient_coar.csv')
