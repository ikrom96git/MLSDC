p = [1 5 -3 1 6 -3 5 -7 1]; % higher degree polynomial p(x), which shall be approximated  by a lower degree polynomial q(x) 

m = 2;                 % prescribed degree of the approximating polynomial q(x) 

a = -1;              % domain x in [a,b]
b = 1; 

%plot_flag = 0; % no plots
plot_flag = 1;

[q,qc] = poly_approx(p,m,a,b,plot_flag)

