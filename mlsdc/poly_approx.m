function [q,qc] = poly_approx(p,m,a,b,plot_flag)

if nargin < 5
   plot_flag = false;
end
if nargin < 3 || nargin < 4
    a = -1; 
    b = 1;
end
%f = chebfun(@(x) polyval(p,x),[a,b]);
%q_ = minimax(f,m);        
%q_ = minimax(f,m);        

p_ = @(x) polyval(p,x);

q_ = minimax(p_,[a,b],m); % best approximation of degree m       
q = poly(q_);              

qc_ = chebfun(p_,[a,b],'trunc',m+1); % Chebzshev approximation of degree m   
qc = poly(qc_);





%--------------------------------------------------------------------------
% plots
%--------------------------------------------------------------------------

if plot_flag
   x = linspace(a,b,1e5);  
   
%    f_ = poly(f);
%    figure(1); 
%    plot(x,polyval(p,x),'b',x,polyval(f_,x),'--g','LineWidth',2);
%    xlabel('x')
%    ylabel('y')
%    grid on   
%    legend('p(x)','chebfun(p)(x)')
   
   figure(2)
   plot(x,polyval(p,x),'b',x,polyval(q,x),'g',x,polyval(qc,x),'r','LineWidth',2);
   xlabel('x')
   ylabel('y')
   grid on      
   legend('input p(x)','best approx. q(x)','Chebyshev approx. qc(x)')

   figure(3)
   plot(x,polyval(p,x)-polyval(q,x),'g',x,polyval(p,x)-polyval(qc,x),'r','LineWidth',2);
   xlabel('x')
   ylabel('y')
   grid on 
   title('approximation error')
   legend('p(x) - q(x)','p(x) - qc(x)')

end

end

