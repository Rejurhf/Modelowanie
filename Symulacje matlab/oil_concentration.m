function project_numerical 
clc;
clear all;
close all; 
global D c0 L ;   

D = 1.3e-3; %m^2/s;  
c0= 600; %mmol/m^3 
L=5; %m   
t_max= 150;   
t=linspace(0, t_max, 200); 
x=linspace(0,L, 100); 
theta=(linspace(0, 2*pi,200));   

sol_pdepe = pdepe(0,@pdefun,@ic,@bc,x,t);   
sol_pdepe_t=sol_pdepe';   

figure(1) 
surf(t,x,sol_pdepe_t, 'EdgeColor', 'none') 
title('Oil Spill One Dimensional pdepe') 
xlabel('Time [days]') 
ylabel('Length x [m]') 
zlabel('Concentration ')   

figure(2) 
h1=plot(x, sol_pdepe_t(:,1),'b'); 
hold on 
%plot(x, sol_pdepe_t(:,50)); 
h2=plot(x, sol_pdepe_t(:,25), 'r'); 
%plot(x, sol_pdepe_t(:,100)); 
h3=plot(x, sol_pdepe_t(:,150), 'g'); 
legend([h1, h2, h3],{'1 day','25 days', '150 days'}); 
title(['Concentration Profile at Different Times']); 
xlabel('x[m]'); 
ylabel('Concentration');   
end
% function definitions for pdepe: 
% --------------------------------------------------------------   
function [c, f, s] = pdefun(x, t, u, DuDx) 
% PDE coefficients functions   
global D 
c = 1; 
f = D * DuDx; 
% diffusion 
s = 0; % homogeneous, no driving term 
end
% --------------------------------------------------------------   
function u0 = ic(x) 
% Initial conditions function 
u0=(x==0);
end

% --------------------------------------------------------------   
function [pl, ql, pr, qr] = bc(xl, ul, xr, ur, t) 
% Boundary conditions function   
global c0   
pl = ul-600; % c0 value left, chose a large number to simulate infinity 
ql = 0;  % arbitrary flux left boundary condition 
pr = 0; % zero value right boundary condition 
qr = 1;  % no flux right boundary condition   
end