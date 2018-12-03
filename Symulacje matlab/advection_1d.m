% Rejurhf
% 3.12.2018

%% Solving a PDE
clear;
% Equation
% Ct = -uCx + qC

%% Domain
% Space
Lx = 10;
dx = 0.5; % Every 0.5m
nx = fix(Lx/dx);
x = linspace(0, Lx, nx);
v = 1; % velocity

% Time
T = 10; % 10s

CFL = 1;
dt = CFL*dx/v;
nt = fix(T/dt);

%% Field Arrays
% Variables
C = zeros(nx, 1);

% Parameters
u = zeros(nx, 1);

%% Initial Conditions
t = 0;
C(:) = 0;
u(:) = 1; % velocities

%% Time stepping Loop
for n=1:nt
    % Boundary Conditions
    % Optional, period
%     C(1) = C(end);
    
    % Source
    if n==1
        C(1) = 1;
    else
        C(1) = 0;
    end
    
    % Solution
    t = t + dt;
    Cn = C;
    for i=2:nx
        % Advection term
        A = u(i) * (Cn(i) - Cn(i-1))/dx;
        % Euler's Method
        C(i) = Cn(i) + dt*(-A);
    end
    
        
    % Check coverage
    
    % Visualize at selected steps
    clf;
    plot(x, C, '-o');
    title(sprintf('t = %.2f', t));
    axis([0 Lx 0 2]);
    shg; pause(0.1);
end