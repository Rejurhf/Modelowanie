% Rejurhf
% 3.12.2018

% Diffusion-Advection
clear;

%% Solving a PDE
% Equation
% Ct = -uCx + KCxx + qC

%% Domain
% Space
Lx = 10;
dx = 0.5; % Every 0.5m
nx = fix(Lx/dx);
x = linspace(0, Lx, nx);
v = 1; % velocity

% Time
T = 5; % 5s

CFL = 0.15;
dt = CFL*dx/v;
nt = fix(T/dt);

%% Field Arrays
% Variables
C = zeros(nx, 1);

% Parameters
u = zeros(nx, 1); % Velocity
K = zeros(nx, 1); % Diffusion factor

%% Initial Conditions
t = 0;
C(:) = 0;
u(:) = 1;
K(:) = 1;

%% Time stepping Loop
for n=1:nt
    % Boundary Conditions
    C(end) = C(end-1);
    % Optional,if period add this
    C(1) = C(end);
    
    % Source
    if n==1
        C(1) = 1;
%     else
%         % if period delete this
%         C(1) = 0;
    end
    
    % Solution
    t = t + dt;
    Cn = C;
    for i=2:nx-1
        % Advection term
        A = u(i) * (Cn(i) - Cn(i-1))/dx;
        % Diffusion term
        D = K(i) * (Cn(i+1) - 2*Cn(i) + Cn(i-1))/dx^2;
        % Euler's Method
        C(i) = Cn(i) + dt*(-A + D);
    end
    
        
    % Check coverage
    
    % Visualize at selected steps
    clf;
    plot(x, C, '-o');
    title(sprintf('t = %.2f', t));
    axis([0 Lx 0 0.1]);
    shg; pause(0.05);
end