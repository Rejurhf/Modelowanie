% Rejurhf
% 3.12.2018

% Diffusion-Advection 2D
clear;

%% Solving a PDE
% Equation
% Ct = -uCx - vCy + KCxx + KCyy + qC

%% Domain
% Space
Lx = 7; % 7m
Ly = 5; % 5m
dx = 0.2; dy = dx; % Every 0.2m
nx = fix(Lx/dx); ny = fix(Ly/dy);
X = linspace(0, Lx, nx);
Y = linspace(0, Ly, ny);
[x,y] = meshgrid(X,Y);
x = x'; y = y'; % transposition
vs = 1.5;

% Time
T = 5; % 5s

%% Field Arrays
% Variables
C = zeros(nx, ny);

% Parameters
u = zeros(nx, ny);
v = zeros(nx, ny);
K = zeros(nx, ny); % Diffusion factor

%% Initial Conditions
t = 0;
C(:) = 0; % concentration
u(:) = vs;
v = 0.1 + 0.01*(y-Ly) + sin(4*pi*x/Lx); % custom value (strzalki)
K(:) = 0.01;

CFL = 0.1;
dt = CFL*min(dx./abs(u(:)) + dy./abs(v(:)));
f = 1;

%% Time stepping Loop
while (t < T)
    % Boundary Conditions
    % 1st col = 2nd col and last col = last-1 col, same with rows
    C(:,[1 end]) = C(:,[2 end-1]);
    C([1 end],:) = C([2 end-1],:);
    
    % Source
    if t < 2 % less than 2s
        C(4,12) = C(4,12) + dt*50; % spawn 50 units there
    end
    
    % Solution
    t = t + dt;
    Cn = C;
    for i=2:nx-1
        for j=2:ny-1
            % Advection term
            A = u(i,j) * (Cn(i+1,j) - Cn(i-1,j))/(2*dx) ...
              + v(i,j) * (Cn(i,j+1) - Cn(i,j-1))/(2*dy);
            % Diffusion term
            D = K(i,j) * (Cn(i+1,j) - 2*Cn(i,j) + Cn(i-1,j))/dx^2 ...
              + K(i,j) * (Cn(i,j+1) - 2*Cn(i,j) + Cn(i,j-1))/dy^2;
            % Euler's Method
            C(i,j) = Cn(i,j) + dt*(-A + D);
            if C(i,j)<0 
                C(i,j)=abs(C(i,j)); 
            end
        end
    end
    
    % Visualize at selected steps
    clf;
    figure(1);
    filename = 'burgers.gif';
    imagesc(X, Y, C'); colorbar;
    hold on
    quiver(x,y,u,v);
    hold off;
    set(gca, 'ydir', 'norm');
    title(sprintf('t = %.2f', t));
    axis([0 Lx 0 Ly]);
    drawnow;  
    frame = getframe(1);     
    im = frame2im(frame);     
    [imind,cm] = rgb2ind(im,256);     
    if f == 1
        f = 0;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);     
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');     
    end
%     shg; pause(0.01);
end