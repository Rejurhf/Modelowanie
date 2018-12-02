clearvars;
close all;
clc;

N = 100;                % grid size
T = 400;                % number of time steps
temperature = 30;       % in Celsius degrees
time = 10;              % time step length in minutes
M0 = 35000;             % initial weight in kg
m = 0.098;              % spreading constant in four cells on the principle directions in water surface
d = 0.18;               % spreading constant in four diagonal cells in water surface

% INITIALIZATION

current_state = zeros(N,N);
current_state(N/2, N/2) = M0;      
next_state = zeros(N,N);

% SIMULATION
for t = 0:T
    for x = 2:(N-1)
        for y = 2:(N-1)
            % Spreading without wind and current:
            next_state(y,x) = current_state(y,x) + ...
                m*(current_state(y,x-1) + current_state(y,x+1) + ...
                current_state(y-1,x) + current_state(y+1,x) - ...
                4*current_state(y,x)) + m*d*(current_state(y+1,x-1) + ...
                current_state(y-1,x+1) + current_state(y-1,x-1) + ...
                current_state(y+1,x+1) - 4*current_state(y,x));
            % evaporation - equation for FCC heavy cycle:
            next_state(y,x) = (100-((0.17 + 0.013*temperature)*...
                sqrt(time)))/100*next_state(y,x);
        end
    end
    
    if(mod(t,20) == 0)
        figure(1);
        filename = 'pre.gif';
        imagesc(current_state);
        title(['T: ',num2str(t)],'FontSize',14);
        truesize([500,500]);
        drawnow;  
        frame = getframe(1);     
        im = frame2im(frame);     
        [imind,cm] = rgb2ind(im,256);     
        if t == 0     
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);     
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');     
        end
    end
    current_state = next_state;
end