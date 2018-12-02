clearvars;
close all;
clc;

%select the time at which to see solution 
t=5*30; 
 
%initiate 
D=(0.5*10^-4)*24; %[m^2/day] 
L=5; %how far are we looking in [m] 
dx=.1; 
xmesh= -L:dx:L; 
nx= length(xmesh); 
dy=.1; 
ymesh= -L:dx:L; 
ny= length(ymesh); 
unum=zeros(nx,ny);   

%set up integration 
a=0; 
b=t; 
n=150; %number of steps 
dh=5*t/n; %stepwidth 
hmesh=0:dh:t; %vector of step locations 
nh=length(hmesh); 
w=2*dh;   

%integrate and plot 
for i=2:nh     
    for x=1:nx         
        for y=1:ny         
            unum(x,y)=unum(x,y)+...             
                w*exp(-(xmesh(x)^2+ymesh(y)^2)/...             
                (4*D*hmesh(i)))/(4*pi*D*hmesh(i));         
        end
    end
    figure(1);     
    filename = 'integrator.gif';     
    surf(xmesh,ymesh,unum,'LineStyle','none')     
    % Create xlabel     
    xlabel('x (m)','FontSize',14);
    xlim([-5 5]);     
    % Create ylabel     
    ylabel('y (m)','FontSize',14);     
    ylim([-5 5]);     
    zlabel('concentration','FontSize',14);     
    zlim([0 650]);     
    % Create title     
    title(['Concentration of Oil at Day ',num2str(hmesh(i))],'FontSize',14);     
    drawnow;  
    frame = getframe(1);     
    im = frame2im(frame);     
    [imind,cm] = rgb2ind(im,256);     
    if i == 2     
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);     
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');     
    end
end