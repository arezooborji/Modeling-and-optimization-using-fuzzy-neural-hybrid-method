clear all
clc
% GD iteration number (epoch)
GDiter=200;
% training input signal, (random between [-2,2] and sin signal
ut=zeros(1000,1);
ut(1:500)=2.*rand(500,1)-1;
ut(501:1000)=1.05*sin(pi.*(501:1000)/45);
% system 1 output for training input signal according to eq. 29
yt=zeros(1000,1);
yt2=yt;
for i=5:1000
    yt(i)=.72*yt(i-1)+.025*yt(i-2)*ut(i-2)+.01*ut(i-3)^2+.2*ut(i-4);
end
% system 2 output for training input signal according to eq. 32
for i=3:1000
    yt2(i)=(yt2(i)*yt2(i-1)*yt2(i-2)*ut(i-1)*(yt2(i-2)-1)+ut(i))/(1+yt2(i-2)^2+yt2(i-1)^2);
end
% test input signal as eq. 30
u=zeros(1000,1);
u(1:250)=sin(pi*(0:249)/25);
u(251:500)=1;
u(501:750)=-1;
u(751:1000)=.3*sin(pi*(750:999)/25)+.1*sin(pi*(750:999)/32)+ ...
    .6*sin(pi*(750:999)/10);
y=zeros(1000,1);
y2=zeros(1000,1);
% output of system 1 for test input signal as eq. 29
for i=5:1000
    y(i)=.72*y(i-1)+.025*y(i-2)*u(i-2)+.01*u(i-3)^2+.2*u(i-4);
end
% output of system 2 for test input signal as eq. 32
for i=3:1000
    y2(i)=(y2(i)*y2(i-1)*y2(i-2)*u(i-1)*(y2(i-2)-1)+u(i))/(1+y2(i-2)^2+y2(i-1)^2);
end
%% training system 1
% system 1 parameter initialization for PSO method, random numbers between
% 1 and 2 to avoid small numbers
aa=2;
veloc=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
posi=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
velocc=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
posii=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
velocPSO=veloc;
posiPSO=posi;
% internal variables
y_out=zeros(20,1);
fitness=y_out;
% iteration number for PSO method
maxit=50;
% PSO parameters
r1=.4*rand;
r2=.4*rand; 
fitInPSO1=zeros(50,1);
% PSO method implementation
for k=1:maxit
    % variable w for PSO method as in eq. 20
    w1=.9-.5*k/maxit;
    for i=1:20
        % set parameters of FWNN network from initial matrices for each
        % particle
        s=[posi(1:2,i) posi(3:4,i)];
        c=[posi(5:6,i) posi(7:8,i)];
        a1=[posi(9:10,i) posi(11:12,i)];
        a2=[posi(13:14,i) posi(15:16,i)];
        b1=[posi(17:18,i) posi(19:20,i)];
        b2=[posi(21:22,i) posi(23:24,i)];
        w=[posi(25:26,i) posi(27:28,i)];
        ybar=posi(29:30,i);
        % FWNN output computation for each instance of signal
        for j=1:20
            y_out(j)=fuzzy_knn(ut(j),yt(j),s,c,a1,a2,b1,b2,w,ybar);
        end
        % fitness computation with eq. 19
        fitness(i)=sqrt(.05*sum((y_out-yt(1:20)).^2));
    end
    % choosing best fitness and its index
    [val,indx]=min(fitness);
    % pbest and gbest update according to fitness value
    if k==1
        pbest=posi;
        gbest=posi(:,indx);
    else
        for i1=1:20
            if fitness(i1)<=fitness1(i1)
                pbest(:,i1)=posi(:,i1);
            else
                fitness(i1)=fitness1(i1);
            end
        end
        if val <= val1
            gbest=posi(:,indx);
        end
    end
    % PSO parameters update
    for i2=1:20
        veloc(:,i2)=w1.*veloc(:,i2)+2*r1.*(pbest(:,i2)-posi(:,i2))+2*r2.*(gbest-posi(:,i2));
        posi(:,i2)=posi(:,i2)+veloc(:,i2);
    end
    % entering InLine segment of InPSO method
    % random permutation of input signal bunch
    p=randperm(20);
    posii=posi;
    u1=ut(p);
    y1=yt(p);
    for l=1:20 
        % variable w for InPSO method as in eq. 20
        w2=.9-.5*l/20;
        % set parameters of FWNN network from PSO output for each particle
        for i3=1:20
            s=[posii(1:2,i3) posii(3:4,i3)];
            c=[posii(5:6,i3) posii(7:8,i3)];
            a1=[posii(9:10,i3) posii(11:12,i3)];
            a2=[posii(13:14,i3) posii(15:16,i3)];
            b1=[posii(17:18,i3) posii(19:20,i3)];
            b2=[posii(21:22,i3) posii(23:24,i3)];
            w=[posii(25:26,i3) posii(27:28,i3)];
            ybar=posii(29:30,i3);
            % FWNN output computation for each instance of signal
            y_outt=fuzzy_knn(u1(l),y1(l),s,c,a1,a2,b1,b2,w,ybar);
            % fitness computation with eq. 18
            fitnesss(i3)=.5*(y_outt-y1(l)).^2;
        end
        % choosing best fitness and its index
        [vall,indxx]=min(fitnesss);
        % pbest and gbest update according to fitness value
        if l==1
            pbestt=posii;
            gbestt=posii(:,indxx);
        else
            for i4=1:20
                if fitnesss(i4)<=fitnesss1(i4)
                    pbestt(:,i4)=posii(:,i4);
                else
                    fitnesss(i4)=fitnesss1(i4);
                end
            end
            if vall <= vall1
                gbestt=posii(:,indxx);
            end
        end
        % InPSO parameter update
        for i5=1:20
            velocc(:,i5)=w2.*velocc(:,i5)+2*r1.*(pbestt(:,i5)-posii(:,i5))+2*r2.*(gbestt-posii(:,i5));
            posii(:,i5)=posii(:,i5)+velocc(:,i5);
        end
        fitnesss1=fitnesss;
        vall1=vall;
    end     
    % retaining previous fitness for next iteration reference
    fitness1=fitness;
    fitInPSO1(k,1)=min(fitness);
    val1=val;
    % solution update from PSO to InPSO 
    posi=posii;
end
% PSO only for comparison with InPSO
% different lines are same as previous method omitting InPSO section
veloc=velocPSO;
posi=posiPSO;
fitPSO1=zeros(50,1);
for k=1:maxit
    w1=.9-.5*k/maxit;
    for i=1:20
        s=[posi(1:2,i) posi(3:4,i)];
        c=[posi(5:6,i) posi(7:8,i)];
        a1=[posi(9:10,i) posi(11:12,i)];
        a2=[posi(13:14,i) posi(15:16,i)];
        b1=[posi(17:18,i) posi(19:20,i)];
        b2=[posi(21:22,i) posi(23:24,i)];
        w=[posi(25:26,i) posi(27:28,i)];
        ybar=posi(29:30,i);
        for j=1:20
            y_out(j)=fuzzy_knn(ut(j),yt(j),s,c,a1,a2,b1,b2,w,ybar);
        end
        fitness(i)=sqrt(.05*sum((y_out-yt(1:20)).^2));
    end
    [val,indx]=min(fitness);
    if k==1
        pbest=posi;
        gbest=posi(:,indx);
    else
        for i1=1:20
            if fitness(i1)<=fitness1(i1)
                pbest(:,i1)=posi(:,i1);
            else
                fitness(i1)=fitness1(i1);
            end
        end
        if val <= val1
            gbest=posi(:,indx);
        end
    end
    for i2=1:20
        veloc(:,i2)=w1.*veloc(:,i2)+2*r1.*(pbest(:,i2)-posi(:,i2))+2*r2.*(gbest-posi(:,i2));
        posi(:,i2)=posi(:,i2)+veloc(:,i2);
    end  
    fitness1=fitness;
    fitPSO1(k,1)=min(fitness);
    val1=val;
end
% parameter initialization computed by InlinePSO
t_sol=gbest;
% FWNN parameter set from Inline solution
s=[t_sol(1:2) t_sol(3:4)];
c=[t_sol(5:6) t_sol(7:8)];
a1=[t_sol(9:10) t_sol(11:12)];
a2=[t_sol(13:14) t_sol(15:16)];
b1=[t_sol(17:18) t_sol(19:20)];
b2=[t_sol(21:22) t_sol(23:24)];
w=[t_sol(25:26) t_sol(27:28)];
ybar=t_sol(29:30);
y_est1=zeros(1000,1);
y_est2=y_est1;
y_est3=y_est1;
% Gradient descent method
for l=1:GDiter
    for k=1:1000
        % FWNN output computation
        y_out=fuzzy_knn(ut(k),yt(k),s,c,a1,a2,b1,b2,w,ybar);
        % difference computation of eqs. 22-28 by function delta_theta
        [ds dc da1 da2 db1 db2 dw dybar]=delta_theta(ut(k),yt(k),s,c,a1,a2,b1,b2,w,ybar);
        % GD stem size multiplied by output difference
        dgamma=.7*(y_out-yt(k));
        % FWNN parameter update by GD method, eq.22
        s=s-dgamma.*ds;
        c=c-dgamma.*dc;
        a1=a1-dgamma.*da1;
        a2=a2-dgamma.*da2;
        b1=b1-dgamma.*db1;
        b2=b2-dgamma.*db2;
        w=w-dgamma.*dw;
        ybar=ybar-dgamma.*dybar';
        y_est1(k,1)=y_out;
    end
    % RMSE for each epoch
    rms_error1(l,1)=sqrt((sum(y_est1-yt)).^2);
end
%% testing system 1
% system 1 identification with FWNN parameters computed previously and for
% system input u(i) as in eq.29
for i=1:1000
    y_est2(i,1)=fuzzy_knn(u(i),y(i),s,c,a1,a2,b1,b2,w,ybar);
end
%real output and identified output depiction
plot(y);
hold on;
plot(y_est2,'r');
title('system 1 identification');
legend('target signal','system 1 output');
% PSO and InPSO fitness figures
figure;
plot(1:50,fitPSO1,'r',1:50,fitInPSO1,'b');
title('system 1 fitness');
legend('PSO','InPSO');
% RMSE figure for training session
figure;
plot(1:GDiter,rms_error1);
title('RMSE for system 1');
% remaining code is same as above but for different signals so comment is
% the same for equivalent sections
%% training system 2
% system 2 parameter initialization for PSO method
veloc=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
posi=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
velocc=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
posii=aa.*ones(30,20)+2*aa.*rand(30,20)-aa.*ones(30,20);
velocPSO=veloc;
posiPSO=posi;

y_out=zeros(20,1);
fitness=y_out;
fitInPSO2=zeros(50,1);
% PSO method implementation
for k=1:maxit
    w1=.9-.5*k/maxit;
    for i=1:20
        s=[posi(1:2,i) posi(3:4,i)];
        c=[posi(5:6,i) posi(7:8,i)];
        a1=[posi(9:10,i) posi(11:12,i)];
        a2=[posi(13:14,i) posi(15:16,i)];
        b1=[posi(17:18,i) posi(19:20,i)];
        b2=[posi(21:22,i) posi(23:24,i)];
        w=[posi(25:26,i) posi(27:28,i)];
        ybar=posi(29:30,i);
        for j=1:20
            y_out(j)=fuzzy_knn(ut(j),yt2(j),s,c,a1,a2,b1,b2,w,ybar);
        end
        fitness(i)=sqrt(.05*sum((y_out-yt2(1:20)).^2));
    end
    [val,indx]=min(fitness);
    if k==1
        pbest=posi;
        gbest=posi(:,indx);
    else
        for i1=1:20
            if fitness(i1)<=fitness1(i1)
                pbest(:,i1)=posi(:,i1);
            else
                fitness(i1)=fitness1(i1);
            end
        end
        if val <= val1
            gbest=posi(:,indx);
        end
    end
    for i2=1:20
        veloc(:,i2)=w1.*veloc(:,i2)+2*r1.*(pbest(:,i2)-posi(:,i2))+2*r2.*(gbest-posi(:,i2));
        posi(:,i2)=posi(:,i2)+veloc(:,i2);
    end
    p=randperm(20);
    posii=posi;
    u1=ut(p);
    y1=yt2(p);
    for l=1:20 
        w2=.9-.5*l/20;
        for i3=1:20
            s=[posii(1:2,i3) posii(3:4,i3)];
            c=[posii(5:6,i3) posii(7:8,i3)];
            a1=[posii(9:10,i3) posii(11:12,i3)];
            a2=[posii(13:14,i3) posii(15:16,i3)];
            b1=[posii(17:18,i3) posii(19:20,i3)];
            b2=[posii(21:22,i3) posii(23:24,i3)];
            w=[posii(25:26,i3) posii(27:28,i3)];
            ybar=posii(29:30,i3);
            y_outt=fuzzy_knn(u1(l),y1(l),s,c,a1,a2,b1,b2,w,ybar);
            fitnesss(i3)=.5*(y_outt-y1(l)).^2;
        end
        [vall,indxx]=min(fitnesss);
        if l==1
            pbestt=posii;
            gbestt=posii(:,indxx);
        else
            for i4=1:20
                if fitnesss(i4)<=fitnesss1(i4)
                    pbestt(:,i4)=posii(:,i4);
                else
                    fitnesss(i4)=fitnesss1(i4);
                end
            end
            if vall <= vall1
                gbestt=posii(:,indxx);
            end
        end
        for i5=1:20
            velocc(:,i5)=w2.*velocc(:,i5)+2*r1.*(pbestt(:,i5)-posii(:,i5))+2*r2.*(gbestt-posii(:,i5));
            posii(:,i5)=posii(:,i5)+velocc(:,i5);
        end
        fitnesss1=fitnesss;
        vall1=vall;
    end     
    fitness1=fitness;
    fitInPSO2(k,1)=min(fitness);
    val1=val;
    posi=posii;
end
% PSO only for comparison
veloc=velocPSO;
posi=posiPSO;
fitPSO2=zeros(50,1);
for k=1:maxit
    w1=.9-.5*k/maxit;
    for i=1:20
        s=[posi(1:2,i) posi(3:4,i)];
        c=[posi(5:6,i) posi(7:8,i)];
        a1=[posi(9:10,i) posi(11:12,i)];
        a2=[posi(13:14,i) posi(15:16,i)];
        b1=[posi(17:18,i) posi(19:20,i)];
        b2=[posi(21:22,i) posi(23:24,i)];
        w=[posi(25:26,i) posi(27:28,i)];
        ybar=posi(29:30,i);
        for j=1:20
            y_out(j)=fuzzy_knn(ut(j),yt2(j),s,c,a1,a2,b1,b2,w,ybar);
        end
        fitness(i)=sqrt(.05*sum((y_out-yt2(1:20)).^2));
    end
    [val,indx]=min(fitness);
    if k==1
        pbest=posi;
        gbest=posi(:,indx);
    else
        for i1=1:20
            if fitness(i1)<=fitness1(i1)
                pbest(:,i1)=posi(:,i1);
            else
                fitness(i1)=fitness1(i1);
            end
        end
        if val <= val1
            gbest=posi(:,indx);
        end
    end
    for i2=1:20
        veloc(:,i2)=w1.*veloc(:,i2)+2*r1.*(pbest(:,i2)-posi(:,i2))+2*r2.*(gbest-posi(:,i2));
        posi(:,i2)=posi(:,i2)+veloc(:,i2);
    end  
    fitness1=fitness;
    fitPSO2(k,1)=min(fitness);
    val1=val;
end
% parameter initialization computed by InlinePSO
t_sol=gbest;
s=[t_sol(1:2) t_sol(3:4)];
c=[t_sol(5:6) t_sol(7:8)];
a1=[t_sol(9:10) t_sol(11:12)];
a2=[t_sol(13:14) t_sol(15:16)];
b1=[t_sol(17:18) t_sol(19:20)];
b2=[t_sol(21:22) t_sol(23:24)];
w=[t_sol(25:26) t_sol(27:28)];
ybar=t_sol(29:30);
y_est1=zeros(1000,1);
y_est2=y_est1;
y_est3=y_est1;
% Gradient descent method
for l=1:GDiter
    for k=1:1000
        y_out=fuzzy_knn(ut(k),yt2(k),s,c,a1,a2,b1,b2,w,ybar);
        [ds dc da1 da2 db1 db2 dw dybar]=delta_theta(ut(k),yt2(k),s,c,a1,a2,b1,b2,w,ybar);
        dgamma=.7*(y_out-yt2(k));
        s=s-dgamma.*ds;
        c=c-dgamma.*dc;
        a1=a1-dgamma.*da1;
        a2=a2-dgamma.*da2;
        b1=b1-dgamma.*db1;
        b2=b2-dgamma.*db2;
        w=w-dgamma.*dw;
        ybar=ybar-dgamma.*dybar';
        y_est1(k,1)=y_out;
    end
    rms_error2(l,1)=sqrt((sum(y_est1-yt2)).^2);
end
%% testing system 2
for i=1:1000
    y_est3(i,1)=fuzzy_knn(u(i),y2(i),s,c,a1,a2,b1,b2,w,ybar);
end
figure;
plot(y2);
hold on;
plot(y_est3,'r');
title('system 2 identification');
legend('target signal','system 2 output');
figure;
plot(1:50,fitPSO2,'r',1:50,fitInPSO2,'b');
title('system 2 fitness');
legend('PSO','InPSO');
figure;
plot(1:GDiter,rms_error2);
title('RMSE for system 2');