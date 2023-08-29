% function computing difference amount in GD method
function [ds dc da1 da2 db1 db2 dw dybar]=delta_theta(x1,x2,s,c,a1,a2,b1,b2,w,ybar)
% FWNN output computation as described in fuzzy_knn.m function
mu11=exp(-((x1-c(1,1))/s(1,1))^2);
mu12=exp(-((x2-c(1,2))/s(1,2))^2);
mu21=exp(-((x1-c(2,1))/s(2,1))^2);
mu22=exp(-((x2-c(2,2))/s(2,2))^2);
o1=mu11*mu12;
o2=mu21*mu22;
t1=(x1-b1(1,1))/a1(1,1);
t2=(x2-b1(1,2))/a1(1,2);
phi(1,1)=gauswavf(t1,t1,1,2)*gauswavf(t2,t2,1,2);

t1=(x1-b1(2,1))/a1(2,1);
t2=(x2-b1(2,2))/a1(2,2);
phi(2,1)=gauswavf(t1,t1,1,2)*gauswavf(t2,t2,1,2);

t1=(x1-b2(1,1))/a2(1,1);
t2=(x2-b2(1,2))/a2(1,2);
phi(1,2)=gauswavf(t1,t1,1,2)*gauswavf(t2,t2,1,2);

t1=(x1-b2(2,1))/a2(2,1);
t2=(x2-b2(2,2))/a2(2,2);
phi(2,2)=gauswavf(t1,t1,1,2)*gauswavf(t2,t2,1,2);

Y1=w(1,1)*phi(1,1)+w(2,1)*phi(2,1)+ybar(1);
Y2=w(1,2)*phi(1,2)+w(2,2)*phi(2,2)+ybar(2);

% eq. 23
dc(1,1)=((Y1*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu12*mu11*2*((x1-c(1,1))/(s(1,1)^2));
dc(1,2)=((Y1*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu11*mu12*2*((x2-c(1,2))/(s(1,2)^2));
dc(2,1)=((Y2*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu22*mu21*2*((x1-c(2,1))/(s(2,1)^2));
dc(2,2)=((Y2*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu21*mu22*2*((x2-c(2,2))/(s(2,2)^2));
% eq. 24
ds(1,1)=((Y1*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu11*mu12*((x1-c(1,1))^2)/(s(1,1)^3);
ds(1,2)=((Y1*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu11*mu12*((x2-c(1,2))^2)/(s(1,2)^3);
ds(2,1)=((Y2*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu21*mu22*((x1-c(2,1))^2)/(s(2,1)^3);
ds(2,2)=((Y2*(o1+o2)+o1*Y1+o2*Y2)/(o1+o2))*mu22*mu21*((x2-c(2,2))^2)/(s(2,2)^3);
% eq. 25, b1 stands for parameters of wavelet function for k=1 as in eq. 10
t1=(x1-b1(1,1))/a1(1,1);
t2=(x2-b1(1,2))/a1(1,2);
db1(1,1)=(o1/(o1+o2))*w(1,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a1(1,1));
t1=(x2-b1(1,2))/a1(1,2);
t2=(x1-b1(1,1))/a1(1,1);
db1(1,2)=(o1/(o1+o2))*w(1,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a1(1,2));
t1=(x1-b1(2,1))/a1(2,1);
t2=(x2-b1(2,2))/a1(2,2);
db1(2,1)=(o1/(o1+o2))*w(2,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a1(2,1));
t1=(x2-b1(2,2))/a1(2,2);
t2=(x1-b1(2,1))/a1(2,1);
db1(2,2)=(o1/(o1+o2))*w(2,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a1(2,2));
% eq. 25, b2 stands for parameters of wavelet function for k=2 as in eq. 10
t1=(x1-b2(1,1))/a2(1,1);
t2=(x2-b2(1,2))/a2(1,2);
db2(1,1)=(o2/(o1+o2))*w(1,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a2(1,1));
t1=(x2-b2(1,2))/a2(1,2);
t2=(x1-b2(1,1))/a2(1,1);
db2(1,2)=(o2/(o1+o2))*w(1,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a2(1,2));
t1=(x1-b2(2,1))/a2(2,1);
t2=(x2-b2(2,2))/a2(2,2);
db2(2,1)=(o2/(o1+o2))*w(2,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a2(2,1));
t1=(x2-b2(2,2))/a2(2,2);
t2=(x1-b2(2,1))/a2(2,1);
db2(2,2)=(o2/(o1+o2))*w(2,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-1/a2(2,2));
% eq. 26, b1 stands for parameters of wavelet function for k=1 as in eq. 10
t1=(x1-b1(1,1))/a1(1,1);
t2=(x2-b1(1,2))/a1(1,2);
da1(1,1)=(o1/(o1+o2))*w(1,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x1-b1(1,1))/(a1(1,1))^2);
t1=(x2-b1(1,2))/a1(1,2);
t2=(x1-b1(1,1))/a1(1,1);
da1(1,2)=(o1/(o1+o2))*w(1,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x2-b1(1,2))/(a1(1,2))^2);
t1=(x1-b1(2,1))/a1(2,1);
t2=(x2-b1(2,2))/a1(2,2);
da1(2,1)=(o1/(o1+o2))*w(2,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x1-b1(2,1))/(a1(2,1))^2);
t1=(x2-b1(2,2))/a1(2,2);
t2=(x1-b1(2,1))/a1(2,1);
da1(2,2)=(o1/(o1+o2))*w(2,1)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x2-b1(2,2))/(a1(2,2))^2);

% eq. 26, b1 stands for parameters of wavelet function for k=2 as in eq. 10
t1=(x1-b2(1,1))/a2(1,1);
t2=(x2-b2(1,2))/a2(1,2);
da2(1,1)=(o2/(o1+o2))*w(1,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x1-b1(1,1))/(a1(1,1))^2);
t1=(x2-b2(1,2))/a2(1,2);
t2=(x1-b2(1,1))/a2(1,1);
da2(1,2)=(o2/(o1+o2))*w(1,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x2-b1(1,2))/(a1(1,2))^2);
t1=(x1-b2(2,1))/a2(2,1);
t2=(x2-b2(2,2))/a2(2,2);
da2(2,1)=(o2/(o1+o2))*w(2,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x1-b1(2,1))/(a1(2,1))^2);
t1=(x2-b2(2,2))/a2(2,2);
t2=(x1-b2(2,1))/a2(2,1);
da2(2,2)=(o2/(o1+o2))*w(2,2)*gauswavf(t2,t2,1,2)*gauswavf(t1,t1,1,3)*(-(x2-b1(2,2))/(a1(2,2))^2);
% eq. 27
dw(1,1)=(o1/(o1+o2))*phi(1,1);
dw(1,2)=(o2/(o1+o2))*phi(1,2);
dw(2,1)=(o1/(o1+o2))*phi(2,1);
dw(2,2)=(o2/(o1+o2))*phi(2,2);
% eq. 28
dybar(1)=o1/(o1+o2);
dybar(2)=o2/(o1+o2);