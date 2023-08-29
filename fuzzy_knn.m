% function computing FWNN output as in Fig.2
function y=fuzzy_knn(x1,x2,s,c,a1,a2,b1,b2,w,ybar)
% membership computation as in eq. 2
mu11=exp(-((x1-c(1,1))/s(1,1))^2);
mu12=exp(-((x2-c(1,2))/s(1,2))^2);
mu21=exp(-((x1-c(2,1))/s(2,1))^2);
mu22=exp(-((x2-c(2,2))/s(2,2))^2);
% eq. 11
o1=mu11*mu12;
o2=mu21*mu22;
% wavelet functions as in eq. 8
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

% eq. 12
Y1=w(1,1)*phi(1,1)+w(2,1)*phi(2,1)+ybar(1);
Y2=w(1,2)*phi(1,2)+w(2,2)*phi(2,2)+ybar(2);
% eq. 15
y=(o1*Y1+o2*Y2)/(o1+o2);