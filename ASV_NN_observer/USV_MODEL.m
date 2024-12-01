function [eta_next,nV_next,eta_d,utilde,vtilde,nV_star,f]=USV_MODEL(eta,nV,tau,k)

T=0.01;
t=k*T;

x=eta(1);
y=eta(2);
sie=eta(3);
J_eta=[cos(sie) -sin(sie)   0;
       sin(sie)  cos(sie)   0;
       0,        0,         1];

%% reference trajectory
eta_d = 1*[t t pi/4]'; % desired trajectory 
eta_dd = 1*[1 1 0]'; % derivative 


%% virtual control 
kx=0.9*1;
ky=0.9*1;
kr=.9*1;
xtilde = x -eta_d(1);
ytilde = y -eta_d(2);

xdot_d= eta_dd(1);
ydot_d= eta_dd(2);

xdot_t = -kx*xtilde + xdot_d; % $ \dot x = -k_x \tilde x +\dot x_d (23) Relaxed Control Barrier Function Based Control for Closest Approach by Underactuated USVs
ydot_t = -ky*ytilde + ydot_d;
ut = xdot_t*cos(sie) + ydot_t*sin(sie); % virtual control policies
vt = -xdot_t*sin(sie) + ydot_t*cos(sie);
rt= -kr*(sie - eta_d(3))+eta_dd(3);
nV_star = [ut;vt;rt];

%% Kinematics

eta_next=eta+T*J_eta*nV;

%% reference trajectory

u=nV(1);v=nV(2);r=nV(3); % dynamic states
uc=nV_star(1);vc=nV_star(2); % reference trajectory == virtual control polic
utilde = (u-uc);
vtilde = (v-vc);

%% system parameters

m11=189;m22=1036;m33=2411.1; 
m23=-543.5;m32=-543.5;
ay=0.595; a_psi=1.134;
d_x = 50*u+70*u*abs(u);
d_y = 948.2*v+385.4*r;
d_psi = 385.4*v +1926.9*r;

%% system dynamics
% Compute intermediate forces f_y' and f_psi'
f_y_prime = - (1/m22) * (m11 * u * r + d_y);
f_psi_prime = - (1/m33) * ((m22 - m11) * u * v + ((m23 + m32)/2) * u * r + d_psi);

% Compute the forces f_x, f_y, and f_psi
f_x = (1/m11) * (m22 * v * r + ((m23 + m32)/2) * r^2 - d_x);
f_y = a_psi * (f_y_prime - (m23/m22) * f_psi_prime);
f_psi = a_psi * (f_psi_prime - (m32/m33) * f_y_prime);

f= [f_x,f_y,f_psi]';

%% 
d_nV=[f_x+1/m11*tau(1);
      f_y+ay/m33*tau(2);
      f_psi+a_psi/m33*tau(2)];

nV_next=nV+T*d_nV;

end