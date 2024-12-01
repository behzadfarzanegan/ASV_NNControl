clear all;
%close all;
clc;
%rng('default');
% format short
NF=3000;
eta(:,1) = [10;-10;0];
eta(:,2) = eta(:,1);

eta_d(:,1) = [1,0,0]';
eta_d(:,2) = eta_d(:,1);

nV(:,1) = [-1;-1;0];
nV(:,2) = nV(:,1);

xhat(:,1)=[2;2;0];
xhat(:,2)=xhat(:,1);

eta_d3=atan2(eta(2,1),eta(1,1));
nV_star(:,1)=[0;0;0];
nV_star(:,2)=nV_star(:,1);

e(:,1)=nV(:,1)-nV_star(:,1);
e(:,2)=nV(:,2)-nV_star(:,2);

AugX(:,1) = [xhat(:,1);nV_star(:,1)];
AugX(:,2) = [xhat(:,2);nV_star(:,2)];
rc=0;
T=0.01;

%%%%%%%%%%%%%%%%%%
tau(:,1)=[0 0]';
tau(:,2)=tau(:,1);
X(:,1)=[0,0,0,0,0,0,0,0,1]';
X(:,2)=X(:,1);
% %%%%%%%%%%%%%%%%%%

%% system parameters

m11=189;m22=1036;m33=2411.1; 
m23=-543.5;m32=-543.5;
ay=0.595; a_psi=1.134;
%% control parameters
Ku=0.75*1;
Kv=0.75*10;
Kr=0.075*1;

%% observer 
load('WfVfall.mat') 
load('WgVgall.mat') 
Vf=Vf;
Wf=Wf;

Vg=1*Vg;
Wg=Wg;
v_obserf(:,:,1)=Vf;
W_obserf(:,:,1)=Wf;
v_obserf(:,:,2)=v_obserf(:,:,1);
W_obserf(:,:,2)=W_obserf(:,:,1);
alphao=0.0001;
betao=0.0001;

v_obserg(:,:,1)=Vg;
W_obserg(:,:,1)=Wg;
v_obserg(:,:,2)=v_obserg(:,:,1);
W_obserg(:,:,2)=W_obserg(:,:,1);

for k = 2:NF
   
    [eta(:,k+1),nV(:,k+1),eta_d(:,k),utilde,vtilde,nV_star(:,k),f] = USV_MODEL(eta(:,k),nV(:,k),tau(:,k),k);
    
%% observer
A= -3*eye(3);
C = eye(3);
observer_poles = 5*[-7, -3, -5];
L = place(A', C', observer_poles)';
Xfhat = [xhat(:,k);1];
Fnhat = Wf'*Actor_NL_gamma_bah(Xfhat,Vf,Neuron_Num_f);
% Fnnhat=(Fnhat+A*xhat(:,k)-xhat(:,k))/T;
f_x_hat=Fnhat(1);
f_y_hat=Fnhat(2);
f_psi_hat=Fnhat(3);
f_hat= [f_x_hat,f_y_hat,f_psi_hat]'/T;

Gnhat = Wg'*Actor_NL_gamma_bah(Xfhat,Vg,Neuron_Num_g);
Gnnhat=reshape(Gnhat,3,2);
% G = T*[1/m11 0;0 0;0 a_psi/m33];
% Gnnhat=G;
Xf = [nV(1,k);nV(2,k);nV(3,k);1];
y = C*Xf(1:end-1);
yhat= C*xhat(:,k);
ytilde = y-yhat;
xhat(:,k+1) = xhat(:,k)+ (T*A*xhat(:,k)+Fnhat+Gnnhat*tau(:,k)+ T*L*(y-yhat));

%% observer update law
temp = Actor_NL_gamma_bah(Xfhat,Vf,Neuron_Num_f)/(Actor_NL_gamma_bah(Xfhat,Vf,Neuron_Num_f)'*Actor_NL_gamma_bah(Xfhat,Vf,Neuron_Num_f)+1);
l=[0.1 0.2 0.3];
W_obserf(:,:,k+1) = (1-alphao)*W_obserf(:,:,k)+betao*temp*ytilde'*l';
Wf=W_obserf(:,:,k+1);
%%%%%%%%%%%%%%
temp = Actor_NL_gamma_bah(Xfhat,Vg,Neuron_Num_g)/(Actor_NL_gamma_bah(Xfhat,Vg,Neuron_Num_g)'*Actor_NL_gamma_bah(Xfhat,Vg,Neuron_Num_g)+1);
l=0.0001*eye(3);

W_obserg(:,:,k+1) = (1-alphao)*W_obserg(:,:,k)+.0000001*betao*[[temp*tau(1,k)*ytilde'*l']';[temp*tau(2,k)*ytilde'*l']']';
% Wg=W_obserg(:,:,k+1);


%%%%%%%%%%%%%%%%%
%% Dynamic control
u=nV(1,k+1);v=nV(2,k+1);r=nV(3,k+1); % dynamic states
uc=nV_star(1,k);vc=nV_star(2,k); % reference trajectory == virtual control policy

nV_stard=(nV_star(:,k)-nV_star(:,k-1))/T; % derevitave of refenece trajecory
ucd=nV_stard(1);vcd=nV_stard(2);

utilde = (u-uc);
vtilde = (v-vc);

% f_x=f(1);f_y=f(2);f_psi=f(3);
f_x=f_hat(1);f_y=f_hat(2);f_psi=f_hat(3);

rcd=a_psi/ay*(-Kv*vtilde-f_y+vcd)+f_psi;
rc = rc + T *rcd;
nV_star(3,k)=rc;
rtilde = (r-rc);
%% control

rcd=a_psi/ay*(-Kv*vtilde-f_y+vcd)+f_psi; % (56)
tau_x =  m11*(-f_x-Ku*utilde+ucd);
tau_psi= m33*(-f_psi-Kr*rtilde+rcd);
tau(:,k+1)=[tau_x;tau_psi];
    
end



figure(1);hold on;
% nn = 1:iter+1;
subplot 331; plot(eta(1,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(eta_d(1,:),'--r','LineWidth',2);
ylabel('x1,r1','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x1 and r1','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 332; plot(eta(2,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(eta_d(2,:),'--r','LineWidth',2);
ylabel('x2,r2','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 333; plot(eta(3,:),'b','LineWidth',2);
grid on;box on;
hold on
% plot(eta_d(3,:),'--r','LineWidth',2);
ylabel('\psi','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );


subplot 334; plot(nV(1,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(nV_star(1,:),'--r','LineWidth',2);
hold on
plot(xhat(1,:),'.-g','LineWidth',2);
ylabel('u ,uc','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );


subplot 335; plot(nV(2,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(nV_star(2,:),'--r','LineWidth',2);
hold on
plot(xhat(2,:),'.-g','LineWidth',2);
ylabel('v,vc','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );


subplot 336; plot(nV(3,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(nV_star(3,:),'--r','LineWidth',2);
hold on
plot(xhat(3,:),'.-g','LineWidth',2);
ylabel('r,rc','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );

