%% simple_2d_orbit_collocation_casadi.m
% Direct collocation optimal control example in 2D orbital mechanics.
%
% State:
% X = [x; y; vx; vy]
%
% Control:
% U = [ux; uy]
%
% Dynamics:
% xdot = vx
% ydot = vy
% vxdot = -mu*x/r^3 + ux
% vydot = -mu*y/r^3 + uy
%
% Objective:
% minimize integral of control effort: int ||u||^2 dt
%
% Method:
% trapezoidal direct collocation
%
% Requires:
% CasADi installed and on MATLAB path
clear; clc; close all;
addpath('casadi-3.7.2-windows64-matlab2018b')
import casadi.*
%% ------------------------------------------------------------
% Problem parameters
% ------------------------------------------------------------
mu = 1.0; % nondimensional gravitational parameter
r0 = 1.0; % initial circular orbit radius
rf = 1.5; % target circular orbit radius
tf = 12.0; % fixed final time
N = 80; % number of mesh intervals
h = tf/N; % time step
umax = 0.08; % maximum thrust acceleration
nx = 4; % [x, y, vx, vy]
nu = 2; % [ux, uy]
%% ------------------------------------------------------------
% Create Opti problem
% ------------------------------------------------------------
opti = Opti();
% State variables at nodes
% X(:,k) = [x_k; y_k; vx_k; vy_k]
X = opti.variable(nx, N+1);
% Control variables at nodes
% U(:,k) = [ux_k; uy_k]
U = opti.variable(nu, N+1);
%% ------------------------------------------------------------
% Define dynamics as a CasADi Function
% ------------------------------------------------------------
Xs = MX.sym('Xs', nx);
Us = MX.sym('Us', nu);
x = Xs(1);
y = Xs(2);
vx = Xs(3);
vy = Xs(4);
ux = Us(1);
uy = Us(2);
r = sqrt(x^2 + y^2);
xdot = [
vx;
vy;
-mu*x/r^3 + ux;
-mu*y/r^3 + uy
];
f = Function('f', {Xs, Us}, {xdot});
%% ------------------------------------------------------------
% Initial condition: circular orbit at radius r0
% ------------------------------------------------------------
x0 = r0;
y0 = 0;
vx0 = 0;
vy0 = sqrt(mu/r0);
X_initial = [x0; y0; vx0; vy0];
opti.subject_to(X(:,1) == X_initial);
%% ------------------------------------------------------------
% Trapezoidal collocation constraints
% ------------------------------------------------------------
for k = 1:N
Xk = X(:,k);
Xkp1 = X(:,k+1);
Uk = U(:,k);
Ukp1 = U(:,k+1);
fk = f(Xk, Uk);
fkp1 = f(Xkp1, Ukp1);
defect = Xkp1 - Xk - h/2*(fk + fkp1);
opti.subject_to(defect == 0);
end
%% ------------------------------------------------------------
% Final orbit constraints
% ------------------------------------------------------------
% We do NOT fix the final angle.
% We only require the spacecraft to be on a circular orbit of radius rf.
%
% Conditions:
% ||r_f|| = rf
% ||v_f|| = sqrt(mu/rf)
% r_f dot v_f = 0
% angular momentum positive, i.e. prograde orbit
r_final = X(1:2,end);
v_final = X(3:4,end);
opti.subject_to(r_final.'*r_final == rf^2);
opti.subject_to(v_final.'*v_final == mu/rf);
opti.subject_to(r_final.'*v_final == 0);
h_final = r_final(1)*v_final(2) - r_final(2)*v_final(1);
opti.subject_to(h_final >= 0);
%% ------------------------------------------------------------
% Path constraints
% ------------------------------------------------------------
for k = 1:N+1
rk = X(1:2,k);
uk = U(:,k);
% Avoid collision / singularity near central body
opti.subject_to(rk.'*rk >= 0.5^2);
% Control acceleration bound
opti.subject_to(uk.'*uk <= umax^2);
end
%% ------------------------------------------------------------
% Objective function
% ------------------------------------------------------------
% Minimize integral ||u||^2 dt using trapezoidal quadrature.
J = 0;
for k = 1:N
Uk = U(:,k);
Ukp1 = U(:,k+1);
J = J + h/2*(Uk.'*Uk + Ukp1.'*Ukp1);
end
opti.minimize(J);
%% ------------------------------------------------------------
% Initial guess
% ------------------------------------------------------------
% A rough spiral-like guess from r0 to rf.
% This does not need to satisfy the dynamics exactly.
theta0 = 0;
thetaf = 2.5*pi;
for k = 1:N+1
s = (k-1)/N;
r_guess = (1-s)*r0 + s*rf;
theta = theta0 + s*thetaf;
xg = r_guess*cos(theta);
yg = r_guess*sin(theta);
vg = sqrt(mu/r_guess);
vxg = -vg*sin(theta);
vyg = vg*cos(theta);
opti.set_initial(X(:,k), [xg; yg; vxg; vyg]);
% Small tangential thrust guess
tx = -sin(theta);
ty = cos(theta);
opti.set_initial(U(:,k), 0.01*[tx; ty]);
end
%% ------------------------------------------------------------
% Solver settings
% ------------------------------------------------------------
p_opts = struct();
s_opts = struct();
p_opts.expand = true;
s_opts.max_iter = 2000;
s_opts.tol = 1e-8;
s_opts.print_level = 5;
opti.solver('ipopt', p_opts, s_opts);
%% ------------------------------------------------------------
% Solve
% ------------------------------------------------------------
sol = opti.solve();
X_sol = sol.value(X);
U_sol = sol.value(U);
J_sol = sol.value(J);
fprintf('\nSolved successfully.\n');
fprintf('Objective int ||u||^2 dt = %.8e\n', J_sol);
fprintf('Final radius = %.8f\n', norm(X_sol(1:2,end)));
fprintf('Target radius = %.8f\n', rf);
%% ------------------------------------------------------------
% Post-process
% ------------------------------------------------------------
tgrid = linspace(0, tf, N+1);
r_sol = sqrt(X_sol(1,:).^2 + X_sol(2,:).^2);
v_sol = sqrt(X_sol(3,:).^2 + X_sol(4,:).^2);
u_sol = sqrt(U_sol(1,:).^2 + U_sol(2,:).^2);
%% ------------------------------------------------------------
% Plot trajectory
% ------------------------------------------------------------
figure; hold on; grid on; axis equal;
plot(X_sol(1,:), X_sol(2,:), 'LineWidth', 2);
theta_plot = linspace(0, 2*pi, 400);
plot(r0*cos(theta_plot), r0*sin(theta_plot), '--', 'LineWidth', 1.2);
plot(rf*cos(theta_plot), rf*sin(theta_plot), '--', 'LineWidth', 1.2);
plot(0, 0, 'ko', 'MarkerFaceColor', 'k');
xlabel('x');
ylabel('y');
title('2D Direct Collocation Low-Thrust Transfer');
legend('Optimized trajectory', 'Initial circular orbit', ...
'Target circular orbit', 'Central body');
%% ------------------------------------------------------------
% Plot radius
% ------------------------------------------------------------
figure; hold on; grid on;
plot(tgrid, r_sol, 'LineWidth', 2);
yline(r0, '--');
yline(rf, '--');
xlabel('Time');
ylabel('Radius');
title('Orbital Radius');
%% ------------------------------------------------------------
% Plot speed
% ------------------------------------------------------------
figure; hold on; grid on;
plot(tgrid, v_sol, 'LineWidth', 2);
yline(sqrt(mu/r0), '--');
yline(sqrt(mu/rf), '--');
xlabel('Time');
ylabel('Speed');
title('Spacecraft Speed');
%% ------------------------------------------------------------
% Plot control magnitude
% ------------------------------------------------------------
figure; hold on; grid on;
plot(tgrid, u_sol, 'LineWidth', 2);
yline(umax, '--');
xlabel('Time');
ylabel('||u||');
title('Control Acceleration Magnitude');
%% ------------------------------------------------------------
% Plot control components
% ------------------------------------------------------------
figure; hold on; grid on;
plot(tgrid, U_sol(1,:), 'LineWidth', 2);
plot(tgrid, U_sol(2,:), 'LineWidth', 2);
xlabel('Time');
ylabel('Control acceleration');
title('Control Components');
legend('u_x', 'u_y');