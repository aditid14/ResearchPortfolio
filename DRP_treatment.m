%% Tumor-Normal-Immune Model with Forward Euler Scheme
%% Optimized for Dark Mode Plotting + Enhanced Clarity on Drug Effects
clearvars; close all; clc;
set(0, 'DefaultLineLineWidth', 2); % Global setting for thicker, clearer lines
fprintf('--- Starting T-N-I Model Simulation ---\n');
%% ========================================================================
%% PART 1: DIMENSIONAL SYSTEM SIMULATION (Reference for scaling)
%% ========================================================================
fprintf('1. Running Dimensional ODE System...\n');
% --- Parameters (dimensional system) ---
r1 = 0.2; K1 = 1e5;  % Tumor growth (r1) and carrying capacity (K1)
r2 = 0.15; K2 = 5e4; % Normal cell growth (r2) and carrying capacity (K2)
c2 = 5e-6; c3 = 5e-8; c4 = 1e-7; % Interaction coefficients
c1 = 1e-7; d1 = 0.05; s  = 0.5; alpha = 0.05; % Immune dynamics
% Initial conditions
N0 = 4e4; T0 = 1000; I0 = 500;
y_dim = [N0; T0; I0];
% Simulation setup
dt_dim = 0.1; tmax_dim = 200; nsteps_dim = floor(tmax_dim/dt_dim);
% Storage
tvals_dim = zeros(nsteps_dim,1); Nvals_dim = zeros(nsteps_dim,1);
Tvals_dim = zeros(nsteps_dim,1); Ivals_dim = zeros(nsteps_dim,1);
% Euler integration (dimensional)
for k = 1:nsteps_dim
    N = y_dim(1); T = y_dim(2); I = y_dim(3);
    dN = r2*N*(1-N/K2) - c4*T*N;
    dT = r1*T*(1-T/K1) - c2*I*T - c3*T*N;
    dI = s + alpha*T - c1*I*T - d1*I;
    y_dim = max(y_dim + dt_dim*[dN; dT; dI],0); % Enforce nonnegativity
    tvals_dim(k) = k*dt_dim; Nvals_dim(k) = y_dim(1);
    Tvals_dim(k) = y_dim(2); Ivals_dim(k) = y_dim(3);
end
%% ========================================================================
%% PART 2 & 3: DIMENSIONLESS SYSTEM - BASELINE (No Treatment)
%% ========================================================================
fprintf('2. Running Dimensionless Baseline ODE System...\n');
% The user is using known oscillation parameters for demonstration, overriding
% the natural scaling from Part 2.
r2_dim = 0.8; c2_dim = 0.8; c3_dim = 0.1; c4_dim = 0.05;
c1_dim = 0.05; d1_dim = 0.1; s_dim = 0.05; alpha_dim = 0.5;
% Initial conditions (dimensionless baseline)
n0 = 0.6; t0 = 0.1; i0 = 0.1;
y = [n0; t0; i0];
dt = 0.01; tmax = 400; nsteps = floor(tmax/dt);
tvals = zeros(nsteps,1); Nvals = zeros(nsteps,1);
Tvals = zeros(nsteps,1); Ivals = zeros(nsteps,1);
% Euler integration (dimensionless baseline)
for k = 1:nsteps
    n = y(1); t = y(2); i = y(3);
    dn = r2_dim*n*(1-n) - c4_dim*n*t;
    dtum = t*(1-t) - c2_dim*i*t - c3_dim*t*n;
    di = s_dim + alpha_dim*t - c1_dim*i*t - d1_dim*i;
    y = max(y + dt*[dn; dtum; di],0);
    tvals(k) = k*dt; Nvals(k) = y(1); Tvals(k) = y(2); Ivals(k) = y(3);
end
%% ========================================================================
%% PART 4-9: STEADY STATE, STABILITY, NULLCLINES & VISUALIZATIONS
%% Figures 1, 2, 3 show the intrinsic dynamics (limit cycle)
%% ========================================================================
fprintf('3. Calculating Steady States and Stability...\n');
steady_fun = @(x) [
    r2_dim*x(1)*(1-x(1)) - c4_dim*x(1)*x(2);
    x(2)*(1-x(2)) - c2_dim*x(2)*x(3) - c3_dim*x(2)*x(1);
    s_dim + alpha_dim*x(2) - c1_dim*x(2)*x(3) - d1_dim*x(3)
];
x0 = [0.5,0.1,0.1]; % Initial guess near coexistence
options = optimoptions('fsolve','Display','off','TolFun',1e-12,'TolX',1e-12);
[ss, ~, exitflag] = fsolve(steady_fun,x0,options);
if exitflag <= 0
    fprintf('Warning: fsolve did not converge (exitflag=%d).\n', exitflag);
else
    n_ss = ss(1); t_ss = ss(2); i_ss = ss(3);
    fprintf('   Coexistence Steady State (FP): n=%.4f, t=%.4f, i=%.4f\n', n_ss, t_ss, i_ss);
end
% Jacobian at Coexistence Fixed Point
J = [ r2_dim*(1-2*n_ss)-c4_dim*t_ss, -c4_dim*n_ss, 0;
     -c3_dim*t_ss, 1-2*t_ss - c2_dim*i_ss - c3_dim*n_ss, -c2_dim*t_ss;
      0, alpha_dim - c1_dim*i_ss, -c1_dim*t_ss - d1_dim ];
eigvals = eig(J);
fprintf('   Eigenvalues at FP: %.4f + %.4fi\n', real(eigvals), imag(eigvals));
fprintf('   (Real part > 0 indicates unstable spiral/limit cycle)\n');
% Nullclines
t_range = linspace(0.001,0.5,200);
% Tumor nullcline (n fixed to n_ss) for the t-i plane
i_tumor_nullcline = max((1 - t_range - c3_dim*n_ss)./c2_dim,0);
% Immune nullcline
i_immune_nullcline = max((s_dim + alpha_dim*t_range)./(c1_dim*t_range + d1_dim),0);
% --- Figure 1: Time Series (Dimensional vs Dimensionless) ---
figure('Position',[100 100 1200 400],'Name','Time Series Comparison');
subplot(1,2,1);
plot(tvals_dim,Nvals_dim,'Color','#44AAFF'); hold on;
plot(tvals_dim,Tvals_dim,'Color','#FF4444');
plot(tvals_dim,Ivals_dim,'Color','#00CC66');
xlabel('Time (days)'); ylabel('Population');
legend('Normal (N)','Tumor (T)','Immune (I)','Location','best');
title('Dimensional ODE System: Baseline Dynamics'); grid on;
subplot(1,2,2);
plot(tvals,Nvals,'Color','#44AAFF'); hold on;
plot(tvals,Tvals,'Color','#FF4444');
plot(tvals,Ivals,'Color','#00CC66');
xlabel('Time (\tau)'); ylabel('Population (scaled)');
legend('Normal (n)','Tumor (t)','Immune (i)','Location','best');
title('Dimensionless ODE System: Baseline Dynamics (Oscillation)'); grid on;
% --- Figure 2: Phase Plane & Nullclines (t-i plane) ---
figure('Position',[200 200 800 600],'Name','Phase Plane Analysis (t-i)');
plot(t_range,i_tumor_nullcline,'--','Color','#FFD700','LineWidth',3,'DisplayName','Tumor Nullcline (dt/d\tau=0)'); hold on;
plot(t_range,i_immune_nullcline,'--','Color','#00FFFF','LineWidth',3,'DisplayName','Immune Nullcline (di/d\tau=0)');
plot(Tvals,Ivals,'Color','#FF00FF','LineWidth',1.5,'DisplayName','Trajectory (Limit Cycle)');
plot(t_ss,i_ss,'ko','MarkerFaceColor','#FF4444','MarkerSize',10,'DisplayName','Fixed Point (FP)');
plot(t0,i0,'bs','MarkerFaceColor','#44AAFF','MarkerSize',10,'DisplayName','Initial Condition');
xlabel('Tumor (t)'); ylabel('Immune (i)');
title('Tumor-Immune Phase Plane with Nullclines (Baseline)'); legend('show','Location','best'); grid on; axis tight;
% --- Figure 3: Vector Field Plot (t-i plane) ---
figure('Position',[300 300 800 800],'Name','Vector Field (t-i) at Fixed n');
t_grid = linspace(0, 0.6, 20);
i_grid = linspace(0, 1.2, 20);
[T_mesh, I_mesh] = meshgrid(t_grid, i_grid);
% Fix n to the Coexistence Steady State value (n_ss)
n_fixed = n_ss;
dT_mesh = T_mesh.*(1-T_mesh) - c2_dim.*I_mesh.*T_mesh - c3_dim.*T_mesh.*n_fixed;
dI_mesh = s_dim + alpha_dim*T_mesh - c1_dim.*I_mesh.*T_mesh - d1_dim.*I_mesh;
magnitude = sqrt(dT_mesh.^2 + dI_mesh.^2);
% Normalize vectors for clean visualization
dT_norm = dT_mesh ./ (magnitude + 1e-8);
dI_norm = dI_mesh ./ (magnitude + 1e-8);
quiver(T_mesh, I_mesh, dT_norm, dI_norm, 0.8, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
hold on;
plot(t_range, i_tumor_nullcline, '--', 'Color', '#FFD700', 'LineWidth', 3, 'DisplayName', 'Tumor Nullcline');
plot(t_range, i_immune_nullcline, '--', 'Color', '#00FFFF', 'LineWidth', 3, 'DisplayName', 'Immune Nullcline');
plot(Tvals, Ivals, 'Color', '#FF00FF', 'LineWidth', 2, 'DisplayName', 'Trajectory');
plot(t_ss, i_ss, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', '#FF4444', 'LineWidth', 1.5, 'DisplayName', 'Fixed Point');
plot(t0, i0, 'bs', 'MarkerSize', 10, 'MarkerFaceColor', '#44AAFF', 'DisplayName', 'Initial Condition');
xlabel('Tumor (t)'); ylabel('Immune (i)');
title('Tumor-Immune Vector Field (n fixed at Steady State)');
legend('show','Location','best'); grid on; axis([0 0.6 0 1.2]);
%% ========================================================================
%% PART 10: ADD TREATMENT ODE MODEL (4th equation for drug u)
%% ========================================================================
fprintf('4. Running Treatment ODE System...\n');
% --- Drug Killing Rates (a1, a2, a3) ---
a1 = 0.4;   % Killing coefficient on Normal (n)
a2 = 0.6;   % Killing coefficient on Tumor (t) - MUST be highest for efficacy
a3 = 0.2;   % Killing coefficient on Immune (i) - MUST be lowest to prevent suppression
du = 0.2;   % Decay rate of drug u (dimensionless)
fprintf('   Drug parameters: a1 (Normal)=%.2f, a2 (Tumor)=%.2f, a3 (Immune)=%.2f\n', a1, a2, a3);
% --- Treatment Schedule ---
v_mode = 'pulse'; % 'constant', 'pulse', or 'single'
v0 = 1.5;            % Drug infusion amplitude (max rate of u production)
pulse_period = 40;   % Period in tau units
pulse_width = 5;     % Width of each pulse
t_bolus = 50;        % For single bolus option
fprintf('   Schedule: %s (v0=%.1f, Period=%.0f, Width=%.0f)\n', v_mode, v0, pulse_period, pulse_width);
% Simulation setup
dt_treat = 0.01; tmax_treat = 400; nsteps_treat = floor(tmax_treat/dt_treat);
y_t = [n0; t0; i0; 0]; % y = [n; t; i; u] (initial drug u=0)
tvals_t = zeros(nsteps_treat,1); Nvals_t = zeros(nsteps_treat,1);
Tvals_t = zeros(nsteps_treat,1); Ivals_t = zeros(nsteps_treat,1); Uvals_t = zeros(nsteps_treat,1);
% Euler integration (treatment ODE)
for k = 1:nsteps_treat
    n = y_t(1); tt = y_t(2); ii = y_t(3); u = y_t(4);
    tau = (k-1)*dt_treat;
    
    % Determine drug infusion rate v_t
    switch v_mode
        case 'constant'
            v_t = v0;
        case 'pulse'
            in_pulse = mod(tau,pulse_period) < pulse_width;
            v_t = v0 * double(in_pulse);
        case 'single'
            v_t = v0 * double( abs(tau - t_bolus) < dt_treat*2 );
        otherwise
            v_t = 0;
    end
    
    % Non-linear kill factor based on drug concentration u
    kill_factor = (1 - exp(-u)); 
    
    % ODEs with drug killing terms
    dn = r2_dim*n*(1-n) - c4_dim*n*tt - a1*kill_factor*n;            % Normal cells killed by a1
    dtt = tt*(1-tt) - c2_dim*ii*tt - c3_dim*tt*n - a2*kill_factor*tt; % Tumor cells killed by a2
    dii = s_dim + alpha_dim*tt - c1_dim*ii*tt - d1_dim*ii - a3*kill_factor*ii; % Immune cells killed by a3
    duu = v_t - du*u;                                                 % Drug dynamics
    
    y_t = max(y_t + dt_treat*[dn; dtt; dii; duu],0);
    tvals_t(k) = tau; Nvals_t(k) = y_t(1); Tvals_t(k) = y_t(2);
    Ivals_t(k) = y_t(3); Uvals_t(k) = y_t(4);
end
% --- Figure 4: Treatment ODE Time Series ---
figure('Position',[100 100 1200 400],'Name','Treatment ODE Time Series');
subplot(1,2,1);
plot(tvals_t,Nvals_t,'Color','#44AAFF','DisplayName','Normal (n)'); hold on;
plot(tvals_t,Tvals_t,'Color','#FF4444','DisplayName','Tumor (t)');
plot(tvals_t,Ivals_t,'Color','#00CC66','DisplayName','Immune (i)');
xlabel('Time (\tau)'); ylabel('Population (scaled)');
title(sprintf('Dimensionless + Treatment: %s Schedule', upper(v_mode))); grid on;
legend('show','Location','best');
subplot(1,2,2);
% Drug concentration should be a cyan color for visibility
plot(tvals_t,Uvals_t,'Color','#00FFFF','LineWidth',2,'DisplayName','Drug u(\tau)');
xlabel('Time (\tau)'); ylabel('Treatment variable u(\tau)'); title('Drug Concentration Profile'); grid on;
% --- Figure 5: Phase Plane Comparison (t-i) ---
figure('Position',[200 200 800 600],'Name','Treatment Trajectory Comparison');
plot(Tvals_t,Ivals_t,'Color','#FF00FF','LineWidth',1.5,'DisplayName','Trajectory (Treatment)'); hold on;
plot(Tvals,Ivals,'--','Color','#44AAFF','LineWidth',1.5,'DisplayName','Trajectory (No Treatment)');
plot(t_ss,i_ss,'ko','MarkerFaceColor','#FF4444','MarkerSize',10,'DisplayName','Fixed Point (Baseline)');
xlabel('Tumor (t)'); ylabel('Immune (i)');
title('Tumor vs Immune: Treatment vs Baseline'); grid on;
legend('show','Location','best');
%% ========================================================================
%% PART 11: SPATIAL PDE TREATMENT MODEL (Nutrient Diffusion)
%% Figure 6 shows the final spatial distribution
%% ========================================================================
fprintf('5. Running Spatial PDE System...\n');
% Domain and discretization
Lx = 1.0; Ly = 1.0; Nx = 80; Ny = 80;
dx = Lx/(Nx-1); dy = Ly/(Ny-1);
x = linspace(0,Lx,Nx); ygrid = linspace(0,Ly,Ny);
% Diffusion coefficients
D = 0.1;        % Nutrient diffusion (D)
smallD = 5e-2; % <--- ADJUSTMENT HERE: Increased cell motility (T and I) from 1e-2 to 5e-2. 
               % This higher motility relative to D=0.1 is critical for breaking the 
               % strong vertical nutrient stratification and generating non-uniform patterns.
% Time stepping: enforce CFL stability
dt_max = 1/(2*D*(1/dx^2 + 1/dy^2));
dt_pde = min(1e-3, 0.8*dt_max); % Safe dt
tmax_pde = 100; nsteps_pde = floor(tmax_pde/dt_pde);
save_every = max(1,floor(nsteps_pde/50)); % Save status periodically
% Consumption and coupling params (Nutrient N is the PDE variable)
alpha_n = 0.5;    % Baseline consumption
lambda_T = 2.0;   % Tumor excess consumption multiplier
H = 0.8 * ones(Ny,Nx); % Host/Normal cell background density
% Initialize fields
Ifield = 0.05 * ones(Ny,Nx);
[Xg, Yg] = meshgrid(x, ygrid);
xc = 0.5; yc = 0.5; sigma = 0.08;
% Localized tumor initial condition
Tfield = 0.4 * exp(-((Xg-xc).^2 + (Yg-yc).^2)/(2*sigma^2)); % Correct initial condition
% Add small spatial noise to encourage pattern formation
Tfield = Tfield + 0.01 * randn(Ny,Nx); 
Tfield = max(Tfield, 0); 

Nfield = 1.0 * ones(Ny,Nx); % Nutrient N, uniform start
% Treatment state for PDE run (spatially uniform systemic u)
u_p = 0;
uvals_p = zeros(nsteps_pde,1);
meanT_p = zeros(nsteps_pde,1);
% PDE Integration Loop
for k = 1:nsteps_pde
    tau_p = (k-1)*dt_pde;
    
    % 1. Drug Update (Uniform ODE in space)
    switch v_mode
        case 'constant'
            v_t_p = v0;
        case 'pulse'
            v_t_p = v0 * double(mod(tau_p,pulse_period) < pulse_width);
        case 'single'
            v_t_p = v0 * double( abs(tau_p - t_bolus) < dt_pde*2 );
        otherwise
            v_t_p = 0;
    end
    duu = v_t_p - du*u_p;
    u_p = max(u_p + dt_pde*duu, 0);
    kill_local = a2*(1 - exp(-u_p)); % Local kill factor for tumor (a2 used)
    
    % 2. Nutrient N Update (Diffusion-Consumption PDE)
    % Vectorized Laplacian (interior) using circshift for periodic/approx
    N_xp = circshift(Nfield, [0, -1]); N_xm = circshift(Nfield, [0, 1]);
    N_yp = circshift(Nfield, [-1, 0]); N_ym = circshift(Nfield, [1, 0]);
    lapN = (N_xp - 2*Nfield + N_xm)/dx^2 + (N_yp - 2*Nfield + N_ym)/dy^2;
    
    consume = alpha_n.*(H + Ifield).*Nfield + lambda_T*alpha_n.*Tfield.*Nfield;
    N_new = Nfield + dt_pde*( D*lapN - consume );
    
    % Impose Dirichlet Boundary Conditions (BCs) at top/bottom (vessel source)
    N_new(1,:) = 1.0;
    N_new(end,:) = 1.0;
    % Impose Periodic BCs Left/Right (copy second/second-last columns)
    N_new(:,1) = N_new(:,end-1);
    N_new(:,end) = N_new(:,2);
    Nfield = max(N_new, 0);
    % 3. Tumor T Update (Reaction-Diffusion)
    % Note: Nutrient Nfield acts as the logistic factor for T growth
    growth_T = Tfield .* (1 - Tfield);
    death_by_immune = c2_dim .* Ifield .* Tfield;
    death_by_host = c3_dim .* Tfield .* H; % Host H is now the N cell analogue
    growth_factor = max(Nfield, 0); % Growth is proportional to local nutrient N
    
    dT_dt = growth_T .* growth_factor - death_by_immune - death_by_host - kill_local.*Tfield;
    Tfield = Tfield + dt_pde*dT_dt;
    
    % 4. Immune I Update (Reaction-Diffusion)
    dI_dt = s_dim + alpha_dim.*Tfield - c1_dim.*Ifield.*Tfield - d1_dim.*Ifield - a3*(1 - exp(-u_p)).*Ifield;
    Ifield = Ifield + dt_pde*dI_dt;
    
    % Small cell diffusion/motility (simple 5-point laplacian via circshift)
    lapT = (circshift(Tfield, [0,1]) + circshift(Tfield,[0,-1]) - 2*Tfield)/dx^2 + ...
           (circshift(Tfield,[1,0]) + circshift(Tfield,[-1,0]) - 2*Tfield)/dy^2;
    lapI = (circshift(Ifield, [0,1]) + circshift(Ifield,[0,-1]) - 2*Ifield)/dx^2 + ...
           (circshift(Ifield,[1,0]) + circshift(Ifield,[-1,0]) - 2*Ifield)/dy^2;
    Tfield = max(Tfield + smallD*dt_pde*lapT, 0);
    Ifield = max(Ifield + smallD*dt_pde*lapI, 0);
    
    % Record and log
    uvals_p(k) = u_p;
    meanT_p(k) = mean(Tfield(:));
    
    if mod(k,save_every) == 0
        fprintf('   PDE step %d/%d (tau=%.4f) meanT=%.6f u=%.4f\n', k, nsteps_pde, tau_p, meanT_p(k), u_p);
    end
end
% --- Figure 6: Final Spatial Fields ---
figure('Position',[100 100 1200 500],'Name','Final Spatial Fields (T, I, N)');
subplot(1,3,1);
imagesc(x,ygrid,Tfield); axis xy image; colorbar;
title(sprintf('Tumor density T(x,y) at \\tau=%.1f',tmax_pde));
xlabel('x'); ylabel('y'); colormap(hot);
subplot(1,3,2);
imagesc(x,ygrid,Ifield); axis xy image; colorbar;
title(sprintf('Immune density I(x,y) at \\tau=%.1f',tmax_pde));
colormap(spring);
subplot(1,3,3);
imagesc(x,ygrid,Nfield); axis xy image; colorbar;
title(sprintf('Nutrient N(x,y) at \\tau=%.1f',tmax_pde));
colormap(winter);
% --- Figure 7: PDE Time Series and Comparison ---
t_pde_time = linspace(0,tmax_pde,nsteps_pde);
figure('Position',[100 100 1000 400],'Name','Treatment Comparison (Mean Tumor)');
subplot(1,2,1);
plot(t_pde_time, meanT_p,'Color','#FF4444','LineWidth',2.5,'DisplayName','PDE Mean T');
xlabel('Time (\tau)'); ylabel('Mean Tumor Density'); title('Mean Tumor Density (PDE Run)'); grid on;
subplot(1,2,2);
% Determine comparison window
t_compare_max = min(tmax_pde, max(tvals_t));
idx_ode = find(tvals_t <= t_compare_max);
if isempty(idx_ode)
    fprintf('Warning: Skipping direct AUC comparison due to no time overlap.\n');
else
    plot(tvals_t(idx_ode), Tvals_t(idx_ode), 'Color', '#44AAFF', 'LineWidth', 2.5, 'DisplayName', 'ODE Tumor (t)'); hold on;
    plot(t_pde_time, meanT_p, 'Color', '#FF4444', 'LineWidth', 2.5, 'DisplayName', 'PDE Mean Tumor (T)');
    xlabel('Time (\tau)'); ylabel('Tumor Population'); legend('show','Location','best');
    title('Comparison: ODE Treatment vs Spatial PDE Mean Tumor');
    grid on;
    
    % Compute AUCs
    AUC_ode = trapz(tvals_t(idx_ode), Tvals_t(idx_ode));
    AUC_pde = trapz(t_pde_time, meanT_p);
    fprintf('\nAUC over comparison window (tau=0 to %.1f):\n', t_compare_max);
    fprintf('   ODE_treatment = %.6f\n', AUC_ode);
    fprintf('   PDE_spatial_mean = %.6f\n', AUC_pde);
    fprintf('   This shows the difference between well-mixed and spatial models.\n');
end
%% ========================================================================
%% PART 12: SIMPLIFIED BIFURCATION ANALYSIS (Hopf)
%% Vary c2_dim (Immune killing rate) to see fixed point -> limit cycle
%% ========================================================================
fprintf('6. Running Simplified Bifurcation Analysis (c2_dim vs Tumor t)...\n');
% 1. Define the parameter range
c2_range = linspace(0.1, 1.5, 30); 
L = length(c2_range);
% 2. Initialize storage arrays with NaN
stable_pts = NaN(L, 1);       % To store stable fixed points
min_amplitude = NaN(L, 1);    % To store min of oscillation (cycle)
max_amplitude = NaN(L, 1);    % To store max of oscillation (cycle)
% 3. Loop using INTEGER index 'i' (1, 2, 3...)
for i = 1:L
    c2_curr = c2_range(i); % Get the actual parameter value
    
    % Solve for Fixed Point with current c2_curr
    steady_fun_bif = @(x) [
        r2_dim*x(1)*(1-x(1)) - c4_dim*x(1)*x(2);
        x(2)*(1-x(2)) - c2_curr*x(2)*x(3) - c3_dim*x(2)*x(1);
        s_dim + alpha_dim*x(2) - c1_dim*x(2)*x(3) - d1_dim*x(3)
    ];
    
    [ss_bif, ~, flag] = fsolve(steady_fun_bif, [0.5, 0.1, 0.1], options);
    
    if flag > 0
        t_fp = ss_bif(2);
        n_fp = ss_bif(1); 
        i_fp = ss_bif(3);
        
        % Check Stability (Eigenvalues)
        J_bif = [ r2_dim*(1-2*n_fp)-c4_dim*t_fp, -c4_dim*n_fp, 0;
                 -c3_dim*t_fp, 1-2*t_fp - c2_curr*i_fp - c3_dim*n_fp, -c2_curr*t_fp;
                  0, alpha_dim - c1_dim*i_fp, -c1_dim*t_fp - d1_dim ];
        eigs_bif = eig(J_bif);
        
        if all(real(eigs_bif) < 0)
            % Stable Fixed Point
            stable_pts(i) = t_fp; 
        else
            % Unstable -> Limit Cycle -> Run short simulation
            dt_bif = 0.5; steps_bif = 1000; % Fast, coarse simulation
            y_temp = [n_fp; t_fp+0.01; i_fp]; % Slight perturbation
            
            % Store temporary trajectory
            traj_segment = zeros(steps_bif, 1);
            for k=1:steps_bif
                n=y_temp(1); t=y_temp(2); imm=y_temp(3);
                dn = r2_dim*n*(1-n) - c4_dim*n*t;
                dtum = t*(1-t) - c2_curr*imm*t - c3_dim*t*n;
                dimm = s_dim + alpha_dim*t - c1_dim*imm*t - d1_dim*imm;
                y_temp = max(y_temp + dt_bif*[dn; dtum; dimm], 0);
                traj_segment(k) = y_temp(2);
            end
            % Take the last 50% of points to capture the cycle
            analysis_data = traj_segment(floor(end/2):end);
            
            min_amplitude(i) = min(analysis_data);
            max_amplitude(i) = max(analysis_data);
        end
    end
end
% --- Figure 8: Bifurcation Diagram ---
figure('Position',[100 100 700 500],'Name','Bifurcation Diagram');
% Plot Stable points
plot(c2_range, stable_pts, 's-', 'Color', '#44AAFF', 'LineWidth', 2, 'MarkerFaceColor', '#44AAFF', 'DisplayName', 'Stable Fixed Point'); hold on;
% Plot Oscillations (Min and Max)
plot(c2_range, max_amplitude, 'o-', 'Color', '#FF4444', 'LineWidth', 1.5, 'DisplayName', 'Oscillation Max');
plot(c2_range, min_amplitude, 'o-', 'Color', '#FF4444', 'LineWidth', 1.5, 'HandleVisibility', 'off'); % Hide duplicate legend
xlabel('Immune Killing Parameter (c2_{dim})');
ylabel('Tumor Population (t)');
title('Hopf Bifurcation: Transition from Stable to Oscillatory');
legend('show', 'Location', 'best');
grid on;
fprintf('\n--- Script fully finished. Check Figure 8 for Bifurcation. ---\n');