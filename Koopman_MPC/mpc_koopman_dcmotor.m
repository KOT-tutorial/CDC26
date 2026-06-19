%% =========================================================
%  mpc_koopman_dcmotor.m   — CDC 2026 KOT tutorial paper
%
%  PART 2 of 2.  Loads the four trained Koopman models from
%  Part 1 (DC motor) and runs reference-tracking MPC in one of
%  three modes:
%    Mode 1 : pick one form; compare its ONE-STEP and MULTI-STEP
%             versions in closed loop.
%    Mode 2 : run all four forms (MULTI-STEP) and cross-compare.
%    Mode 3 : run all four forms (ONE-STEP) and cross-compare.
%  Reports per-step CPU (mean / median / worst) and tracking ISE
%  in a single table, and plots closed-loop trajectories.
%
%  Reference: angular velocity (x2) sinusoid centred at -40 rad/s.
%  ref_period_s = 1.5 s, so T_mpc * Ts = 3 s contains 2 periods.
%
%  Input  : koopman_dcmotor_model_<kernel>.mat   (saved by Part 1)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  USER SETTINGS
%% =========================================================

% --- MPC mode ---
%   1 : single form, compare ONE-STEP vs MULTI-STEP
%   2 : all 4 forms (MULTI-STEP), cross-comparison
%   3 : all 4 forms (ONE-STEP),   cross-comparison
mpc_mode = 2;

% --- Choice of Koopman form (used only in mode 1) ---
%   1 : Linear     2 : Bilinear     3 : GeKo     4 : KCF
mpc_form = 3;

% --- Which trained model file to load (one per state kernel) ---
%   Part 1 saves koopman_dcmotor_model_<kernel>.mat, one per state kernel.
%     0 : AUTO -- load whatever Part 1 wrote (newest, if several kernels exist)
%     1 : RBF    2 : Rational Quadratic    3 : Matern 5/2    4 : Polynomial
%   Auto (default) needs no edit when you switch kernels in Part 1; pick a
%   specific 1..4 only to force one kernel when several files are on disk.
%   NOTE: filenames are tagged by KERNEL only. If you retrain the SAME kernel
%   with the other INPUT_NONLINEARITY, Part 1 overwrites the file (rename it,
%   or ask to add a nonlinearity tag).
kernel_choice = 0;
ktag_map = {'rbf','rq','matern','poly'};
if kernel_choice == 0
    cand = dir('koopman_dcmotor_model_*.mat');
    if isempty(cand)
        error(['No koopman_dcmotor_model_*.mat in the current directory:\n' ...
               '   %s\n   Run Part 1 (learn_koopman_dcmotor.m) first.'], pwd);
    end
    [~, newest] = max([cand.datenum]);
    model_file  = cand(newest).name;
    if numel(cand) > 1
        fprintf('   Auto-detect: %d model files found; using newest (%s).\n', ...
            numel(cand), model_file);
        fprintf('   Set kernel_choice = 1..4 to force a specific kernel.\n');
    end
elseif ismember(kernel_choice, 1:4)
    model_file = sprintf('koopman_dcmotor_model_%s.mat', ktag_map{kernel_choice});
else
    error('kernel_choice must be 0 (auto) or 1..4 (got %g).', kernel_choice);
end

% --- MPC settings (reference tracking on x2) ---
Hp           = 20;            % prediction horizon (steps)
Q_y          = 4.0;           % tracking weight on x2 output
R_u          = 0.1;          % physical input magnitude weight
R_du         = 0.01;          % physical delta-u weight
u_max_phys   = 4.0;           % physical input bound
T_mpc        = 600;           % closed-loop steps (3 s with Ts=5 ms)
x0_mpc_phys  = [0; 0];        % motor at rest

% Reference (physical x2): mild sinusoid centred at -40 rad/s.
% Period 1.5 s, so 3 s simulation = 2 periods.
ref_offset   = -40;           % rad/s  (mean operating point)
ref_amp      =  15;           % rad/s  (mildly more testing)
ref_period_s = 1.5;           % s
ref_fun      = @(t) ref_offset + ref_amp * sin(2*pi*t / ref_period_s);

% --- fmincon solver algorithm ---
%   1 : SQP            (default; fast, good for smooth problems)
%   2 : interior-point (more robust for non-convex problems)
solver_choice = 1;
switch solver_choice
    case 1, fmincon_algo = 'sqp';
    case 2, fmincon_algo = 'interior-point';
    otherwise, error('solver_choice must be 1 or 2.');
end

% --- Safety: warn if iteration 1 is unexpectedly slow ---
slow_iter_thresh_s = 60;
%   1 :  3 minutes
%   2 :  5 minutes
%   3 : 10 minutes
cumulative_cap_choice = 2;
switch cumulative_cap_choice
    case 1, cumulative_thresh_s =  3 * 60;
    case 2, cumulative_thresh_s =  5 * 60;
    case 3, cumulative_thresh_s = 10 * 60;
    otherwise, error('cumulative_cap_choice must be 1, 2, or 3.');
end

%% =========================================================
%  LOAD TRAINED MODELS  (with safety checks)
%% =========================================================
fprintf('\n[1] Loading trained models from %s...\n', model_file);

% --- Check #1: file exists ---
if ~isfile(model_file)
    error(['Cannot find ''%s'' in the current directory.\n' ...
           '   Run Part 1 (learn_koopman_dcmotor.m) first; it saves this file.\n' ...
           '   If you ran Part 1 elsewhere, update the model_file path at the top.'], ...
           model_file);
end

% --- Check #2: required variables are present ---
required = {'M_ms','M_1s','model_names','params','D_lin','nz','nv', ...
            'deg_cheb_u','deg_poly_x','use_centres_x','C_x','hp_x', ...
            'KERNEL_TYPE','INPUT_NONLINEARITY','kernel_name_x','kernel_name_u','prepend_state'};
info = whos('-file', model_file);
present = {info.name};
missing = setdiff(required, present);
if ~isempty(missing)
    error(['The file ''%s'' is missing required variable(s):\n   %s\n' ...
           '   It looks like an older save. Re-run Part 1 to refresh it.'], ...
           model_file, strjoin(missing, ', '));
end

S = load(model_file);

% --- Check #3: 4 models with expected kinds ---
M_ms = S.M_ms;  M_1s = S.M_1s;  names = S.model_names;
expected_kinds = {'linear','bilinear','geko','kcf'};
if numel(M_ms) ~= 4 || numel(M_1s) ~= 4 || numel(names) ~= 4
    error('Expected 4 forms; got M_ms=%d, M_1s=%d, names=%d.', ...
        numel(M_ms), numel(M_1s), numel(names));
end
for m = 1:4
    if ~isfield(M_ms{m}, 'kind') || ~strcmp(M_ms{m}.kind, expected_kinds{m})
        error('M_ms{%d} kind ''%s'', expected ''%s''.', m, M_ms{m}.kind, expected_kinds{m});
    end
    if ~isfield(M_1s{m}, 'kind') || ~strcmp(M_1s{m}.kind, expected_kinds{m})
        error('M_1s{%d} kind ''%s'', expected ''%s''.', m, M_1s{m}.kind, expected_kinds{m});
    end
end

% Pull remaining variables
params        = S.params;
D_lin         = S.D_lin;
nz            = S.nz;
nv            = S.nv;
deg_cheb_u    = S.deg_cheb_u;
deg_poly_x    = S.deg_poly_x;
use_centres_x = S.use_centres_x;
C_x           = S.C_x;
hp_x          = S.hp_x;
KERNEL_TYPE   = S.KERNEL_TYPE;
INPUT_NONLINEARITY = S.INPUT_NONLINEARITY;
kernel_name_x = S.kernel_name_x;
kernel_name_u = S.kernel_name_u;
prepend_state = S.prepend_state;

% Reconstruct the state-kernel handle from KERNEL_TYPE
switch KERNEL_TYPE
    case 1, kfun_x = @(X,C,hp) rbf_kernel(X,C,hp(1));
    case 2, kfun_x = @(X,C,hp) rq_kernel(X,C,hp(1),hp(2));
    case 3, kfun_x = @(X,C,hp) matern52_kernel(X,C,hp(1));
    case 4, kfun_x = [];
    otherwise, error('Unknown KERNEL_TYPE in saved file.');
end
% Reconstruct the physical nonlinearity handle from INPUT_NONLINEARITY
switch INPUT_NONLINEARITY
    case 1, params.f_fun = @(u) 2 * tanh(u);
    case 2, params.f_fun = @(u) 2 * tanh(u .* cos(u));
    otherwise, error('Unknown INPUT_NONLINEARITY in saved file.');
end

% Validate user choices
if ~ismember(mpc_mode, [1, 2, 3])
    error('mpc_mode must be 1, 2, or 3.');
end
if mpc_mode == 1 && ~ismember(mpc_form, 1:numel(names))
    error('mpc_form must be in 1..%d (got %d).', numel(names), mpc_form);
end

fprintf('   State kernel : %s\n', kernel_name_x);
fprintf('   Input kernel : %s\n', kernel_name_u);
fprintf('   Nonlinearity : %s\n', params.f_name);
fprintf('   MPC solver   : fmincon / %s\n', fmincon_algo);
fprintf('   Ref period   : %.2f s   (T_mpc = %.2f s = %.2f periods)\n', ...
    ref_period_s, T_mpc * params.Ts, (T_mpc * params.Ts) / ref_period_s);
switch mpc_mode
    case 1
        fprintf('   Mode 1 form  : %s  (one-step vs multi-step)\n', names{mpc_form});
    case 2
        fprintf('   Mode 2       : all 4 forms (multi-step), cross-comparison\n');
    case 3
        fprintf('   Mode 3       : all 4 forms (one-step), cross-comparison\n');
end

%% =========================================================
%  CLOSED-LOOP MPC
%% =========================================================

fmin_opts = optimoptions('fmincon', ...
    'Algorithm', fmincon_algo, 'Display','off', 'MaxIterations',100, ...
    'OptimalityTolerance',1e-4, 'StepTolerance',1e-6, ...
    'SpecifyObjectiveGradient', true);   % analytic gradient (mpc_cost_track_with_grad)
lb = -u_max_phys * ones(Hp, 1);   ub = u_max_phys * ones(Hp, 1);
u_init_default = zeros(Hp, 1);

% Output map: y = x2 (physical) = C_out * Sx_inv * D_lin * z
C_out  = [0, 1];
H_lift = C_out * params.Sx_inv * D_lin;   % 1 x nz

% Run MPC per mode
runs    = cell(0, 1);
headers = cell(0, 1);

switch mpc_mode
    case 1
        fprintf('\n[2] Mode 1: %s  (one-step vs multi-step)\n', names{mpc_form});
        M_pair     = {M_1s{mpc_form}, M_ms{mpc_form}};
        label_pair = {'one-step',     'multi-step'};
        for r = 1:2
            fprintf('\n[2.%d] MPC with %s %s model...\n', r, names{mpc_form}, label_pair{r});
            runs{end+1}    = run_mpc_for_model(M_pair{r}, x0_mpc_phys, H_lift, ...
                Q_y, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
                fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
                deg_poly_x, deg_cheb_u, ref_fun, slow_iter_thresh_s, ...
                cumulative_thresh_s, label_pair{r}); 
            headers{end+1} = label_pair{r}; 
        end
    case 2
        fprintf('\n[2] Mode 2: all 4 forms (multi-step), cross-comparison\n');
        for m = 1:numel(names)
            fprintf('\n[2.%d] MPC with %s multi-step model...\n', m, names{m});
            runs{end+1}    = run_mpc_for_model(M_ms{m}, x0_mpc_phys, H_lift, ...
                Q_y, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
                fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
                deg_poly_x, deg_cheb_u, ref_fun, slow_iter_thresh_s, ...
                cumulative_thresh_s, names{m}); 
            headers{end+1} = names{m}; 
        end
    case 3
        fprintf('\n[2] Mode 3: all 4 forms (one-step), cross-comparison\n');
        for m = 1:numel(names)
            fprintf('\n[2.%d] MPC with %s one-step model...\n', m, names{m});
            runs{end+1}    = run_mpc_for_model(M_1s{m}, x0_mpc_phys, H_lift, ...
                Q_y, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
                fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
                deg_poly_x, deg_cheb_u, ref_fun, slow_iter_thresh_s, ...
                cumulative_thresh_s, names{m}); 
            headers{end+1} = names{m}; 
        end
end
n_runs = numel(runs);

%% =========================================================
%  SUMMARY TABLE : metrics as rows, runs as columns
%% =========================================================
metric_names = {'mean CPU [s]', 'median CPU', 'worst CPU', 'ISE_track'};
M_vals = nan(numel(metric_names), n_runs);
for r = 1:n_runs
    cpu = runs{r}.cpu;
    if ~isempty(cpu)
        M_vals(1, r) = mean(cpu);
        M_vals(2, r) = median(cpu);
        M_vals(3, r) = max(cpu);
    end
    M_vals(4, r) = runs{r}.ise;
end

fprintf('\n=========================================\n');
switch mpc_mode
    case 1
        fprintf('   MPC summary  (form = %s, one-step vs multi-step)\n', names{mpc_form});
    case 2
        fprintf('   MPC summary  (multi-step, all 4 forms)\n');
    case 3
        fprintf('   MPC summary  (one-step, all 4 forms)\n');
end
fprintf('=========================================\n');
fprintf('   %-14s', 'Metric');
for r = 1:n_runs, fprintf('  %12s', headers{r}); end
fprintf('\n');
for k = 1:numel(metric_names)
    fprintf('   %-14s', metric_names{k});
    row = M_vals(k, :);
    if all(isnan(row))
        is_best = false(size(row));
    else
        best_val = min(row, [], 'omitnan');
        tol      = max(1e-12, 5e-5 * abs(best_val));
        is_best  = ~isnan(row) & abs(row - best_val) <= tol;
    end
    for r = 1:n_runs
        if isnan(row(r))
            fprintf('  %12s', 'n/a');
        elseif is_best(r)
            fprintf('  <strong>%12.4e</strong>', row(r));
        else
            fprintf('  %12.4e', row(r));
        end
    end
    fprintf('\n');
end
fprintf('   %-14s', 'steps');
for r = 1:n_runs
    fprintf('  %12s', sprintf('%d/%d', runs{r}.t_done, T_mpc));
end
fprintf('\n');
fprintf('=========================================\n');

%% =========================================================
%  PLOTS  (physical units)
%
%  Mode 1: one figure per run (one-step, multi-step).
%  Mode 2/3: ONE overlaid figure with 4 coloured curves (one per form).
%% =========================================================
colors = [0.00 0.45 0.74;    % Linear   — blue
          0.00 0.60 0.20;    % Bilinear — green
          0.85 0.10 0.10;    % GeKo     — red
          0.75 0.00 0.75];   % KCF      — magenta

% Distinct markers per form so overlapping curves stay legible in the
% all-4-forms overlays (GeKo and KCF in particular can run close together).
markers = {'o','s','^','d'};
mk_idx  = @(npts) unique([1, round(linspace(1, npts, 12)), npts]);

if mpc_mode == 1
    for r = 1:n_runs
        t_done = runs{r}.t_done;
        t_ax_r = (0:t_done) * params.Ts;
        r_ax_r = ref_fun(t_ax_r);
        figure('Name', sprintf('MPC closed-loop %s  (%s)', runs{r}.label, names{mpc_form}));
        % x1 (current, A)
        subplot(3,1,1);  hold on;
        plot(t_ax_r, runs{r}.X(1,:), 'b-', 'LineWidth', 2);
        ylabel('x_1  (A)');
        title(sprintf('%s MPC, %s  |  ISE=%.3e  |  %d/%d steps', ...
            names{mpc_form}, runs{r}.label, runs{r}.ise, t_done, T_mpc));
        grid on;
        % x2 (angular velocity, rad/s) with reference
        subplot(3,1,2);  hold on;
        plot(t_ax_r, runs{r}.X(2,:), 'b-', 'LineWidth', 2);
        plot(t_ax_r, r_ax_r,         'k--', 'LineWidth', 1.5);
        ylabel('x_2  (rad/s)');
        legend('x_2', 'reference', 'Location','best');
        grid on;
        % u (physical V)
        subplot(3,1,3);  hold on;
        stairs(t_ax_r(1:end-1), runs{r}.U, 'b-', 'LineWidth', 1.5);
        yline( u_max_phys, 'r--', 'HandleVisibility','off');
        yline(-u_max_phys, 'r--', 'HandleVisibility','off');
        ylabel('u  (V)');  xlabel('Time [s]');
        grid on;
    end
else
    % Mode 2: all 4 forms overlaid
    if mpc_mode == 2
        model_set_label = 'multi-step';
    else
        model_set_label = 'one-step';
    end
    figure('Name', sprintf('MPC closed-loop — all 4 forms (%s)', model_set_label));
    % x1
    subplot(3,1,1);  hold on;
    for r = 1:n_runs
        t_done = runs{r}.t_done;
        t_ax_r = (0:t_done) * params.Ts;
        plot(t_ax_r, runs{r}.X(1,:), '-', 'Color', colors(r,:), 'LineWidth', 1.8, ...
        'Marker', markers{r}, 'MarkerIndices', mk_idx(numel(t_ax_r)), 'MarkerSize', 5);
    end
    ylabel('x_1  (A)');
    title(sprintf('MPC closed-loop  |  all 4 forms (%s)', model_set_label));
    legend(headers, 'Location','best');  grid on;
    % x2 with reference
    subplot(3,1,2);  hold on;
    for r = 1:n_runs
        t_done = runs{r}.t_done;
        t_ax_r = (0:t_done) * params.Ts;
        plot(t_ax_r, runs{r}.X(2,:), '-', 'Color', colors(r,:), 'LineWidth', 1.8, ...
        'Marker', markers{r}, 'MarkerIndices', mk_idx(numel(t_ax_r)), 'MarkerSize', 5);
    end
    % Plot reference once, on the longest time axis available
    t_max_done = max(cellfun(@(rr) rr.t_done, runs));
    t_ref_ax   = (0:t_max_done) * params.Ts;
    plot(t_ref_ax, ref_fun(t_ref_ax), 'k--', 'LineWidth', 1.5);
    ylabel('x_2  (rad/s)');
    legend([headers, {'reference'}], 'Location','best');  grid on;
    % u
    subplot(3,1,3);  hold on;
    hLegU = gobjects(1, n_runs);
    for r = 1:n_runs
        t_done = runs{r}.t_done;
        t_ax_r = (0:t_done) * params.Ts;
        % Stair line itself carries NO legend entry (markers can't sit on a
        % Stair); a line+marker proxy below supplies the legend swatch.
        stairs(t_ax_r(1:end-1), runs{r}.U, '-', 'Color', colors(r,:), ...
            'LineWidth', 1.5, 'HandleVisibility', 'off');
        % Sparse markers overlaid on the step curve (hidden from the legend).
        idxU = mk_idx(numel(runs{r}.U));
        plot(t_ax_r(idxU), runs{r}.U(idxU), 'LineStyle', 'none', ...
            'Marker', markers{r}, 'Color', colors(r,:), 'MarkerSize', 5, ...
            'HandleVisibility', 'off');
        % Proxy (line+marker, NaN data so nothing is drawn) -> legend swatch
        % matches the state subplots above.
        hLegU(r) = plot(NaN, NaN, '-', 'Color', colors(r,:), 'Marker', markers{r}, ...
            'LineWidth', 1.5, 'MarkerSize', 5);
    end
    yline( u_max_phys, 'r--', 'HandleVisibility','off');
    yline(-u_max_phys, 'r--', 'HandleVisibility','off');
    ylabel('u  (V)');  xlabel('Time [s]');
    legend(hLegU, headers, 'Location','best');  grid on;
end

fprintf('\nPart 2 complete.\n');


%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

% ---------- Run MPC for one model: returns result struct ----------
function res = run_mpc_for_model(M_use, x0_mpc_phys, H_lift, ...
        Q_y, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
        fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
        deg_poly_x, deg_cheb_u, ref_fun, slow_iter_thresh_s, ...
        cumulative_thresh_s, label)
    X_phys   = zeros(2, T_mpc+1);   X_phys(:,1) = x0_mpc_phys;
    U_phys   = zeros(1, T_mpc);     % decision variable is physical u
    J_cl     = zeros(1, T_mpc);
    cpu_step = zeros(1, T_mpc);
    R_log    = zeros(1, T_mpc);

    u_init       = u_init_default;
    u_prev_phys  = 0;
    t_done       = 0;
    cum_warned   = false;

    for t = 1:T_mpc
        % Lift current state (scaled coordinates)
        x_scaled = params.Sx * X_phys(:, t);
        if use_centres_x
            z0_t = lift_state(x_scaled, C_x, hp_x, kfun_x, prepend_state);
        else
            z0_t = poly_lift_state(x_scaled, deg_poly_x);
        end
        % Future reference window: r(t+1), r(t+2), ..., r(t+Hp)
        % (post-step cost: stage k penalises predicted z_k against r_k)
        t_now    = (t-1) * params.Ts;
        t_future = t_now + (1:Hp) * params.Ts;
        r_window = ref_fun(t_future)';   % Hp x 1

        cost_fn = @(u_phys_seq) mpc_cost_track_with_grad(u_phys_seq, z0_t, r_window, ...
                                                         M_use, H_lift, ...
                                                         Q_y, R_u, R_du, u_prev_phys, ...
                                                         Hp, deg_cheb_u, params.Su);
        tic;
        [u_opt, J_opt] = fmincon(cost_fn, u_init, [],[],[],[], lb, ub, [], fmin_opts);
        cpu_step(t) = toc;

        u0_phys      = u_opt(1);
        U_phys(t)    = u0_phys;
        J_cl(t)      = J_opt;
        R_log(t)     = r_window(1);
        X_phys(:, t+1) = step_dcmotor(X_phys(:, t), u0_phys, params);
        u_init       = [u_opt(2:end); u_opt(end)];   % warm-shift
        u_prev_phys  = u0_phys;
        t_done       = t;

        % Per-iteration safety check
        if t == 1 && cpu_step(1) > slow_iter_thresh_s
            est_total = cpu_step(1) * T_mpc;
            fprintf('\n   *** SLOW MPC WARNING (per-iteration) ***\n');
            fprintf('   Iteration 1 took %.2f s (threshold %.1f s).\n', ...
                cpu_step(1), slow_iter_thresh_s);
            fprintf('   At this pace the full %d-step run will take ~%.1f minutes.\n', ...
                T_mpc, est_total / 60);
            ans_str = input('   Continue? [y/n] : ', 's');
            if ~strcmpi(strtrim(ans_str), 'y')
                fprintf('   User aborted. Returning partial results (%d step done).\n', t_done);
                break;
            else
                fprintf('   Continuing...\n');
            end
        end

        % Cumulative safety check
        cum_elapsed = sum(cpu_step(1:t));
        if ~cum_warned && cum_elapsed > cumulative_thresh_s
            cum_warned = true;
            steps_left   = T_mpc - t;
            mean_so_far  = cum_elapsed / t;
            est_remaining = steps_left * mean_so_far;
            fprintf('\n   *** SLOW MPC WARNING (cumulative) ***\n');
            fprintf('   %d / %d steps done in %.1f s (threshold %.1f s).\n', ...
                t, T_mpc, cum_elapsed, cumulative_thresh_s);
            fprintf('   Mean per-step so far: %.2f s.  Estimated remaining: ~%.1f min.\n', ...
                mean_so_far, est_remaining / 60);
            ans_str = input('   Continue? [y/n] : ', 's');
            if ~strcmpi(strtrim(ans_str), 'y')
                fprintf('   User aborted. Returning partial results (%d steps done).\n', t_done);
                break;
            else
                fprintf('   Continuing (no further cumulative prompts this run)...\n');
            end
        end
    end

    % Truncate on abort
    X_phys   = X_phys(:, 1:t_done+1);
    U_phys   = U_phys(1:t_done);
    J_cl     = J_cl(1:t_done);
    cpu_step = cpu_step(1:t_done);
    R_log    = R_log(1:t_done);

    % ISE (tracking) — error on x2 vs reference
    if t_done > 0
        x2_phys = X_phys(2, 2:end);   % x2 from step 1 onward (where ref is matched)
        ise_track = params.Ts * sum((x2_phys - R_log).^2);
    else
        ise_track = 0;
    end

    res.X      = X_phys;
    res.U      = U_phys;
    res.cpu    = cpu_step;
    res.J      = J_cl;
    res.R      = R_log;
    res.ise    = ise_track;
    res.label  = label;
    res.t_done = t_done;
end

% ---------- DC motor one-step simulator (physical, RK45) ----------
function x_next = step_dcmotor(x, u, p)
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
    fu = p.f_fun(u);
    f = @(~,xv)[ -(p.Ra/p.La)*xv(1) - (p.km/p.La)*xv(2)*fu + p.ua/p.La;
                 -(p.B /p.J )*xv(2) + (p.km/p.J )*xv(1)*fu - p.tau_l/p.J ];
    [~, xo] = ode45(f, [0, p.Ts], x, opts);
    x_next = xo(end, :)';
end

% ---------- Kernel functions ----------
function k = rbf_kernel(A, B, sg)
    d2 = sum((A-B).^2, 1);  k = exp(-d2 / (2 * sg^2));
end
function k = rq_kernel(A, B, l, al)
    d2 = sum((A-B).^2, 1);  k = (1 + d2 / (2 * al * l^2)).^(-al);
end
function k = matern52_kernel(A, B, l)
    r  = sqrt(sum((A-B).^2, 1));
    s5 = sqrt(5);  rs = s5 * r / l;
    k  = (1 + rs + (5/3) * (r/l).^2) .* exp(-rs);
end

% ---------- Lifting ----------
function Z = lift_state(X, C_x, hp, kfun, prepend)
    nzc = size(C_x, 2);  M = size(X, 2);  Zk = zeros(nzc, M);
    for i = 1:nzc
        Zk(i,:) = kfun(X, C_x(:,i) * ones(1, M), hp);
    end
    Zk = sqrt(2/nzc) * Zk;
    if prepend
        % Choice 1: prepend [1; x1; x2] (constant + physical state); state at
        % rows 1:3, decoder = projection on rows 2:3. (Choice 2: pure kernel.)
        Z = [ones(1, M); X(1,:); X(2,:); Zk];
    else
        Z = Zk;
    end
end
function Z = poly_lift_state(X, d)
    x1 = X(1,:);  x2 = X(2,:);  rows = {};
    for total = 0:d
        for i = total:-1:0
            j = total - i;
            rows{end+1} = x1.^i .* x2.^j; %#ok<AGROW>
        end
    end
    Z = cell2mat(rows');
end
function V = cheb_lift_input(U, d)
    M = numel(U);  V = zeros(d+1, M);
    V(1,:) = 1;
    if d >= 1, V(2,:) = U; end
    for k = 2:d
        V(k+1,:) = 2 .* U .* V(k,:) - V(k-1,:);
    end
end

% ---------- Chebyshev lift + derivative (scalar u, for analytic grad) ----------
function [v, dv] = cheb_lift_and_deriv(u, d)
    v  = zeros(d+1, 1);
    dv = zeros(d+1, 1);
    v(1)  = 1;       dv(1) = 0;
    if d >= 1
        v(2)  = u;   dv(2) = 1;
    end
    for k = 2:d
        v(k+1)  = 2*u*v(k)  - v(k-1);
        dv(k+1) = 2*v(k) + 2*u*dv(k) - dv(k-1);
    end
end

% ---------- Predictors dispatched on M.kind ----------
function z_next = predict_dispatch(M, z, v, u)
    switch M.kind
        case 'linear'
            z_next = M.A * z + M.B * u;
        case 'bilinear'
            z_next = M.A * z + M.B * u + M.N * (z * u);
        case 'geko'
            z_next = M.K * kron(z, v);
        case 'kcf'
            if isfield(M, 'v_const_dropped') && M.v_const_dropped
                v_eff = v(2:end);
            elseif isfield(M, 'v_const_dropped')
                v_eff = v;
            else
                v_eff = v(2:end);   % legacy default
            end
            if isfield(M, 'couple_idx'), ci = M.couple_idx; else, ci = 1:numel(z); end
            z_next = M.A11 * z + M.A12 * kron(v_eff, z(ci));   % subset coupling
        otherwise
            error('Unknown M.kind: %s', M.kind);
    end
end

% ---------- Predictor with Jacobians (for analytic gradient) ----------
%
% Returns:
%   z_next     : next state (nz x 1)
%   dz_dz      : Jacobian w.r.t. z          (nz x nz)
%   dz_du      : Jacobian w.r.t. u (scalar) (nz x 1)
% v and dv are Chebyshev features and their derivative at u (u_scaled here).
%
function [z_next, dz_dz, dz_du] = predict_and_jacobian(M, z, v, dv, u)
    nz = numel(z);
    switch M.kind
        case 'linear'
            z_next = M.A * z + M.B * u;
            dz_dz  = M.A;
            dz_du  = M.B;
        case 'bilinear'
            z_next = M.A * z + M.B * u + M.N * (z * u);
            dz_dz  = M.A + u * M.N;
            dz_du  = M.B + M.N * z;
        case 'geko'
            % K * kron(z, v) with MATLAB's kron(z,v) ordering:
            % the slice for v_ell is K(:, ell : nv : end), an nz x nz matrix.
            nv_loc = numel(v);
            dz_dz  = zeros(nz, nz);
            Aprime = zeros(nz, nz);
            for ell = 1:nv_loc
                K_ell  = M.K(:, ell : nv_loc : end);
                dz_dz  = dz_dz  + v(ell)  * K_ell;
                Aprime = Aprime + dv(ell) * K_ell;
            end
            z_next = dz_dz * z;
            dz_du  = Aprime * z;
        case 'kcf'
            % Sparse-coupling KCF: A12 * kron(v_eff, z(couple_idx)).
            % MATLAB kron(v_eff, zc) ordering -> the slice for v_eff_i is
            % A12(:, (i-1)*nc+1 : i*nc), size nz x nc, acting on zc=z(ci).
            % d/dz only touches the coupled columns ci; the rest is A11.
            if isfield(M, 'v_const_dropped') && M.v_const_dropped
                v_eff  = v(2:end);
                dv_eff = dv(2:end);
            elseif isfield(M, 'v_const_dropped')
                v_eff  = v;
                dv_eff = dv;
            else
                v_eff  = v(2:end);
                dv_eff = dv(2:end);
            end
            if isfield(M, 'couple_idx'), ci = M.couple_idx; else, ci = 1:nz; end
            nc = numel(ci);
            zc = z(ci);
            nv_eff = numel(v_eff);
            sum_v_A12  = zeros(nz, nc);
            sum_dv_A12 = zeros(nz, nc);
            for i = 1:nv_eff
                Ai          = M.A12(:, (i-1)*nc+1 : i*nc);   % nz x nc
                sum_v_A12   = sum_v_A12  + v_eff(i)  * Ai;
                sum_dv_A12  = sum_dv_A12 + dv_eff(i) * Ai;
            end
            z_next = M.A11 * z + sum_v_A12 * zc;
            dz_dz  = M.A11;
            dz_dz(:, ci) = dz_dz(:, ci) + sum_v_A12;
            dz_du  = sum_dv_A12 * zc;
        otherwise
            error('Unknown M.kind: %s', M.kind);
    end
end

% ---------- MPC cost: tracking, with analytic gradient ----------
%
%  J = sum_{k=1..Hp} [ Q_y (H z_k - r_k)^2
%                     + R_u u_phys_k^2 + R_du (u_phys_k - u_prev)^2 ]
%
%  Post-step output cost (z_k is AFTER applying u_k). Decision variable is
%  u_phys; predictor sees u_scaled = Su * u_phys. Hence
%      dz/d(u_phys) = Su * dz/d(u_scaled).
%
%  Gradient computed by adjoint backpropagation:
%    Lambda_{Hp} = 2 H' Q_y (H z_{Hp} - r_{Hp})
%    Lambda_{k}  = 2 H' Q_y (H z_k - r_k) + dz_dz^T Lambda_{k+1}    for k < Hp
%    dJ/du_phys_k = 2 R_u u_phys_k + 2 R_du (u_phys_k - u_{k-1})
%                  -2 R_du (u_phys_{k+1} - u_phys_k)  [if k < Hp]
%                  + Su * (dz_du)^T Lambda_k
%
function [J, dJ] = mpc_cost_track_with_grad(u_phys_seq, z0, r_window, M, ...
                                            H_lift, Q_y, R_u, R_du, ...
                                            u_prev_phys, Hp, deg_cheb_u, Su)
    nz = numel(z0);

    % --- Forward pass ---
    Z_seq    = zeros(nz, Hp+1);   Z_seq(:,1) = z0;
    dz_dz_st = cell(Hp, 1);
    dz_du_st = zeros(nz, Hp);    % dz/du_scaled at each stage
    J = 0;
    u_km1 = u_prev_phys;
    for k = 1:Hp
        uk_phys   = u_phys_seq(k);
        uk_scaled = Su * uk_phys;
        [v_k, dv_k] = cheb_lift_and_deriv(uk_scaled, deg_cheb_u);
        [Z_seq(:, k+1), dz_dz_st{k}, dz_du_st(:, k)] = ...
            predict_and_jacobian(M, Z_seq(:, k), v_k, dv_k, uk_scaled);
        % Post-step output cost on Z_seq(:, k+1)
        y_pred = H_lift * Z_seq(:, k+1);
        e_y    = y_pred - r_window(k);
        J = J + Q_y * e_y^2 + R_u * uk_phys^2 + R_du * (uk_phys - u_km1)^2;
        u_km1 = uk_phys;
    end

    if nargout < 2
        return;
    end

    % --- Backward adjoint pass ---
    %  Lambda_k = dJ/dz_k  (z_k = Z_seq(:, k+1) in 1-indexed array)
    %  Stage k cost depends on z_k (post-step): contribution 2 H' Q_y (H z_k - r_k).
    % Initialize with stage-Hp contribution (terminal-equivalent here since
    % cost is post-step).
    y_end  = H_lift * Z_seq(:, Hp+1);
    Lambda = 2 * (H_lift') * (Q_y * (y_end - r_window(Hp)));   % nz x 1

    dJ = zeros(Hp, 1);
    for k = Hp:-1:1
        uk_phys = u_phys_seq(k);
        if k == 1
            u_km1 = u_prev_phys;
        else
            u_km1 = u_phys_seq(k-1);
        end
        d_input = 2 * R_u * uk_phys + 2 * R_du * (uk_phys - u_km1);
        if k < Hp
            d_input = d_input - 2 * R_du * (u_phys_seq(k+1) - uk_phys);
        end
        % Chain via z_k: dJ/du_phys_k = d_input + Su * dz_du_scaled' * Lambda_k
        dJ(k) = d_input + Su * (dz_du_st(:, k)' * Lambda);

        % Push Lambda back to Lambda_{k-1}:
        %   Lambda_{k-1} = 2 H' Q_y (H z_{k-1} - r_{k-1})  [stage k-1's output cost]
        %                  + dz_dz_st{k}' * Lambda_k       [propagation back]
        % Stage k-1 exists only when k > 1 (otherwise z_0 has no stage cost,
        % it's the initial condition).
        if k > 1
            y_km1  = H_lift * Z_seq(:, k);   % z_{k-1} in 0-indexed
            stage_grad = 2 * (H_lift') * (Q_y * (y_km1 - r_window(k-1)));
            Lambda = stage_grad + dz_dz_st{k}' * Lambda;
        else
            % After processing k=1, Lambda would be Lambda_0 = dJ/dz_0.
            % z_0 is the initial condition (not a decision), so we don't
            % need it. Loop is about to exit anyway.
        end
    end
end
