%% =========================================================
%  mpc_koopman_duffing.m   — CDC 2026 KOT tutorial paper
%
%  PART 2 of 2.  Loads the four trained Koopman models from
%  Part 1 and runs regulating MPC in one of three modes:
%    Mode 1 : pick one form; compare its ONE-STEP and MULTI-STEP
%             versions in closed loop.
%    Mode 2 : run all four forms (MULTI-STEP) and cross-compare.
%    Mode 3 : run all four forms (ONE-STEP) and cross-compare.
%  Reports per-step CPU statistics (mean / median / worst) and
%  tracking ISE in a single table, and plots closed-loop trajectories.
%
%  Input  : koopman_duffing_model_<kernel>.mat   (saved by Part 1)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  USER SETTINGS
%% =========================================================

% --- MPC mode ---
%   1 : single form, compare ONE-STEP vs MULTI-STEP of that form
%   2 : all 4 forms (MULTI-STEP), cross-comparison
%   3 : all 4 forms (ONE-STEP),   cross-comparison
mpc_mode = 2;

% --- Choice of Koopman form (used only in mode 1) ---
%   1 : Linear     2 : Bilinear     3 : GeKo     4 : KCF
mpc_form = 1;

% --- Which trained model file to load (one per state kernel) ---
%   Part 1 saves koopman_duffing_model_<kernel>.mat, one per state kernel.
%     0 : AUTO -- load whatever Part 1 wrote (newest, if several kernels exist)
%     1 : RBF    2 : Rational Quadratic    3 : Matern 5/2    4 : Polynomial
%   Auto (default) needs no edit when you switch kernels in Part 1; pick a
%   specific 1..4 only to force one kernel when several files are on disk.
kernel_choice = 0;
ktag_map = {'rbf','rq','matern','poly'};
if kernel_choice == 0
    cand = dir('koopman_duffing_model_*.mat');
    if isempty(cand)
        error(['No koopman_duffing_model_*.mat in the current directory:\n' ...
               '   %s\n   Run Part 1 (learn_koopman_duffing.m) first.'], pwd);
    end
    [~, newest] = max([cand.datenum]);
    model_file  = cand(newest).name;
    if numel(cand) > 1
        fprintf('   Auto-detect: %d model files found; using newest (%s).\n', ...
            numel(cand), model_file);
        fprintf('   Set kernel_choice = 1..4 to force a specific kernel.\n');
    end
elseif ismember(kernel_choice, 1:4)
    model_file = sprintf('koopman_duffing_model_%s.mat', ktag_map{kernel_choice});
else
    error('kernel_choice must be 0 (auto) or 1..4 (got %g).', kernel_choice);
end

% --- MPC settings (regulation to zero) ---
Hp     = 15;            % prediction horizon (steps)
Q_x    = [10 0; 0 1];   % state penalty (physical coordinates)
R_u    = 0.1;           % input penalty
R_du   = 0.01;          % delta-u penalty (smooths the input)
% NOTE: input bound |u| <= u_max_phys is loaded from the .mat file so it
% matches what the model was trained on. Set it in Part 1, not here.
T_mpc  = 100;           % closed-loop horizon (steps)
x0_mpc = [1.5; 0.5];    % initial state

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
% If the very first fmincon call takes longer than this, we estimate the
% total run time and ask the user whether to continue or abort.
slow_iter_thresh_s = 60;
% Cumulative cap: total elapsed time across all iterations of a single
% MPC run. Once exceeded, the user is prompted ONCE per run.
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
%  LOAD TRAINED MODELS
%% =========================================================
fprintf('\n[1] Loading trained models from %s...\n', model_file);

% --- Safety check #1: file exists ---
if ~isfile(model_file)
    error(['Cannot find ''%s'' in the current directory.\n' ...
           '   Run Part 1 (learn_koopman_duffing.m) first; it saves this file.\n' ...
           '   If you ran Part 1 elsewhere, update the model_file path at the top.'], ...
           model_file);
end

% --- Safety check #2: required variables are present ---
required = {'M_ms','M_1s','model_names','params','D_lin','nz','nv', ...
            'deg_cheb_u','deg_poly_x','use_centres_x','C_x','hp_x', ...
            'KERNEL_TYPE','kernel_name_x','kernel_name_u','Su','u_max_phys','prepend_state'};
info = whos('-file', model_file);
present = {info.name};
missing = setdiff(required, present);
if ~isempty(missing)
    error(['The file ''%s'' is missing required variable(s):\n   %s\n' ...
           '   It looks like an older save. Re-run Part 1 to refresh it.'], ...
           model_file, strjoin(missing, ', '));
end

S = load(model_file);

% --- Safety check #3: 4 models present, each with expected kind ---
M_ms      = S.M_ms;
M_1s      = S.M_1s;
names     = S.model_names;
expected_kinds = {'linear','bilinear','geko','kcf'};
if numel(M_ms) ~= 4 || numel(M_1s) ~= 4 || numel(names) ~= 4
    error(['Saved model file expected to contain 4 forms.\n' ...
           '   Got M_ms=%d, M_1s=%d, names=%d.'], ...
           numel(M_ms), numel(M_1s), numel(names));
end
for m = 1:4
    if ~isfield(M_ms{m}, 'kind') || ~strcmp(M_ms{m}.kind, expected_kinds{m})
        error('M_ms{%d} has kind ''%s'' but expected ''%s''.', ...
              m, M_ms{m}.kind, expected_kinds{m});
    end
    if ~isfield(M_1s{m}, 'kind') || ~strcmp(M_1s{m}.kind, expected_kinds{m})
        error('M_1s{%d} has kind ''%s'' but expected ''%s''.', ...
              m, M_1s{m}.kind, expected_kinds{m});
    end
end

% Pull the remaining variables
params    = S.params;
D_lin     = S.D_lin;
nz        = S.nz;
nv        = S.nv;
deg_cheb_u    = S.deg_cheb_u;
deg_poly_x    = S.deg_poly_x;
use_centres_x = S.use_centres_x;
C_x       = S.C_x;
hp_x      = S.hp_x;
KERNEL_TYPE   = S.KERNEL_TYPE;
kernel_name_x = S.kernel_name_x;
kernel_name_u = S.kernel_name_u;
Su            = S.Su;            % input pre-scaling: Su * u_phys -> [-1, 1]
u_max_phys    = S.u_max_phys;    % physical input bound used at training time
prepend_state = S.prepend_state; % shared lift/decoder convention (Choice 1/2)

% Reconstruct the state-kernel handle locally (handles to local functions
% don't survive .mat reliably; we rebuild from KERNEL_TYPE here).
switch KERNEL_TYPE
    case 1, kfun_x = @(X,C,hp) rbf_kernel(X,C,hp(1));
    case 2, kfun_x = @(X,C,hp) rq_kernel(X,C,hp(1),hp(2));
    case 3, kfun_x = @(X,C,hp) matern52_kernel(X,C,hp(1));
    case 4, kfun_x = [];          % polynomial — no kernel handle needed
    otherwise, error('Unknown KERNEL_TYPE in saved file.');
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
fprintf('   MPC solver   : fmincon / %s\n', fmincon_algo);
switch mpc_mode
    case 1
        fprintf('   Mode 1 form  : %s  (one-step vs multi-step)\n', names{mpc_form});
    case 2
        fprintf('   Mode 2       : all 4 forms (multi-step), cross-comparison\n');
    case 3
        fprintf('   Mode 3       : all 4 forms (one-step), cross-comparison\n');
end

%% =========================================================
%  CLOSED-LOOP MPC : ONE-STEP MODEL  then  MULTI-STEP MODEL
%
%  Same cost, same constraints, same initial state — the only
%  difference between the two runs is which trained operator
%  drives the predictor inside the cost function.
%% =========================================================

fmin_opts = optimoptions('fmincon', ...
    'Algorithm', fmincon_algo, 'Display','off', 'MaxIterations',100, ...
    'OptimalityTolerance',1e-4, 'StepTolerance',1e-6, ...
    'SpecifyObjectiveGradient', true);   % analytic gradient (mpc_cost_with_grad)
lb = -u_max_phys * ones(Hp, 1);   ub = u_max_phys * ones(Hp, 1);
u_init_default = zeros(Hp, 1);

% Decoder-weighted Q in lifted coordinates: ||D z - 0||_Q^2 = z' (D'QD) z.
Q_z_mat = D_lin' * Q_x * D_lin;

% Lifted zero-state: phi(x = 0). In kernel coordinates phi(0) is NOT
% the zero vector, so the cost must be (z - z_ref)' Q_z (z - z_ref),
% not z' Q_z z. Compute once outside the MPC loop.
if use_centres_x
    z_ref_mpc = lift_state(zeros(2,1), C_x, hp_x, kfun_x, prepend_state);
else
    z_ref_mpc = poly_lift_state(zeros(2,1), deg_poly_x);
end

% Run MPC according to mode:
%   1 -> single form, ONE-STEP vs MULTI-STEP
%   2 -> all 4 forms, MULTI-STEP only
%   3 -> all 4 forms, ONE-STEP only

runs    = cell(0, 1);   % cell array of result structs
headers = cell(0, 1);   % column headers for the summary table (per run)

switch mpc_mode
    case 1
        % --- Mode 1: chosen form, one-step then multi-step ---
        fprintf('\n[2] Mode 1: %s  (one-step vs multi-step)\n', names{mpc_form});
        M_pair     = {M_1s{mpc_form}, M_ms{mpc_form}};
        label_pair = {'one-step',     'multi-step'};
        for r = 1:2
            fprintf('\n[2.%d] MPC with %s %s model...\n', r, names{mpc_form}, label_pair{r});
            runs{end+1}    = run_mpc_for_model(M_pair{r}, x0_mpc, z_ref_mpc, ...
                Q_z_mat, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
                fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
                deg_poly_x, deg_cheb_u, Su, slow_iter_thresh_s, cumulative_thresh_s, ...
                label_pair{r}); %#ok<SAGROW>
            headers{end+1} = label_pair{r}; %#ok<SAGROW>
        end
    case 2
        % --- Mode 2: all 4 forms, multi-step only ---
        fprintf('\n[2] Mode 2: all 4 forms (multi-step), cross-comparison\n');
        for m = 1:numel(names)
            fprintf('\n[2.%d] MPC with %s multi-step model...\n', m, names{m});
            runs{end+1}    = run_mpc_for_model(M_ms{m}, x0_mpc, z_ref_mpc, ...
                Q_z_mat, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
                fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
                deg_poly_x, deg_cheb_u, Su, slow_iter_thresh_s, cumulative_thresh_s, ...
                names{m}); %#ok<SAGROW>
            headers{end+1} = names{m}; %#ok<SAGROW>
        end
    case 3
        % --- Mode 3: all 4 forms, one-step only ---
        fprintf('\n[2] Mode 3: all 4 forms (one-step), cross-comparison\n');
        for m = 1:numel(names)
            fprintf('\n[2.%d] MPC with %s one-step model...\n', m, names{m});
            runs{end+1}    = run_mpc_for_model(M_1s{m}, x0_mpc, z_ref_mpc, ...
                Q_z_mat, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
                fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
                deg_poly_x, deg_cheb_u, Su, slow_iter_thresh_s, cumulative_thresh_s, ...
                names{m}); %#ok<SAGROW>
            headers{end+1} = names{m}; %#ok<SAGROW>
        end
    otherwise
        error('mpc_mode must be 1, 2, or 3.');
end
n_runs = numel(runs);

%% =========================================================
%  SUMMARY TABLE : metrics as rows, runs as columns
%
%  Bolds the best value per metric row (lowest CPU/ISE).
%% =========================================================
% Collect the 4 numeric metrics into a (4 x n_runs) matrix; rows:
%   1 = mean CPU, 2 = median CPU, 3 = worst CPU, 4 = ISE_track
% If a run has 0 completed steps (early abort), the CPU stats are NaN.
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
% Header row
fprintf('   %-14s', 'Metric');
for r = 1:n_runs, fprintf('  %12s', headers{r}); end
fprintf('\n');
% Numeric rows with bold-on-best (lower is better; NaNs ignored; ties bold all)
for k = 1:numel(metric_names)
    fprintf('   %-14s', metric_names{k});
    row = M_vals(k, :);
    if all(isnan(row))
        is_best = false(size(row));
    else
        best_val = min(row, [], 'omitnan');
        % Display is "%12.4e" -> 4 significant digits -> relative tol 5e-5.
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
% Steps row (not bolded)
fprintf('   %-14s', 'steps');
for r = 1:n_runs
    fprintf('  %12s', sprintf('%d/%d', runs{r}.t_done, T_mpc));
end
fprintf('\n');
fprintf('=========================================\n');

%% =========================================================
%  PLOTS
%
%  Mode 1: one figure per model (one-step, multi-step).
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
        figure('Name', sprintf('MPC closed-loop %s  (%s)', runs{r}.label, names{mpc_form}));
        subplot(3,1,1);  hold on;
        plot(t_ax_r, runs{r}.X(1,:), 'b-', 'LineWidth', 2);
        yline(0, 'k--', 'HandleVisibility','off');
        ylabel('x_1');
        title(sprintf('%s MPC, %s model  |  ISE=%.3e  |  %d/%d steps', ...
            names{mpc_form}, runs{r}.label, runs{r}.ise, t_done, T_mpc));
        grid on;
        subplot(3,1,2);  hold on;
        plot(t_ax_r, runs{r}.X(2,:), 'b-', 'LineWidth', 2);
        yline(0, 'k--', 'HandleVisibility','off');
        ylabel('x_2');
        grid on;
        subplot(3,1,3);  hold on;
        stairs(t_ax_r(1:end-1), runs{r}.U, 'b-', 'LineWidth', 1.5);
        yline( u_max_phys, 'r--', 'HandleVisibility','off');
        yline(-u_max_phys, 'r--', 'HandleVisibility','off');
        ylabel('u');  xlabel('Time [s]');
        grid on;
    end
else
    % Mode 2 or 3: all 4 forms overlaid in one figure
    if mpc_mode == 2
        model_set_label = 'multi-step';
    else
        model_set_label = 'one-step';
    end
    figure('Name', sprintf('MPC closed-loop — all 4 forms (%s)', model_set_label));
    subplot(3,1,1);  hold on;
    for r = 1:n_runs
        t_done = runs{r}.t_done;
        t_ax_r = (0:t_done) * params.Ts;
        plot(t_ax_r, runs{r}.X(1,:), '-', 'Color', colors(r,:), 'LineWidth', 1.8, ...
        'Marker', markers{r}, 'MarkerIndices', mk_idx(numel(t_ax_r)), 'MarkerSize', 5);
    end
    yline(0, 'k--', 'HandleVisibility','off');
    ylabel('x_1');
    title(sprintf('MPC closed-loop  |  all 4 forms (%s)', model_set_label));
    legend(headers, 'Location','best');  grid on;

    subplot(3,1,2);  hold on;
    for r = 1:n_runs
        t_done = runs{r}.t_done;
        t_ax_r = (0:t_done) * params.Ts;
        plot(t_ax_r, runs{r}.X(2,:), '-', 'Color', colors(r,:), 'LineWidth', 1.8, ...
        'Marker', markers{r}, 'MarkerIndices', mk_idx(numel(t_ax_r)), 'MarkerSize', 5);
    end
    yline(0, 'k--', 'HandleVisibility','off');
    ylabel('x_2');
    legend(headers, 'Location','best');  grid on;

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
    ylabel('u');  xlabel('Time [s]');
    legend(hLegU, headers, 'Location','best');  grid on;
end

fprintf('\nPart 2 complete.\n');


%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

% ---------- Run MPC for one model: returns result struct ----------
%
% Same closed-loop logic for both modes. Returns a struct with
% fields X, U, cpu, J, ise, label, t_done. Truncates on user-aborted
% slow-iteration or cumulative-time prompts.
function res = run_mpc_for_model(M_use, x0_mpc, z_ref_mpc, ...
        Q_z_mat, R_u, R_du, Hp, T_mpc, u_init_default, lb, ub, ...
        fmin_opts, params, use_centres_x, C_x, hp_x, kfun_x, prepend_state, ...
        deg_poly_x, deg_cheb_u, Su, slow_iter_thresh_s, cumulative_thresh_s, ...
        label)
    X_cl     = zeros(2, T_mpc+1);   X_cl(:,1) = x0_mpc;
    U_cl     = zeros(1, T_mpc);
    J_cl     = zeros(1, T_mpc);
    cpu_step = zeros(1, T_mpc);

    u_init     = u_init_default;
    u_prev     = 0;
    t_done     = 0;
    cum_warned = false;

    for t = 1:T_mpc
        if use_centres_x
            z0_t = lift_state(X_cl(:, t), C_x, hp_x, kfun_x, prepend_state);
        else
            z0_t = poly_lift_state(X_cl(:, t), deg_poly_x);
        end
        cost_fn = @(u_seq) mpc_cost_with_grad(u_seq, z0_t, z_ref_mpc, M_use, ...
                                              Q_z_mat, R_u, R_du, u_prev, ...
                                              Hp, deg_cheb_u, Su);
        tic;
        [u_opt, J_opt] = fmincon(cost_fn, u_init, [],[],[],[], lb, ub, [], fmin_opts);
        cpu_step(t) = toc;

        u0 = u_opt(1);  U_cl(t) = u0;  J_cl(t) = J_opt;
        X_cl(:, t+1) = step_duffing(X_cl(:, t), u0, params);
        u_init = [u_opt(2:end); u_opt(end)];   % warm-shift for next MPC step
        u_prev = u0;
        t_done = t;

        % Per-iteration safety check (only after iteration 1)
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

        % Cumulative safety check (fires at most once per run)
        cum_elapsed = sum(cpu_step(1:t));
        if ~cum_warned && cum_elapsed > cumulative_thresh_s
            cum_warned = true;
            steps_left = T_mpc - t;
            mean_so_far = cum_elapsed / t;
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

    % Truncate to the steps that actually ran (on abort)
    X_cl     = X_cl(:, 1:t_done+1);
    U_cl     = U_cl(1:t_done);
    J_cl     = J_cl(1:t_done);
    cpu_step = cpu_step(1:t_done);

    % ISE (tracking) — regulation to zero, so e(t) = x(t)
    ise_track = params.Ts * sum(sum(X_cl(:, 1:end-1).^2, 1));

    res.X      = X_cl;
    res.U      = U_cl;
    res.cpu    = cpu_step;
    res.J      = J_cl;
    res.ise    = ise_track;
    res.label  = label;
    res.t_done = t_done;
end

% ---------- Duffing one-step simulator (RK45) ----------
function x_next = step_duffing(x, u, p)
    opts = odeset('RelTol',1e-7,'AbsTol',1e-9);
    f = @(~,xv)[ xv(2);
                -p.delta*xv(2) - p.alpha*xv(1) - p.beta*xv(1)^3 + u ];
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

% ---------- Lifting (state + Chebyshev input) ----------
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
%   T_0(u) = 1,                T_0'(u) = 0
%   T_1(u) = u,                T_1'(u) = 1
%   T_{k+1}(u) = 2u T_k(u) - T_{k-1}(u)
%   T_{k+1}'(u) = 2 T_k(u) + 2u T_k'(u) - T_{k-1}'(u)
% Returns (d+1)x1 column vectors.
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

% ---------- Predictor with Jacobians  (for analytic gradient) ----------
%
% Given (M, z, u) at one stage, returns:
%   z_next     : next state (nz x 1)
%   dz_dz_next : Jacobian of z_next w.r.t. z          (nz x nz)
%   dz_du_next : Jacobian of z_next w.r.t. u (scalar) (nz x 1)
%
% v and dv are the Chebyshev features and their derivative evaluated at u.
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
            % K * kron(z, v) with z (nz x 1) and v (nv x 1).
            % MATLAB convention: kron(z, v)[k] = z(ceil(k/nv)) * v(mod(k-1,nv)+1).
            % Group by v_ell: coefficient of v_ell is sum_i K(:, (i-1)*nv + ell) * z_i.
            % So the "slice for v_ell" is K(:, ell : nv : end), which is nz x nz.
            nv_loc = numel(v);
            dz_dz  = zeros(nz, nz);
            Aprime = zeros(nz, nz);          % d/du of dz_dz
            for ell = 1:nv_loc
                K_ell  = M.K(:, ell : nv_loc : end);   % nz x nz
                dz_dz  = dz_dz  + v(ell)  * K_ell;
                Aprime = Aprime + dv(ell) * K_ell;
            end
            z_next = dz_dz * z;
            dz_du  = Aprime * z;
        case 'kcf'
            % Sparse-coupling KCF: A12 * kron(v_eff, z(couple_idx)).
            % MATLAB kron(v_eff, zc) -> slice for v_eff_i is
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

% ---------- MPC cost with analytic gradient ----------
%
%  J = sum_{k=1..Hp} [ (z_{k-1}-z_ref)' Q_z (z_{k-1}-z_ref)
%                     + R_u u_k^2 + R_du (u_k - u_{k-1})^2 ]
%    +                (z_Hp - z_ref)' Q_z (z_Hp - z_ref)         (terminal)
%
%  Gradient computed by adjoint backpropagation through the predictor.
%  Lambda_k = dJ/dz_k satisfies:
%     Lambda_{Hp} = 2 Q_z (z_{Hp} - z_ref)                 (terminal)
%     Lambda_{k}  = 2 Q_z (z_k - z_ref) + dz_dz^T Lambda_{k+1}   (stage k cost
%                                                          + propagation back)
%  Then dJ/du_k = 2 R_u u_k + 2 R_du (u_k - u_{k-1})
%               - 2 R_du (u_{k+1} - u_k) [if k < Hp]
%               + dz_du^T Lambda_k
%
function [J, dJ] = mpc_cost_with_grad(u_seq, z0, z_ref, M, Q_z_mat, R_u, R_du, ...
                                       u_prev, Hp, deg_cheb_u, Su)
    % Decision variable u_seq is in PHYSICAL units (the bounds lb/ub use
    % physical u, and R_u / R_du penalize physical u). The predictor was
    % trained on SCALED input u_s = Su * u_phys, so we scale here before
    % evaluating the Chebyshev features and the predictor.
    %
    % Chain rule:  dz/du_phys = Su * dz/du_scaled.
    nz = numel(z0);

    % --- Forward pass: store z_seq and Jacobians per stage ---
    Z_seq    = zeros(nz, Hp+1);   Z_seq(:,1) = z0;
    dz_dz_st = cell(Hp, 1);       % dz_dz at each stage
    dz_du_st = zeros(nz, Hp);     % dz/du_scaled at each stage (nz x Hp)
    J = 0;
    u_km1 = u_prev;
    for k = 1:Hp
        uk_phys = u_seq(k);
        uk_s    = Su * uk_phys;
        [v_k, dv_k] = cheb_lift_and_deriv(uk_s, deg_cheb_u);
        % Stage cost on z_{k-1} (input penalties on PHYSICAL u)
        dz_stage = Z_seq(:, k) - z_ref;
        J = J + dz_stage' * Q_z_mat * dz_stage ...
              + R_u * uk_phys^2 + R_du * (uk_phys - u_km1)^2;
        % Predictor + Jacobians (in scaled input)
        [Z_seq(:, k+1), dz_dz_st{k}, dz_du_k] = ...
            predict_and_jacobian(M, Z_seq(:, k), v_k, dv_k, uk_s);
        dz_du_st(:, k) = dz_du_k;
        u_km1 = uk_phys;
    end
    % Terminal cost
    dz_term = Z_seq(:, Hp+1) - z_ref;
    J = J + dz_term' * Q_z_mat * dz_term;

    if nargout < 2
        return;
    end

    % --- Backward pass: adjoint Lambda_k = dJ/dz_k ---
    Lambda = 2 * Q_z_mat * (Z_seq(:, Hp+1) - z_ref);   % terminal
    dJ = zeros(Hp, 1);
    for k = Hp:-1:1
        uk_phys = u_seq(k);
        if k == 1
            u_km1 = u_prev;
        else
            u_km1 = u_seq(k-1);
        end
        d_input = 2 * R_u * uk_phys + 2 * R_du * (uk_phys - u_km1);
        if k < Hp
            d_input = d_input - 2 * R_du * (u_seq(k+1) - uk_phys);
        end
        % Chain: dJ/du_phys_k = d_input + Su * (dz/du_scaled)' * Lambda_k
        dJ(k) = d_input + Su * (dz_du_st(:, k)' * Lambda);

        Lambda = 2 * Q_z_mat * (Z_seq(:, k) - z_ref) + dz_dz_st{k}' * Lambda;
    end
end
