%% =========================================================
%  learn_koopman_duffing.m   — CDC 2026 KOT tutorial paper
%
%  PART 1 of 2.  Trains four Koopman model structures on the
%  forced Duffing oscillator, evaluates them open-loop on a set
%  of held-out test trajectories, and saves the trained models
%  to disk for Part 2 (MPC).
%
%  Reference equations (z = phi_x(x), v = phi_u(u)):
%    1. Linear EDMD :  z+ = A z + B u
%    2. Bilinear    :  z+ = A z + B u + N z u
%    3. GeKo        :  z+ = K (z (x) v)
%    4. KCF         :  z+ = A_{11} z + A_{12} (v (x) z)
%
%  KCF auto-drops the constant first input feature (T_0 = 1) to
%  avoid column-space redundancy with its [z; ...] block —
%  see fit_kcf below for details.
%
%  Output : koopman_duffing_model_<kernel>.mat  (loaded by Part 2)
%% =========================================================

clear; clc; close all;
rng(42);

%% =========================================================
%  USER SETTINGS
%% =========================================================

% --- State kernel (input kernel is fixed: Chebyshev T_0..T_d) ---
%   1 : Gaussian RBF
%   2 : Rational Quadratic
%   3 : Matérn 5/2
%   4 : Polynomial (no centres)
KERNEL_TYPE = 4;

% --- KCF coupling + shared lift (3 knobs) ---------------------------------
%   KCF lets the input modulate the lifted dynamics through only a SUBSET of
%   lift features (z+ = A11*z + A12*Gtilde(u)*z). More coupled features = more
%   input leverage and a larger model; fewer = leaner.
%     prepend_state  : put [1;x1;x2] in the lift (shared by all forms).
%                      false -> pure-feature kernel lift + one regressed decoder
%                               (poly has the state natively); true -> state at
%                               rows 2:3, decoder is a plain projection.
%     kcf_couple_deg : POLY lift   -- couple u to monomials up to this degree.
%     kcf_s_ratio    : KERNEL lift -- couple u to the top-variance features,
%                      sized so the coupled model is ~ kcf_s_ratio * nz wide.
prepend_state  = false;
kcf_couple_deg = 2;
kcf_s_ratio    = 2;

% --- Duffing oscillator parameters ---
params.delta = 0.5;
params.alpha = -1.0;
params.beta  =  1.0;
params.Ts    =  0.05;

% --- Feature-space dimensions and hyperparameters ---
% RBF
nz_rbf      = 300;
sigma_rbf_x =  1.0;
% Rational Quadratic
nz_rq       = 300;
ell_rq_x    =  1.0;
alpha_rq_x  =  2.0;
% Matérn 5/2
nz_mat      = 300;
ell_mat_x   =  1.0;
% Polynomial
deg_poly_x  =   4;

% Chebyshev input kernel (fixed)
deg_cheb_u  =   3;

% --- Input range and scaling ---
%   u_max_phys : largest input used in training (and the MPC bound).
%   Su = 1/u_max_phys pre-scales the input into [-1, 1], the domain where the
%   Chebyshev features T_k stay well-behaved. No state scaling: the Duffing
%   state is already O(1) (wells at x1 = +/-1).

u_max_phys = 3.0;
Su         = 1 / u_max_phys;

% --- Data collection ---
%   L  = number of training trajectories; balanced across both wells
%        (left, right, saddle) so kernel centres cover the state space.
%   Smaller L + M_raw + M_target acts as implicit regularization for the
%   multi-step solve, which otherwise over-inflates operator norms (esp.
%   ||N|| for Bilinear) and produces oscillating rollout errors.
L         = 6;
T_long    = 2000;
M_raw_max = 6000;
M_target  = 800;
N         = 20;        % multi-step horizon used in training
% gamma : Tikhonov (ridge) regulariser in the least-squares fit
%   Theta = (Zp*Phi') / (Phi*Phi' + gamma*I). The lifted features are many and
%   highly correlated, so Phi*Phi' is near-singular; gamma*I makes the solve
%   well-conditioned and keeps operators and rollouts stable. Larger gamma =
%   smoother/more damped; too small = unstable fit.
gamma     = 1e-4;

% --- Open-loop validation ---
T_test  = 150;
x0_test = [1.5; 0.5];  % representative IC (also IC #1 in validation set)
N_test  = 10;

%% =========================================================
%  INPUT VALIDATION  (fail fast, before any heavy computation)
%% =========================================================

% Toolbox dependency: kmeans is in the Statistics and ML Toolbox.
% If it isn't on the path or licensed, halt now with a clear message.
if ~exist('kmeans', 'file')
    error(['kmeans is required but not available.\n' ...
           '   Install / license the Statistics and Machine Learning Toolbox.']);
end

% Kernel type
if ~ismember(KERNEL_TYPE, 1:4)
    error('KERNEL_TYPE must be 1, 2, 3, or 4 (got %d).', KERNEL_TYPE);
end

% Numeric settings: positive, non-zero
must_pos = struct( ...
    'sigma_rbf_x', sigma_rbf_x, 'nz_rbf', nz_rbf, ...
    'ell_rq_x', ell_rq_x, 'nz_rq', nz_rq, 'alpha_rq_x', alpha_rq_x, ...
    'ell_mat_x', ell_mat_x, 'nz_mat', nz_mat, ...
    'deg_poly_x', deg_poly_x, 'deg_cheb_u', deg_cheb_u, ...
    'L', L, 'T_long', T_long, 'M_raw_max', M_raw_max, 'M_target', M_target, ...
    'N', N, 'gamma', gamma, ...
    'T_test', T_test, 'N_test', N_test, 'Ts', params.Ts);
flds = fieldnames(must_pos);
for ii = 1:numel(flds)
    if must_pos.(flds{ii}) <= 0
        error('%s must be > 0 (got %g).', flds{ii}, must_pos.(flds{ii}));
    end
end

% Logical consistency
if L < 3
    error('L must be >= 3 to balance ICs across left/right/saddle (got %d).', L);
end
if N >= T_long
    error('N (=%d) must be < T_long (=%d) to extract %d-step windows.', N, T_long, N);
end
if M_target > M_raw_max
    warning('M_target (%d) > M_raw_max (%d); k-means will select from a smaller pool.', ...
        M_target, M_raw_max);
end
if numel(x0_test) ~= 2
    error('x0_test must be a 2x1 vector (got %d elements).', numel(x0_test));
end

% Per-kernel model filename, so trying a new kernel does not overwrite the
% previous model. Part 2 (MPC) selects the matching file via kernel_choice.
switch KERNEL_TYPE
    case 1, ktag = 'rbf';
    case 2, ktag = 'rq';
    case 3, ktag = 'matern';
    case 4, ktag = 'poly';
end
save_path = sprintf('koopman_duffing_model_%s.mat', ktag);

% Output directory writable: try a small probe file
save_path_probe = save_path;
[fid, msg] = fopen(save_path_probe, 'a');
if fid == -1
    error(['Cannot write ''%s'' in the current directory:\n   %s\n' ...
           '   Change directory to one with write permission and try again.'], ...
          save_path_probe, msg);
else
    fclose(fid);
    % Delete the empty probe file if we created it new (don't clobber an
    % existing valid save; just confirm writability).
    info = dir(save_path_probe);
    if ~isempty(info) && info.bytes == 0
        delete(save_path_probe);
    end
end

%% =========================================================
%  RESOLVE KERNEL-SPECIFIC PARAMETERS
%% =========================================================

switch KERNEL_TYPE
    case 1
        kernel_name_x = 'Gaussian RBF';
        use_centres_x = true;
        n_centres_x = nz_rbf;
        nz     = nz_rbf + 3*prepend_state;   % +3 only when prepending
        kfun_x = @(X,C,hp) rbf_kernel(X,C,hp(1));
        hp_x   = sigma_rbf_x;
    case 2
        kernel_name_x = 'Rational Quadratic';
        use_centres_x = true;
        n_centres_x = nz_rq;
        nz     = nz_rq + 3*prepend_state;   % +3 only when prepending
        kfun_x = @(X,C,hp) rq_kernel(X,C,hp(1),hp(2));
        hp_x   = [ell_rq_x, alpha_rq_x];
    case 3
        kernel_name_x = 'Matérn 5/2';
        use_centres_x = true;
        n_centres_x = nz_mat;
        nz     = nz_mat + 3*prepend_state;   % +3 only when prepending
        kfun_x = @(X,C,hp) matern52_kernel(X,C,hp(1));
        hp_x   = ell_mat_x;
    case 4
        kernel_name_x = 'Polynomial';
        use_centres_x = false;
        n_centres_x = 0;   % poly lift already starts with [1, x1, x2]
        nz     = nchoosek(2 + deg_poly_x, deg_poly_x);
        kfun_x = [];
        hp_x   = [];
    otherwise
        error('KERNEL_TYPE must be 1..4.');
end

% Input kernel: Chebyshev T_0..T_d, always
kernel_name_u = sprintf('Chebyshev T_0..T_%d', deg_cheb_u);
nv            = deg_cheb_u + 1;

fprintf('\n=========================================\n');
fprintf(' Koopman Form Comparison — Duffing (Part 1)\n');
fprintf(' State kernel : %s\n', kernel_name_x);
fprintf(' Input kernel : %s\n', kernel_name_u);
fprintf(' nz=%d  nv=%d\n', nz, nv);
fprintf('=========================================\n');

%% =========================================================
%  SECTION 1 : GENERATE TRAINING TRAJECTORIES
%
%  Balanced IC sampling across both Duffing wells:
%    - 1/3 in the LEFT well   (x1 in [-2.5, -0.3])
%    - 1/3 in the RIGHT well  (x1 in [+0.3, +2.5])
%    - 1/3 NEAR THE SADDLE    (x1 in [-0.5, +0.5])
%  Without this, random ICs concentrate in one well and the
%  Koopman model later predicts the wrong basin for test ICs.
%% =========================================================
fprintf('\n[1] Generating training trajectories...\n');
X_all = [];  U_all = [];
X_trajs = cell(L,1);  U_trajs = cell(L,1);
n_per_group = ceil(L / 3);
ic_pool     = zeros(2, 3 * n_per_group);
for g = 1:3
    for k = 1:n_per_group
        col = (g-1)*n_per_group + k;
        switch g
            case 1, x1 = -2.5 + (-0.3 - (-2.5)) * rand;
            case 2, x1 =  0.3 + ( 2.5 -  0.3 ) * rand;
            case 3, x1 = -0.5 + ( 0.5 - (-0.5)) * rand;
        end
        x2 = (2*rand - 1) * 2.5;
        ic_pool(:, col) = [x1; x2];
    end
end
ic_pool   = ic_pool(:, randperm(size(ic_pool,2)));
X0_train  = ic_pool(:, 1:L);
for l = 1:L
    x0_l   = X0_train(:, l);
    t_vec  = (0:T_long-1) * params.Ts;
    freqs  = [0.3, 0.7, 1.1, 1.9, 3.1, 5.3];
    phases = 2*pi*rand(1,6);
    % Amplitude 0.6 per sine, 6 sines: rms ~ sqrt(6)*0.6 ~ 1.5 with peaks
    % reaching ~3.6, so clipping at u_max_phys actually saturates regularly.
    % This exercises the full input range MPC will use.
    u_l_phys = sum(0.6 * sin(freqs .* t_vec' + phases), 2)';
    u_l_phys = max(-u_max_phys, min(u_max_phys, u_l_phys));
    X_l      = simulate_duffing(x0_l, u_l_phys, params);
    % Store SCALED u (in [-1,1]) for consistency with Chebyshev domain;
    X_trajs{l} = X_l;  U_trajs{l} = Su * u_l_phys;
    X_all = [X_all, X_l(:, 1:end-1)]; 
    U_all = [U_all, u_l_phys];        
end
fprintf('   L=%d trajectories, total snapshots: %d\n', L, size(X_all, 2));
fprintf('   IC distribution: x1<0 -> %d   x1>0 -> %d   |x1|<0.5 -> %d\n', ...
    sum(X0_train(1,:) < -0.1), sum(X0_train(1,:) > 0.1), sum(abs(X0_train(1,:)) < 0.5));

%% =========================================================
%  SECTION 2 : KERNEL CENTRES (state only; input has no centres)
%% =========================================================
fprintf('\n[2] Kernel centres...\n');
if use_centres_x
    [~, C_x] = kmeans(X_all', n_centres_x, 'MaxIter', 300, 'Replicates', 3, 'Display', 'off');
    C_x = C_x';
    fprintf('   C_x: %dx%d\n', size(C_x, 1), size(C_x, 2));
else
    C_x = [];
    fprintf('   State: polynomial (no centres)\n');
end

%% =========================================================
%  SECTION 3 : LIFT TRAINING DATA
%% =========================================================
fprintf('\n[3] Lifting training trajectories...\n');
Z_trajs = cell(L, 1);  V_trajs = cell(L, 1);
for l = 1:L
    if use_centres_x
        Z_trajs{l} = lift_state(X_trajs{l}, C_x, hp_x, kfun_x, prepend_state);
    else
        Z_trajs{l} = poly_lift_state(X_trajs{l}, deg_poly_x);
    end
    V_trajs{l} = cheb_lift_input(U_trajs{l}, deg_cheb_u);
end

%% =========================================================
%  SECTION 4 : COLLECT MULTI-STEP WINDOWS
%
%  Each window is of length N+1. We extract windows as cells
%  first (cell growth in MATLAB is cheap; matrix concat is not),
%  then select a representative subset via k-means on the first
%  feature of each window, then build preallocated matrices.
%% =========================================================
fprintf('\n[4] Collecting windows (N=%d)...\n', N);
all_Z = {};  all_V = {};  all_U = {};
for l = 1:L
    Zl = Z_trajs{l};  Vl = V_trajs{l};  Ul = U_trajs{l};
    for t = 1 : size(Vl, 2) - N
        all_Z{end+1} = Zl(:, t : t+N);     
        all_V{end+1} = Vl(:, t : t+N-1);  
        all_U{end+1} = Ul(:, t : t+N-1);  
    end
end
M_raw = numel(all_Z);
if M_raw > M_raw_max
    keep  = randperm(M_raw, M_raw_max);
    all_Z = all_Z(keep);  all_V = all_V(keep);  all_U = all_U(keep);
    M_raw = M_raw_max;
end
fprintf('   Raw windows: %d\n', M_raw);

%% =========================================================
%  SECTION 4b : K-MEANS WINDOW SELECTION  (subset S, |S| = M_a)
%
%  Cluster each window by its trajectory SHAPE -- the stacked
%  triple [z_0; z_mid; z_N] (start / midpoint / end of the lifted
%  window) -- and keep the window closest to each cluster centre.
%  Using start+mid+end instead of z_0 alone selects windows that
%  differ in how they EVOLVE, not just where they begin, so the
%  multi-step GeKo fit sees residuals spread over distinct regions
%  of the lifted flow z_0..z_N rather than near-duplicate rollouts
%  sharing an initial condition. The three blocks are lifted states
%  on the same scale, so plain concatenation is commensurate.
%% =========================================================
fprintf('\n[4b] Window selection (M_target=%d)...\n', M_target);
% Window columns 1..N+1 hold z_t..z_{t+N}; pick start / middle / end.
c0 = 1;  cmid = 1 + round(N/2);  cN = N + 1;
desc_all = zeros(3*nz, M_raw);
for i = 1:M_raw
    desc_all(:, i) = [all_Z{i}(:, c0); all_Z{i}(:, cmid); all_Z{i}(:, cN)];
end
[~, C_km] = kmeans(desc_all', M_target, 'MaxIter', 200, 'Replicates', 3, 'Display', 'off');
sel = zeros(1, M_target);
for k = 1:M_target
    [~, sel(k)] = min(sum((desc_all' - C_km(k,:)).^2, 2));
end
sel   = unique(sel);
M_a   = numel(sel);        % |S|, selected subset size (paper notation)
M_aug = N * M_a;           % augmented multi-step pairs, M_aug = N*M_a
fprintf('   Selected windows |S|=%d   Multi-step pairs M_aug=%d\n', M_a, M_aug);

%% =========================================================
%  SECTION 4c : BUILD SHARED DESIGN MATRICES
%
%  Multi-step pairs (z_k, v_k, u_k, z_{k+1}) come from k=0..N-1
%  columns of the M_a selected windows in S -- M_aug = N * M_a pairs.
%
%  One-step pairs are built INDEPENDENTLY of windowing: a random
%  subsample of all one-step transitions in the training data, as
%  standard EDMDc would use. Sample count is set equal to M_aug so
%  the one-step vs multi-step comparison sees the same amount of
%  data and isolates the regression objective.
%% =========================================================
fprintf('\n[4c] Building shared design matrices...\n');

% --- Multi-step pairs (unchanged) ---
Z_pairs  = zeros(nz, M_aug);   V_pairs  = zeros(nv, M_aug);
U_pairs  = zeros(1,  M_aug);   Zp_pairs = zeros(nz, M_aug);
p = 0;
for k = 0:N-1
    for ii = 1:M_a
        p = p + 1;
        wi = sel(ii);
        Z_pairs (:, p) = all_Z{wi}(:, k+1);
        V_pairs (:, p) = all_V{wi}(:, k+1);
        U_pairs (1, p) = all_U{wi}(:, k+1);
        Zp_pairs(:, p) = all_Z{wi}(:, k+2);
    end
end

% --- One-step pairs: standard EDMDc data set ---
% Pool ALL one-step transitions (z_k, v_k, u_k, z_{k+1}) from every
% training trajectory, then randomly subsample to M_aug pairs.
Z_one_pool  = [];  V_one_pool  = [];
U_one_pool  = [];  Zp_one_pool = [];
for l = 1:L
    Zl = Z_trajs{l};  Vl = V_trajs{l};  Ul = U_trajs{l};
    Z_one_pool  = [Z_one_pool,  Zl(:, 1:end-1)];  
    V_one_pool  = [V_one_pool,  Vl              ];  
    U_one_pool  = [U_one_pool,  Ul              ];  
    Zp_one_pool = [Zp_one_pool, Zl(:, 2:end  )];  
end
M_one_pool = size(Z_one_pool, 2);
M_one      = min(M_aug, M_one_pool);   % match multi-step sample count
idx_one    = randperm(M_one_pool, M_one);
Z_pairs_1  = Z_one_pool (:, idx_one);
V_pairs_1  = V_one_pool (:, idx_one);
U_pairs_1  = U_one_pool (:, idx_one);
Zp_pairs_1 = Zp_one_pool(:, idx_one);

fprintf('   Multi-step pairs: %d\n', M_aug);
fprintf('   One-step pool:    %d   sampled: %d\n', M_one_pool, M_one);

%% =========================================================
%  SECTION 5 : LINEAR DECODER  D : z -> x
%
%  z = phi(x) is high-dimensional; we fit a linear decoder so
%  predicted z can be projected back to state space for error
%  computation and plots.
%% =========================================================
fprintf('\n[5] Fitting linear decoder D...\n');
if use_centres_x
    Z_all = lift_state(X_all, C_x, hp_x, kfun_x, prepend_state);
else
    Z_all = poly_lift_state(X_all, deg_poly_x);
end
if prepend_state
    % Choice 1: decoder = exact projection onto the state rows (2:3), the same
    % selection for all four forms (state sits at rows 2:3 in every lift).
    n_state = size(X_all, 1);
    D_lin = [zeros(n_state,1), eye(n_state), zeros(n_state, nz-1-n_state)];
else
    % Choice 2: one regressed decoder, shared by all four forms.
    D_lin = (X_all * Z_all') / (Z_all * Z_all' + gamma * eye(nz));
end
fprintf('   Decode RMSE: %.4f\n', ...
    mean(sqrt(sum((D_lin * Z_all - X_all).^2, 1))));

%% =========================================================
%  SECTION 6 : FIT ALL 4 KOOPMAN FORMS  (multi-step and one-step)
%% =========================================================
fprintf('\n[6] Fitting all 4 Koopman forms...\n');
models = {
    'Linear',      @fit_linear,   @predict_linear;
    'Bilinear',    @fit_bilinear, @predict_bilinear;
    'GeKo',        @fit_geko,     @predict_geko;
    'KCF',         @fit_kcf,      @predict_kcf;
};
n_models = size(models, 1);
M_ms = cell(n_models, 1);  fit_times_ms = zeros(n_models, 1);
M_1s = cell(n_models, 1);  fit_times_1s = zeros(n_models, 1);
% --- Build the KCF coupling index (see knobs above) ---
%   Poly lift  : the low-degree monomials (total degree <= kcf_couple_deg).
%   Kernel lift: the highest-variance features; if the state was prepended,
%                keep {1,x1,x2} and rank the rest, else rank all rows.
nv_eff_k = nv - 1;                       % Chebyshev T_0 dropped inside KCF
if ~use_centres_x
    if kcf_couple_deg < 1 || kcf_couple_deg > deg_poly_x || ...
            kcf_couple_deg ~= round(kcf_couple_deg)
        error('kcf_couple_deg must be an integer in 1..deg_poly_x (=%d).', deg_poly_x);
    end
    kcf_couple_idx = 1:nchoosek(2 + kcf_couple_deg, kcf_couple_deg);
else
    if kcf_s_ratio <= 1
        error('kcf_s_ratio must be > 1 (target augmented size s = ratio*nz).');
    end
    if prepend_state
        [~, ord]      = sort(var(Z_all(4:end, :), 0, 2), 'descend');  % kernel rows
        kcf_var_order = ord + 3;                                      % -> rows 4..nz
        p_couple      = max(3, min(nz, round((kcf_s_ratio - 1) * nz / nv_eff_k)));
        kcf_couple_idx = sort([1 2 3, kcf_var_order(1:(p_couple-3))']);
    else
        [~, kcf_var_order] = sort(var(Z_all, 0, 2), 'descend');       % all rows
        p_couple       = max(1, min(nz, round((kcf_s_ratio - 1) * nz / nv_eff_k)));
        kcf_couple_idx = sort(kcf_var_order(1:p_couple))';
    end
end
fprintf('   KCF couples input to %d of %d lift features (s = %d).\n', ...
    numel(kcf_couple_idx), nz, nz + nv_eff_k*numel(kcf_couple_idx));

for m = 1:n_models
    tic; M_ms{m} = models{m,2}(Z_pairs,   V_pairs,   U_pairs,   Zp_pairs,   gamma, nz, nv, kcf_couple_idx);
    fit_times_ms(m) = toc;
    tic; M_1s{m} = models{m,2}(Z_pairs_1, V_pairs_1, U_pairs_1, Zp_pairs_1, gamma, nz, nv, kcf_couple_idx);
    fit_times_1s(m) = toc;
    fprintf('   %-14s multi-step: %6.3fs   one-step: %6.3fs\n', ...
        models{m,1}, fit_times_ms(m), fit_times_1s(m));
end

%% =========================================================
%  SECTION 7 : OPERATOR-NORM DIAGNOSTICS
%
%  Frobenius norms give a quick sense of which block of each
%  operator carries most of the dynamics. They are a DIAGNOSTIC, not an
%  accuracy score -- read them RELATIVELY, within one model:
%    * which block dominates tells you how the input acts (pure additive
%      ||B|| vs state-modulated ||N|| / ||A_12||) -- i.e. how nonlinear the
%      learned input channel is;
%    * an input-coupling norm (||N||, ||A_12||) that dwarfs the autonomous
%      block usually signals an over-inflated, ill-conditioned fit and tends
%      to predict oscillatory / diverging rollouts (raise gamma or shrink
%      the data set if you see it);
%    * comparing the MULTI-STEP vs ONE-STEP tables shows whether multi-step
%      training tamed those norms (it usually lowers the coupling blocks).
%  Absolute values depend on the lift scaling (sqrt(2/nz), Su), so don't
%  compare norms across different kernels or benchmarks.
%
%    Linear   :  z+ = A z + B u
%       ||A|| = autonomous evolution      ||B|| = input gain
%       ratio A/B large ==> almost autonomous (input has little effect)
%
%    Bilinear :  z+ = A z + B u + N z u
%       ||A|| = autonomous evolution      ||B|| = pure input effect
%       ||N|| = state-modulated input effect; large means the input
%               acts strongly through the state (true nonlinearity)
%
%    GeKo     :  z+ = K (z (x) v)
%       ||K|| = one combined operator on the Kronecker space; no
%               natural decomposition of autonomous vs input parts.
%
%    KCF      :  z+ = A_11 z + A_12 (v (x) z)
%       ||A_11|| = autonomous evolution (input-independent channel)
%       ||A_12|| = input-modulated evolution
%       Comparable magnitudes mean both contribute meaningfully.
%% =========================================================
fprintf('\n[7] Operator norm diagnostics\n');
fprintf('   ----- Multi-step models -----\n');
ML = M_ms{1};  fprintf('   Linear      :  ||A||_F = %.4e   ||B||_F = %.4e\n', ...
    norm(ML.A,'fro'), norm(ML.B,'fro'));
MB = M_ms{2};  fprintf('   Bilinear    :  ||A||_F = %.4e   ||B||_F = %.4e   ||N||_F = %.4e\n', ...
    norm(MB.A,'fro'), norm(MB.B,'fro'), norm(MB.N,'fro'));
MG = M_ms{3};  fprintf('   GeKo        :  ||K||_F = %.4e\n', norm(MG.K,'fro'));
MK = M_ms{4};
fprintf('   KCF         :  ||A_11||_F = %.4e   ||A_12||_F = %.4e\n', ...
    norm(MK.A11,'fro'), norm(MK.A12,'fro'));
fprintf('   ----- One-step models -----\n');
ML1 = M_1s{1}; fprintf('   Linear      :  ||A||_F = %.4e   ||B||_F = %.4e\n', ...
    norm(ML1.A,'fro'), norm(ML1.B,'fro'));
MB1 = M_1s{2}; fprintf('   Bilinear    :  ||A||_F = %.4e   ||B||_F = %.4e   ||N||_F = %.4e\n', ...
    norm(MB1.A,'fro'), norm(MB1.B,'fro'), norm(MB1.N,'fro'));
MG1 = M_1s{3}; fprintf('   GeKo        :  ||K||_F = %.4e\n', norm(MG1.K,'fro'));
MK1 = M_1s{4};
fprintf('   KCF         :  ||A_11||_F = %.4e   ||A_12||_F = %.4e\n', ...
    norm(MK1.A11,'fro'), norm(MK1.A12,'fro'));

%% =========================================================
%  SECTION 8 : OPEN-LOOP VALIDATION  (10 test ICs)
%
%  Each model is rolled out from t = 0 only (no feedback from
%  later true states). Per-trajectory error ||x_hat(t) - x*(t)||
%  is computed and stored for plotting (IC #1 only) and for the
%  per-trajectory MAE summary tables (all ICs).
%% =========================================================
fprintf('\n[8] Open-loop validation (T=%d, N_test=%d ICs)...\n', T_test, N_test);

t_ax   = (0:T_test) * params.Ts;
u_test = 0.5*sin(0.5*t_ax) + 0.3*sin(1.3*t_ax) + ...
         0.4*sin(2.7*t_ax) + 0.2*sin(4.1*t_ax);
u_test = max(-u_max_phys, min(u_max_phys, u_test(1:T_test)));
u_test_scaled = Su * u_test;     % what predictors expect (matches training)

% Test ICs: IC #1 = x0_test fixed; rest random in [-2.5, 2.5]^2
rng_state_ic = rng;
rng(2025, 'twister');
X0_test         = zeros(2, N_test);
X0_test(:,1)    = x0_test;
X0_test(:,2:end) = (2*rand(2, N_test-1) - 1) * 2.5;
rng(rng_state_ic);

V_test = cheb_lift_input(u_test_scaled, deg_cheb_u);

err_z_all_ms = cell(n_models, 1);  err_x_all_ms = cell(n_models, 1);
err_z_all_1s = cell(n_models, 1);  err_x_all_1s = cell(n_models, 1);
for m = 1:n_models
    err_z_all_ms{m} = zeros(N_test, T_test+1);
    err_x_all_ms{m} = zeros(N_test, T_test+1);
    err_z_all_1s{m} = zeros(N_test, T_test+1);
    err_x_all_1s{m} = zeros(N_test, T_test+1);
end

X_test_rep = [];  % IC #1 ground truth, kept for the trajectory plots

for i = 1:N_test
    X_i = simulate_duffing(X0_test(:,i), u_test, params);   % simulator: physical u
    if i == 1, X_test_rep = X_i; end
    if use_centres_x
        Z_true_i = lift_state(X_i, C_x, hp_x, kfun_x, prepend_state);
    else
        Z_true_i = poly_lift_state(X_i, deg_poly_x);
    end
    for m = 1:n_models
        pred_fn = models{m, 3};
        % Predictors: pass scaled u (consistent with training)
        Z_hat_ms = rollout(M_ms{m}, pred_fn, Z_true_i(:,1), V_test, u_test_scaled, nz, T_test);
        Z_hat_1s = rollout(M_1s{m}, pred_fn, Z_true_i(:,1), V_test, u_test_scaled, nz, T_test);
        X_hat_ms = D_lin * Z_hat_ms;
        X_hat_1s = D_lin * Z_hat_1s;
        err_z_all_ms{m}(i,:) = sqrt(sum((Z_hat_ms - Z_true_i).^2, 1));
        err_x_all_ms{m}(i,:) = sqrt(sum((X_hat_ms - X_i      ).^2, 1));
        err_z_all_1s{m}(i,:) = sqrt(sum((Z_hat_1s - Z_true_i).^2, 1));
        err_x_all_1s{m}(i,:) = sqrt(sum((X_hat_1s - X_i      ).^2, 1));
    end
end

% Per-trajectory aggregated MAE (one number per IC per form)
%   MAE_i^{(m)} = (1/(T+1)) sum_t || x_hat^{(i,m)}(t) - x*^{(i)}(t) ||
mae_x_ms = zeros(N_test, n_models);
mae_x_1s = zeros(N_test, n_models);
for m = 1:n_models
    mae_x_ms(:, m) = mean(err_x_all_ms{m}, 2);   % multi-step, N_test x 1
    mae_x_1s(:, m) = mean(err_x_all_1s{m}, 2);   % one-step,   N_test x 1
end

fprintf('\n   Per-trajectory aggregated error (state, MULTI-STEP)\n');
fprintf('     mean_t ||x_hat - x*||,  averaged over T+1 = %d steps.\n', T_test+1);
fprintf('     (bold value = lowest of the 4 forms for that IC)\n');
fprintf('     %-3s', 'IC');
for m = 1:n_models, fprintf('  %12s', models{m,1}); end
fprintf('\n');
for i = 1:N_test
    fprintf('     %-3d', i);
    row      = mae_x_ms(i, :);
    best_val = min(row);
    tol      = max(1e-12, 5e-5 * abs(best_val));
    for m = 1:n_models
        if abs(row(m) - best_val) <= tol
            fprintf('  <strong>%12.4e</strong>', row(m));
        else
            fprintf('  %12.4e', row(m));
        end
    end
    fprintf('\n');
end

fprintf('\n   Per-trajectory aggregated error (state, ONE-STEP)\n');
fprintf('     mean_t ||x_hat - x*||,  averaged over T+1 = %d steps.\n', T_test+1);
fprintf('     (bold value = lowest of the 4 forms for that IC)\n');
fprintf('     %-3s', 'IC');
for m = 1:n_models, fprintf('  %12s', models{m,1}); end
fprintf('\n');
for i = 1:N_test
    fprintf('     %-3d', i);
    row      = mae_x_1s(i, :);
    best_val = min(row);
    tol      = max(1e-12, 5e-5 * abs(best_val));
    for m = 1:n_models
        if abs(row(m) - best_val) <= tol
            fprintf('  <strong>%12.4e</strong>', row(m));
        else
            fprintf('  %12.4e', row(m));
        end
    end
    fprintf('\n');
end

%% =========================================================
%  SECTION 8b : KCF coupling sweep (informational)
%
%  Refits KCF over a range of coupling sizes and scores each with the same
%  open-loop state error as the table above, so you can see how accuracy
%  varies with the number of coupled features and pick a good operating point.
%% =========================================================
fprintf('\n[8b] KCF coupling sweep (open-loop state RMSE):\n');
% Cache test ground truth once (independent of the coupling budget).
Xgt = cell(N_test, 1);  z0gt = zeros(nz, N_test);
for i = 1:N_test
    Xi = simulate_duffing(X0_test(:,i), u_test, params);   % physical state
    if use_centres_x
        Zi = lift_state(Xi, C_x, hp_x, kfun_x, prepend_state);
    else
        Zi = poly_lift_state(Xi, deg_poly_x);
    end
    Xgt{i} = Xi;  z0gt(:,i) = Zi(:,1);
end
% Build the list of coupling-index sets to score.
if ~use_centres_x
    budgets = cell(1, deg_poly_x);  blabel = cell(1, deg_poly_x);
    for q = 1:deg_poly_x
        budgets{q} = 1:nchoosek(2 + q, q);   blabel{q} = sprintf('deg=%d', q);
    end
else
    if prepend_state
        base    = max(3, round(nz / nv_eff_k));
        nc_grid = unique(min(nz, [max(3,round(base/2)), base, 2*base, 4*base, nz]));
        budgets = cell(1, numel(nc_grid));  blabel = cell(1, numel(nc_grid));
        for k = 1:numel(nc_grid)
            nE         = nc_grid(k) - 3;
            budgets{k} = sort([1 2 3, kcf_var_order(1:nE)']);
            blabel{k}  = sprintf('p=%d', nc_grid(k));
        end
    else
        base   = max(1, round(nz / nv_eff_k));
        p_grid = unique(min(nz, [max(1,round(base/2)), base, 2*base, 4*base, nz]));
        budgets = cell(1, numel(p_grid));  blabel = cell(1, numel(p_grid));
        for k = 1:numel(p_grid)
            budgets{k} = sort(kcf_var_order(1:p_grid(k)))';
            blabel{k}  = sprintf('p=%d', p_grid(k));
        end
    end
end
fprintf('       %-8s %-9s %-5s %-12s\n', 'budget', '#coupled', 's', 'state RMSE');
for k = 1:numel(budgets)
    ci = budgets{k};
    Mq = fit_kcf(Z_pairs, V_pairs, U_pairs, Zp_pairs, gamma, nz, nv, ci);
    e_ic = zeros(N_test, 1);
    for i = 1:N_test
        Z_hat   = rollout(Mq, @predict_kcf, z0gt(:,i), V_test, u_test_scaled, nz, T_test);
        X_hat   = D_lin * Z_hat;
        e_ic(i) = mean(sqrt(sum((X_hat - Xgt{i}).^2, 1)));
    end
    tag = '';
    if numel(ci) == numel(kcf_couple_idx), tag = '  <- operative'; end
    if numel(ci) == nz,                    tag = [tag '  (full coupling)']; end
    fprintf('       %-8s %-9d %-5d %12.4e%s\n', ...
        blabel{k}, numel(ci), nz + nv_eff_k*numel(ci), mean(e_ic), tag);
end

%% =========================================================
%  SECTION 9 : PLOTS
%
%  All plots use 4 solid colours (one per form). Two figures
%  for the rollout error (feature, state), each with multi-step
%  on top and one-step on bottom.  No 3D feature error surface.
%% =========================================================
fprintf('\n[9] Plotting...\n');
colors = [0.00 0.45 0.74;    % Linear   — blue
          0.00 0.60 0.20;    % Bilinear — green
          0.85 0.10 0.10;    % GeKo     — red
          0.75 0.00 0.75];   % KCF      — magenta

% Distinct markers per form so overlapping curves stay legible (GeKo and
% KCF in particular can run close together). ~12 marks/curve, evenly spaced.
markers = {'o','s','^','d'};
mk_idx  = @(npts) unique([1, round(linspace(1, npts, 12)), npts]);

% --- Figure 1: feature-space RMSE for IC #1, multi-step + one-step ---
figure('Name','Figure 1: feature-space rollout error (IC #1)');
subplot(2,1,1);  hold on;
for m = 1:n_models
    semilogy(t_ax, err_z_all_ms{m}(1,:), '-', 'Color', colors(m,:), 'LineWidth', 2, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(numel(t_ax)), 'MarkerSize', 5);
end
set(gca,'YScale','log');
xlabel('Time [s]');  ylabel('|z_{hat} - z*|');
title(sprintf('Feature error (IC #1)  |  multi-step  |  %s', kernel_name_x));
legend(models(:,1), 'Location', 'best');  grid on;

subplot(2,1,2);  hold on;
for m = 1:n_models
    semilogy(t_ax, err_z_all_1s{m}(1,:), '-', 'Color', colors(m,:), 'LineWidth', 2, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(numel(t_ax)), 'MarkerSize', 5);
end
set(gca,'YScale','log');
xlabel('Time [s]');  ylabel('|z_{hat} - z*|');
title('Feature error (IC #1)  |  one-step');
legend(models(:,1), 'Location', 'best');  grid on;

% --- Figure 2: state-space RMSE for IC #1, multi-step + one-step ---
figure('Name','Figure 2: state-space rollout error (IC #1)');
subplot(2,1,1);  hold on;
for m = 1:n_models
    semilogy(t_ax, err_x_all_ms{m}(1,:), '-', 'Color', colors(m,:), 'LineWidth', 2, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(numel(t_ax)), 'MarkerSize', 5);
end
set(gca,'YScale','log');
xlabel('Time [s]');  ylabel('|x_{hat} - x*|');
title(sprintf('State error (IC #1)  |  multi-step  |  %s', kernel_name_x));
legend(models(:,1), 'Location', 'best');  grid on;

subplot(2,1,2);  hold on;
for m = 1:n_models
    semilogy(t_ax, err_x_all_1s{m}(1,:), '-', 'Color', colors(m,:), 'LineWidth', 2, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(numel(t_ax)), 'MarkerSize', 5);
end
set(gca,'YScale','log');
xlabel('Time [s]');  ylabel('|x_{hat} - x*|');
title('State error (IC #1)  |  one-step');
legend(models(:,1), 'Location', 'best');  grid on;

% --- Figure 3: state trajectories at IC #1, true + 4 multi-step models ---
% Decode each predicted z back to x via D_lin, plot x1 and x2.
figure('Name','Figure 3: state trajectories (IC #1)');
ax_x1 = [-2.5, 2.5];  ax_x2 = [-3, 3];
ax_x1_lo = min(X_test_rep(1,:));  ax_x1_hi = max(X_test_rep(1,:));
ax_x2_lo = min(X_test_rep(2,:));  ax_x2_hi = max(X_test_rep(2,:));
X_hat_all = cell(n_models, 1);
for m = 1:n_models
    pred_fn  = models{m, 3};
    if use_centres_x
        z0_m = lift_state(X_test_rep(:,1), C_x, hp_x, kfun_x, prepend_state);
    else
        z0_m = poly_lift_state(X_test_rep(:,1), deg_poly_x);
    end
    Z_hat = rollout(M_ms{m}, pred_fn, z0_m, V_test, u_test_scaled, nz, T_test);
    X_hat_all{m} = D_lin * Z_hat;
    ax_x1_lo = min(ax_x1_lo, min(X_hat_all{m}(1,:)));
    ax_x1_hi = max(ax_x1_hi, max(X_hat_all{m}(1,:)));
    ax_x2_lo = min(ax_x2_lo, min(X_hat_all{m}(2,:)));
    ax_x2_hi = max(ax_x2_hi, max(X_hat_all{m}(2,:)));
end
m1 = 0.05*(ax_x1_hi - ax_x1_lo);  m2 = 0.05*(ax_x2_hi - ax_x2_lo);
subplot(2,1,1);  hold on;
plot(t_ax, X_test_rep(1,:), 'k-', 'LineWidth', 2.5);
for m = 1:n_models
    plot(t_ax, X_hat_all{m}(1,:), '-', 'Color', colors(m,:), 'LineWidth', 1.5, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(numel(t_ax)), 'MarkerSize', 5);
end
ylabel('x_1');
title(sprintf('Open-loop state prediction (IC #1)  |  %s', kernel_name_x));
legend([{'True'}; models(:,1)], 'Location', 'best');  grid on;
ylim([ax_x1_lo - m1, ax_x1_hi + m1]);

subplot(2,1,2);  hold on;
plot(t_ax, X_test_rep(2,:), 'k-', 'LineWidth', 2.5);
for m = 1:n_models
    plot(t_ax, X_hat_all{m}(2,:), '-', 'Color', colors(m,:), 'LineWidth', 1.5, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(numel(t_ax)), 'MarkerSize', 5);
end
ylabel('x_2');  xlabel('Time [s]');
legend([{'True'}; models(:,1)], 'Location', 'best');  grid on;
ylim([ax_x2_lo - m2, ax_x2_hi + m2]);

% --- Figure 4: phase portrait at IC #1, true + 4 multi-step models ---
figure('Name','Figure 4: phase portrait (IC #1)');
hold on;
plot(X_test_rep(1,:), X_test_rep(2,:), 'k-', 'LineWidth', 2.5);
for m = 1:n_models
    plot(X_hat_all{m}(1,:), X_hat_all{m}(2,:), '-', ...
        'Color', colors(m,:), 'LineWidth', 1.5, ...
        'Marker', markers{m}, 'MarkerIndices', mk_idx(size(X_hat_all{m},2)), 'MarkerSize', 5);
end
% Duffing equilibria: stable wells at x1 = +/- 1, saddle at x1 = 0
plot([-1, 0, 1], [0, 0, 0], 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
% Starting IC marker
plot(X_test_rep(1,1), X_test_rep(2,1), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'y');
xlabel('x_1');  ylabel('x_2');
title(sprintf('Phase portrait (IC #1)  |  %s', kernel_name_x));
legend([{'True'}; models(:,1); {'Equilibria'; 'x_0'}], 'Location', 'best');
grid on;  axis equal;

%% =========================================================
%  SECTION 10 : SAVE TRAINED MODELS FOR PART 2 (MPC)
%% =========================================================
fprintf('\n[10] Saving trained models...\n');
% save_path was set per-kernel during INPUT VALIDATION
% (e.g. koopman_duffing_model_rbf.mat).
% Note: function handles to local functions (kfun_x, kfun_u, predictors)
% don't survive .mat reliably. We save KERNEL_TYPE and hyperparameters
% so Part 2 can reconstruct the kernel handle from its own local copies.
model_names = models(:, 1);
save(save_path, ...
    'M_ms', 'M_1s', 'model_names', ...
    'KERNEL_TYPE', 'kernel_name_x', 'kernel_name_u', ...
    'use_centres_x', 'C_x', 'hp_x', ...
    'deg_poly_x', 'deg_cheb_u', ...
    'nz', 'nv', 'params', 'D_lin', ...
    'Su', 'u_max_phys', 'prepend_state', '-v7.3');
fprintf('   Saved to %s\n', save_path);
fprintf('\nPart 1 complete.\n');

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

% ---------- Duffing simulator (RK45) ----------
function X = simulate_duffing(x0, u_seq, p)
    T = numel(u_seq);  X = zeros(2, T+1);  X(:,1) = x0;
    opts = odeset('RelTol',1e-7,'AbsTol',1e-9);
    for t = 1:T
        ut = u_seq(t);
        f = @(~,x)[ x(2);
                   -p.delta*x(2) - p.alpha*x(1) - p.beta*x(1)^3 + ut ];
        [~, xo] = ode45(f, [0, p.Ts], X(:,t), opts);
        X(:, t+1) = xo(end, :)';
    end
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
        % Choice 1: prepend [1; x1; x2] (constant + physical state) so the
        % state lives in the lift (rows 1:3). The decoder is then a projection
        % onto rows 2:3 and KCF can couple the input to {1,x1,x2}. Shared by
        % all four forms. (Choice 2 / prepend=false: pure kernel features.)
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
    % Chebyshev polynomials of the first kind, T_0..T_d, evaluated at U.
    % Stable three-term recurrence:
    %   T_0(u) = 1,  T_1(u) = u,  T_{k+1}(u) = 2 u T_k(u) - T_{k-1}(u)
    M = numel(U);  V = zeros(d+1, M);
    V(1,:) = 1;
    if d >= 1, V(2,:) = U; end
    for k = 2:d
        V(k+1,:) = 2 .* U .* V(k,:) - V(k-1,:);
    end
end

% ---------- Generic rollout ----------
function Z_hat = rollout(M, pred_fn, z0, V_seq, U_seq, nz, T)
    Z_hat = zeros(nz, T+1);  Z_hat(:,1) = z0;
    for t = 1:T
        Z_hat(:, t+1) = pred_fn(M, Z_hat(:,t), V_seq(:,t), U_seq(t));
    end
end

% ---------- Fitters and predictors (4 Koopman forms) ----------

function M = fit_linear(Z, ~, U, Zp, gamma, nz, ~, ~)
    Phi   = [Z; U];
    Theta = (Zp * Phi') / (Phi * Phi' + gamma * eye(nz + 1));
    M.A = Theta(:, 1:nz);  M.B = Theta(:, nz+1);  M.kind = 'linear';
end
function z_next = predict_linear(M, z, ~, u)
    z_next = M.A * z + M.B * u;
end

function M = fit_bilinear(Z, ~, U, Zp, gamma, nz, ~, ~)
    Phi   = [Z; U; bsxfun(@times, Z, U)];
    d     = 2*nz + 1;
    Theta = (Zp * Phi') / (Phi * Phi' + gamma * eye(d));
    M.A = Theta(:, 1:nz);  M.B = Theta(:, nz+1);  M.N = Theta(:, nz+2:end);
    M.kind = 'bilinear';
end
function z_next = predict_bilinear(M, z, ~, u)
    z_next = M.A * z + M.B * u + M.N * (z * u);
end

function M = fit_geko(Z, V, ~, Zp, gamma, nz, nv, ~)
    % GeKo ridge fit (paper notation): K = W Phi' (Phi Phi' + gamma I)^-1,
    % with feature matrix Phi = [z (x) v] and target matrix W = [z^+].
    nznv = nz * nv;
    Phi  = zeros(nznv, size(Z, 2));
    for p = 1:size(Z, 2)
        Phi(:, p) = kron(Z(:, p), V(:, p));
    end
    M.K = (Zp * Phi') / (Phi * Phi' + gamma * eye(nznv));
    M.kind = 'geko';
end
function z_next = predict_geko(M, z, v, ~)
    z_next = M.K * kron(z, v);
end

function M = fit_kcf(Z, V, ~, Zp, gamma, nz, nv, couple_idx)
    % KCF (Haseli & Cortes 2025), input-state separable form:
    %   z+ = A(u) z,  with  A(u) = A11 + A12*Gtilde(u).  Gtilde couples the
    %   input to the lift rows in couple_idx only. We fit [A11 A12] by
    %   regressing z+ on the augmented features [ z ; Gtilde(u)*z ] with a
    %   uniform ridge gamma. The top block keeps the full lift, so the rank
    %   condition holds for any subset.
    if max(couple_idx) > nz
        error('kcf_couple_idx max (%d) exceeds lift dim nz=%d.', max(couple_idx), nz);
    end
    v_var = var(V(1, :));
    if v_var < 1e-10
        v_const = true;  V_eff = V(2:end, :);
    else
        v_const = false; V_eff = V;
    end
    nv_eff = size(V_eff, 1);
    nc     = numel(couple_idx);
    Zc     = Z(couple_idx, :);              % coupled state rows only
    nznv   = nz + nv_eff * nc;
    Phi    = zeros(nznv, size(Z, 2));
    for p = 1:size(Z, 2)
        Phi(:, p) = [Z(:, p); kron(V_eff(:, p), Zc(:, p))];
    end
    Theta = (Zp * Phi') / (Phi * Phi' + gamma * eye(nznv));
    M.A11 = Theta(:, 1:nz);                 % nz x nz   (autonomous)
    M.A12 = Theta(:, nz+1:end);             % nz x (nv_eff*nc)  (input coupling)
    M.v_const_dropped = v_const;
    M.nv_eff          = nv_eff;
    M.couple_idx      = couple_idx(:)';     % stored for predict + MPC
    M.nc              = nc;
    M.kind = 'kcf';
end
function z_next = predict_kcf(M, z, v, ~)
    if M.v_const_dropped
        v_eff = v(2:end);
    else
        v_eff = v;
    end
    if isfield(M, 'couple_idx'), ci = M.couple_idx; else, ci = 1:numel(z); end
    z_next = M.A11 * z + M.A12 * kron(v_eff, z(ci));   % couple to subset only
end
