clc; clear all; close all;
rng(1);

%% Duffing oscillator dynamics
% parameters taken from https://arxiv.org/pdf/1408.4408

delta = 0.5;
beta = -1;
alpha = 1;
dt = 0.25;

dyn = @(t,x) [x(2); -delta*x(2)-x(1)*(beta+alpha*x(1)^2)];

%% generate grid data for testing

N_test = [101,101];
range_test = [-2,2; -2,2];

X_test = generate_n_d_grid(range_test,N_test);

for i=1:size(X_test,2)
    [~,y] = ode23(dyn,[0,dt],X_test(:,i));
    Y_test(:,i) = y(end,:);
    
end



%% generate training data 

data_type = "grid"; %choices: trajectory, grid
N_train = [100,100]; %can fail for smaller number of samples
range_train = [-2,2; -2,2];
horizon = 10; %only for trajectory data


if(data_type == "grid")
    X_train = generate_n_d_grid(range_train,N_train);
    Y_train = zeros(size(X_train));
    for i=1:size(X_train,2)
        [~,y] = ode23(dyn,[0,dt],X_train(:,i));
        Y_train(:,i) = y(end,:);
    end

elseif(data_type == "trajectory")
    X = generate_n_d_grid(range_train,N_train);
    Ts = linspace(0,horizon-1,horizon)*dt;
    X_train = zeros(length(N_train),(horizon-1)*prod(N_train));
    Y_train = zeros(length(N_train),(horizon-1)*prod(N_train));
    for i=1:size(X,2)
        [~,traj] = ode23(dyn,Ts,X(:,i));
        X_train(:,(i-1)*(horizon-1)+1:i*(horizon-1)) = traj(1:end-1,:)';
        Y_train(:,(i-1)*(horizon-1)+1:i*(horizon-1)) = traj(2:end,:)';
    end

else
    error('Unknown requested type of training data!')
end

%% define dictionary

dictionary_type = "rbf"; %choices: poly, fourier, rbf
num_feat = 600; %for rbf and fourier; min of 600 rbf or 300 fourier features necessary
degree = 7; %for poly; min degree of 7 necessary
include_state = true; %state is automatically included for poly features


dim = size(X_train,1);

if(dictionary_type=="poly")
    psi = poly_dict(degree,dim);

elseif(dictionary_type=="fourier")
    omegas = randn([dim,num_feat]);
    if(include_state)
        psi = @(x) [x; sin(omegas'*x); cos(omegas'*x)];
    else
        psi = @(x) [sin(omegas'*x); cos(omegas'*x)];
    end

elseif(dictionary_type=="rbf")
    num_feat = min(num_feat,size(X_train,2));
    [~,centers] = kmeans(X_train',num_feat);
    if(include_state)
        psi = @(x) [x; squeeze(exp(-sum((repmat(reshape(x,[1,size(x,1),size(x,2)]),num_feat,1,1)-centers).^2,2)/2))];
    else
        psi = @(x) squeeze(exp(-sum((repmat(reshape(x,[1,size(x,1),size(x,2)]),num_feat,1,1)-centers).^2,2)/2));
    end

else
    error('Unknown requested type of dictionary!')
end


%% EDMD predictor computation

psi_X = psi(X_train);
psi_Y = psi(Y_train);

A = psi_Y*psi_X' * pinv(psi_X*psi_X');

if(include_state)
    C = [eye(dim),zeros(dim, size(psi_Y,1)-dim)];
else
    C = Y_train*psi_Y' * pinv(psi_Y*psi_Y');
end

%% Determine eigenvalues and eigenfunctions

[eigvec, eigval] = eig(A);
eigval = diag(eigval);
Rinv = inv(eigvec);

zetahat = @(x) Rinv * psi(x);
vhat = eigvec*C';


%% find two eigenvalues with value ~1

idcs = find(abs(imag(eigval))<1e-3);
eig1 = real(eigval(idcs));
[~,idcs2] = sort(abs(eig1-1));
eig1 = eig1(idcs2(1:2));
idx1 = idcs(idcs2(1:2));


eigfun_vals = zetahat(X_test);
eigfun_vals = real(eigfun_vals(idx1,:));

%first eigenfunction with eigenvalue ~1 is giving approximately a constant function
figure();
surf(reshape(X_test(1,:),N_test(1),N_test(2)),reshape(X_test(2,:),N_test(1),N_test(2)),...
    reshape(eigfun_vals(1,:),N_test(1),N_test(2)));

%second eigenfunction with eigenvalue ~1 indicates the regions of attraction
figure()
surf(reshape(X_test(1,:),N_test(1),N_test(2)),reshape(X_test(2,:),N_test(1),N_test(2)),...
    reshape(eigfun_vals(2,:),N_test(1),N_test(2)));


%% Separate regions of attraction

roa1 = X_test(:,eigfun_vals(2,:)<0);
roa2 = X_test(:,eigfun_vals(2,:)>0);

figure();
hold on;
plot(roa1(1,:),roa1(2,:),'r*');
plot(roa2(1,:),roa2(2,:),'g*');


%% count non-trivial eigenvalues

num_sig = sum(abs(eigval)>1e-2)