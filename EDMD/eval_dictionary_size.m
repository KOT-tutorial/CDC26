clc; clear all; close all;

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
N_train = [15,15];
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

dictionary_type = "fourier"; %choices: poly, fourier, rbf
include_state = true; %state is automatically included for poly features
switch(dictionary_type)
    case("fourier")
        dict_range = 100;
    case("rbf")
        dict_range = 200;
    case("poly")
        dict_range = 19;
end

for k=2:dict_range

    dim = size(X_train,1);

    if(dictionary_type=="poly")
        psi = poly_dict(k,dim);

    elseif(dictionary_type=="fourier")
        omegas = randn([dim,k]);
        if(include_state)
            psi = @(x) [x; sin(omegas'*x); cos(omegas'*x)];
        else
            psi = @(x) [sin(omegas'*x); cos(omegas'*x)];
        end

    elseif(dictionary_type=="rbf")
        num_feat = min(k,size(X_train,2));
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

    num_dict(k) = size(psi_X,1);

    A = psi_Y*psi_X' * pinv(psi_X*psi_X');

    if(include_state)
        C = [eye(dim),zeros(dim, size(psi_Y,1)-dim)];
    else
        C = Y_train*psi_Y' * pinv(psi_Y*psi_Y');
    end


    %% evaluate predictor

    Y_pred = C*A*psi(X_test);
    err = sum((Y_pred-Y_test).^2,1);

    err_mean(k) = mean(err);
    err_max(k) = max(err);

    %% propagation and projection error

    eprop_test = sum((psi(Y_test) - A*psi(X_test)).^2,1);
    eproj_test = sum((X_test - C*psi(X_test)).^2,1);

    eprop_train = sum((psi(Y_train) - A*psi(X_train)).^2,1);
    eproj_train = sum((X_train - C*psi(X_train)).^2,1);

    eprop_test_mean(k) = mean(eprop_test);
    eprop_test_max(k) = max(eprop_test);

    eproj_test_mean(k) = mean(eproj_test);
    eproj_test_max(k) = max(eproj_test);

    eprop_train_mean(k) = mean(eprop_train);
    eprop_train_max(k) = max(eprop_train);

    eproj_train_mean(k) = mean(eproj_train);
    eproj_train_max(k) = max(eproj_train);
end

%% plot results

figure();
plot(num_dict,err_mean);
hold on;
plot(num_dict,eprop_test_mean);
plot(num_dict,eprop_train_mean);

figure();
plot(num_dict,err_max);
hold on;
plot(num_dict,eprop_test_max);
plot(num_dict,eprop_train_max);
