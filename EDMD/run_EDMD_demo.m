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
N_train = [5,5];
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
num_feat = 100; %for rbf and fourier
degree = 4; %for poly
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


%% evaluate predictor

downsample = 4;

Y_pred = C*A*psi(X_test);
err = sum((Y_pred-Y_test).^2,1);

Xmat1 = reshape(X_test(1,:),N_test(1),N_test(2));
Xmat2 = reshape(X_test(2,:),N_test(1),N_test(2));
errmat = reshape(err,N_test(1),N_test(2)); 

D = (Y_pred-X_test)./ sqrt(sum((Y_pred-X_test).^2,1));
Dmat1 = reshape(D(1,:),N_test(1),N_test(2));
Dmat2 = reshape(D(2,:),N_test(1),N_test(2));

Xmat1_down = Xmat1(1:downsample:end, 1:downsample:end);
Xmat2_down = Xmat2(1:downsample:end, 1:downsample:end);
Dmat1_down = Dmat1(1:downsample:end, 1:downsample:end);
Dmat2_down = Dmat2(1:downsample:end, 1:downsample:end);

figure();
surf(Xmat1,Xmat2,errmat);

figure();
quiver(Xmat1_down,Xmat2_down, Dmat1_down,Dmat2_down);

disp(['mean error: ', num2str(mean(err))]);
disp(['max error: ', num2str(max(err))]);

%% propagation and projection error

e_miss = sum((psi(Y_test) - A*psi(X_test)).^2,1);
e_proj = sum((X_test - C*psi(X_test)).^2,1);

e_miss_mat = reshape(e_miss,N_test(1),N_test(2)); 
e_proj_mat = reshape(e_proj,N_test(1),N_test(2));

figure();
surf(Xmat1,Xmat2,e_miss_mat);

figure();
surf(Xmat1,Xmat2,e_proj_mat);

disp(['mean error: ', num2str(mean(e_miss))]);
disp(['max error: ', num2str(max(e_miss))]);

disp(['mean error: ', num2str(mean(e_proj))]);
disp(['max error: ', num2str(max(e_proj))]);

%% long term prediction
hor_pred = 50;
reproj_after = 4;

% X0 = [-1.75, -1.75, -1.75, 0, 0, 1.75, 1.75, 1.75; -1.75, 0, 1.75, -1.75, 1.75, -1.75, 0, 1.75];
X0 = [-1.25,-1.25,1.25,1.25; -1.25,1.25,-1.25,1.25];
Ytraj1 = zeros(hor_pred+1,size(X0,2));
Ytraj2 = zeros(hor_pred+1,size(X0,2));

psi_aux = psi(X0);
Ytraj1(1,:) = X0(1,:);
Ytraj2(1,:) = X0(2,:);

Ytrajtrue1(1,:) = X0(1,:);
Ytrajtrue2(1,:) = X0(2,:);

for i=1:hor_pred
    psi_aux = A*psi_aux;
    aux = C*psi_aux;
    Ytraj1(i+1,:) = psi_aux(1,:);
    Ytraj2(i+1,:) = psi_aux(2,:);
    if(mod(i,reproj_after)==0)
        psi_aux = psi([Ytraj1(i+1,:);Ytraj2(i+1,:)]);
    end
    for j=1:size(X0,2)
        [~,y] = ode23(dyn,[0,dt],[Ytrajtrue1(i,j);Ytrajtrue2(i,j)]);
        Ytrajtrue1(i+1,j) = y(end,1);
        Ytrajtrue2(i+1,j) = y(end,2);
    end
end

figure();
hold on;
for i=1:size(Ytraj1,2)
    plot(Ytraj1(:,i),Ytraj2(:,i));
    plot(Ytrajtrue1(:,i),Ytrajtrue2(:,i),'--');
end