% Implements the FIST algorithm.
% 
%     Parameters
%     ----------
%     T : tl.tensor
%          input tensor (dim: np-n1-n2-...)
%     W_g : tl.tensor
%          PPI network in adjacency matrix form (dim: npxnp)
%     l : float, default 0.1
%     rank_k : int, default None
%          rank of the tensor in CPD form. if None, will guess based on PCA
%          value of hyperparameter lambda
%     stop_crit : float, default 1e-3
%          stopping criteria for FIST
%     max_iters : int, default 500
%          maximum number of iterations to run
%     val_indices : tl.tensor, default None
%          indices of validation data, if None not used (dim: numindices-ndim)
%     val_values : tl.tensor, default None
%          values corresponding to validation indices (dim: nimdices)
% 
%     Returns
%     -------
%     A : list of tl.tensors
%         list of the CPD factor matrices learned by FIST

input_tensor = './FIST_output/Visium/HBA1/tensorPPI.mat';
metadata_path = './FIST_output/Visium/HBA1/metadata.json';
output_dir = './FIST_output/Visium/HBA1_matlab/';

verbose = 1;
lambda = 0.1;
rank = 200;
stopcrit = 1e-4;
MaxIters = 100;

load(input_tensor); %loads T, W_g
M = zeros(size(T));
M(T>0) = 1;


val_indices = VI+1;
val_values = VV;

%open and read metadata
file = fopen(metadata_path); 
text = fread(file,inf); 
str = char(text'); 
fclose(file); 
metadata = jsondecode(str);

cd("./FIST_utils");

T = tensor(T);
M = tensor(M);
    
n = size(T); % tensor dimensions
net_num = length(n); 

%construct spatial chain graphs
W = cell(net_num, 1);
W{1} = W_g;

for netid = 2:net_num
    W{netid} = diag(ones(n(netid)-1,1),-1) + diag(ones(n(netid)-1,1),1);
end

% graph normalization
D=cell(net_num,1);
for netid = 1:net_num
    W{netid} = W{netid} - diag(diag(W{netid}));
    d = sum(W{netid},2);
    d(d~=0) = (d(d~=0)).^-(0.5);
    W{netid} = W{netid}.*d;
    W{netid}=d'.*W{netid};
    D{netid}=diag(sum(W{netid},2));
end

% initialization
A=cell(net_num,1); % tensor in CPD form
phi = A; % A^T dot A
WA = A; % W dot A
DA = A; % D dot A
theta_W = A; % A^T dot W dot A
theta_D = A; % A^T dot D dot A
for netid=1:net_num
    rng(0);
    A{netid} = rand(n(netid),rank);  
    phi{netid} = A{netid}'*A{netid};
    WA{netid} = W{netid}*A{netid};
    DA{netid} = diag(D{netid}).*A{netid};
    theta_W{netid} = A{netid}'*WA{netid};
    theta_D{netid} = A{netid}'*DA{netid};
end

val_mae = zeros(MaxIters, 1);
val_mape = zeros(MaxIters, 1);
val_r2 = zeros(MaxIters, 1);

% training
for iter=1:MaxIters
    A_old = A;
    for i=1:net_num
        num = mttkrp(T, A, i);
        Y_hat = ktensor(A).*M;
        denom = mttkrp(Y_hat, A, i);

        % calculate dJ2dAi-
        to_multiply = zeros(size(phi{1}));
        for j=1:net_num
            if j ~= i
                to_sum = theta_W{j};
                for k=1:net_num
                    if k~=i && k~=j
                        to_sum = to_sum.*phi{k};
                    end
                end
                to_multiply = to_multiply+to_sum;
            end
        end
        num_kronsum = A{i}*to_multiply;

        hadamard_product = ones(size(phi{1}));
        for j=1:net_num
            if i~=j
                hadamard_product = hadamard_product.*phi{j};
            end
        end
        num_kronsum = num_kronsum + (WA{i}*hadamard_product);

        % calculate dJ2dAi+, very similar to above
        to_multiply = zeros(size(phi{1}));
        for j=1:net_num
            if j ~= i
                to_sum = theta_D{j};
                for k=1:net_num
                    if k~=i && k~=j
                        to_sum = to_sum.*phi{k};
                    end
                end
                to_multiply = to_multiply+to_sum;
            end
        end
        denom_kronsum = A{i}*to_multiply;

        denom_kronsum = denom_kronsum + (DA{i}*hadamard_product);

        % add together the two parts of the derivative and update Ai
        num=num+lambda*num_kronsum+10^-10;
        denom=denom+lambda*denom_kronsum+10^-10;
        A{i}=A{i}.*(num./denom);

        % update variables for the next go-around
        phi{i}=A{i}'*A{i};
        WA{i}=W{i}*A{i};
        DA{i}=diag(D{i}).*A{i};
        theta_W{i}=A{i}'*WA{i};
        theta_D{i}=A{i}'*DA{i};
    end

    res = compute_res(A,A_old);    
    [mae, mape, r2] = compute_metrics(A, val_indices, val_values);
    val_mae(iter) = mae;
    val_mape(iter) = mape;
    val_r2(iter) = r2;
    if verbose
        disp(['FIST Iter: ',num2str(iter), ' Res: ', num2str(res), ' MAE: ', num2str(mae),' MAPE: ',num2str(mape),' R^2: ',num2str(r2)]);
    end
    if res<stopcrit
        break;
    end
end

T_imputed = double(tensor(ktensor(A)));
cd("..");
save([output_dir, 'imputedtensor.mat'], 'T_imputed');

metadata.val_mae = val_mae;
metadata.val_mape = val_mape;
metadata.val_r2 = val_r2;

metadata_str = jsonencode(metadata);
%write metadata
file = fopen(metadata_path, 'wt'); 
fprintf(file,metadata_str); 
fclose(file); 

function res = compute_res(Q,Qold)
% compute residual
res_num = 0;
res_denom = 0;
for i = 1:length(Q)
    res_num = res_num + sum(sum((Q{i}-Qold{i}).^2));
    res_denom = res_denom + sum(sum(Qold{i}.^2));
end
res=sqrt(res_num/res_denom);
end


function [mae, mape, r2] = compute_metrics(A, val_indices, val_values)
    T = double(tensor(ktensor(A)));
    val_indices_args = num2cell(val_indices, 1);
    indices = sub2ind(size(T), val_indices_args{:});
    predicted = double(T(indices)).';
    errors = abs(val_values-predicted);
    good_vals = find(val_values>0);
    mae = mean(errors(good_vals));
    mape = mean(errors(good_vals)/val_values(good_vals));
    se_line = sum((predicted - val_values).^2);
    se_y = sum((val_values - mean(val_values)).^2);
    r2 = 1-se_line/se_y;
end
