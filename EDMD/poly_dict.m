function psi = poly_dict(degree,dim)

exp = generate_n_d_grid(ones(dim,1)*[0,degree],...
    ones(dim,1)*(degree+1));
idelete = [];
ifirst = [];
notfirst = [];
for i=1:size(exp,2)
    if(sum(exp(:,i))>degree)
        idelete = [idelete,i];
    end
end
exp(:,idelete) = [];
for i=1:size(exp,2)
    if(sum(exp(:,i))==1)
        ifirst = [ifirst, i];
    else
        notfirst = [notfirst, i];
    end
end
exp = [exp(:,ifirst), exp(:,notfirst)];
psi = @(x)squeeze(prod(reshape(x,[size(x,1),1,size(x,2)]).^exp,1));

end