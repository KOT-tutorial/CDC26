function x=generate_n_d_grid(range,N)

x = linspace(range(1,1),range(1,2),N(1));
for i=2:length(N)
    x0 = linspace(range(i,1),range(i,2),N(i));
    x0 = reshape(x0.*ones(size(x,2),1),1,N(i)*size(x,2));
    x = [repmat(x,1,N(i));x0];
end


end