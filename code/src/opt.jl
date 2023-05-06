using Gurobi, JuMP

function gamma_formulation(data, K)
    N = size(data, 1)
    D = size(data, 2)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, a[1:N, 1:K], Bin)
    @variable(model, f[1:N, 1:N, 1:K] >= 0)
    @variable(model, b[N-K+1, 1:K], Bin)
    @variable(model, γ[1:N, 1:N, 1:K, 1:N-K+1] >= 0)

    @objective(model, Min, sum(sum(sum(norm(sum((1/l) * (sum(γ[i,j,k,l]*(x[i,d] - x[j,d]) for j=1:N)) for l=1:N-K+1))^2 for k=1:K) for d=1:D) for i=1:N))

    @constraint(model, [i=1:N], sum(a[i,k] for k=1:K) == 1)
    @constraint(model, [k=1:K], sum(a[i,k] for i=1:N) >= 1)
    @constraint(model, [i=1:N, j=1:N, k=1:K], f[i,j,k] <= a[i,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K], f[i,j,k] <= a[j,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K], f[i,j,k] >= a[i,k] + a[j,k] - 1)
    @constraint(model, [i=1:N, j=1:N, k=1:K, l=1:N-K+1], γ[i,j,k,l] <= f[i,j,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K, l=1:N-K+1], γ[i,j,k,l] <= b[l,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K, l=1:N-K+1], γ[i,j,k,l] >= f[i,j,k] + b[l,k] - 1)

    optimize!(model)

    return value.(a)
end