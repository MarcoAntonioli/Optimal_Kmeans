using Gurobi, JuMP

function gamma_formulation(data::Matrix{Float64}, K::Int64)
    # Get the number of data points and dimensions in the data
    N = size(data, 1)
    #D = size(data, 2)

    model = JuMP.Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "TimeLimit", 300)

    #--------------Decision Variables----------------#

    @variable(model, θ[1:N,1:K] >= 0) # Auxiliary variable used to move the objective function
    @variable(model, a[1:N, 1:K], Bin) # Binary indicator variable for assigning data points to clusters
    @variable(model, f[1:N, 1:N, 1:K] >= 0) # Auxiliary variable used to linearize
    @variable(model, b[1:N-K+1, 1:K], Bin) # Auxiliary variable used to linearize
    @variable(model, γ[1:N, 1:N, 1:K, 1:N-K+1] >= 0) # Auxiliary variable used to linearize

    #--------------Objective Function----------------#

    # Objective function to minimize the sum of the squared Euclidean distances between data points and their assigned cluster centers
    @objective(model, Min, sum(sum(θ[i,k] for i=1:N) for k=1:K))
    #@objective(model, Min, sum(sum(norm(sum((1/l) * (sum(γ[i,j,k,l]*(data[i,:] - data[j,:]) for j=1:N)) for l=1:N-K+1))^2 for k=1:K) for i=1:N))

    #----------------Constraints---------------------#

    # Constraint to move the objective function
    @constraint(model, [i=1:N, k=1:K], [θ[i,k]; sum((1/l) * (sum(γ[i,j,k,l] * (data[i,:] - data[j,:]) for j=1:N)) for l=1:N-K+1)] in SecondOrderCone())
    
    # Assignment constraints
    @constraint(model, [i=1:N], sum(a[i,k] for k=1:K) == 1) # Each data point is assigned to exactly one cluster
    @constraint(model, [k=1:K], sum(a[i,k] for i=1:N) >= 1) # Each cluster has at least one data point assigned to it
    
    # f[i,j,k] is equal to the product a[i,k]*a[j,k]
    @constraint(model, [i=1:N, j=1:N, k=1:K], f[i,j,k] <= a[i,k]) 
    @constraint(model, [i=1:N, j=1:N, k=1:K], f[i,j,k] <= a[j,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K], f[i,j,k] >= a[i,k] + a[j,k] - 1)

    # γ[i,j,k,l] is equal to the product b[l,k] * f[i,j,k]
    @constraint(model, [i=1:N, j=1:N, k=1:K, l=1:N-K+1], γ[i,j,k,l] <= f[i,j,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K, l=1:N-K+1], γ[i,j,k,l] <= b[l,k])
    @constraint(model, [i=1:N, j=1:N, k=1:K, l=1:N-K+1], γ[i,j,k,l] >= f[i,j,k] + b[l,k] - 1)

    @constraint(model, [k=1:K], sum(b[l,k] for l=1:N-K+1) == 1) # Each cluster has exactly one b[l,k] equal to 1

    optimize!(model)
    return value.(a), solution_summary(model)#, value.(θ), value.(f), value.(b), value.(γ)
end