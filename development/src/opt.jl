using Gurobi, JuMP

function direct_solve(data::Matrix{Float64}, K::Int64)
    """
    Solve the k-means problem using a direct MIO formulation.

        input:
            - data: a matrix of size (N,D) containing the points in the dataset
            - K: number of clusters
        output:
            - assignments: a matrix of size (N,K) containing the cluster assignment of each point
    """
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

function new_cluster(data::Matrix{Float64}, p::Vector{Float64}, q::Float64, K::Int64, Iter::Int64)
    """
    Subproblem of the column generation, it returns a new cluster and its reduced cost

        input: 
            - data: Nx2 matrix of points
            - p: dual solution of the master problem
            - q: dual solution of the master problem
            - K: number of clusters
            - Iter: iteration of the column generation, this is used in the heuristics to avoid the convex hull constraint of the subproblem

        output:
            - Cluster: new cluster
            - reduced cost of the cluster
    """

    N = size(data,1)
    D = size(data,2)

    ###Subproblem
    sp_model = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attributes( sp_model, "OutputFlag" => 0)

    @variable(sp_model, θ[1:N])
    @variable(sp_model, a[1:N], Bin)
    @variable(sp_model, f[1:N, 1:N])
    @variable(sp_model, b[1:N-K+1], Bin)
    @variable(sp_model, γ[1:N,1:N,1:N-K+1])
    @variable(sp_model, z[1:N,1:N-K+1])

    @constraint(sp_model, [i=1:N, j=1:N], f[i,j] <= a[i])
    @constraint(sp_model, [i=1:N, j=1:N], f[i,j] <= a[j])
    @constraint(sp_model, [i=1:N, j=1:N], f[i,j] >= a[i] + a[j] - 1)
    @constraint(sp_model, sum(a[i] for i=1:N) >= 1)

    @constraint(sp_model, [i=1:N, j=1:N, l=1:N-K+1], γ[i,j,l] <= f[i,j])
    @constraint(sp_model, [i=1:N, j=1:N, l=1:N-K+1], γ[i,j,l] <= b[l])
    @constraint(sp_model, [i=1:N, j=1:N, l=1:N-K+1], γ[i,j,l] >= f[i,j] + b[l] - 1)

    @constraint(sp_model, sum(b[l] for l=1:N-K+1) == 1)

    @constraint(sp_model, [i=1:N], [θ[i]; sum( 1/l .* sum( γ[i,j,l] .* (data[i,:] .- data[j,:]) for j=1:N) for l=1:N-K+1)] in SecondOrderCone()) #
    
    ### heuristics constraint
    @constraint(sp_model, [i=1:N, l=1:N-K+1], z[i,l] >= b[l] + a[i] -1)
    @constraint(sp_model, [i=1:N, l=1:N-K+1], z[i,l] <= b[l])
    @constraint(sp_model, [i=1:N, l=1:N-K+1], z[i,l] <= a[i])
    @constraint(sp_model, [i=1:N], [5/(Iter/4); (data[i,:] .- sum(1/l .* sum( data[j,:] * z[j,l] for j in 1:N)  for l in 1:N-K+1))] in SecondOrderCone())

    @objective(sp_model, Min, sum( θ[i] - a[i]*p[i] for i in 1:N))

    optimize!(sp_model)

    #### Retrieve cluster from solution
    assignments = vec(findall(x->x>0.9, value.(a)))
    cost, centroid = compute_cluster_centroid_cost(data, assignments)
    Clust = Cluster(assignments, centroid, cost)
    reduced_cost = compute_reduced_cost(Clust, p, q)

    return Clust, reduced_cost

end 

function column_generation(data::Matrix{Float64}, K::Int64, n_initial_clusters::Int64, max_iterations::Int = 10000, verbose::Bool = true)

    """
    Column generation algorithm for the k-means problem

        input: 
            - data: Nx2 matrix of points
            - K: number of clusters
            - n_initial_clusters: number of initial clusters
            - max_iterations: maximum number of iterations
            - verbose: if true prints the progress of the algorithm

        output:
            - dictionary with the following keys:
                - CGIP_solution
                - CGIP_objective 
                - CGLP_solution
                - CGLP_objective
                - clusters
                - p_values
                - q_values
                - upper_bounds
                - lower_bounds
                - MP_time
                - SP_time
                - time_taken
    """

    N = size(data,1)
    
    # Initialization
    CG_solution = nothing
    CG_objective = 0.0
    upper_bounds = []
    lower_bounds = []
    MP_time = []
    SP_time = []
    p_values = []
    q_values = []

    if verbose
        @printf("              | Objective | Lower bound | MP time (s) | SP time (s) | Reduced cost\n")
    end

    # Creates initial clusters
    clusters = initial_clusters(data, K, n_initial_clusters, 1)
    counter = 0

    while true
        counter += 1
        # Make restricted master problem with current clusters 
        MP_start_time = time()
        model = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attributes(
            model,
            "OutputFlag" => 0,
        )

        @variable(model, x[r in 1:length(clusters)] ≥ 0)

        @constraint(model, points_assigned[r=1:N], sum(x[r] for r in 1:length(clusters) if r in clusters[r].assignments) == 1)
        @constraint(model, number_clusters, sum(x[r] for r in 1:length(clusters)) == K)

        @objective(model, Min, sum(x[r] * clusters[r].cost for r in 1:length(clusters)))

        optimize!(model)
        MP_end_time = time()

        ### Get duals
        p = vec(dual.(points_assigned))
        push!(p_values, p)

        q = dual.(number_clusters)
        push!(q_values, q)

        # Update column generation metrics
        push!(upper_bounds, objective_value(model))
        push!(MP_time, MP_end_time - MP_start_time)

        # Compute a new route to add and its corresponding reduced cost
        SP_start_time = time()
        #clust, reduced_cost = new_cluster(data, p, q, K, counter)
        clust, reduced_cost = subproblem_heuristic(data, p, q, counter, K)
        SP_end_time = time()

        # Update column generation metrics
        push!(lower_bounds, objective_value(model) + min(N * reduced_cost,0))
        push!(SP_time, SP_end_time - SP_start_time)

        if verbose
            @printf(
                "Iteration %3d | %9.3f | %11.3f |      %6.3f |      %6.3f | %12.3f\n",
                counter, 
                upper_bounds[end],
                lower_bounds[end],
                MP_time[end],
                SP_time[end],
                reduced_cost,
            )
        end

        # Termination criteria
        if (reduced_cost > -1e-6 || counter > max_iterations)  
            CG_solution = value.(x)
            CG_objective = objective_value(model)
            break
        end
        push!(clusters, clust)

    end

    #### solve MP with integer variables
    # Solve model again with the same variables, enforcing integrality

    model_bin = JuMP.Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attributes( model_bin, "OutputFlag" => 0)

    @variable(model_bin, y[r in 1:length(clusters)], Bin)

    @constraint(model_bin, points_assigned[r=1:N], sum(y[r] for r in 1:length(clusters) if r in clusters[r].assignments) == 1)
    @constraint(model_bin, number_clusters, sum(y[r] for r in 1:length(clusters)) == K)

    @objective(model_bin, Min, sum(y[r] * clusters[r].cost for r in 1:length(clusters)))

    optimize!(model_bin)

    return Dict(
        "CGIP_solution" => value.(y),
        "CGIP_objective" => objective_value(model_bin),
        "CGLP_solution" => CG_solution,
        "CGLP_objective" => CG_objective,
        "clusters" => clusters,
        "p_values" => p_values,
        "q_values" => q_values,
        "upper_bounds" => upper_bounds,
        "lower_bounds" => lower_bounds,
        "MP_time" => MP_time,
        "SP_time" => SP_time,
        "time_taken" => sum(MP_time) + sum(SP_time),
    )
end