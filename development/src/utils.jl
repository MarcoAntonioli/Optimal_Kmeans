using CSV, Tables, LinearAlgebra, Random, Gurobi, JuMP, DataFrames, Statistics, MLJ, Plots, Clustering, Distances

function generate_points(K::Int64, N::Int64, D::Int64 , std::Float64, seed = 1234)
    """
    Generates points for the k-means problem

        input: 
            - K: number of clusters
            - N: number of points
            - D: dimension of the points
            - std: standard deviation of the clusters
            - seed: seed for the random number generator

        output:
            - data: NxD matrix of standardized points
    """
    Random.seed!(seed);
    X, yy = MLJ.make_blobs(N, D; centers=K, cluster_std=std)
    data = Matrix(DataFrame(X));
    min = minimum(data, dims=1);
    max = maximum(data, dims=1);
    data = (data .- min) ./ (max .- min);

    return data
end

function euclidean_distance(data::Matrix{Float64})
    """
    Compute the pairwise Euclidean distance between all points in the dataset.

        input: 
            - data: a matrix of size (N,D) containing the points in the dataset
        
        output:
            - distances: a matrix of size (N,N) containing the pairwise Euclidean distance between all points
    """
    N = size(data, 1)
    distances = sqrt.(sum(abs2.(data'), dims=1)' .+ sum(abs2.(data'), dims=1) .- 2 .* (data * data'))

    return distances
end

function manhattan_distance(data::Matrix{Float64})
    """
    Compute the pairwise Manhattan distance between all points in the dataset.

        input:
            - data: a matrix of size (N,D) containing the points in the dataset
        
        output:
            - distances: a matrix of size (N,N) containing the pairwise Manhattan distance between all points
    """
    N = size(data, 1)
    X = reshape(repeat(data, outer = (N, 1)), N, N, :)
    Y = reshape(repeat(data', inner = (N, 1)), N, N, :)

    return sum(abs.(X .- Y), dims = 3)
end

function get_centroids(assignments::Matrix{Float64}, data::Matrix{Float64})
    """
    Compute the centroids of the clusters given a clustering assignment and the data.

        input: 
            - assignments: a matrix of size (N,K) containing the cluster assignment of each point
            - data: a matrix of size (N,D) containing the points in the dataset
        
        output:
            - centroids: a matrix of size (K,D) containing the centroids of the clusters
    """
    K = size(assignments, 2)
    N, D = size(data)

    centroids = zeros(K, D)
    for k=1:K
        centroids[k,:] = sum(assignments[i,k] * data[i,:] for i=1:N) / sum(assignments[i,k] for i=1:N)
    end

    return centroids
end

function compute_cluster_centroid_cost(data::Matrix{Float64}, assignments::Vector{Int64})
    """
    Computes the centroid and cost of a cluster given the indeces of the points in the cluster

        input: 
            - assignments: Nx1 matrix of cluster assignments

        output:
            - cost of the clustering assignment
            - Vector representing the centroid of the cluster
    """

    cluster_points = data[assignments,:]
    centroid = mean(cluster_points, dims=1)

    return sum( euclidean(cluster_points[i,:],centroid) for i in 1:size(cluster_points,1)), vec(centroid)
end

struct Cluster 
    """
    Cluster data structure
        methods: 
            - assignments: Nx1 matrix of cluster assignments
            - centroid: Vector representing the centroid of the cluster
            - cost: cost of the clustering assignment
    """
    assignments::Vector{Int64}
    centroid::Vector{Float64}
    cost::Float64
end

function compute_reduced_cost(cluster::Cluster, p::Vector{Float64}, q::Float64)
    """
    Computes the reduced cost given a dual solution of the master problem
    
        input: 
            - cluster: cluster to compute the reduced cost for
            - p: dual solution of the master problem
            - q: dual solution of the master problem
    
        output: reduced cost of the cluster
    """
    
    return cluster.cost - sum(p[i] for i in cluster.assignments) - q
end

function subproblem_heuristic(data::Matrix{Float64}, p::Vector{Float64}, q::Float64, Iter::Int64, K::Int64)
    """
    Computes the subproblem heuristic for the k-means problem

        input: 
            - data: a matrix of size (N,D) containing the points in the dataset
            - p: dual solution of the master problem
            - q: dual solution of the master problem
            - Iter: number of iterations to run the heuristic
            - K: number of clusters

        output:
            - cluster: a Cluster object containing the cluster assignment, centroid and cost
            - reduced_cost: reduced cost of the cluster
    """
    clusters = Cluster[]
    
    for i = 1:15
        append!(clusters, initial_clusters(data, K, 1, Iter * i ))
    end
    idx = argmin([compute_reduced_cost(cluster, p, q) for cluster in clusters])

    return clusters[idx], compute_reduced_cost(clusters[idx], p, q)
end 

function initial_clusters(data::Matrix{Float64}, K::Int64, n_initial_clusters::Int64, seed::Int64 = 1)
    """
    Computes the initial clusters for the k-means problem

        input: 
            - data: a matrix of size (N,D) containing the points in the dataset
            - K: number of clusters
            - n_initial_clusters: number of initial clusters to compute
            - seed: seed for the random number generator

        output:
            - clusters: a vector of size (n_initial_clusters*K) containing the initial clusters
    """

    clusters = Cluster[]

    for i in 1:n_initial_clusters
        Random.seed!(seed * i)
        km =  kmeans(data', K).assignments
        for k in 1:K
            assinments = findall(x->x==k, km); 
            cost, centroid = compute_cluster_centroid_cost(data, assinments)
            push!(clusters, Cluster(assinments, centroid, cost))
        end
    end
    return clusters
end