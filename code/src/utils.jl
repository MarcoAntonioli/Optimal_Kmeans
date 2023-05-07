using CSV, Tables, LinearAlgebra, Random, Gurobi, JuMP, DataFrames, Statistics, MLJ, Plots, Clustering, Distances

function generate_points(N, K, D, std, seed = 42)
    """
    Generate a random dataset of N points with K clusters of dimensionality D.

    N: number of points in the dataset
    K: number of clusters
    D: dimensionality of the points
    std: standard deviation of each cluster
    seed: random seed (optional)
    """
    Random.seed!(seed)
    X, yy = make_blobs(N, D; centers=K, cluster_std=std)
    
    points = Matrix(DataFrame(X))
    min = minimum(points, dims=1)
    max = maximum(points, dims=1)
    points = (points .- min) ./ (max .- min)
    
    return points
end

function mean_silhouette_score(assignment, counts, points)
    """
    Compute the mean silhouette score of a clustering assignment.

    assignment: a vector of length N containing the cluster assignment of each point
    counts: a vector of length K containing the number of points in each cluster
    points: a matrix of size (D,N) containing the points in the dataset
    """
    distances = pairwise(Euclidean(), points')
    
    return mean(silhouettes(assignment, counts, distances))
end

function euclidean_distance_1(points)
    """
    Compute the pairwise Euclidean distance between all points in the dataset.

    points: a matrix of size (N,D) containing the points in the dataset
    """
    N = size(points,1)
    distances = ones((N,N))
    
    for i in 1:N
        for j in 1:N
            distances[i,j] = sqrt(sum((points[i,:] - points[j,:]).^2))
        end
    end
    
    return distances
end

function euclidean_distance_2(points)
    """
    Compute the pairwise Euclidean distance between all points in the dataset.

    points: a matrix of size (N,D) containing the points in the dataset
    """
    N = size(points, 1)
    distances = zeros(N,N)

    # Compute upper triangle of distance matrix
    for i in 1:N-1
        for j in i+1:N
            distances[i,j] = sqrt(sum((points[i,:] .- points[j,:]).^2))
        end
    end

    # Copy upper triangle to lower triangle
    distances = distances + distances'

    return distances
end

function euclidean_distance_3(points)
    """
    Compute the pairwise Euclidean distance between all points in the dataset.

    points: a matrix of size (N,D) containing the points in the dataset
    """
    N = size(points, 1)
    distances = sqrt.(sum(abs2.(points'), dims=1)' .+ sum(abs2.(points'), dims=1) .- 2 .* (points * points'))

    return distances
end

function manhattan_distance_1(points)
    """
    Compute the pairwise Manhattan distance between all points in the dataset.

    points: a matrix of size (N,D) containing the points in the dataset
    """
    N = size(points, 1)
    distances = ones((N,N))

    for i in 1:N
        for j in 1:N
            distances[i,j] = sum(abs.(points[i,:] - points[j,:]))
        end
    
    return distances
end

function manhattan_distance_2(points)
    """
    Compute the pairwise Manhattan distance between all points in the dataset.

    points: a matrix of size (N,D) containing the points in the dataset
    """
    N = size(points, 1)
    distances = zeros(N,N)

    # Compute upper triangle of distance matrix
    for i in 1:N-1
        for j in i+1:N
            distances[i,j] = sum(abs.(points[i,:] .- points[j,:]))
        end
    end

    # Copy upper triangle to lower triangle
    distances = distances + distances'

    return distances
end

function manhattan_distance_3(points)
    """
    Compute the pairwise Manhattan distance between all points in the dataset.

    points: a matrix of size (N,D) containing the points in the dataset
    """
    N = size(points, 1)
    X = reshape(repeat(points, outer = (N, 1)), N, N, :)
    Y = reshape(repeat(points', inner = (N, 1)), N, N, :)
    return sum(abs.(X .- Y), dims = 3)
end

function get_centroids(assignments, data)
    """
    Compute the centroids of the clusters given a clustering assignment and the data.

    assignments: a matrix of size (N,K) containing the cluster assignment of each point
    data: a matrix of size (N,D) containing the points in the dataset
    """
    K = size(assignments, 2)
    N, D = size(data)

    centroids = zeros(K, D)
    for k=1:K
        centroids[k,:] = sum(assignments[i,k] * data[i,:] for i=1:N) / sum(assignments[i,k] for i=1:N)
    end
    return centroids
end