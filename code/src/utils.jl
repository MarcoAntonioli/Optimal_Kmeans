using CSV, Tables, LinearAlgebra, Random, Gurobi, JuMP, DataFrames, Statistics, MLJ, Plots, Clustering, Distances

"""
N is the number of points in the dataset
K is the number of clusters
D is the dimensionality of the points
"""

function generate_points(N, K, D, std, seed = 1234)
    Random.seed!(seed);
    X, yy = make_blobs(N, D; centers=K, cluster_std=std)
    points = Matrix(DataFrame(X));
    min = minimum(points, dims=1);
    max = maximum(points, dims=1);
    points = (points .- min) ./ (max .- min);
    return points
end


function mean_silhouette_score(assignment, counts, points)
    distances = pairwise(Euclidean(), points')
    return mean(silhouettes(assignment, counts, distances))
end

function euclidean_distance(points)
    n,m = size(points)
    distances = ones((n,n))
    for i in 1:n
        for j in 1:n
            distances[i,j] = sqrt(sum((points[i,:] - points[j,:]).^2))
        end
    end
    return distances
end

function manhattan_distance(points)
    n,m = size(points)
    distances = ones((n,n))
    for i in 1:n
        for j in 1:n
            distances[i,j] = sum(abs.(points[i,:] - points[j,:]))
        end
    end
    return distances
end


function get_centroids(assignments, data)
    centroids = zeros(K, D)
    for k=1:K
        centroids[k,:] = sum(assignments[i,k] * data[i,:] for i=1:N) / sum(assignments[i,k] for i=1:N)
    end
    return centroids
end