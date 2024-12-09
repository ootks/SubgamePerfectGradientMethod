include("fom_interface.jl")
import Base.+
import Base.*
using LinearAlgebra

# Quadratic Function
struct quadratic <: Oracle
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Float64
end

function (q::quadratic)(x::Vector{Float64})
    return dot(x, q.A, x)/2 + dot(q.b, x) + q.c
end

function gradient(q::quadratic, x::Vector{Float64})
    return q.A*x + q.b
end

function smoothness(q::quadratic)
    return svdvals(q.A)[1]
end

# Logistic Regression
struct logisticRegression <: Oracle
    y::Vector{Float64}
end

function (q::logisticRegression)(x::Vector{Float64})
    ret = 0
    for i in 1:length(q.y)
        ret += max(0,q.y[i]*x[i]) + log(exp(-1*max(0,q.y[i]*x[i]))+exp(q.y[i]*x[i] - max(0,q.y[i]*x[i])))
    end
    return ret / length(q.y)
end

function gradient(q::logisticRegression, x::Vector{Float64})
    ret = zeros(length(x))
    for i in 1:length(q.y)
        ret[i] += q.y[i] / (1 + exp(-1*q.y[i] * x[i]))
    end
    return ret/length(q.y)
end

function smoothness(q::logisticRegression)
    return 0.25
end

# LogSumExp function
struct logsumexp <: Oracle
    n::Int64 
    rho::Float64
end

function (q::logsumexp)(x::Vector{Float64})
    max_x = maximum(x)
    return max_x + q.rho*log(sum(exp.((x .- max_x)/q.rho)))
end

function gradient(q::logsumexp, x::Vector{Float64})
    max_x = maximum(x)
    exp_shifted = exp.( (x .- max_x)/q.rho )  # Compute e^(x_i - max_x) rescaled by rho
    sum_exp_shifted = sum(exp_shifted)
    return exp_shifted ./ sum_exp_shifted  # Softmax
end

function smoothness(q::logsumexp)
    return 1/q.rho
end


#Moreau of Finite Maximum
struct moreauOfMax <: Oracle
    n::Int64 
    rho::Float64
end

function proxStepOnMax(q::moreauOfMax, x::Vector{Float64})
    sorted_x = sort(x, rev=true)
    t = 1
    sum_t = sorted_x[1]
    c = (sum_t - q.rho)/t

    for i in 1:q.n
        # Check if c is between sorted_x[i] and sorted_x[i+1]
        if i == q.n || c >= sorted_x[i + 1]
            break
        end
        # Accumulate (x_i - c)
        sum_t += sorted_x[i+1]
        t = i+1
        c = (sum_t - q.rho)/t
    end
    return [min(x[i], c) for i in 1:q.n]
end

function (q::moreauOfMax)(x::Vector{Float64})
    ppm_x = proxStepOnMax(q, x)
    return maximum(ppm_x) + (0.5/q.rho)*norm(x-ppm_x,2)^2
end

function gradient(q::moreauOfMax, x::Vector{Float64})
    ppm_x = proxStepOnMax(q, x)
    return (x-ppm_x)/q.rho
end

function smoothness(q::moreauOfMax)
    return 1/q.rho
end


# Linear Function
struct linear <: Oracle
    b::Vector{Float64}
end

function (q::linear)(x::Vector{Float64})
    return dot(q.b, x)
end

function gradient(q::linear, x::Vector{Float64})
    return q.b
end

function smoothness(q::linear)
    return 0
end


# Huber of L2 Norm of x
struct huberL2 <: Oracle
    L::Float64
    r::Float64
end

function (q::huberL2)(x::Vector{Float64})
    n = norm(x,2)
    if n <= q.r
        return q.L*n^2/2
    else
        return q.L*q.r*n - q.L*q.r^2/2
    end
end

function gradient(q::huberL2, x::Vector{Float64})
    n = norm(x,2)
    if n< q.r
        return q.L*x
    else
        return q.L*q.r*x/n
    end
end

function smoothness(q::huberL2)
    return q.L
end


# Huber of Components Summed Up
struct huberL1 <: Oracle
    L::Float64
    r::Float64
end

function (q::huberL1)(x::Vector{Float64})
    return sum([abs(xi) <= q.r ? q.L*xi^2/2 : q.L*q.r*abs(xi) - q.L*q.r^2/2 for xi in x])
end

function gradient(q::huberL1, x::Vector{Float64})
    return     [abs(xi) <= q.r ? q.L*xi : q.L*q.r*sign(xi) for xi in x]
end

function smoothness(q::huberL1)
    return q.L
end

