include("fom_interface.jl")
using LinearAlgebra
import Base.+
import Base.*

# Sum of two oracles
struct SumOracle <: Oracle
    f::Oracle
    g::Oracle
end

+(f::Oracle, g::Oracle) = SumOracle(f, g)

function (q::SumOracle)(x::Vector{Float64})
    return q.f(x) + q.g(x)
end

function gradient(q::SumOracle, x::Vector{Float64})
    return gradient(q.f, x) + gradient(q.g, x)
end

function smoothness(q::SumOracle)
    return smoothness(q.f) + smoothness(q.g)
end

# Product of oracle by a constant.
struct ProdOracle <: Oracle
    c
    f::Oracle
end

*(c, g::Oracle) = ProdOracle(c, g)

function (q::ProdOracle)(x::Vector{Float64})
    return q.c*q.f(x)
end

function gradient(q::ProdOracle, x::Vector{Float64})
    return q.c*gradient(q.f, x)
end

function smoothness(q::ProdOracle)
    return q.c*smoothness(q.f)
end


# Composition of oracle with linear function
struct LinearCompositionOracle <: Oracle
    f::Oracle
    A::Matrix{Float64}
    b::Vector{Float64}
end

function (q::LinearCompositionOracle)(x::Vector{Float64})
    return q.f(q.A*x-q.b)
end

function gradient(q::LinearCompositionOracle, x::Vector{Float64})
    return q.A'*gradient(q.f, q.A*x-q.b)
end

function smoothness(q::LinearCompositionOracle)
    return svdvals(q.A)[1]^2[end]*smoothness(q.f)
end


# Composition of oracle with eigenvalue function
struct SpectralCompositionOracle <: Oracle
    f::Oracle
end
function (q::SpectralCompositionOracle)(X::Matrix{Float64})
    eigen_decomp = eigen(X)
    λ = eigen_decomp.values
    return q.f(λ)
end

function gradient(q::SpectralCompositionOracle, X::Matrix{Float64})
    eigen_decomp = eigen(X)
    λ = eigen_decomp.values
    V = eigen_decomp.vectors
    return V * Diagonal(gradient(q,λ)) * V'
end

function smoothness(q::SpectralCompositionOracle)
    return smoothness(q.f)
end

# Utility oracle for remembering all calls to an oracle
mutable struct TraceOracle <: Oracle
    f::Oracle
    xs
    vals
end
TraceOracle(f) = TraceOracle(f, [], [])
function (q::TraceOracle)(x::Vector{Float64})
    push!(q.xs, x)
    val = q.f(x)
    push!(q.vals, val)
    return val
end

function gradient(q::TraceOracle, x::Vector{Float64})
    return gradient(q.f, x)
end

function smoothness(q::TraceOracle)
    return smoothness(q.f)
end
