include("fom_interface.jl")
using Parameters
using JuMP
import LinearAlgebra
import MathOptInterface as MOI
import Mosek
import MosekTools

@with_kw mutable struct SPGM <: FOM
    memory_size
    max_steps
    x0
    x
    Z
    G
    zprime
    psi_prev
    taus
    h
    q
    v
    vmin
    best_update
    iteration
end

SPGM(memory_size) = SPGM(memory_size,missing,missing,missing,missing,missing,
                           missing,missing,missing,missing,missing,missing,
                           missing,missing,missing)
SPGM(memory_size, max_steps) = SPGM(memory_size,max_steps,missing,missing,
                                    missing,missing,missing,missing,missing,
                                    missing,missing,missing,missing,missing)

function initialize(method::SPGM, start::Vector{Float64})
    method.x0 = start
    method.x = start - start
    method.Z = Array{Float64}(undef, length(start), 0)
    method.G = Array{Float64}(undef, length(start), 0)
    method.zprime = start - start
    method.psi_prev = 2.
    method.taus = [2.]
    method.h = []
    method.q = []
    method.v = []
    method.vmin = missing
    method.iteration = 0
end

# Push an element of v into x, taking the tail if the size grows over k
function push_over_mat(x,v,k)
    x = hcat(x, v)
    if size(x)[end] > k
        x = x[:, end-k+1:end]
    end
    return x
end

# Push an element of v into x, taking the tail if the size grows over k
function push_over(x,v,k)
    push!(x, v)
    if length(x) > k
        return x[end-k+1:end]
    else
        return x
    end
end

function update(method::SPGM, value::Float64, gradient::Vector{Float64};
        tol=1e-9)
    @unpack_SPGM method

    # updates that depend on g_n
    z = zprime - psi_prev * gradient
    # Terminate if the next iterate is too good.
    if norm(z) < tol || taus[end] > 1/tol
        return x + x0
    end
    G = push_over_mat(G, gradient, memory_size)
    Z = push_over_mat(Z, z, memory_size)
    v = push_over(v, value - norm(gradient)^2/2, memory_size)
    if ismissing(vmin) || v[end] < vmin
        method.vmin = v[end]
        method.best_update = x - gradient
    end
    h = push_over(h, taus[end] * v[end] + norm(z)^2/2, memory_size)
    q = push_over(q, value - dot(gradient, x) + norm(gradient)^2/2, memory_size)

    method.q = q
    method.h  = h
    method.Z = Z
    method.G = G
    method.v = v
    method.iteration += 1

    # compute update
    mu, lambdas = optimize_lambdas(taus, Z, G, h, q, v, method.vmin)
    phi = dot(taus, mu) + sum(lambdas)
    if  !ismissing(max_steps) && method.iteration == max_steps
        psi = (1+sqrt(1+4*phi))/2
    else
        psi = 1 + sqrt(1 + 2 * phi)
    end

    # updates that don't depend on g_{n+1}
    method.psi_prev = psi
    method.zprime = Z * mu - G * lambdas
    tau = phi + psi
    method.taus = push_over(taus, tau, memory_size)
    method.x = (psi * method.zprime + phi * method.best_update)/tau
    return method.x + x0
end

function optimize_lambdas(taus, Z, G, h, q, v, vmin; tol=1e-8)
    k = length(taus)

    mucoeffs = h - vmin * taus
    lambdascoeffs = q .- vmin

    y = nonnegSecondOrderCone(hcat(Z, -G), vcat(mucoeffs, lambdascoeffs),
                              vcat(taus, ones(k)), tol=tol)
    mu = y[1:k]
    lambdas = y[k+1:end]
 
    # Clean solution
    a = dot(mucoeffs, mu) + dot(lambdascoeffs, lambdas)
    b = norm(Z * mu - G * lambdas)^2/2
    return clean_solution(mu, lambdas, taus, a/b)
end

function clean_solution(mu, lambdas, taus, s)
    mu .= max.(mu, 0)
    lambdas .= max.(lambdas, 0)

    s = max(0, s)
    lambdas = s * lambdas
    mu = s * mu

    if dot(taus, mu) + sum(lambdas) < taus[end]
        # println("Falling back to OTF(1).")
        i = length(mu)
        mu .= 0
        mu[end] = 1
        lambdas .= 0
    end
    return mu, lambdas
end

# Computes 
# max <v, y>
# s.t. ||A y||^2 <= <b, y>
#       y >= 0
function nonnegSecondOrderCone(A, b, v; tol=1e-6)
    k = length(b)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # _, R = qr(A)

    # Include tolerance for numerical stability
    @variable(model, 1/tol >= y[1:k] >= 0)
    @constraint(model, [dot(b, y); 1; A*y] in RotatedSecondOrderCone())
    @objective(model, Max, dot(v, y))
    optimize!(model)
    return value.(y)
end


function guarantee(method::SPGM)
    # todo fix this
    return 1/method.taus[end]
end

function methodTitle(method::SPGM)
    return "spgm-" * string(method.memory_size)
end
