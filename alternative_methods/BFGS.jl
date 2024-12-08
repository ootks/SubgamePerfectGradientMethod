include("../fom_interface.jl")
include("../oracle_utilities.jl")
using Optim

mutable struct BFGS <: FOM
end

function test(method::BFGS, oracle::Oracle, start::Vector{Float64}, steps::Int64, minVal=0, earlyStop = 0.0)

    oracle = TraceOracle(oracle)

    res = Optim.optimize(oracle, (x -> gradient(oracle, x)), start, Optim.BFGS(), Optim.Options(f_calls_limit=steps); inplace=false)
    return oracle.vals .- minVal, [], oracle.xs[end]
end

function methodTitle(method::BFGS)
    return "BFGS"
end


mutable struct LBFGS <: FOM
    mem_size
end

function test(method::LBFGS, oracle::Oracle, start::Vector{Float64}, steps::Int64, minVal=0, earlyStop = 0.0)

    oracle = TraceOracle(oracle)

    res = Optim.optimize(oracle, (x -> gradient(oracle, x)), start, Optim.LBFGS(;m=method.mem_size), Optim.Options(f_calls_limit=steps); inplace=false)
    return oracle.vals .- minVal, [], oracle.xs[end]
end

function methodTitle(method::LBFGS)
    return "LBFGS-"*string(method.mem_size)
end

# function bfgs_error(oracle, x0)
# end

# trace_q = TraceOracle(q)
# bfgs_error(trace_q, x0)
# errors = [log.((trace_q.vals .- (minVal - 1e-6))/(0.5*smoothness(q)*norm(x0-xStar,2)^2))]
