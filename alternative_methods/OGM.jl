include("../fom_interface.jl")

mutable struct OGM <: FOM
    theta
    z
    x
    max_iteration
    iteration
    prev_theta
end

OGM() = OGM(missing, missing, missing, missing, missing, missing)
OGM(max_iteration::Int64) = OGM(missing, missing, missing, max_iteration, missing, missing)

function initialize(method::OGM, start::Vector{Float64})
    method.x = start
    method.z = method.x
    method.prev_theta = 1
    method.theta = 1
    method.iteration = 0
end

function update(method::OGM, value::Float64, gradient::Vector{Float64})
    method.iteration += 1

    y = method.x - gradient

    method.z = method.z - 2*method.theta*gradient
    
    theta = method.theta
    if !ismissing(method.max_iteration) &&  method.iteration == method.max_iteration
        theta = (1+sqrt(1+8*theta^2))/2
    else 
        theta = (1+sqrt(1+4*theta^2))/2
    end

    method.x = (1-1/theta) * y + method.z / theta
    method.prev_theta = method.theta
    method.theta = theta

    return method.x
end

function guarantee(method::OGM)
#     if !ismissing(method.max_iteration) &&  method.iteration == method.max_iteration
#         return 1/(method.theta^2)
#     end
    if method.iteration==0
        return 1.0
    end
    return 1/(2*method.prev_theta^2) #Currently returning the online guarantee until the last step when it gets improved
end

function methodTitle(method::OGM)
    return "OGM"
end

# mutable struct OGM <: FOM
#     theta
#     epsilon
#     z
#     x
# end
# 
# function initialize(method::OGM, start::Vector{Float64})
#     method.x = start
#     method.z = missing
#     method.theta = 1
#     method.epsilon = 1
# end
# 
# function update(method::OGM, value::Float64, gradient::Vector{Float64})
#     if ismissing(method.z)
#         method.z = method.x - gradient
#     end
# 
#     z = method.z
#     theta = method.theta
# 
#     l6 = (1+sqrt(1+4*theta^2))/2
#     alpha = 2 * theta^2 / (1 + 2 * theta^2 + sqrt(1 + 4 * theta^2))
# 
#     method.x = z + alpha * (method.x-z) - alpha * gradient
#     method.epsilon = method.epsilon - theta^2 * norm(gradient)^2/2
#     method.z = z - l6 * gradient
#     method.theta = sqrt(theta^2 + l6)
# 
#     return method.x
# end
