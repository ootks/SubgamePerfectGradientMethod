using JLD2
using Printf
using Plots
import LinearAlgebra: dot 

include("../fom_interface.jl")
include("../oracle_utilities.jl")
include("../example_oracles.jl")

# Tests a FOM on a function defined by an oracle.
function test(method::FOM, oracle::Oracle, start::Vector{Float64}, steps::Int64,
        minVal=0, earlyStop = 0.0)
    initialize(method, start)
    L = smoothness(oracle)
    x = start
    errors = []
    guarantees = []
    for i = 1:steps
        val = oracle(x)
        outputValue = val

        #If the method we are looking at has a different iterate for output values than x, we also need to evaluate there.
        if hasproperty(method, :outputIterate)
           outputValue = oracle(method.outputIterate)
        end
        
        push!(errors, outputValue-minVal)
        push!(guarantees, guarantee(method))
        
        if (outputValue-minVal)/errors[1] <= earlyStop && guarantee(method) <= earlyStop
            break
        end
        
        grad = gradient(oracle, x)
        x = update(method, val/L, grad/L) #wlog run the method on the 1-smooth rescaling
    end
    return errors, guarantees, x
end

# method = constant_step(1., missing)
# q = logisticRegression([1.])
# display(test(method, q, [1.], 30))
# 
# method = constant_step(0.5, missing)
# q = quadratic(hcat(2.), [0.], 0)
# display(test(method, q, [1.], 10))
# 
# method = constant_step(1, missing)
# q = logsumexp(1)
# display(test(method, q, [1.], 10))

# method = OGM(missing, missing, missing, missing)
# q = quadratic(hcat(2.), [0.], 0)
# display(test(method, q, [1.], 10))

# method = constant_step(1, missing)
# q = logisticRegression([1.])
# display(test(method, q, [1.], 10))

# method = OGM(missing, missing, missing, missing)
# q = logisticRegression([1.])
# display(test(method, q, [1.], 10))



function report(title, flag, N, errors)
    if flag
        println("\t\t ",title,": ", N)
    else
        println("\t\t ",title,": failed\t\t only reaching ",errors[N])
    end
end

function reportCompare(title, flag, N, errors, N_baseline)
    if flag
        println("\t\t ",title,": ", N, "\t\t (improvement of ",(N_baseline-N)*100.0/N_baseline,"% over OGM's ", N_baseline,")")
    else
        println("\t\t ",title,": failed\t\t only reaching ",errors[N])
    end
end


function plotSingleInstance(oracle::Oracle, x0::Vector{Float64}, minVal::Float64, xStar::Vector{Float64}, methods, steps=100; title="")
    errors = []
    guarantees =[]
    for method in methods
        out = test(method, q, x0, steps, minVal)
        push!(errors, out[1]/(0.5*smoothness(q)*norm(x0-xStar,2)^2))
        push!(guarantees, out[2])
    end
    plot1 = plot(errors, labels=permutedims([methodTitle(method) * " scaled error" for method in methods]), lc=[:blue :red :green :orange :purple], yscale=:log10,  linewidth=3, title = title)
    plot!(guarantees, labels=permutedims([methodTitle(method) * " guarantee" for method in methods]), lc=[:blue :red :green :orange :purple])
    return plot1
end

function plotSingleInstance(file_path::String, methods, steps=100)
    
    @load file_path title description q x0 minVal xStar
    errors = []
    guarantees =[]
    for method in methods
        out = test(method, q, x0, steps, minVal)
        push!(errors, out[1]/(0.5*smoothness(q)*norm(x0-xStar,2)^2))
        push!(guarantees, out[2])
    end
    plot1 = plot(errors, labels=permutedims([methodTitle(method) * " scaled error" for method in methods]), lc=[:blue :red :green :orange :purple], yscale=:log10,  linewidth=3, title = title)
    plot!(guarantees, labels=permutedims([methodTitle(method) * " guarantee" for method in methods]), lc=[:blue :red :green :orange :purple])
    return plot1
end

function runSuiteOfTest(directory, methods, targetRelAccuracies=[1e-1,1e-3,1e-5], maxSteps=100)
    for file in readdir(directory)
        file_path = joinpath(directory, file)

        # Ensure only to process .jld2 files
        if endswith(file, ".jld2")
            # Load the data
            @load file_path title description q x0 minVal xStar

            println("Processing:  ", file)
            
            print("Method       ")
            for target in targetRelAccuracies
                @printf("%-10.7f  ", target)
            end
            for method in methods
                println("")
                println(methodTitle(method))
                print(" RelSubopt   ")
                
                errors, guarantees, _ = test(method, q, x0, maxSteps, minVal, targetRelAccuracies[length(targetRelAccuracies)])
                i=1
                for j in 1:length(errors)
                    if errors[j]/(0.5*smoothness(q)*norm(x0-xStar,2)^2) < targetRelAccuracies[i]
                        @printf("%-12d", j)
                        i=i+1
                    end
                    if i > length(targetRelAccuracies)
                        break
                    end
                end
                while i <= length(targetRelAccuracies)
                    print("--          ")
                    i = i+1
                end
                
                println("")
                print(" Guarantee   ")
                i=1
                for j in 1:length(guarantees)
                    if guarantees[j] < targetRelAccuracies[i]
                        @printf("%-12d", j)
                        i=i+1
                    end
                    if i > length(targetRelAccuracies)
                        break
                    end
                end
                while i <= length(targetRelAccuracies)
                    print("--          ")
                    i = i+1
                end
                
            end
            println("")
            println("")    
        end
    end
end
