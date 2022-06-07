using ForwardDiff, LinearAlgebra, BenchmarkTools, Zygote, Plots
import ForwardDiff: gradient
include("symtridiagonal.jl")

### The below approach does not work with Zygote as Zygote does not support mutating arrays
function banded_eigvals(x, R) # x is a vector of polynomial coefficients
    f = -R:R # set the range
    V = f .^ (0:length(x)-1)'
    M = V*x
    M = SymTridiagonal(M, ones(eltype(x), length(f)-1))
    eigvals(M)
end

function update_weights(x, y, R; γ=0.001, α=0.05, β=0.03)
    function loss(x)
        return norm(banded_eigvals(x, R) - y)
    end
    ∇f = ForwardDiff.gradient(loss,x)[1]
    while norm(banded_eigvals(x, R) - y) - norm(banded_eigvals(x .- ∇f .* γ, R) - y) < α * γ * (∇f)' * -(∇f)
        γ *= β
    end
    x .-= ∇f .* γ
end


### Reverse mode
function banded_eigvals_rev(x, R)
    f = -R:R # set the range
    V = f .^ (0:length(x)-1)'
    M = V*x
    M = SymTridiagonal(M, ones(length(f)-1))
    eigvals(M)  # compute the eigenvalues
end

function update_weights_rev(x, y, R; γ=0.001, α=0.05, β=0.03)
    function loss(x)
        return norm(banded_eigvals_rev(x, R) - y)
    end
    ∇f = Zygote.gradient(loss,x)[1]
    while norm(banded_eigvals_rev(x, R) - y) - norm(banded_eigvals_rev(x .- ∇f .* γ, R) - y) < α * γ * (∇f)' * -(∇f)
        γ *= β
    end
    x .-= ∇f .* γ
end

### Benchmarking
function rev_run(target; R=10)
    y = banded_eigvals_rev(target, R)
    x = zeros(length(target))
    for _ = 1:1000
        x = update_weights_rev(x, y, R, γ=0.0001)
    end 
end

function forward_run(target; R=10)
    y = banded_eigvals(target, R)
    x = zeros(length(target))
    for _ = 1:1000
        x = update_weights(x, y, R, γ=0.0001)
    end 
end

suite = BenchmarkGroup()
suite["Rev"] = BenchmarkGroup()
suite["For"] = BenchmarkGroup()
for R = 2:4
    suite["Rev"]["Matrix size = " * string(R)] = BenchmarkGroup()
    suite["For"]["Matrix size = " * string(R)] = BenchmarkGroup()
    for size = 2:4
        suite["Rev"]["Matrix size = " * string(R)]["Polynomial degree = " * string(size-1)] = @benchmarkable rev_run($(rand(size) .* 10), R=$R)
        suite["For"]["Matrix size = " * string(R)]["Polynomial degree = " * string(size-1)] = @benchmarkable forward_run($(rand(size) .* 10), R=$R)
    end
end
#tune!(suite)
results = BenchmarkTools.run(suite, verbose=true)

medians = zeros(3, 3)
for R = 2:4
    for size = 2:4
        medians[R-1, size-1] = median(results["Rev"]["Matrix size = " * string(R)]["Polynomial degree = " * string(size-1)]).time
    end
end
