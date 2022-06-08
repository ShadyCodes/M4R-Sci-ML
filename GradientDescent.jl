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

MAX_POLYDEG = 20  

suite = BenchmarkGroup()
suite["Rev"] = BenchmarkGroup()
suite["For"] = BenchmarkGroup()

for size = 2:MAX_POLYDEG
    suite["Rev"]["Pdeg = " * string(size-1)] = @benchmarkable rev_run(target) setup=(target  = rand($size).*10)
    suite["For"]["Pdeg = " * string(size-1)] = @benchmarkable forward_run(target) setup=(target  = rand($size).*10)
end
tune!(suite)
results = BenchmarkTools.run(suite, verbose=true)

medians = zeros(MAX_POLYDEG-1)
for size = 2:MAX_POLYDEG
    medians[size-1] = median(results["Rev"]["Pdeg = " * string(size-1)]).time
 end

medians_for = zeros(MAX_POLYDEG-1)
for size = 2:MAX_POLYDEG
    medians_for[size-1] = median(results["For"]["Pdeg = " * string(size-1)]).time
end

mins = zeros(MAX_POLYDEG-1)
for size = 2:MAX_POLYDEG
    mins[size-1] = minimum(results["Rev"]["Pdeg = " * string(size-1)]).time
 end

mins_for = zeros(MAX_POLYDEG-1)
for size = 2:MAX_POLYDEG
    mins_for[size-1] = minimum(results["For"]["Pdeg = " * string(size-1)]).time
end


l = @layout [a b]
p1 = plot(
    medians_for ./ 10^9, 
    xlim=[1, MAX_POLYDEG-1],
    xlabel="Polynomial degree",
    ylabel="Time (ns)", 
    title="ForwardDiff Median Time",
    legend=false
)

p2 = plot(
    medians ./ 10^9, 
    xlim=[1, MAX_POLYDEG-1],
    xlabel="Polynomial degree",
    title="Zygote Median Time",
    legend=false
)

plot(p1, p2, layout=l, size=(1200,500), link=:y, margin=7Plots.mm)

p1 = plot(
    mins_for ./ 10^9, 
    xlim=[1, MAX_POLYDEG-1],
    xlabel="Polynomial degree",
    ylabel="Time (ns)", 
    title="ForwardDiff Minimum Time",
    legend=false
)

p2 = plot(
    mins ./ 10^9, 
    xlim=[1, MAX_POLYDEG-1],
    xlabel="Polynomial degree",
    title="Zygote Minimum Time",
    legend=false
)

plot(p1, p2, layout=l, size=(1200,500), link=:y, margin=7Plots.mm)