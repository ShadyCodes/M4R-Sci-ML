using ForwardDiff, LinearAlgebra, BenchmarkTools, Zygote, Plots, DelimitedFiles, Statistics
import ForwardDiff: gradient
include("symtridiagonal.jl")

### The below approach does not work with Zygote as Zygote does not support mutating arrays
function banded_eigvals(x, R) # x is a vector of polynomial coefficients
    f = -R:R # set the range
    V = f .^ (0:length(x)-1)' # Vandermonde matrix
    M = V*x # Vector of polynomial evaluated over range
    M = SymTridiagonal(M, ones(eltype(x), length(f)-1))
    eigvals(M)
end

# Update using forward mode gradient
function update_weights(x, y, R; γ=0.001, α=0.05, β=0.03)
    function loss(x)
        return norm(banded_eigvals(x, R) - y)
    end
    ∇f = ForwardDiff.gradient(loss,x)[1]
    ### Backtracking linesearch according to Armijo rule
    while norm(banded_eigvals(x, R) - y) - norm(banded_eigvals(x .- ∇f .* γ, R) - y) < α * γ * (∇f)' * -(∇f)
        γ *= β
    end
    x .-= ∇f .* γ
end


### Reverse mode
function banded_eigvals_rev(x, R)
    f = -R:R # set the range
    V = f .^ (0:length(x)-1)' # Vandermonde matrix
    M = V*x # Vector of polynomial evaluated over range
    M = SymTridiagonal(M, ones(length(f)-1))
    eigvals(M)
end

# Update using reverse mode gradient
function update_weights_rev!(x, y, R; γ=0.001, α=0.05, β=0.03)
    function loss(x)
        return norm(banded_eigvals_rev(x, R) - y)
    end
    ∇f = Zygote.gradient(loss,x)[1]
    while norm(banded_eigvals_rev(x, R) - y) - norm(banded_eigvals_rev(x .- ∇f .* γ, R) - y) < α * γ * (∇f)' * -(∇f)
        γ *= β
    end
    x .-= ∇f .* γ
end


### Benchmarking (1000 runs each)
function rev_run(target; R=10)
    y = banded_eigvals_rev(target, R)
    x = zeros(length(target))
    for _ = 1:1000
        x = update_weights_rev!(x, y, R, γ=0.0001)
    end 
end

function forward_run(target; R=10)
    y = banded_eigvals(target, R)
    x = zeros(length(target))
    for _ = 1:1000
        x = update_weights(x, y, R, γ=0.0001)
    end 
end


### Setting up benchmarks
MAX_POLYDEG = 1000 # maximum polynomial degree
STEP_SIZE = 25 # check polynomials up to MAX_POLYDEG in steps of STEP_SIZE
POLYIDX = MAX_POLYDEG ÷ STEP_SIZE 

MAX_MATSIZE = 4
MAT_SIZES = collect(1:MAX_MATSIZE).*10

suite = BenchmarkGroup()
suite["for"] = BenchmarkGroup()
suite["rev"] = BenchmarkGroup()

coeffs = rand(MAX_POLYDEG+1)

for R = 1:length(MAT_SIZES)
    for deg = 1:POLYIDX+1
        suite["for"]["deg_" * string(STEP_SIZE*(deg-1)), "mat_" * string(MAT_SIZES[R])] = @benchmarkable forward_run($coeffs[1:$STEP_SIZE*($deg-1)+1], R=$MAT_SIZES[$R])
        suite["rev"]["deg_" * string(STEP_SIZE*(deg-1)), "mat_" * string(MAT_SIZES[R])] = @benchmarkable rev_run($coeffs[1:$STEP_SIZE*($deg-1)+1], R=$MAT_SIZES[$R])
    end
end

for_results = run(suite["for"], verbose=true)
rev_results = run(suite["rev"], verbose=true)

degs = collect(1:POLYIDX).*STEP_SIZE

mins = zeros(POLYIDX, length(MAT_SIZES))
mins_for = zeros(POLYIDX, length(MAT_SIZES))
for R = 1:length(MAT_SIZES)
    for deg = 1:POLYIDX
        mins[deg,R] = minimum(rev_results["deg_" * string(STEP_SIZE*(deg-1)), "mat_" * string(MAT_SIZES[R])]).time
        mins_for[deg,R] = minimum(for_results["deg_" * string(STEP_SIZE*(deg-1)), "mat_" * string(MAT_SIZES[R])]).time
    end
end
mins_for = zeros(POLYIDX-1)
for deg = 1:POLYIDX+1
    mins_for[size-1] = minimum(for_results["deg_" * string(STEP_SIZE*deg - (STEP_SIZE-1))]).time
end


### Plotting
l = @layout [a b]
p1 = plot(
    degs,
    mins_for,
    xlim=[1, MAX_POLYDEG],
    xlabel="Polynomial degree",
    ylabel="Time (ns)", 
    title="ForwardDiff Minimum Time",
    label=(MAT_SIZES.*2 .+1)',
    legendtitle="Matrix size",
    legendtitlefonthalign=:center,
    legend=:topleft
)

p2 = plot(
    degs,
    mins,
    xlim=[1, MAX_POLYDEG],
    xlabel="Polynomial degree",
    title="Zygote Minimum Time",
    legend=false
)

plot(p1, p2, layout=l, size=(1200,500), link=:y, margin=7Plots.mm)


for R = 1:length(MAT_SIZES)
    for deg = 1:MAX_POLYDEG+1
       for_results[deg,R] = @belapsed forward_run((collect(Float64, 1:$STEP_SIZE*$deg-($STEP_SIZE-1))), R=$MAT_SIZES[$R])
       rev_results[deg,R] = @belapsed rev_run((collect(Float64, 1:$STEP_SIZE*$deg-($STEP_SIZE-1))), R=$MAT_SIZES[$R])
    end
 end
 
for_results = zeros(MAX_POLYDEG, length(MAT_SIZES))
rev_results = zeros(MAX_POLYDEG, length(MAT_SIZES))

### Convergence/Accuracy
function rev_run(x, target; R=10, n_iter=10000)
    y = banded_eigvals_rev(target, R)
    function loss(x)
        return norm(banded_eigvals_rev(x, R) - y)
    end
    loss_history = []
    acc_history = []
    for i = 1:n_iter
        x = update_weights_rev!(x, y, R, γ=0.0001)
        if i % 100 == 0
            push!(loss_history,loss(x))
            push!(acc_history, norm(x-target))
        end
    end 

    p1 = plot(
        (collect(1:length(loss_history)).*100),
        loss_history,
        xlabel="Iteration",
        ylabel="Loss",
        yaxis=:log,
        title="Degree = $(length(x)-1)",
        legend=false
    )
    p2 = plot(
        (collect(1:length(loss_history)).*100),
        acc_history,
        xlabel="Iteration",
        ylabel="Coefficient Loss",
        yaxis=:log,
        title="Degree = $(length(x)-1)",
        legend=false
    )

    return loss_history, acc_history, p1, p2
end

# Iterate through polynomial degree 1 to 8
nitr = 100  # number of samples of each degree to take
pmax = 8
fin_loss = zeros(nitr,pmax)
fin_acc = zeros(nitr,pmax)
for iter = 1:nitr
    for p = 1:pmax
        x = rand(p+1).*10
        y = rand(p+1).*10
        loss_history, acc_history, p1, p2 = rev_run(x, y, R=p, n_iter=50000)
        fin_loss[iter,p] = loss_history[lastindex(loss_history)]
        fin_acc[iter,p] = acc_history[lastindex(acc_history)]
    end
    println("Iteration = $iter")
end
loss_means = mean(fin_loss, dims=1)
loss_sds = std(fin_loss, dims=1)
acc_means = mean(fin_acc, dims=1)
acc_sds = std(fin_acc, dims=1)
writedlm("table.csv", vcat(loss_means, loss_sds, acc_means, acc_sds))