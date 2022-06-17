using Flux, LinearAlgebra, Zygote, Plots
include("symtridiagonal.jl")

# extracts eigenvalues given a potential, range and grid size
function eigen_decomp_func(f, R=10; grid_sz=1)
    x = collect(Float64, -R:1/grid_sz:R)
    y = f(x)
    M = SymTridiagonal(y, ones(length(y)-1))
    λ, U = eigen(M)
    return λ, U[1,:]
end

function test_func(x)
    return exp.(sin.(x))
end

R = 10
grid_sz=10
x = collect(Float64, -R:1/grid_sz:R)
target_vals, target_vec = eigen_decomp_func(test_func, R, grid_sz=grid_sz)

# Neural network
model = Chain(
    Dense(Int64(2*R*grid_sz+1),20, relu),
    Dense(20,20, relu),
    Dense(20,20, relu),
    Dense(20,20, relu),
    Dense(20,Int64(2*R*grid_sz+1))
)

# Loss function
function loss(x,y_vals,y_vec; λ=0.0, γ=0.0)
    ŷ = model(x)
    M = SymTridiagonal(ŷ, ones(length(ŷ)-1))
    eig_vals, eig_vecs = eigen(M)
    norm(y_vals - eig_vals).^2 + # eigenvalue loss
    0*norm(eig_vecs[1,:] - y_vec).^2 + # eigenvector loss
    λ * sum((ŷ[2:lastindex(ŷ)] - ŷ[1:lastindex(ŷ)-1]).^2) + # first order smoothing
    γ * sum((ŷ[3:lastindex(ŷ)] - 2*ŷ[2:lastindex(ŷ)-1] + ŷ[1:lastindex(ŷ)-2]).^2) # second order smoothing
end

ps = Flux.params(model)
opt = Flux.Optimise.Descent(0.001) # SGD optimiser

loss_history = []
### May wish to run this loop several times for better convergence
for epoch = 1:50000
    Flux.train!(loss, ps, [(x, target_vals, target_vec)], opt)
    train_loss = loss(x, target_vals, target_vec)
    push!(loss_history, train_loss)
    unreg_loss = loss(x,target_vals, target_vec, λ=0, γ=0)
    if epoch % 100 == 0
        println("Epoch = $epoch, loss = $train_loss, unreg_loss = $unreg_loss")
    end
end

### Plotting
p = plot(
    x, [test_func(x), model(x)],
    title="Functional approximation",
    xlabel="x",
    ylabel="f(x)",
    label=["target" "model"],
    legend=:bottomleft
)

p = plot(
    collect(50000:lastindex(loss_history)),
    loss_history[50000:lastindex(loss_history)],
    title="Loss history",
    xlabel="Iteration",
    yaxis=:log,
    ylabel="Loss",
    legend=:false,
    margin=4Plots.mm
)