using Flux, LinearAlgebra, Zygote, Plots
include("symtridiagonal.jl")

function eigen_decomp_func(f, R=10; grid_sz=1)
    x = collect(Float64, -R:1/grid_sz:R)
    y = f(x)
    M = SymTridiagonal(y, ones(length(y)-1))
    λ, U = eigen(M)
    return λ, U[1,:]
end

function test_func(x)
    return x.^3
end

R = 10
grid_sz=20
x = collect(Float64, -R:1/grid_sz:R)
target_vals, target_vec = eigen_decomp_func(test_func, R, grid_sz=grid_sz)


model = Chain(
    Dense(Int64(2*R*grid_sz+1),1000, relu),
    Dense(1000,1000, relu),
    Dense(1000,1000, relu),
    Dense(1000,Int64(2*R*grid_sz+1))
)

function loss(x,y_vals,y_vec; λ=0.01, γ=0.001)
    ŷ = model(x)
    M = SymTridiagonal(ŷ, ones(length(ŷ)-1))
    eig_vals, eig_vecs = eigen(M)
    norm(y_vals - eig_vals) + 
    norm(eig_vecs[1,:] - y_vec) + 
    λ * sum((ŷ[2:lastindex(ŷ)] - ŷ[1:lastindex(ŷ)-1]).^2 ./length(ŷ)) + 
    γ*sum((ŷ[3:lastindex(ŷ)] - 2*ŷ[2:lastindex(ŷ)-1] + ŷ[1:lastindex(ŷ)-2]).^2 ./length(ŷ)) 
end

ps = Flux.params(model)
opt = Flux.Optimiser(Flux.Optimise.ADAM(), Flux.Optimise.ExpDecay(1))

loss_history = []
for epoch = 1:10000
    Flux.train!(loss, ps, [(x, target_vals, target_vec)], opt)
    train_loss = loss(x, target_vals, target_vec)
    push!(loss_history, train_loss)
    unreg_loss = loss(x,target_vals, target_vec, λ=0, γ=0)
    println("Epoch = $epoch, loss = $train_loss, unreg_loss = $unreg_loss")
end

plot(x, [test_func(x), model(x)])