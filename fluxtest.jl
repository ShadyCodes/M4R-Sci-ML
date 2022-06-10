    using Flux, LinearAlgebra, Zygote
    include("symtridiagonal.jl")

    function nonlin_func(x)
        return exp.(sin.(x))
    end

    function eigvals_from_func(f, R=10)
        x = collect(Float64, -R:R)
        y = f(x)
        M = SymTridiagonal(y, ones(length(y)-1))
        eigvals(M)
    end

    function test_func(x)
        return x.^2
    end

    R = 10
    x = collect(Float64, -R:R)
    target = eigvals_from_func(test_func, R)


    model = Chain(
        Dense(2*R+1,100),
        Dense(100,100),
        Dense(100,100),
        Dense(100,100),
        Dense(100,100),
        Dense(100,100),
        Dense(100,100),
        Dense(100,2*R+1)
    )

    function loss(x,y)
        ŷ = model(x)
        M = SymTridiagonal(ŷ, ones(length(ŷ)-1))
        norm(y - eigvals(M))
    end

    ps = Flux.params(model)
    γ = 0.001
    opt = Flux.Optimise.ADAM(γ)

    loss_history = []
    for epoch = 1:100000
        Flux.train!(loss, ps, [(x, target)], opt)
        train_loss = loss(x, target)
        push!(loss_history, train_loss)
        println("Epoch = $epoch, loss = $train_loss")
    end

    plot(x, [test_func(x), model(x)])