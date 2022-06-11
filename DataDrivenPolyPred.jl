using Flux, LinearAlgebra, Zygote, Plots

deg = 4
R = 10
sz = 100000
function random_poly_gen(deg)
    p = rand((1,deg+1))
    x = rand(Float64, p) .* rand((-1,1), p)
    return vcat(x, zeros(deg+1 - length(x)))
end

function poly_eigvals(x, R=10)
    f = -R:R # set the range
    V = f .^ (0:length(x)-1)'
    M = V*x
    M = SymTridiagonal(M, ones(length(f)-1))
    eigvals(M)  # compute the eigenvalues
end

function gen_data(deg=10, R=10, sz=10000)
    y_vals = zeros(deg+1,sz)
    x_vals = zeros(2*R+1,sz)
    for i = 1:sz
        y_vals[:,i] = random_poly_gen(deg)
        x_vals[:,i] = poly_eigvals(x_vals[:,i], R)      
    end
    return x_vals, y_vals
end

x_train, y_train = gen_data(deg, R, sz)
x_val, y_val = gen_data(deg, R, sz)


model = Chain(
    Dense(Int64(2*R+1),100),
    Dense(100,100),
    Dense(100,Int64(deg+1))
)

ps = Flux.params(model)
opt = Flux.Optimise.ADAM()

function loss3(x,y)
    Flux.Losses.mse(model(x),y)
end

train_loss_history = []
val_loss_history = []
for epoch = 1:1000
    Flux.train!(loss3, ps, [(x_train, y_train)], opt)
    train_loss = loss3(x_train, y_train)
    push!(train_loss_history, train_loss)
    val_loss = loss3(x_val, y_val)
    push!(val_loss_history, val_loss)
    println("Epoch = $epoch, loss = $train_loss, val_loss = $val_loss")
end