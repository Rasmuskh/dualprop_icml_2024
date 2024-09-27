module M

using MLDatasets
using Flux, Flux.Optimise, Flux.Losses
using Flux: onehotbatch, onecold
using Random
using MLUtils
using Zygote
using Base.Iterators: partition
using LinearAlgebra

########################################################################

# Mode 0: regular BP
# Mode 1: old style DP with given alpha at the target loss and 1/2 throughout the layers
# Mode 2: new style DP (adjoint method)
# Mode 4: damped DP/LPOM iterations
# Mode 6: Simple adversarial training
const mode = 2

const α = 0.f0

#const α_k = 0.5f0  # alpha for hidden layers used in modes 2-4
const α_k = α

const β = 1.f0 / 2

const batch_size_train = 50
const nHidden = 128 # 512 #192
const epochs = 20
const η = 1f-3

const n_damped_LPOM_rounds = 15

Random.seed!(43)

########################################################################

accuracy(x, y) = sum(onecold(x, 1:10) .== onecold(y, 1:10)) / size(y, 2)

#my_act = identity
my_act = relu
#my_act = leakyrelu

function dot_products(xs, ys)
    sum(xs .* ys, dims=1)
end

function G_primal(zs)
    return 0.5f0 * sum(abs2.(zs))
end

function G_dual(xs)
    zs = my_act(xs)
    return sum(xs .* zs) - G_primal(zs)
end

########################################################################

@show mode, β, α, α_k

struct Model
    W0 :: Matrix{Float32}
    b0 :: Vector{Float32}
    W1 :: Matrix{Float32}
    b1 :: Vector{Float32}
    W2 :: Matrix{Float32}
    b2 :: Vector{Float32}
end

m_net = Model(randn(Float32, nHidden, 28*28) / 28.f0, zeros(Float32, nHidden),
              randn(Float32, nHidden, nHidden) / sqrt(nHidden), zeros(Float32, nHidden),
              randn(Float32, 10, nHidden) / sqrt(nHidden), zeros(Float32, 10))

########################################################################

function get_γ_power_iter(W)
    n_iters = 3

    b = randn(Float32, size(W, 2))
    for iter = 1:n_iters
        b .= W'*W*b
        b ./= sqrt(sum(abs2.(b)))
    end
    return sqrt(sum(abs2.(W*b))) # sqrt(b'*W'*W*b)
end

########################################################################

# δ is either αβ or -(1-α)β
function infer_outputs!(δ, y, z_L)
    z_L .= (z_L + δ*y) ./ (1.f0 + δ)
end

function infer_zk_DP!(α_cur, a_k, Δ_in, z_k, u_k)
    z_k .= my_act(a_k +     α_cur * Δ_in)
    u_k .= my_act(a_k - (1-α_cur) * Δ_in)
end

function infer_zk_dualprop!(a_k, Δ_in, z_k, u_k)
    infer_zk_DP!(0.5f0, a_k, Δ_in, z_k, u_k)
end

########################################################################

function infer_z!(δ_pos, δ_neg, m, x, y, z1, z2, z3, u1, u2, u3)
    n_rounds = 5

    a1 =  m.W0*x  .+ m.b0; z1 .= my_act(a1); u1 .= z1
    a2 =  m.W1*z1 .+ m.b1; z2 .= my_act(a2); u2 .= z2
    z3 .= m.W2*z2 .+ m.b2; u3 .= z3

    infer_outputs!(δ_pos, y, z3)
    infer_outputs!(δ_neg, y, u3)

    for round = 1:n_rounds
        infer_zk_dualprop!(a2, m.W2'*(z3-u3), z2, u2)
        infer_zk_dualprop!(a1, m.W1'*(z2-u2), z1, u1)

        a2 .= m.W1 * (z1+u1)/2 .+ m.b1
        infer_zk_dualprop!(a2, m.W2'*(z3-u3), z2, u2)

        z3 .= m.W2 * (z2+u2)/2 .+ m.b2; u3 .= z3

        infer_outputs!(δ_pos, y, z3)
        infer_outputs!(δ_neg, y, u3)
    end

    infer_zk_dualprop!(a2, m.W2'*(z3-u3), z2, u2)
    infer_zk_dualprop!(a1, m.W1'*(z2-u2), z1, u1)
end

function eval_loss(δ_pos, δ_neg, m, x, y, z1, z2, z3, u1, u2, u3)
    a1 = m.W0*x         .+ m.b0
    a2 = m.W1*(z1+u1)/2 .+ m.b1
    a3 = m.W2*(z2+u2)/2 .+ m.b2

    res  = δ_pos * sum(abs2.(z3 .- y)) / 2 + δ_neg * sum(abs2.(u3 .- y)) / 2
    res += sum(abs2.(z3 - a3)) / 2 - sum(abs2.(u3 - a3)) / 2
    res += G_primal(z2) - G_primal(u2) - sum((z2-u2) .* a2)
    res += G_primal(z1) - G_primal(u1) - sum((z1-u1) .* a1)

    return res
end

########################################################################

function infer_outputs_adjoint!(δ, y, z_L)
    z_L .= z_L + δ*(y - z_L)
end

function infer_zk_adjoint!(α_cur, a_k, Δ_in, z_k, u_k)
    z_k .= my_act(a_k + (1-α_cur) * Δ_in)
    u_k .= my_act(a_k -     α_cur * Δ_in)
end

function infer_zk_adjoint!(a_k, Δ_in, z_k, u_k)
    infer_zk_adjoint!(α_k, a_k, Δ_in, z_k, u_k)
end

function infer_z_adjoint!(m, x, y, z1, z2, z3, u1, u2, u3)
    α1   = 1-α
    α1_k = 1-α_k

    n_rounds = 5

    a1 =  m.W0*x  .+ m.b0; z1 .= my_act(a1); u1 .= z1
    a2 =  m.W1*z1 .+ m.b1; z2 .= my_act(a2); u2 .= z2
    z3 .= m.W2*z2 .+ m.b2;                   u3 .= z3

    infer_outputs_adjoint!(α*β,   y, z3)
    infer_outputs_adjoint!(-α1*β, y, u3)

    for round = 1:n_rounds
        infer_zk_adjoint!(a2, m.W2'*(z3-u3), z2, u2)
        infer_zk_adjoint!(a1, m.W1'*(z2-u2), z1, u1)

        a2 .= m.W1 * (α_k*z1+α1_k*u1) .+ m.b1
        infer_zk_adjoint!(a2, m.W2'*(z3-u3), z2, u2)

        z3 .= m.W2 * (α_k*z2+α1_k*u2) .+ m.b2; u3 .= z3

        infer_outputs_adjoint!(α*β,   y, z3)
        infer_outputs_adjoint!(-α1*β, y, u3)
    end

    infer_zk_adjoint!(a2, m.W2'*(z3-u3), z2, u2)
    infer_zk_adjoint!(a1, m.W1'*(z2-u2), z1, u1)
end

function eval_loss_adjoint(m, x, y, z1, z2, z3, u1, u2, u3)
    α1   = 1-α
    α1_k = 1-α_k

    a1 = m.W0*x                .+ m.b0
    a2 = m.W1*(α_k*z1+α1_k*u1) .+ m.b1
    a3 = m.W2*(α_k*z2+α1_k*u2) .+ m.b2

    res  = β * (α * sum(abs2.(z3 .- y)) / 2 + α1 * sum(abs2.(u3 .- y))) / 2
    res -= sum((z3-u3) .* a3)
    res -= sum((z2-u2) .* a2)
    res -= sum((z1-u1) .* a1)

    return res
end

########################################################################

function infer_zk_DDP!(L, a_k, Δ_in, z_k, u_k)
    z_k .= my_act((a_k + L*z_k +    α_k  * Δ_in) / (1+L))
    u_k .= my_act((a_k + L*u_k - (1-α_k) * Δ_in) / (1+L))
end

# Dampened DP
function infer_z_DDP!(δ_pos, δ_neg, m, x, y, z1, z2, z3, u1, u2, u3)
    γ1 = get_γ_power_iter(m.W1); γ2 = get_γ_power_iter(m.W2)

    L1 = max(0, γ1^2-1)/2 * abs(1 - 2*α_k); L2 = max(0, γ2^2-1)/2 * abs(1 - 2*α_k)
    #L1 = γ1^2 * abs(1 - 2*α_k); L2 = γ2^2 * abs(1 - 2*α_k)

    n_rounds = n_damped_LPOM_rounds

    a1 =  m.W0*x  .+ m.b0; z1 .= my_act(a1); u1 .= z1
    a2 =  m.W1*z1 .+ m.b1; z2 .= my_act(a2); u2 .= z2
    z3 .= m.W2*z2 .+ m.b2; u3 .= z3

    infer_outputs!(δ_pos, y, z3)
    infer_outputs!(δ_neg, y, u3)

    for round = 1:n_rounds
        infer_zk_DDP!(L2, a2, m.W2'*(z3-u3), z2, u2)
        infer_zk_DDP!(L1, a1, m.W1'*(z2-u2), z1, u1)

        a2 .= m.W1 * (α_k*z1+(1-α_k)*u1) .+ m.b1
        infer_zk_DDP!(L2, a2, m.W2'*(z3-u3), z2, u2)

        z3 .= m.W2 * (α_k*z2+(1-α_k)*u2) .+ m.b2; u3 .= z3

        infer_outputs!(δ_pos, y, z3)
        infer_outputs!(δ_neg, y, u3)
    end

    infer_zk_DDP!(L2, a2, m.W2'*(z3-u3), z2, u2)
    infer_zk_DDP!(L1, a1, m.W1'*(z2-u2), z1, u1)
end

function eval_loss_DP(δ_pos, δ_neg, m, x, y, z1, z2, z3, u1, u2, u3)
    a1 = m.W0*x                   .+ m.b0
    a2 = m.W1*(α_k*z1+(1-α_k)*u1) .+ m.b1
    a3 = m.W2*(α_k*z2+(1-α_k)*u2) .+ m.b2

    res  = δ_pos * sum(abs2.(z3 .- y)) / 2 + δ_neg * sum(abs2.(u3 .- y)) / 2
    res += sum(abs2.(z3 - a3)) / 2 - sum(abs2.(u3 - a3)) / 2
    res += G_primal(z2) - G_primal(u2) - sum((z2-u2) .* a2)
    res += G_primal(z1) - G_primal(u1) - sum((z1-u1) .* a1)

    return res
end

########################################################################

DATASET = MNIST
#DATASET = FashionMNIST

train_x, train_y = DATASET(split=:train)[:]
test_x,  test_y  = DATASET(split=:test)[:]

train_X = Float32.(reshape(train_x, 28*28, :));
test_X  = Float32.(reshape(test_x, 28*28, :));

train_Y = onehotbatch(train_y, 0:9)
test_Y  = onehotbatch(test_y, 0:9)

function train_DP(m)
    train_loader = ([(train_X[:,i], train_Y[:,i]) for i in partition(1:50000, batch_size_train)])

    opt = ADAM(η)

    ps = Flux.params(m.W0, m.b0, m.W1, m.b1, m.W2, m.b2)

    B = batch_size_train
    B1 = B * β

    κ_pos = α*β; κ_neg = (1-α)*β

    zs0 = zeros(Float32, 28*28, B);   us0 = zeros(Float32, 28*28, B);
    zs1 = zeros(Float32, nHidden, B); us1 = zeros(Float32, nHidden, B);
    zs2 = zeros(Float32, nHidden, B); us2 = zeros(Float32, nHidden, B);
    zs3 = zeros(Float32, 10, B);      us3 = zeros(Float32, 10, B);

    for epoch = 0:epochs
        shuffle!(train_loader)
        iter = 1
        total_loss = 0.f0
        train_accuracy = 0.f0

        @time for (xs, ys) in train_loader
            if mode == 0
                cur_loss, gs = withgradient(ps) do
                    zs_L = m.W2 * my_act(m.W1 * my_act(m.W0*xs .+ m.b0) .+ m.b1) .+ m.b2
                    ignore() do
                        zs3 .= zs_L
                    end
                    sum(abs2.(zs_L - ys)) / 2
                end
            elseif mode == 1
                infer_z!(κ_pos, -κ_neg, m, xs, ys, zs1, zs2, zs3, us1, us2, us3)
                cur_loss, gs = withgradient(ps) do
                    eval_loss(κ_pos, κ_neg, m, xs, ys, zs1, zs2, zs3, us1, us2, us3) / B1
                end
            elseif mode == 2
                infer_z_adjoint!(m, xs, ys, zs1, zs2, zs3, us1, us2, us3)
                cur_loss, gs = withgradient(ps) do
                    eval_loss_adjoint(m, xs, ys, zs1, zs2, zs3, us1, us2, us3) / B1
                end
            elseif mode == 4
                infer_z_DDP!(κ_pos, -κ_neg, m, xs, ys, zs1, zs2, zs3, us1, us2, us3)
                cur_loss, gs = withgradient(ps) do
                    eval_loss_DP(κ_pos, κ_neg, m, xs, ys, zs1, zs2, zs3, us1, us2, us3) / B1
                end
            elseif mode == 6
                grad_xs = gradient(xs) do xs_in
                    zs_L = m.W2 * my_act(m.W1 * my_act(m.W0*xs_in .+ m.b0) .+ m.b1) .+ m.b2
                    sum(abs2.(zs_L - ys)) / 2
                end
                xxs = xs + β * grad_xs[1]

                cur_loss, gs = withgradient(ps) do
                    zs_L = m.W2 * my_act(m.W1 * my_act(m.W0*xxs .+ m.b0) .+ m.b1) .+ m.b2
                    ignore() do
                        zs3 .= zs_L
                    end
                    sum(abs2.(zs_L - ys)) / 2
                end
            end

            total_loss     += cur_loss
            train_accuracy += accuracy(zs3, ys)
            if epoch > 0  update!(opt, ps, gs) end

            iter += 1
        end

        total_loss /= iter
        train_accuracy /= iter

        zz3 = m.W2 * my_act(m.W1 * my_act(m.W0*test_X .+ m.b0) .+ m.b1) .+ m.b2

        test_accuracy = accuracy(zz3, test_Y)

        println(epoch, "\t", total_loss, "\t", 100*train_accuracy, "\t", 100*test_accuracy, "\t|W2*W1*W0|: $(opnorm(m.W2*m.W1*m.W0))")
    end
end

train_DP(m_net)

end
