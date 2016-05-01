module MatrixCompletion

import Optim
import NLopt

export complete, cross_complete

function nuclear_norm{T <: AbstractFloat}(m::Matrix{T})
    f = svdfact(m)
    sum(f[:S])
end

function d_nuclear_norm{T <: AbstractFloat}(m::Matrix{T})
    # Alt (http://math.stackexchange.com/users/123685/alt), Derivative
    # of nuclear norm, URL (version: 2015-12-29):
    # http://math.stackexchange.com/q/701104
    f = svdfact(m)
    f[:U] * f[:Vt]
end

type CompletionResults{T <: AbstractFloat}
    method::Symbol
    initial::Matrix{T}
    minimum::Matrix{T}
    f_minimum::Float64
    xtol::T
    ftol::T
end

function complete{T <: AbstractFloat}(m::Matrix{T}, lambda::Real;
                                      method = :LD_LBFGS, tol = eps(T),
                                      x0 = 0.5 * m + mean(m),
                                      maxeval = Inf, maxtime = Inf)
    sz = size(m)

    mones = zeros(sz)
    for i = eachindex(m)
        if m[i] != 0.
            mones[i] = 1.
        end
    end

    function f(vx::Vector, grad::Vector)
        x = reshape(vx, sz)

        fn = 0.
        for i = eachindex(m)
            s = mones[i] * (x[i] - m[i])
            fn += s * s
        end
        fn = sqrt(fn)

        f = svdfact(x)
        sum(f[:S]) + lambda * fn
    end

    function g!(vx::Vector, grad::Vector)
        x = reshape(vx, sz)
        z = reshape(grad, sz)

        fn = 0.
        for i = eachindex(m)
            s = mones[i] * (x[i] - m[i])
            z[i] = s
            fn += s * s
        end
        fn = sqrt(fn)

        f = svdfact(x)
        BLAS.gemm!('N', 'N', 1., f[:U], f[:Vt], lambda / fn, z)
    end

    function fg!(vx::Vector, grad::Vector)
        x = reshape(vx, sz)
        z = reshape(grad, sz)

        # z = mones .* (x - m)
        # fn = vecnorm(z)
        # grad[:] = vec(d_nuclear_norm(x) + (lambda/fn) * z)
        # nuclear_norm(x) + lambda * fn

        fn = 0.
        for i = eachindex(m)
            s = mones[i] * (x[i] - m[i])
            z[i] = s
            fn += s * s
        end
        fn = sqrt(fn)

        f = svdfact(x)
        BLAS.gemm!('N', 'N', 1., f[:U], f[:Vt], lambda / fn, z)
        x = sum(f[:S]) + lambda * fn

        x
    end

    d4 = Optim.DifferentiableFunction(f, g!, fg!)
    res = Optim.optimize(d4, vec(x0))
    optf = res.f_minimum
    optx = res.minimum

    # opt = NLopt.Opt(method, length(m))
    # NLopt.min_objective!(opt, fg!)
    # NLopt.xtol_rel!(opt, tol)
    # NLopt.ftol_rel!(opt, sqrt(tol))

    # if maxeval < Inf
    #     NLopt.maxeval!(opt, maxeval)
    # end
    # if maxtime < Inf
    #     NLopt.maxtime!(opt, maxeval)
    # end

    # (optf,optx,ret) = NLopt.optimize(opt, vec(x0))

    CompletionResults(method, m, reshape(optx, sz), optf, tol, sqrt(tol))
end

function roundn(v, n::Int)
    round(v * 10. ^ n) * 10. ^ (-n)
end

function cross_complete(m; n=5, js=1:n,
                        lambda_min = 0, lambda_max = 20,
                        trace = false)
    subset = rand(1:n, size(m))

    insample = map(j -> subset .!= j, js)
    outsample = map(j -> ! insample[j], js)

    m_in  = map(j -> m .* insample[j], js)

    if trace
        cnt = countnz(m)
        for j = js
            cnt_in = countnz(insample[j] .* m)
            cnt_out = countnz(outsample[j] .* m)

            println("j = ", j)
            println("insample count: ", cnt_in, " / ", cnt, " = ", roundn(cnt_in / cnt, 2))
            println("outsample count: ", cnt_out, " / ", cnt, " = ", roundn(cnt_out / cnt, 2))
            println()
        end
    end

    function x_utility(lambda)
        function j_utility(j)
            # @time r = complete(m_in, lambda, x0=res.minimum)
            # r = complete(m_in[j], lambda, x0=res[j].minimum)
            @time r = complete(m_in[j], lambda)
            u = vecnorm((m - r.minimum) .* outsample[j])

            x = reshape(r.minimum, size(m))
            f = svdfact(x)
            println(j, " : ", @sprintf("%.2f", lambda), " : ", round(Int, f[:S]))

            u
        end

        us = map(j_utility, js)
        u = mean(us)

        if trace
            println(@sprintf("%.2f", u), " ", roundn(us, 2))
            println()
        end

        u
    end

    Optim.optimize(x_utility, lambda_min, lambda_max, rel_tol = 1e-4)
end

end # module
