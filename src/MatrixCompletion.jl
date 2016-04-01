module MatrixCompletion

import NLopt

export complete

function nuclear_norm{T <: AbstractFloat}(m::Matrix{T})
    f = svdfact(m);
    sum(f[:S])
end

function d_nuclear_norm{T <: AbstractFloat}(m::Matrix{T})
    # Alt (http://math.stackexchange.com/users/123685/alt), Derivative
    # of nuclear norm, URL (version: 2015-12-29):
    # http://math.stackexchange.com/q/701104
    f = svdfact(m);
    f[:U] * f[:Vt];
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
                                      x0 = m + std(m) * randn(size(m)),
                                      maxeval = Inf, maxtime = Inf)
    sz = size(m);

    mones = zeros(sz);
    for i = eachindex(m)
        if m[i] != 0.
            mones[i] = 1.
        end
    end

    function fg!(vx::Vector, grad::Vector)
        x = reshape(vx, sz);
        z = reshape(grad, sz);

        # z = mones .* (x - m);
        # fn = vecnorm(z);
        # grad[:] = vec(d_nuclear_norm(x) + (lambda/fn) * z);
        # nuclear_norm(x) + lambda * fn

        fn = 0.;
        for i = eachindex(m)
            s = mones[i] * (x[i] - m[i])
            z[i] = s;
            fn += s * s;
        end
        fn = sqrt(fn);

        f = svdfact(x);
        BLAS.gemm!('N', 'N', 1., f[:U], f[:Vt], lambda / fn, z);
        sum(f[:S]) + lambda * fn
    end

    opt = NLopt.Opt(method, length(m));
    NLopt.min_objective!(opt, fg!);
    NLopt.xtol_rel!(opt, tol);
    NLopt.ftol_rel!(opt, sqrt(tol));

    if maxeval < Inf
        NLopt.maxeval!(opt, maxeval)
    end
    if maxtime < Inf
        NLopt.maxtime!(opt, maxeval)
    end

    (optf,optx,ret) = NLopt.optimize(opt, vec(x0));

    CompletionResults(method, m, reshape(optx, sz), optf, tol, sqrt(tol))
end

end # module
