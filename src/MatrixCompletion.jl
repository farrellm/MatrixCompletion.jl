module MatrixCompletion

import NLopt

export complete

function add!(x::Array, y::Array)
    for i = eachindex(x)
        x[i] += y[i]
    end
end

function mul!(x::Array, y::Array)
    for i = eachindex(x)
        x[i] *= y[i]
    end
end

function nuclear_norm{T <: AbstractFloat}(m::Matrix{T}, grad::Matrix{T})
    # Alt (http://math.stackexchange.com/users/123685/alt), Derivative
    # of nuclear norm, URL (version: 2015-12-29):
    # http://math.stackexchange.com/q/701104
    f = svdfact(m);
    grad[:] = vec(f[:U] * f[:Vt]);
    sum(f[:S])
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
    vjes = vec(m);

    vones = zeros(size(vjes));
    # vones = spzeros(length(vjes), 1);
    for i = eachindex(m)
        if m[i] != 0.
            vones[i] = 1.
        end
    end

    function fg!(vx::Vector, grad::Vector)
        x = reshape(vx, sz);

        # vz = (vx - vjes) .* vones;
        vz = vx - vjes;
        mul!(vz, vones);

        fn = vecnorm(vz);

        if length(grad) > 0
            gradm = reshape(grad, sz);
            n = nuclear_norm(x, gradm) + lambda * fn;

            # grad[:] += (lambda / fn) * vz;
            add!(grad, (lambda / fn) * vz);
        else
            n = nuclear_norm(x) + lambda * fn;
            throw(AssertionError("grad-free algorithms are slow"));
        end

        n
    end

    opt = NLopt.Opt(method, length(vjes));
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
