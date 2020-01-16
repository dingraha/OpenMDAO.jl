using FLOWMath: akima_setup, akima_interp, akima
using ForwardDiff: jacobian, jacobian!
using CompositeArrays: NamedCompositeVector, NamedCompositeMatrix, update!

export AkimaSplineComponent

struct AkimaSplineComponent <: AbstractExplicitComp
    num_control_points::Int
    num_points::Int
    vec_size::Int
    name::String
    input_x::Bool
    input_xcp::Bool
    units::Union{String, Nothing}
    x_units::Union{String, Nothing}
    delta_x::Float64
    eval_at::String
    eps::Float64
    x_cp_grid::Array{Float64}
    x_grid::Array{Float64}
end

function AkimaSplineComponent(; num_control_points, num_points, vec_size=1, name="var", input_x=false, input_xcp=false, units=nothing, x_units=nothing, delta_x=0.1, eval_at="end", eps=1e-30)

    if ! input_xcp
        x_cp_grid = collect(range(0.0, 1.0, length=num_control_points))
        x_cp_grid = copy(reshape(x_cp_grid, (1, 1, num_control_points)))
    else
        x_cp_grid = Vector{Float64}(undef, 0)
    end

    if ! input_x
        if eval_at == "cell_center"
            x_ends = range(0.0, 1.0, length=num_points)
            x_grid = collect(0.5*(x_ends[1:end-1] + x_ends[2:end]))
        elseif eval_at == "end"
            x_grid = collect(range(0.0, 1.0, length=num_points))
        else
            throw(ArgumentError("invalid eval_at value $(eval_at)"))
        end
        x_grid = copy(reshape(x_grid, (1, num_points, 1)))
    else
        x_grid = Vector{Float64}(undef, 0)
    end

    return AkimaSplineComponent(num_control_points, num_points, vec_size, name, input_x, input_xcp, units, x_units, delta_x, eval_at, eps, x_cp_grid, x_grid)
end

function OpenMDAO.setup(self::AkimaSplineComponent)

    num_control_points = self.num_control_points
    num_points = self.num_points
    vec_size = self.vec_size
    name = self.name
    input_x = self.input_x
    input_xcp = self.input_xcp
    units = self.units
    x_units = self.x_units
    x_cp_grid = self.x_cp_grid
    x_grid = self.x_grid

    x_name = self.name * ":x"
    xcp_name = self.name * ":x_cp"
    y_name = self.name * ":y"
    ycp_name = self.name * ":y_cp"

    inputs = VarData[]
    outputs = VarData[]

    if input_xcp
        # push!(inputs, VarData(xcp_name, shape=num_control_points, val=rand(num_control_points), units=x_units))
        push!(inputs, VarData(xcp_name, shape=(1, 1, num_control_points), val=rand(1, 1, num_control_points), units=x_units))
    else
        # push!(outputs, VarData(xcp_name, shape=num_control_points, val=x_cp_grid, units=x_units))
        push!(outputs, VarData(xcp_name, shape=(1, 1, num_control_points), val=x_cp_grid, units=x_units))
    end

    if input_x
        # push!(inputs, VarData(x_name, shape=num_points, val=rand(num_points), units=x_units))
        push!(inputs, VarData(x_name, shape=(1, num_points, 1), val=rand(1, num_points, 1), units=x_units))
    else
        # push!(outputs, VarData(x_name, shape=num_points, val=x_grid, units=x_units))
        push!(outputs, VarData(x_name, shape=(1, num_points, 1), val=x_grid, units=x_units))
    end

    # push!(inputs, VarData(ycp_name, shape=(vec_size, num_control_points), val=rand(vec_size, num_control_points), units=units))
    # push!(inputs, VarData(ycp_name, shape=(vec_size, 1, num_control_points), val=rand(vec_size, num_control_points), units=units))
    push!(inputs, VarData(ycp_name, shape=(vec_size, 1, num_control_points), val=rand(vec_size, 1, num_control_points), units=units))

    # push!(outputs, VarData(y_name, shape=(vec_size, num_points), val=rand(vec_size, num_points), units=units))
    # push!(outputs, VarData(y_name, shape=(vec_size, num_points, 1), val=rand(vec_size, num_points), units=units))
    push!(outputs, VarData(y_name, shape=(vec_size, num_points, 1), val=rand(vec_size, num_points, 1), units=units))

    # Derivatives
    partials = PartialsData[]

    ss_sizes = Dict(:j=>vec_size, :k=>num_points, :l=>num_control_points, :s=>1)

    # rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:j, :k], wrt_ss=[:j, :l])
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:j, :k, :s], wrt_ss=[:j, :s, :l])
    push!(partials, PartialsData(y_name, ycp_name, rows=rows, cols=cols))

    if input_xcp
        # rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:j, :k], wrt_ss=[:l])
        rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:j, :k, :s], wrt_ss=[:s, :s, :l])
        push!(partials, PartialsData(y_name, xcp_name, rows=rows, cols=cols))
    end

    if input_x
        # rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:j, :k], wrt_ss=[:k])
        rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:j, :k, :s], wrt_ss=[:s, :k, :s])
        push!(partials, PartialsData(y_name, x_name, rows=rows, cols=cols))
    end

    return inputs, outputs, partials
end

function OpenMDAO.compute!(self::AkimaSplineComponent, inputs, outputs)
    delta_x = self.delta_x
    xcp = self.input_xcp ? inputs[self.name*":x_cp"] : self.x_cp_grid
    x = self.input_x ? inputs[self.name*":x"] : self.x_grid
    ycp = inputs[self.name*":y_cp"]
    y = outputs[self.name*":y"]

    splines = akima_setup.(Ref(xcp[1, 1, :]), eachrow(ycp[:, 1, :]), delta_x)
    y .= akima_interp.(x[:, :, 1], splines)

end

function OpenMDAO.compute_partials!(self::AkimaSplineComponent, inputs, partials)
    num_points = self.num_points
    num_control_points = self.num_control_points
    vec_size = self.vec_size

    x_name = self.name * ":x"
    xcp_name = self.name * ":x_cp"
    y_name = self.name * ":y"
    ycp_name = self.name * ":y_cp"

    delta_x = self.delta_x
    # x_reshape = reshape(x, 1, :)

    # function wrt_ycp!(Y, X)
    #     spline = akima_setup(xcp, X, delta_x)
    #     Y .= akima_interp(x, spline)
    # end

    # dy_dycp = partials[y_name, ycp_name]
    # dy_dycp = reshape(dy_dycp, num_control_points, num_points, vec_size)
    # dy_dycp = permutedims(dy_dycp, (3, 2, 1))
    # for v in 1:vec_size
    #     dy_dycp[v, :, :] .= jacobian(wrt_ycp!, y[v, :], ycp[v, :])
    # end

    # if self.input_xcp
    #     function wrt_xcp!(Y, X)
    #         splines = akima_setup.(Ref(X), eachrow(ycp), delta_x)
    #         Y .= akima_interp(x_reshape, splines)
    #     end

    #     dy_dxcp = partials[y_name, xcp_name]
    #     dy_dxcp = reshape(dy_dxcp, num_control_points, num_points, vec_size)
    #     dy_dxcp = permutedims(dy_dxcp, (3, 2, 1))
    #     dy_dxcp .= jacobian(wrt_xcp!, y, xcp)
    # end

    # if self.input_x
    #     function wrt_x!(Y, X)
    #         splines = akima_setup.(Ref(xcp), eachrow(ycp), delta_x)
    #         X_reshape = reshape(X, 1, :)
    #         Y .= akima_interp.(x_reshape, splines)
    #     end

    #     dy_dx = partials[y_name, x_name]
    #     dy_dx = reshape(dy_dx, num_points, vec_size)
    #     dy_dx = permutedims(dy_dx, (2, 1))
    #     dy_dx .= jacobian(wrt_x!, y, x)

    # end
    #
    function foo!(Y, X)
        splines = akima_setup.(Ref(X.ddata["xcp"][1, 1, :]), eachrow(X.ddata["ycp"][:, 1, :]), delta_x)
        # @show size(X.ddata["x"])
        # @show size(splines)
        # @show size(Y.ddata["y"])
        Y.ddata["y"] .= akima_interp.(X.ddata["x"], splines)
    end

    xcp = self.input_xcp ? inputs[xcp_name] : self.x_cp_grid
    x = self.input_x ? inputs[x_name] : self.x_grid
    ycp = inputs[ycp_name]
    y = zeros(eltype(ycp), (vec_size, num_points, 1))

    # xcp = reshape(xcp, (1, 1, num_control_points))
    # ycp = reshape(ycp, (vec_size, 1, num_control_points))
    # x = reshape(x, (1, num_points, 1))
    # y = reshape(y, (vec_size, num_points, 1))
    inputs_ncv = NamedCompositeVector([xcp, x, ycp], ["xcp", "x", "ycp"])
    outputs_ncv = NamedCompositeVector([y], ["y"])
    # derivs = jacobian(foo!, outputs_ncv, inputs_ncv)
    # @show size(derivs)
    # println("size(derivs) = $(size(derivs)) (should be $((vec_size*num_points, num_control_points+vec_size*num_control_points+num_points)))")

    # @show valtype(partials), typeof(partials)
    # @show typeof(partials[(y_name, ycp_name)])
    # @show size(partials[(y_name, ycp_name)])
    # @show PyArray{Float64, 2} <: AbstractMatrix{Float64}
    # @show valtype(partials) <: AbstractMatrix{Float64}
    # for (k, v) in partials
    #     @show k, typeof(v)
    # end
    # partials_ncm = NamedCompositeMatrix(valtype(partials), outputs_ncv.sizes, inputs_ncv.sizes, ["y"], ["xcp", "x", "ycp"])
    # update!(partials_ncm, partials)
    # jacobian!(partials_ncm, foo!, outputs_ncv, inputs_ncv)

    out = jacobian(foo!, outputs_ncv, inputs_ncv)
    @show typeof(out), size(out)

    dy_dycp = partials[y_name, ycp_name]
    dy_dycp = reshape(dy_dycp, (num_control_points, num_points, vec_size))
    dy_dycp = permutedims(dy_dycp, [3, 2, 1])

    idx_i = LinearIndices((vec_size, num_points))
    for idx in CartesianIndices((vec_size, num_points, num_control_points))
        # Need to go from 
        i = 
    end

    dy_dycp .= out[:, num_control_points+num_points+1:num_control_points+num_points+num_control_points]

end
