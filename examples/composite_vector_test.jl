using ForwardDiff
include("composite_vector.jl")

x = convert(Array{Float64}, reshape(collect(1:2*5), 5, 2))
y = convert(Array{Float64}, reshape(collect(1:1*5), 5, 1))
z1 = zeros(5, 2)
z2 = zeros(5, 2)

# cv = CompositeVector([x, y])

# @show cv
# @show typeof(cv)
# for e in cv
#     @show e
# end
# cv2 = similar(cv)

# ncv = NamedCompositeVector([x, y], ["x", "y"])
# @show ncv

# @show ncv.ddata["x"]
# ncv.ddata["x"][1, 1] = -5
# @show ncv.data

function compute!(inputs, outputs)
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = x*x + y*y
    @. outputs["z2"] = x + y
end

function compute_partials!(inputs, outputs)
    # Need a CompositeVector for the inputs and the outputs, right?
    inputs_ncv = NamedCompositeVector(inputs)
    outputs_ncv = NamedCompositeVector(outputs)

    function compute_wrapped!(y, x)
        compute!(x.ddata, y.ddata)
        return nothing
    end
    
    partials = ForwardDiff.jacobian(compute_wrapped!, outputs_ncv, inputs_ncv)
    return partials
end


foo = Dict("x" => x, "y" => y)
bar = Dict("z1" => z1, "z2" => z2)

@show foo
@show bar

partials = compute_partials!(foo, bar)
@show partials
@show typeof(partials)
@show size(partials)
