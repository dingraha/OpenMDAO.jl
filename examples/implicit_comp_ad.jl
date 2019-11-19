using OpenMDAO
using PyCall

om = pyimport("openmdao.api")

mutable struct SquareIt{TF} <: OpenMDAO.AbstractADImplicitComp
    a::TF
    ad_data::ImplicitAutoDiffData
    SquareIt{TF}(a, ad_data) where {TF} = new(a, ad_data)
    SquareIt{TF}(a) where {TF} = new(a)
end

function SquareIt(a::TF) where {TF}
    self = SquareIt{TF}(a)
    declare_ad!(self)
    return self
end

function OpenMDAO.setup(self::SquareIt)
    inputs = VarData[]
    push!(inputs, VarData("x", (1,), 2.0))
    push!(inputs, VarData("y", (1,), 3.0))

    outputs = VarData[]
    push!(outputs, VarData("z1", (1,), 2.0))
    push!(outputs, VarData("z2", (1,), 3.0))

    partials = PartialsData[]
    push!(partials, PartialsData("z1", "x"))
    push!(partials, PartialsData("z1", "y"))
    push!(partials, PartialsData("z1", "z1"))
    push!(partials, PartialsData("z1", "z2"))
    push!(partials, PartialsData("z2", "x"))
    push!(partials, PartialsData("z2", "y"))
    push!(partials, PartialsData("z2", "z1"))
    push!(partials, PartialsData("z2", "z2"))

    return inputs, outputs, partials
end

function OpenMDAO.apply_nonlinear!(self::SquareIt, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]
    @. residuals["z1"] = outputs["z1"] - (a*x*x + y*y)
    @. residuals["z2"] = outputs["z2"] - (a*x + y)
end

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(SquareIt(4.0))
comp.linear_solver = om.DirectSolver(assemble_jac=true)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=true, iprint=2, err_on_non_converge=true)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
