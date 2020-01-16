using OpenMDAO
import PyCall
using PyPlot

num_control_points = 8
num_points = 100
vec_size = 3
num_comps = 4

random(n) = 2 .*rand(Float64, n) .- 1

i = range(0.0, 1.0, length=num_control_points)
i = reshape(collect(i), 1, 1, :)
y_cp = sin.(2*pi*i) .+ 0.3 .* random((vec_size, 1, num_control_points))

x_cp = range(0.0, 2.0, length=num_control_points)
x_cp = reshape(x_cp, (1, 1, num_control_points))

x = range(0.5, 0.75, length=num_points)
x = reshape(x, (1, num_points, 1))

prob = om.Problem()
ivc = om.IndepVarComp()
ivc.add_output("var:x_cp", x_cp)
ivc.add_output("var:x", x)
ivc.add_output("var:y_cp", y_cp)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

name = "akima_comp1"
comp = make_component(AkimaSplineComponent(num_control_points=num_control_points, num_points=num_points, vec_size=vec_size))
prob.model.add_subsystem(name, comp)
prob.model.connect("var:y_cp", name*".var:y_cp")

if num_comps >= 2
    name = "akima_comp2"
    comp = make_component(AkimaSplineComponent(num_control_points=num_control_points, num_points=num_points, vec_size=vec_size, input_xcp=true))
    prob.model.add_subsystem(name, comp)
    prob.model.connect("var:x_cp", name*".var:x_cp")
    prob.model.connect("var:y_cp", name*".var:y_cp")
end

if num_comps >= 3
    name = "akima_comp3"
    comp = make_component(AkimaSplineComponent(num_control_points=num_control_points, num_points=num_points, vec_size=vec_size, input_x=true))
    prob.model.add_subsystem(name, comp)
    prob.model.connect("var:x", name*".var:x")
    prob.model.connect("var:y_cp", name*".var:y_cp")
end

if num_comps >= 4
    name = "akima_comp4"
    comp = make_component(AkimaSplineComponent(num_control_points=num_control_points, num_points=num_points, vec_size=vec_size, input_xcp=true, input_x=true))
    prob.model.add_subsystem(name, comp)
    prob.model.connect("var:x_cp", name*".var:x_cp")
    prob.model.connect("var:x", name*".var:x")
    prob.model.connect("var:y_cp", name*".var:y_cp")
end

prob.setup()
prob.run_model()
prob.check_partials()

colors = ["red", "green", "blue", "purple"]
fig, axes = subplots(nrows=num_comps, sharex=true)
if num_comps == 1
    axes = [axes]
end
for i in 1:num_comps
    name = "akima_comp$(i)"
    x_cp = prob.get_val(name*".var:x_cp")
    y_cp = prob.get_val(name*".var:y_cp")
    x = prob.get_val(name*".var:x")
    y = prob.get_val(name*".var:y")
    for v in 1:vec_size
        color = colors[v]
        axes[i].plot(x_cp[1, 1, :], y_cp[v, 1, :], linestyle="None", marker="o", markerfacecolor="None", markeredgecolor=color)
        axes[i].plot(x[1, :, 1], y[v, :, 1], color=color)
    end
    # axes[i].legend()
end
fig.savefig("akima.png")
