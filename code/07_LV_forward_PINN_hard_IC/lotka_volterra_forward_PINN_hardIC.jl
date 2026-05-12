using DifferentialEquations
using Lux
using Optimisers
using Plots
using Printf
using Random
using Statistics
using Zygote

ENV["GKSwstype"] = "100"
Plots.default(show=false)

# ============================================================
# PINN con condición inicial exacta (hard constraint)
# para las ecuaciones de Lotka-Volterra
# ============================================================
#
# Problema con el enfoque soft (06): u = 0 es punto fijo de la ODE,
# por lo que la red puede satisfacer ambos términos de la pérdida de
# manera trivial colapsando a cero.
#
# Solución — ansatz que garantiza la IC algebraicamente:
#
#   u_θ(t) = u₀  +  t · NN(t)
#
# Propiedad clave:   u_θ(0) = u₀  para cualquier valor de θ.
# La red sólo necesita aprender la "corrección" a partir de t = 0.
#
# Derivada por regla del producto:
#   du_θ/dt = NN(t) + t · dNN/dt
# (aproximada por diferencias finitas sobre el ansatz completo)
#
# Función de pérdida — sólo residuo físico, sin término IC:
#
#   L = (1/N) Σᵢ ‖du_θ/dt(tᵢ) − f(u_θ(tᵢ))‖²
#
# Sistema de Lotka-Volterra:
#   dx/dt =  α x − β x y     (presas)
#   dy/dt =  δ x y − γ y     (depredadores)
# ============================================================

# ---------------------------------------------------------------------------
# Parámetros y condición inicial
# ---------------------------------------------------------------------------
const α = 1.0;   const β = 0.1
const δ = 0.075; const γ = 1.5
const u0 = [10.0, 5.0]
const tspan = (0.0, 15.0)

# ---------------------------------------------------------------------------
# Solución de referencia (ODE numérica con Tsit5)
# ---------------------------------------------------------------------------
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] =  α * x - β * x * y
    du[2] =  δ * x * y - γ * y
end

prob_ref = ODEProblem(lotka_volterra!, u0, tspan, [α, β, δ, γ])
sol_ref  = solve(prob_ref, Tsit5(), saveat=0.05)
println("✓ Solución de referencia calculada ($(length(sol_ref.t)) puntos).")

# ---------------------------------------------------------------------------
# Arquitectura de la red neuronal
#
#   NN : t ∈ [0,1] ↦ ℝ²   (entrada escalada, salida sin restricción)
#
# La red NO es la solución directamente — es la corrección NN(t) en:
#   u_θ(t) = u₀ + t · NN(t)
# ---------------------------------------------------------------------------
t0, T = tspan
scale_layer_inp = WrappedFunction(t -> (t .- t0) ./ (T - t0))
scale_layer_out = WrappedFunction(x -> 10.0 .* x)

nn = Chain(
    scale_layer_inp,
    Dense(1 => 5, tanh),
    Dense(5 => 10, tanh),
    Dense(10 => 10, tanh),
    Dense(10 => 2),
    scale_layer_out
)

rng    = MersenneTwister(666)
ps, st = Lux.setup(rng, nn)
st     = Lux.testmode(st)

n_params = sum(p -> sum(length, values(p); init=0), values(ps))
println("✓ Red neuronal creada: $n_params parámetros.")

# ---------------------------------------------------------------------------
# Ansatz con IC exacta
#
#   u_θ(t) = u₀ + t · NN(t)
#
# Garantiza u_θ(0) = u₀ para cualquier θ, eliminando el término IC de la
# pérdida y bloqueando el colapso a cero (u=0 ya no es solución factible).
# ---------------------------------------------------------------------------
function u_ansatz(t_val, ps, st)
    raw = nn([t_val], ps, st)[1]   # NN(t) ∈ ℝ²
    return u0 .+ t_val .* raw      # u₀ + t·NN(t)
end

# Derivada temporal del ansatz por diferencias finitas centrales
function u_ansatz_dudt(t_val, ps, st; h=1e-4)
    u_fwd = u_ansatz(t_val + h, ps, st)
    u_bwd = u_ansatz(t_val - h, ps, st)
    return (u_fwd .- u_bwd) ./ (2h)
end

# ---------------------------------------------------------------------------
# Puntos de colocación
# ---------------------------------------------------------------------------
N_col = 100
t_col = [
    10.0 .^ collect(range(log10(tspan[1] + 1e-10), log10(tspan[2]), length=div(N_col, 2)));
    collect(range(tspan[1], tspan[2], length=div(N_col, 2)))
]

# ---------------------------------------------------------------------------
# Lado derecho de la ODE
# ---------------------------------------------------------------------------
function lv_rhs(u)
    x, y = u[1], u[2]
    return [α * x - β * x * y,
            δ * x * y - γ * y]
end

# ---------------------------------------------------------------------------
# Función de pérdida — sólo residuo físico
# (la IC está garantizada por el ansatz, no necesita penalizarse)
# ---------------------------------------------------------------------------
function pinn_loss(ps)
    phys_residuals = map(t_col) do t_i
        u    = u_ansatz(t_i, ps, st)
        dudt = u_ansatz_dudt(t_i, ps, st)
        rhs  = lv_rhs(u)
        sum((dudt .- rhs) .^ 2)
    end
    return mean(phys_residuals)
end

# ---------------------------------------------------------------------------
# Ciclo de entrenamiento
# ---------------------------------------------------------------------------
n_epochs    = 100000
opt_state   = Optimisers.setup(Adam(1e-3), ps)
phys_losses = Float64[]

println("\nEntrenando PINN con IC exacta ($n_epochs épocas, $N_col puntos de colocación)...")
println("─"^60)

for epoch in 1:n_epochs
    global ps, opt_state
    loss_val, grads = Zygote.withgradient(pinn_loss, ps)
    opt_state, ps   = Optimisers.update(opt_state, ps, grads[1])
    push!(phys_losses, loss_val)

    if epoch % 1000 == 0
        @printf("  Época %6d │ L_phys: %10.3e\n", epoch, loss_val)
    end
end

println("─"^60)
@printf("✓ Finalizado. L_phys final: %.3e\n\n", phys_losses[end])

# Verificar que la IC se satisface exactamente
u_ic_check = u_ansatz(tspan[1], ps, st)
@printf("IC check — u_θ(0) = [%.6f, %.6f]  (debe ser u₀ = %s)\n\n",
        u_ic_check[1], u_ic_check[2], string(u0))

# ---------------------------------------------------------------------------
# Evaluación en grilla fina
# ---------------------------------------------------------------------------
t_fine  = collect(range(tspan[1], tspan[2], length=600))
u_pinn  = hcat([u_ansatz(t, ps, st) for t in t_fine]...)

phys_err = map(t_fine) do t_i
    u    = u_ansatz(t_i, ps, st)
    dudt = u_ansatz_dudt(t_i, ps, st)
    rhs  = lv_rhs(u)
    sqrt(sum((dudt .- rhs) .^ 2))
end

x_ref_interp = [sol_ref(t)[1] for t in t_fine]
y_ref_interp = [sol_ref(t)[2] for t in t_fine]
err_x = abs.(u_pinn[1, :] .- x_ref_interp)
err_y = abs.(u_pinn[2, :] .- y_ref_interp)

@printf("Error L∞ en x(t): %.4f\n", maximum(err_x))
@printf("Error L∞ en y(t): %.4f\n", maximum(err_y))

# ===========================================================================
# Figuras
# ===========================================================================

# ---------------------------------------------------------------------------
# Figura 1: Comparación de soluciones
# ---------------------------------------------------------------------------
fig1 = plot(
    sol_ref.t, sol_ref[1, :],
    label="Presas — ODE (Tsit5)", linewidth=2.5, color=:royalblue,
    xlabel="Tiempo", ylabel="Población",
    title="PINN (IC exacta) vs Solución Numérica\nLotka-Volterra  (α=$α, β=$β, δ=$δ, γ=$γ)",
    legend=:topright, size=(750, 430)
)
plot!(fig1, sol_ref.t, sol_ref[2, :],
      label="Depredadores — ODE (Tsit5)", linewidth=2.5, color=:firebrick)
plot!(fig1, t_fine, u_pinn[1, :],
      label="Presas — PINN (hard IC)", linewidth=2, color=:dodgerblue, linestyle=:dash)
plot!(fig1, t_fine, u_pinn[2, :],
      label="Depredadores — PINN (hard IC)", linewidth=2, color=:orangered, linestyle=:dash)
scatter!(fig1, [tspan[1]], [u0[1]],
         label="u₀ (exacto por construcción)", color=:black,
         markersize=7, markershape=:circle, markerstrokewidth=0)
scatter!(fig1, [tspan[1]], [u0[2]],
         label=false, color=:black, markersize=7, markershape=:circle, markerstrokewidth=0)
scatter!(fig1, t_col, zeros(length(t_col)),
         label="Puntos de colocación", color=:gray40, markersize=5,
         markershape=:xcross, markerstrokewidth=1.5, markerstrokecolor=:gray40)
savefig(fig1, "lv_pinn_hardIC_solution.png")
println("Figura guardada: lv_pinn_hardIC_solution.png")

# ---------------------------------------------------------------------------
# Figura 2: Curva de entrenamiento (sólo pérdida física)
# ---------------------------------------------------------------------------
fig2 = plot(
    phys_losses, label="L_phys  (residuo ODE)", color=:darkorange, linewidth=2,
    xlabel="Época", ylabel="Loss (escala log)",
    title="Curva de entrenamiento — PINN con IC exacta\n(N_col = $N_col,  IC garantizada por ansatz)",
    yscale=:log10, legend=:topright, size=(700, 420)
)
savefig(fig2, "lv_pinn_hardIC_loss.png")
println("Figura guardada: lv_pinn_hardIC_loss.png")

# ---------------------------------------------------------------------------
# Figura 3: Espacio de fase
# ---------------------------------------------------------------------------
fig3 = plot(
    sol_ref[1, :], sol_ref[2, :],
    label="ODE (Tsit5)", linewidth=2.5, color=:steelblue,
    xlabel="Presas  x(t)", ylabel="Depredadores  y(t)",
    title="Espacio de fase — PINN con IC exacta",
    legend=:topright, size=(550, 480)
)
plot!(fig3, u_pinn[1, :], u_pinn[2, :],
      label="PINN (hard IC)", linewidth=2, color=:orangered, linestyle=:dash)
scatter!(fig3, [u0[1]], [u0[2]],
         label="Condición inicial (exacta)", color=:black,
         markersize=9, markershape=:star5, markerstrokewidth=0)
savefig(fig3, "lv_pinn_hardIC_phase.png")
println("Figura guardada: lv_pinn_hardIC_phase.png")

# ---------------------------------------------------------------------------
# Figura 4: Residuo físico y error puntual
# ---------------------------------------------------------------------------
fig4 = plot(layout=(2, 1), size=(720, 550))

plot!(fig4[1], t_fine, phys_err,
      label="‖du/dt − f(u)‖₂", color=:darkgreen, linewidth=2,
      xlabel="", ylabel="Residuo físico",
      title="Residuo ODE y Error Puntual — PINN con IC exacta")

plot!(fig4[2], t_fine, err_x,
      label="|x_PINN − x_ODE|", color=:royalblue, linewidth=2,
      xlabel="Tiempo", ylabel="Error absoluto")
plot!(fig4[2], t_fine, err_y,
      label="|y_PINN − y_ODE|", color=:firebrick, linewidth=2)

savefig(fig4, "lv_pinn_hardIC_error.png")
println("Figura guardada: lv_pinn_hardIC_error.png")

# ---------------------------------------------------------------------------
# Figura 5: Componentes separadas con bandas de error
# ---------------------------------------------------------------------------
fig5 = plot(layout=(1, 2), size=(900, 380),
            plot_title="PINN (IC exacta) — Presas y Depredadores")

plot!(fig5[1],
      sol_ref.t, sol_ref[1, :],
      label="ODE", linewidth=2, color=:royalblue,
      xlabel="Tiempo", ylabel="x(t)  (presas)", title="Presas")
plot!(fig5[1], t_fine, u_pinn[1, :],
      label="PINN", linewidth=2, color=:dodgerblue, linestyle=:dash)
plot!(fig5[1], t_fine, u_pinn[1, :] .+ err_x, fillrange=u_pinn[1, :] .- err_x,
      alpha=0.15, color=:dodgerblue, label=false)

plot!(fig5[2],
      sol_ref.t, sol_ref[2, :],
      label="ODE", linewidth=2, color=:firebrick,
      xlabel="Tiempo", ylabel="y(t)  (depredadores)", title="Depredadores")
plot!(fig5[2], t_fine, u_pinn[2, :],
      label="PINN", linewidth=2, color=:orangered, linestyle=:dash)
plot!(fig5[2], t_fine, u_pinn[2, :] .+ err_y, fillrange=u_pinn[2, :] .- err_y,
      alpha=0.15, color=:orangered, label=false)

savefig(fig5, "lv_pinn_hardIC_components.png")
println("Figura guardada: lv_pinn_hardIC_components.png")

println("\n¡Listo! Se generaron 5 figuras.")
