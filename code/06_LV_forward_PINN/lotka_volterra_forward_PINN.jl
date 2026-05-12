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
# PINN — Red Neuronal Informada por la Física
# para las ecuaciones de Lotka-Volterra
# ============================================================
#
# Idea central: la red neuronal ES la solución.
#
#   u_θ : t ↦ [x(t), y(t)]        (presas, depredadores)
#
# Función de pérdida:
#
#   L = λ · L_IC + L_phys
#
#   L_IC   = ‖u_θ(t₀) − u₀‖²                         (condición inicial)
#   L_phys = (1/N) Σᵢ ‖du_θ/dt(tᵢ) − f(u_θ(tᵢ))‖²  (residuo físico)
#
# Sistema de Lotka-Volterra:
#   dx/dt =  α x − β x y     (presas)
#   dy/dt =  δ x y − γ y     (depredadores)
#
# Parámetros fijos (sin inversión):
#   α = 1.0,  β = 0.1,  δ = 0.075,  γ = 1.5
# ============================================================

# ---------------------------------------------------------------------------
# Parámetros y condición inicial
# ---------------------------------------------------------------------------
const α = 1.0;   const β = 0.1
const δ = 0.075; const γ = 1.5
const u0 = [10.0, 5.0]          # [presas₀, depredadores₀]
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
#   Entrada: t ∈ ℝ (tiempo escalar, pasado como vector de longitud 1)
#   Salida:  [x(t), y(t)] ∈ ℝ²
#
#   Primera capa: escalado lineal t ↦ (t − t₀)/(T − t₀) ∈ [0, 1]
#   Capas ocultas: tanh — elección estándar para PINNs por su suavidad infinita
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
st     = Lux.testmode(st)   # desactiva comportamientos de entrenamiento (dropout, etc.)

n_params = sum(p -> sum(length, values(p); init=0), values(ps))
println("✓ Red neuronal creada: $n_params parámetros.")

# ---------------------------------------------------------------------------
# Puntos de colocación: dónde se impone el residuo físico
# ---------------------------------------------------------------------------
N_col = 100
# t_col = collect(range(tspan[1], tspan[2], length=N_col))
# t_col = 10.0.^collect(range(log10(tspan[1] + 1e-10), log10(tspan[2]), length = N_col))
t_col = [
    10.0.^collect(range(log10(tspan[1] + 1e-10), log10(tspan[2]), length = div(N_col, 2)));
    collect(range(tspan[1], tspan[2], length=div(N_col, 2)))
]


# ---------------------------------------------------------------------------
# Lado derecho de las ecuaciones (f en du/dt = f(u))
# ---------------------------------------------------------------------------
function lv_rhs(u)
    x, y = u[1], u[2]
    return [α * x - β * x * y,
            δ * x * y - γ * y]
end

# ---------------------------------------------------------------------------
# Derivada temporal de la red usando diferencias finitas centrales
#
#   du_θ/dt(t) ≈ [u_θ(t+h) − u_θ(t−h)] / (2h)
#
# La aproximación de 2do orden (O(h²)) es suficiente para el entrenamiento.
# Zygote puede diferenciar esta expresión con respecto a los parámetros θ
# sin necesidad de diferenciación anidada.
# ---------------------------------------------------------------------------
function nn_dudt(t_val, ps, st; h=1e-4)
    u_fwd = nn([t_val + h], ps, st)[1]
    u_bwd = nn([t_val - h], ps, st)[1]
    return (u_fwd .- u_bwd) ./ (2h)
end

# ---------------------------------------------------------------------------
# Función de pérdida del PINN
# ---------------------------------------------------------------------------
const λ_physics = 1.0   # peso relativo de la pérdida física

# Returns (ic_loss, λ·phys_loss) separately — useful for logging and plotting.
function pinn_loss_components(ps)
    u_ic    = nn([tspan[1]], ps, st)[1]
    ic_loss = sum((u_ic .- u0) .^ 2)

    phys_residuals = map(t_col) do t_i
        u    = nn([t_i], ps, st)[1]
        dudt = nn_dudt(t_i, ps, st)
        rhs  = lv_rhs(u)
        sum((dudt .- rhs) .^ 2)
    end
    phys_loss_weighted = λ_physics * mean(phys_residuals)

    return ic_loss, phys_loss_weighted
end

pinn_loss(ps) = sum(pinn_loss_components(ps))

# ---------------------------------------------------------------------------
# Ciclo de entrenamiento
# ---------------------------------------------------------------------------
n_epochs    = 100000
opt_state   = Optimisers.setup(Adam(1e-3), ps)
losses      = Float64[]
ic_losses   = Float64[]
phys_losses = Float64[]

println("\nEntrenando PINN ($n_epochs épocas, $N_col puntos de colocación)...")
println("─"^72)

for epoch in 1:n_epochs
    global ps, opt_state
    loss_val, grads = Zygote.withgradient(pinn_loss, ps)
    opt_state, ps   = Optimisers.update(opt_state, ps, grads[1])

    ic_l, phys_l = pinn_loss_components(ps)
    push!(losses,      ic_l + phys_l)
    push!(ic_losses,   ic_l)
    push!(phys_losses, phys_l)

    if epoch % 1000 == 0
        @printf("  Época %6d │ Total: %10.3e │ IC: %10.3e │ Física (×λ): %10.3e\n",
                epoch, ic_l + phys_l, ic_l, phys_l)
    end
end

println("─"^72)
@printf("✓ Finalizado. Total: %.3e │ IC: %.3e │ Física (×λ): %.3e\n\n",
        losses[end], ic_losses[end], phys_losses[end])

# ---------------------------------------------------------------------------
# Evaluación: predicción en una grilla fina
# ---------------------------------------------------------------------------
t_fine  = collect(range(tspan[1], tspan[2], length=600))
u_pinn  = hcat([nn([t], ps, st)[1] for t in t_fine]...)

# Residuo físico punto a punto (para visualizar qué tan bien se satisface la ODE)
phys_err = map(t_fine) do t_i
    u    = nn([t_i], ps, st)[1]
    dudt = nn_dudt(t_i, ps, st)
    rhs  = lv_rhs(u)
    sqrt(sum((dudt .- rhs) .^ 2))
end

# Error puntual respecto a la solución de referencia (interpolada)
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
    title="PINN vs Solución Numérica\nLotka-Volterra  (α=$α, β=$β, δ=$δ, γ=$γ)",
    legend=:topright, size=(750, 430)
)
plot!(fig1, sol_ref.t, sol_ref[2, :],
      label="Depredadores — ODE (Tsit5)", linewidth=2.5, color=:firebrick)
plot!(fig1, t_fine, u_pinn[1, :],
      label="Presas — PINN", linewidth=2, color=:dodgerblue,
      linestyle=:dash)
plot!(fig1, t_fine, u_pinn[2, :],
      label="Depredadores — PINN", linewidth=2, color=:orangered,
      linestyle=:dash)
scatter!(fig1, [tspan[1]], [u0[1]],
         label="u₀", color=:black, markersize=7, markershape=:circle, markerstrokewidth=0)
scatter!(fig1, [tspan[1]], [u0[2]],
         label=false, color=:black, markersize=7, markershape=:circle, markerstrokewidth=0)
# Collocation points as crosses on the x-axis (y=0)
scatter!(fig1, t_col, zeros(length(t_col)),
         label="Puntos de colocación", color=:gray40, markersize=5,
         markershape=:xcross, markerstrokewidth=1.5, markerstrokecolor=:gray40)
savefig(fig1, "lv_pinn_solution.png")
println("Figura guardada: lv_pinn_solution.png")

# ---------------------------------------------------------------------------
# Figura 2: Curva de entrenamiento
# ---------------------------------------------------------------------------
fig2 = plot(
    losses, label="Loss total", color=:mediumpurple, linewidth=2,
    xlabel="Época", ylabel="Loss (escala log)",
    title="Curva de entrenamiento del PINN\n(λ_physics = $λ_physics,  N_col = $N_col)",
    yscale=:log10, legend=:topright, size=(700, 420)
)
plot!(fig2, ic_losses,   label="L_IC  =  ‖u_θ(t₀) − u₀‖²",
      color=:crimson, linewidth=2, linestyle=:dash)
plot!(fig2, phys_losses, label="λ · L_phys  (residuo ODE)",
      color=:darkorange, linewidth=2, linestyle=:dot)
savefig(fig2, "lv_pinn_loss.png")
println("Figura guardada: lv_pinn_loss.png")

# ---------------------------------------------------------------------------
# Figura 3: Espacio de fase
# ---------------------------------------------------------------------------
fig3 = plot(
    sol_ref[1, :], sol_ref[2, :],
    label="ODE (Tsit5)", linewidth=2.5, color=:steelblue,
    xlabel="Presas  x(t)", ylabel="Depredadores  y(t)",
    title="Espacio de fase — Lotka-Volterra",
    legend=:topright, size=(550, 480)
)
plot!(fig3, u_pinn[1, :], u_pinn[2, :],
      label="PINN", linewidth=2, color=:orangered, linestyle=:dash)
scatter!(fig3, [u0[1]], [u0[2]],
         label="Condición inicial", color=:black, markersize=9,
         markershape=:star5, markerstrokewidth=0)
savefig(fig3, "lv_pinn_phase.png")
println("Figura guardada: lv_pinn_phase.png")

# ---------------------------------------------------------------------------
# Figura 4: Residuo físico y error puntual
# ---------------------------------------------------------------------------
fig4 = plot(layout=(2, 1), size=(720, 550))

plot!(fig4[1], t_fine, phys_err,
      label="‖du/dt − f(u)‖₂", color=:darkgreen, linewidth=2,
      xlabel="", ylabel="Residuo físico",
      title="Residuo ODE e Error Puntual del PINN")

plot!(fig4[2], t_fine, err_x,
      label="|x_PINN − x_ODE|", color=:royalblue, linewidth=2,
      xlabel="Tiempo", ylabel="Error absoluto")
plot!(fig4[2], t_fine, err_y,
      label="|y_PINN − y_ODE|", color=:firebrick, linewidth=2)

savefig(fig4, "lv_pinn_error.png")
println("Figura guardada: lv_pinn_error.png")

# ---------------------------------------------------------------------------
# Figura 5 (bonus): Componentes separadas con bandas de error
# ---------------------------------------------------------------------------
fig5 = plot(layout=(1, 2), size=(900, 380),
            plot_title="PINN — Presas y Depredadores por separado")

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

savefig(fig5, "lv_pinn_components.png")
println("Figura guardada: lv_pinn_components.png")

println("\n¡Listo! Se generaron 5 figuras.")
