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
# PINN Inverso — Lotka-Volterra
# ============================================================
#
# Problema: dados datos observados de una trayectoria de LV,
# aprender simultáneamente:
#   (1) la solución  u_θ(t) ≈ [x(t), y(t)]
#   (2) los parámetros de la ODE  p = [α, β, δ, γ]
#
# Función de pérdida:
#
#   L = L_data + λ · L_phys
#
#   L_data = (1/M) Σⱼ ‖u_θ(tⱼ) − u_obs(tⱼ)‖²     (ajuste a datos)
#   L_phys = (1/N) Σᵢ ‖du_θ/dt(tᵢ) − f(u_θ(tᵢ), p̂)‖²  (residuo ODE)
#
# Los parámetros estimados p̂ = [α̂, β̂, δ̂, γ̂] se optimizan junto con
# los pesos de la red neuronal.
#
# Sistema de Lotka-Volterra:
#   dx/dt =  α x − β x y     (presas)
#   dy/dt =  δ x y − γ y     (depredadores)
# ============================================================

# ---------------------------------------------------------------------------
# Parámetros verdaderos (desconocidos para el PINN)
# ---------------------------------------------------------------------------
const α_true = 1.0;   const β_true = 0.1
const δ_true = 0.075; const γ_true = 1.5
const u0    = [10.0, 5.0]
const tspan = (0.0, 15.0)

# ---------------------------------------------------------------------------
# Generar datos observados (trayectoria "verdadera", sin ruido)
# ---------------------------------------------------------------------------
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] =  α * x - β * x * y
    du[2] =  δ * x * y - γ * y
end

prob_true = ODEProblem(lotka_volterra!, u0, tspan, [α_true, β_true, δ_true, γ_true])
sol_true  = solve(prob_true, Tsit5(), saveat=0.01)

# Muestras de observación: M puntos uniformes en el tiempo
M_obs    = 40
t_obs    = collect(range(tspan[1], tspan[2], length=M_obs))
u_obs    = hcat([sol_true(t) for t in t_obs]...)   # (2, M_obs)

println("✓ Trayectoria de referencia calculada.")
println("  Observaciones: $M_obs puntos en [$(tspan[1]), $(tspan[2])]")

# ---------------------------------------------------------------------------
# Red neuronal:  t ↦ [x(t), y(t)]
# ---------------------------------------------------------------------------
t0_val, T_val = tspan
scale_inp = WrappedFunction(t -> (t .- t0_val) ./ (T_val - t0_val))
scale_out = WrappedFunction(x -> 10.0 .* x)

nn = Chain(
    scale_inp,
    Dense(1 => 32, tanh),
    Dense(32 => 32, tanh),
    Dense(32 => 16, tanh),
    Dense(16 => 2),
    scale_out
)

rng    = MersenneTwister(42)
nn_ps, st = Lux.setup(rng, nn)
st     = Lux.testmode(st)

n_params = sum(p -> sum(length, values(p); init=0), values(nn_ps))
println("✓ Red neuronal: $n_params parámetros.")

# ---------------------------------------------------------------------------
# Parámetros ODE entrenables — inicializados lejos de los valores verdaderos
# ---------------------------------------------------------------------------
p_ode_init = [0.5, 0.05, 0.04, 1.0]   # [α̂, β̂, δ̂, γ̂]  (conjeturas iniciales)

println("\nParámetros verdaderos:  α=$(α_true), β=$(β_true), δ=$(δ_true), γ=$(γ_true)")
println("Conjetura inicial:      α=$(p_ode_init[1]), β=$(p_ode_init[2]), δ=$(p_ode_init[3]), γ=$(p_ode_init[4])\n")

# ---------------------------------------------------------------------------
# Parámetros combinados: pesos de la NN + parámetros ODE
# Optimisers.jl trabaja recursivamente sobre cualquier NamedTuple de arrays.
# ---------------------------------------------------------------------------
theta = (nn = nn_ps, ode = copy(p_ode_init))

# ---------------------------------------------------------------------------
# Puntos de colocación
# ---------------------------------------------------------------------------
N_col = 100
t_col = [
    10.0 .^ collect(range(log10(tspan[1] + 1e-10), log10(tspan[2]), length=div(N_col, 2)));
    collect(range(tspan[1], tspan[2], length=div(N_col, 2)))
]

# ---------------------------------------------------------------------------
# Derivada temporal de la NN por diferencias finitas
# ---------------------------------------------------------------------------
function nn_dudt(t_val, nn_ps, st; h=1e-4)
    u_fwd = nn([t_val + h], nn_ps, st)[1]
    u_bwd = nn([t_val - h], nn_ps, st)[1]
    return (u_fwd .- u_bwd) ./ (2h)
end

# ---------------------------------------------------------------------------
# Función de pérdida inversa
# ---------------------------------------------------------------------------
const λ_phys = 1.0

function inverse_loss_components(theta)
    nn_ps = theta.nn
    α̂, β̂, δ̂, γ̂ = theta.ode

    # --- Pérdida de datos (ajuste a las observaciones) ---
    data_residuals = map(1:M_obs) do j
        u_pred = nn([t_obs[j]], nn_ps, st)[1]
        sum((u_pred .- u_obs[:, j]) .^ 2)
    end
    data_loss = mean(data_residuals)

    # --- Pérdida física (residuo de la ODE con parámetros estimados) ---
    phys_residuals = map(t_col) do t_i
        u    = nn([t_i], nn_ps, st)[1]
        dudt = nn_dudt(t_i, nn_ps, st)
        x, y = u[1], u[2]
        rhs  = [α̂*x - β̂*x*y,
                δ̂*x*y - γ̂*y]
        sum((dudt .- rhs) .^ 2)
    end
    phys_loss = λ_phys * mean(phys_residuals)

    return data_loss, phys_loss
end

inverse_loss(theta) = sum(inverse_loss_components(theta))

# ---------------------------------------------------------------------------
# Ciclo de entrenamiento
# ---------------------------------------------------------------------------
n_epochs    = 20000
opt_state   = Optimisers.setup(Adam(1e-3), theta)
losses      = Float64[]
data_losses = Float64[]
phys_losses = Float64[]

# Track parameter estimates over time
param_history = zeros(n_epochs, 4)

println("Entrenando PINN inverso ($n_epochs épocas)...")
println("─"^72)

for epoch in 1:n_epochs
    global theta, opt_state
    loss_val, grads = Zygote.withgradient(inverse_loss, theta)
    opt_state, theta = Optimisers.update(opt_state, theta, grads[1])

    d_l, p_l = inverse_loss_components(theta)
    push!(losses,      d_l + p_l)
    push!(data_losses, d_l)
    push!(phys_losses, p_l)
    param_history[epoch, :] .= theta.ode

    if epoch % 2000 == 0
        α̂, β̂, δ̂, γ̂ = theta.ode
        @printf("  Época %6d │ Total: %8.3e │ Data: %8.3e │ Física: %8.3e\n",
                epoch, d_l + p_l, d_l, p_l)
        @printf("             │ α̂=%.4f  β̂=%.4f  δ̂=%.4f  γ̂=%.4f\n",
                α̂, β̂, δ̂, γ̂)
    end
end

println("─"^72)
α̂, β̂, δ̂, γ̂ = theta.ode
println("\nResultados finales:")
@printf("  α:  verdadero = %.4f   estimado = %.4f   error = %.2f%%\n",
        α_true, α̂, 100*abs(α̂ - α_true)/α_true)
@printf("  β:  verdadero = %.4f   estimado = %.4f   error = %.2f%%\n",
        β_true, β̂, 100*abs(β̂ - β_true)/β_true)
@printf("  δ:  verdadero = %.4f   estimado = %.4f   error = %.2f%%\n",
        δ_true, δ̂, 100*abs(δ̂ - δ_true)/δ_true)
@printf("  γ:  verdadero = %.4f   estimado = %.4f   error = %.2f%%\n\n",
        γ_true, γ̂, 100*abs(γ̂ - γ_true)/γ_true)

# ---------------------------------------------------------------------------
# Evaluación en grilla fina
# ---------------------------------------------------------------------------
t_fine  = collect(range(tspan[1], tspan[2], length=600))
u_pinn  = hcat([nn([t], theta.nn, st)[1] for t in t_fine]...)

x_ref_interp = [sol_true(t)[1] for t in t_fine]
y_ref_interp = [sol_true(t)[2] for t in t_fine]
err_x = abs.(u_pinn[1, :] .- x_ref_interp)
err_y = abs.(u_pinn[2, :] .- y_ref_interp)

@printf("Error L∞ en x(t): %.4f\n", maximum(err_x))
@printf("Error L∞ en y(t): %.4f\n\n", maximum(err_y))

# ===========================================================================
# Figuras
# ===========================================================================

# ---------------------------------------------------------------------------
# Figura 1: Solución aprendida vs datos observados vs referencia
# ---------------------------------------------------------------------------
fig1 = plot(
    sol_true.t, sol_true[1, :],
    label="Presas — ODE verdadera", linewidth=2, color=:royalblue,
    xlabel="Tiempo", ylabel="Población",
    title="PINN Inverso — Lotka-Volterra\nSolución aprendida con parámetros desconocidos",
    legend=:topright, size=(800, 450)
)
plot!(fig1, sol_true.t, sol_true[2, :],
      label="Depredadores — ODE verdadera", linewidth=2, color=:firebrick)
plot!(fig1, t_fine, u_pinn[1, :],
      label="Presas — PINN inverso", linewidth=2, color=:dodgerblue, linestyle=:dash)
plot!(fig1, t_fine, u_pinn[2, :],
      label="Depredadores — PINN inverso", linewidth=2, color=:orangered, linestyle=:dash)
scatter!(fig1, t_obs, u_obs[1, :],
         label="Observaciones x(t)", color=:royalblue, markersize=4,
         markershape=:circle, markerstrokewidth=0, alpha=0.7)
scatter!(fig1, t_obs, u_obs[2, :],
         label="Observaciones y(t)", color=:firebrick, markersize=4,
         markershape=:circle, markerstrokewidth=0, alpha=0.7)
savefig(fig1, "lv_inverse_pinn_solution.png")
println("Figura guardada: lv_inverse_pinn_solution.png")

# ---------------------------------------------------------------------------
# Figura 2: Curva de entrenamiento
# ---------------------------------------------------------------------------
fig2 = plot(
    losses, label="Loss total", color=:mediumpurple, linewidth=2,
    xlabel="Época", ylabel="Loss (escala log)",
    title="Curva de entrenamiento — PINN Inverso\n(λ_phys = $λ_phys,  M_obs = $M_obs,  N_col = $N_col)",
    yscale=:log10, legend=:topright, size=(700, 420)
)
plot!(fig2, data_losses, label="L_data  (ajuste a observaciones)",
      color=:steelblue, linewidth=2, linestyle=:dash)
plot!(fig2, phys_losses, label="λ · L_phys  (residuo ODE)",
      color=:darkorange, linewidth=2, linestyle=:dot)
savefig(fig2, "lv_inverse_pinn_loss.png")
println("Figura guardada: lv_inverse_pinn_loss.png")

# ---------------------------------------------------------------------------
# Figura 3: Convergencia de los parámetros ODE
# ---------------------------------------------------------------------------
fig3 = plot(layout=(2, 2), size=(850, 600),
            plot_title="Convergencia de parámetros — PINN Inverso")

param_labels  = ["α", "β", "δ", "γ"]
param_true    = [α_true, β_true, δ_true, γ_true]
param_colors  = [:royalblue, :firebrick, :darkgreen, :darkorange]

for (k, (lbl, p_true, col)) in enumerate(zip(param_labels, param_true, param_colors))
    plot!(fig3[k], param_history[:, k],
          label="$lbl estimado", color=col, linewidth=2,
          xlabel="Época", ylabel=lbl, title="Parámetro $lbl")
    hline!(fig3[k], [p_true], label="$lbl verdadero = $p_true",
           color=col, linewidth=1.5, linestyle=:dash)
    hline!(fig3[k], [p_ode_init[k]], label="Inicial = $(p_ode_init[k])",
           color=:gray, linewidth=1, linestyle=:dot)
end

savefig(fig3, "lv_inverse_pinn_params.png")
println("Figura guardada: lv_inverse_pinn_params.png")

# ---------------------------------------------------------------------------
# Figura 4: Espacio de fase
# ---------------------------------------------------------------------------
fig4 = plot(
    sol_true[1, :], sol_true[2, :],
    label="ODE verdadera", linewidth=2.5, color=:steelblue,
    xlabel="Presas  x(t)", ylabel="Depredadores  y(t)",
    title="Espacio de fase — PINN Inverso",
    legend=:topright, size=(550, 480)
)
plot!(fig4, u_pinn[1, :], u_pinn[2, :],
      label="PINN inverso", linewidth=2, color=:orangered, linestyle=:dash)
scatter!(fig4, u_obs[1, :], u_obs[2, :],
         label="Observaciones", color=:black, markersize=4,
         markershape=:circle, markerstrokewidth=0, alpha=0.6)
savefig(fig4, "lv_inverse_pinn_phase.png")
println("Figura guardada: lv_inverse_pinn_phase.png")

# ---------------------------------------------------------------------------
# Figura 5: Error puntual
# ---------------------------------------------------------------------------
fig5 = plot(layout=(2, 1), size=(720, 500))

plot!(fig5[1], t_fine, err_x,
      label="|x_PINN − x_ODE|", color=:royalblue, linewidth=2,
      xlabel="", ylabel="Error absoluto", title="Error puntual — PINN Inverso")
plot!(fig5[1], t_fine, err_y,
      label="|y_PINN − y_ODE|", color=:firebrick, linewidth=2)
scatter!(fig5[1], t_obs, zeros(M_obs),
         label="Observaciones", color=:black, markersize=4,
         markershape=:xcross, markerstrokewidth=1.5)

plot!(fig5[2], t_col, zeros(length(t_col)),
      label=false, alpha=0)
scatter!(fig5[2], t_obs, zeros(M_obs),
         label="Observaciones (M=$M_obs)", color=:steelblue,
         markersize=5, markershape=:circle, markerstrokewidth=0)
scatter!(fig5[2], t_col, zeros(length(t_col)) .- 0.5,
         label="Colocación (N=$N_col)", color=:gray40,
         markersize=4, markershape=:xcross, markerstrokewidth=1.5,
         xlabel="Tiempo", ylabel="", title="Distribución de puntos",
         yticks=false, ylims=(-1.5, 1))

savefig(fig5, "lv_inverse_pinn_error.png")
println("Figura guardada: lv_inverse_pinn_error.png")

println("\n¡Listo! Se generaron 5 figuras.")
