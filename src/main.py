import numpy as np
import matplotlib.pyplot as plt
from linear_algebra import lu_factor_fast, lu_solve_fast
from norms import L2norm

#Defining right handside of differential equations

def f_rhs(t, y):
    """
    Defining the rhs of
    differential equations
    """
    k1 = 4.72
    k2 = 3e9
    k3 = 1.5e4
    k4 = 4e7
    k5 = 1.0
    dCA = -k1*y[0]*y[5]
    dCB = -k3*y[1]*y[4]
    dCP =  k2*y[4]*y[5]
    dCQ =  k4*y[4]**2
    dCX =  k1*y[0]*y[5] - k2*y[4]*y[5] + k3*y[1]*y[4] - 2*k4*y[4]**2
    dCY = -k1*y[0]*y[5] - k2*y[4]*y[5] + k5*y[6]
    dCZ =  k3*y[1]*y[4] - k5*y[6]
    return np.array([dCA, dCB, dCP, dCQ, dCX, dCY, dCZ], dtype=float)


#Weighted mean square of vector v, sclaed by tolerances based on current state y(Error norm for ODE solvers)
#Shows relative importance of the error
def wrms(v, y, atol, rtol):
    """
    Scaling the error vector using atol, rtol and u to build scale vector
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    atol = np.asarray(atol, dtype=float)  
    scale = atol + rtol*np.abs(y)        
    e = v/scale
    return np.sqrt(np.mean(e*e))

#Analytical Jacobian
def Analytic_Jac(t, y):
    """
    Making analytic Jacobian
    """
    # y = [CA, CB, CP, CQ, CX, CY, CZ]   (your ordering)
    CA, CB, CP, CQ, CX, CY, CZ = y
    k1 = 4.72
    k2 = 3e9
    k3 = 1.5e4
    k4 = 4e7
    k5 = 1.0

    J = np.zeros((7, 7), dtype=float)

    # r1 = k1*CA*CY
    # r2 = k2*CX*CY
    # r3 = k3*CB*CX
    # r4 = k4*CX^2
    # r5 = k5*CZ
    dr1_dCA = k1 * CY
    dr1_dCY = k1 * CA

    dr2_dCX = k2 * CY
    dr2_dCY = k2 * CX

    dr3_dCB = k3 * CX
    dr3_dCX = k3 * CB

    dr4_dCX = 2.0 * k4 * CX

    dr5_dCZ = k5

    # dCA = -r1
    J[0, 0] = -dr1_dCA
    J[0, 5] = -dr1_dCY

    # dCB = -r3
    J[1, 1] = -dr3_dCB
    J[1, 4] = -dr3_dCX

    # dCP = +r2
    J[2, 4] = +dr2_dCX
    J[2, 5] = +dr2_dCY

    # dCQ = +r4
    J[3, 4] = +dr4_dCX

    # dCX = +r1 - r2 + r3 - 2*r4
    J[4, 0] = +dr1_dCA
    J[4, 1] = +dr3_dCB
    J[4, 4] = -dr2_dCX + dr3_dCX - 2.0*dr4_dCX
    J[4, 5] = +dr1_dCY - dr2_dCY

    # dCY = -r1 - r2 + r5
    J[5, 0] = -dr1_dCA
    J[5, 4] = -dr2_dCX
    J[5, 5] = -dr1_dCY - dr2_dCY
    J[5, 6] = +dr5_dCZ

    # dCZ = +r3 - r5
    J[6, 1] = +dr3_dCB
    J[6, 4] = +dr3_dCX
    J[6, 6] = -dr5_dCZ

    return J

#Numeric Jacobian(For bigger systems)
def Jac_fd_central(f, t, y, rel=1e-7, abs_step=1e-12):
    """
    Numeric Jacobian for bigger systems
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    f0 = f(t, y)
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        hj = rel * max(abs(y[j]), 1.0) + abs_step
        yp = y.copy(); yp[j] += hj
        ym = y.copy(); ym[j] -= hj
        fp = f(t, yp)
        fm = f(t, ym)
        J[:, j] = (fp - fm) / (2.0 * hj)

    return J


# Newton with backtracking line search
# Solve: G(y)=A*y - rhs - h*f(t,y) = 0
#Start with an initial guess, compute the residual(G), then check weight mean square of G, if its <1 Ok
#Else find the Jacobian matrix of residual, solve for the line search direction(delta), then check if full newton step is required or not by dividing lambda in half!
def newton_solve(f, tnp1, h, A, rhs, y_init,
                 atol_newton=1e-12, rtol_newton=1e-9,
                 max_iter=15,
                 refresh_every=3,           # reuse LU for a few Newton its
                 stall_ratio=0.9,           # refresh if not improving enough
                 max_linesearch=12):
    """
    Newton with backtracking line search
    Solve: G(y)=A*y - rhs - h*f(t,y) = 0
    """

    y = np.array(y_init, dtype=float).copy()
    n = y.size
    I = np.eye(n)

    # cached LU(JG)
    JG_fact = None
    last_normG = None

    for it in range(max_iter):
        fval = f(tnp1, y)
        G = A*y - rhs - h*fval

        # Checking the residual of G
        if wrms(G, y, atol_newton, rtol_newton) < 1.0:
            return y, True, it + 1

        normG = float(L2norm(G))

        # decide if we recompute Jacobian/LU
        need_refresh = (
            (JG_fact is None) or
            (it % refresh_every == 0) or
            (last_normG is not None and normG > stall_ratio * last_normG)
        )

        if need_refresh:
            # analytic Jacobian of f
            Jf = Analytic_Jac(tnp1, y)
            # Jacobian of G(y): JG = A*I - h*Jf
            JG = A*I - h*Jf

            try:
                LU, p = lu_factor_fast(JG)
            except Exception:
                return y, False, it + 1
            JG_fact = (LU, p)

        # Newton direction computation
        try:
            LU, p = JG_fact
            delta = lu_solve_fast(LU, p, -G)
        except Exception:
            # if LU got stale/singular, force refresh once
            try:
                Jf = Analytic_Jac(tnp1, y)
                JG = A*I - h*Jf
                LU, p = lu_factor_fast(JG)
                JG_fact = (LU, p)
                delta = lu_solve_fast(LU, p, -G)

            except Exception:
                return y, False, it + 1

        # backtracking line search (monotone decrease in ||G||)
        lam = 1.0
        accepted = False
        for _ in range(max_linesearch):
            y_try = y + lam * delta
            if not np.all(np.isfinite(y_try)):
                lam *= 0.5
                continue

            f_try = f(tnp1, y_try)
            G_try = A*y_try - rhs - h*f_try

            if float(L2norm(G_try)) <= (1.0 - 1e-4*lam) * normG:
                y = y_try
                accepted = True
                last_normG = normG
                break

            lam *= 0.5

        if not accepted:
            return y, False, it + 1

    return y, False, max_iter


# BDF steps
#Explicit Euler Predictor
#Build next time step, build initial guess, Solve G(y_n+1) = 1*y_n+1-y_n-hf(t_n+1, y_n+1) = 0 with newton solver
def step_bdf1(f, tn, y_n, h, **newton_kw):
    """
    Building BDF1 Nonlinear equation
    """
    tnp1 = tn + h
    y_pred = y_n + h*f(tn, y_n)  # explicit Euler predictor
    A = 1.0
    rhs = y_n
    
    y, ok, nit = newton_solve(f, tnp1, h, A, rhs, y_pred, **newton_kw)
    return y, ok, nit

def step_bdf2_var(f, tn, y_nm1, y_n, h_nm1, h_n, **newton_kw):
    """
    Variable-step BDF2:
      a0*y_{n+1} + a1*y_n + a2*y_{n-1} = h_n * f(t_{n+1}, y_{n+1})
    where r = h_n/h_{n-1} and:
      a0 = (1+2r)/(1+r), a1 = -(1+r), a2 = r^2/(1+r)
    """
    tnp1 = tn + h_n
    r = h_n / h_nm1 if h_nm1 > 0 else 1.0

    a0 = (1.0 + 2.0*r) / (1.0 + r)
    a1 = -(1.0 + r)
    a2 = (r*r) / (1.0 + r)

    # variable-step extrapolation predictor
    y_pred = y_n + r*(y_n - y_nm1)

    A = a0
    rhs = -(a1*y_n + a2*y_nm1)   # makes A*y - rhs - h*f = 0 equivalent

    y, ok, nit = newton_solve(f, tnp1, h_n, A, rhs, y_pred, **newton_kw)
    return y, ok, nit, y_pred



# Adaptive integrator: Favoring BDF2
def integrate_bdf(f, t0, y0, tf,
                  h0=1e-6, h_min=1e-18, h_max=0.1,
                  atol=1e-12, rtol=1e-6,
                  max_reject=60,
                  safety=0.9, fac_min=0.2, fac_max=5.0,
                  PRINT_EVERY=2000, SAVE_EVERY=3):
    """
   Using BDF Technique in order that calculate the integral
    """


    # Newton settings (tune if you want)
    newton_kw = dict(atol_newton=1e-12, rtol_newton=1e-9, max_iter=12)

    t = float(t0)
    y_n = np.array(y0, dtype=float)

    ts = [t]
    Ys = [y_n.copy()]

    h = float(h0)
    accepted = 0
    rejected = 0

    # Start with one BDF1 step  
    while True:
        if t + h > tf:
            h = tf - t
        y1, ok, nit = step_bdf1(f, t, y_n, h, **newton_kw)
        if ok:
            h_nm1 = h          # store accepted startup step
            t += h
            y_nm1 = y_n
            y_n = y1
            accepted += 1
            ts.append(t)
            Ys.append(y_n.copy())
            break

        h *= 0.5
        rejected += 1
        if h < h_min:
            raise RuntimeError("Failed to start BDF1 (h hit h_min).")
    # Main loop: variable-step BDF2
    while t < tf - 1e-15:
        if t + h > tf:
            h = tf - t

        stuck_counter = 0

        for _ in range(max_reject):
            y2, ok2, nit2, y_pred = step_bdf2_var(f, t, y_nm1, y_n, h_nm1, h, **newton_kw)

            # If Newton solver converges
            if ok2:
                # Estimating error with an easy error estimator
                err = (y2 - y_pred) / 1.5
                En = wrms(err, y2, atol, rtol)
                # If step got accepted
                if En <= 1.0:
                    t += h
                    y_nm1, y_n = y_n, y2
                    accepted += 1

                    if (accepted % SAVE_EVERY == 0) or (t >= tf - 1e-15):
                        ts.append(t)
                        Ys.append(y_n.copy())

                    if accepted % PRINT_EVERY == 0:
                        print(f"[accept] t={t:.3e}, h={h:.3e}, En={En:.2e}, nit={nit2}")

                    # Updating step history for r ratio
                    h_nm1 = h

                    # Step-size update if En is too small for faster convergence
                    if En == 0.0:
                        fac = 1.5
                    else:
                        fac = safety * (1.0 / En) ** 0.5  # p=2
                    fac = np.clip(fac, 0.5, 1.5)
                    h = np.clip(h * fac, h_min, h_max)
                    break

                # If En>1, hence, the step size were too aggressive
                fac = safety * (1.0 / max(En, 1e-300)) ** 0.5
                fac = np.clip(fac, 0.05, 0.5)
                h = max(h_min, h * fac)
                rejected += 1
                stuck_counter += 1

                if stuck_counter % 10 == 0:
                    print(f"[reject err] t={t:.3e}, h->{h:.3e}, En={En:.2e}")
            # If Newton solver does not converge
            else:
                # Reducing the step size
                h = max(h_min, 0.5 * h)
                rejected += 1
                stuck_counter += 1

                if stuck_counter % 10 == 0:
                    print(f"[reject Newton] t={t:.3e}, shrink h->{h:.3e}")

            # emergency fallback if we're stuck: take a BDF1 step
            if stuck_counter >= 25:
                y1f, okf, nitf = step_bdf1(f, t, y_n, h, **newton_kw)
                if okf:
                    t += h
                    y_nm1, y_n = y_n, y1f
                    accepted += 1

                    # update history for next BDF2
                    h_nm1 = h

                    ts.append(t)
                    Ys.append(y_n.copy())

                    if accepted % PRINT_EVERY == 0:
                        print(f"[accept BDF1 fallback] t={t:.3e}, h={h:.3e}, nit={nitf}")

                    h = min(h_max, 1.2 * h)
                    break
                else:
                    h = max(h_min, 0.5 * h)
                    rejected += 1

            if h <= h_min:
                raise RuntimeError("h hit h_min (stiff/stuck).")

        else:
            raise RuntimeError("Too many rejected steps in a row (stuck).")

    return np.array(ts), np.array(Ys), accepted, rejected



# RUN

Tfinal = 10000

y0 = np.array([0.066, 0.066, 0.0, 0.0, 0.0, 0.0, 0.002], dtype=float)

# These are based on ODE15s Plots in MATLAB
yref = np.array([0.066, 0.066, 1e-3, 1e-8, 1e-10, 1e-3, 2e-3], dtype=float)

# Starting tolerances
rtol = 1e-4
atol = 1e-2 * yref   # vector atol

t_arr, Y_arr, acc, rej = integrate_bdf(
    f_rhs, 0.0, y0, Tfinal,
    h0=1e-6, h_min=1e-18, h_max=0.5,
    atol=atol, rtol=rtol,
    PRINT_EVERY=2000, SAVE_EVERY=1
)

print(f"Done. Accepted={acc}, Rejected={rej}, points={len(t_arr)}")

fig, axs = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
axs = axs.ravel()

names = ["CA", "CB", "CP", "CQ", "CX", "CY", "CZ"]
for i in range(7):
    axs[i].plot(t_arr, Y_arr[:, i])
    axs[i].set_title(names[i])
    axs[i].grid(True)

axs[7].axis("off")
axs[6].set_xlabel("t [s]")
axs[6].tick_params(labelbottom=True)  # in case it gets hidden

fig.suptitle(f"BDF Solver (t_end = {Tfinal}s)")
fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle
plt.tight_layout()
plt.show()


# BDF2 method residuals 
# D_n = a0*y_{n+1} + a1*y_n + a2*y_{n-1} - h_n*f(t_{n+1}, y_{n+1})
res2 = np.full((len(t_arr), 7), np.nan)

EPS_H = 1e-18

for n in range(1, len(t_arr)-1):
    h_n   = t_arr[n+1] - t_arr[n]
    h_nm1 = t_arr[n]   - t_arr[n-1]
    if (not np.isfinite(h_n)) or (not np.isfinite(h_nm1)):
        continue
    if h_n <= EPS_H or h_nm1 <= EPS_H:
        continue

    r = h_n / h_nm1
    a0 = (1.0 + 2.0*r) / (1.0 + r)
    a1 = -(1.0 + r)
    a2 = (r*r) / (1.0 + r)

    res2[n+1] = (a0*Y_arr[n+1] + a1*Y_arr[n] + a2*Y_arr[n-1]) - h_n * f_rhs(t_arr[n+1], Y_arr[n+1])

print("Max |BDF2-derivative residual| per variable (finite only):")
max_abs = np.nanmax(np.abs(res2), axis=0)
for name, v in zip(names, max_abs):
    print(f"  {name}: {v:.3e}")

# Plot
for i, name in enumerate(names):
    plt.figure()
    plt.plot(t_arr, res2[:, i])
    plt.xlabel("time (s)")
    plt.ylabel(f"ydot - f residual {name}")
    plt.title(f"Derivative residual (BDF2): {name} till t={Tfinal}s")
    plt.grid(True)
    # annotate final finite residual value on the plot
    finite_idx = np.where(np.isfinite(res2[:, i]))[0]
    if finite_idx.size > 0:
        j = finite_idx[-1]
        t_last = t_arr[j]
        r_last = res2[j, i]
        plt.plot(t_last, r_last, "ko")
        plt.text(t_last, r_last, f"  final={r_last:.3e}", va="center")
    

plt.show()



# CA..CZ plots vs time (after residual plots)
for i, name in enumerate(names):
    plt.figure()
    plt.plot(t_arr, Y_arr[:, i])
    plt.title(f"{name} till t={Tfinal}s")
    plt.xlabel("time (s)")
    plt.ylabel(name)
    plt.grid(True)

plt.show()
