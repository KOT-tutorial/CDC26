# Koopman MPC

We train and compare **four EDMD-with-control (EDMDc) model structures** on two benchmarks, then deploy each as the prediction model inside a Koopman MPC controller. The code is written to be read alongside Section IV (EDMD with control, implementation) and Section VI-C (the hands-on Koopman MPC tutorial) of the paper, and uses the same notation.


## The four model forms

With state lift $z = \Psi_x(x)$ and input lift $v = \Psi_u(u)$,
the toolbox fits:
| Form       | Predictor            | Input enters through  |
| ----       | ---------            | --------------------  |
| **Linear** | $z^+ = A z + B u$    | raw input, affinely   |
| **Bilinear** |$z^+ = A z + B u + \sum_k N_k z u_k$ | raw input, bilinearly |
| **GeKo**   | $z^+ = K (z \otimes v)$ | full input dictionary, every mode coupled |
| **KCF**    | $z^+ = A_{11} z + A_{12} (v \otimes z)$ | input dictionary coupled to a *subset*|


Notation matches the paper: 
* $d$ = state dimension 
* $n$ = state-lift dimension 
* $p$ = input-lift dimension ($= m + 1$ Chebyshev modes)
* $m$ = inputs 
* $H$ = multi-step training horizon
* $N = H\cdot M$ = training pairs from $M$ selected windows

GeKo and KCF differ only in how much of
the input-state coupling they retain; they coincide only at full coupling.

## Folder structure

Two systems, two scripts each &mdash; **Part 1 (learn)** then **Part 2 (MPC)**:

```
learn_koopman_duffing.m   Part 1: train + open-loop eval, forced Duffing oscillator
mpc_koopman_duffing.m     Part 2: regulating MPC to the origin, Duffing
learn_koopman_dcmotor.m   Part 1: train + open-loop eval, DC motor w/ input nonlinearity
mpc_koopman_dcmotor.m     Part 2: reference-tracking MPC on angular velocity, DC motor
```

Part 1 saves a model file <tt>koopman\_\<system\>\_model\_\<kernel\>.mat</tt>; Part 2 loads it. **Run Part~1 first, then Part 2, from the same directory.** Requires MATLAB with the Statistics and Machine Learning Toolbox (<tt>kmeans</tt>) and the Optimization Toolbox (<tt>fmincon</tt>).

## Quick start

```
--- Duffing oscillator ---
learn_koopman_duffing      % Part 1: trains 4 forms, saves koopman_duffing_model_<kernel>.mat
mpc_koopman_duffing        % Part 2: auto-loads the newest model, runs MPC

--- DC motor ---
learn_koopman_dcmotor
mpc_koopman_dcmotor
```

Each Part 2 auto-detects the newest matching <tt>\*\_model\_\*.mat</tt> in
the working directory, so no path edits are needed when you switch kernels.


## User Settings

The whole comparison is driven by a handful of settings at the top of the
learn files.

### State dictionary &mdash; <tt>KERNEL\_TYPE</tt>
```
1 : Gaussian RBF        2 : Rational Quadratic
3 : Matern 5/2          4 : Polynomial (no centres)   <- used in the paper
```
For the kernel choices (1-3), centres are placed by $k$-means on the state
data and the $n$ lifted features are variance-normalised. For polynomial (4), the lift is the monomials up to <tt>deg\_poly\_x</tt> and the state
appears natively (no centres). The **input dictionary is always
Chebyshev** $T_0,\dots,T_{d_u}$ on the pre-scaled input.



### KCF coupling + shared lift &mdash; the three knobs

```
prepend_state    false -> pure-feature lift + one regressed decoder D (default;
                          polynomial already contains the state)
                 true  -> prepend [1; x]; the decoder is then a plain projection
                          onto those rows, shared by all four forms
kcf_couple_deg   POLY   lift: couple the input to monomials of degree <= this
                        (paper uses q = 2 -> 6 of 15 features on the Duffing)
kcf_s_ratio      KERNEL lift: couple the input to the top-variance features,
                        sized so the coupled model is ~ kcf_s_ratio * n wide
```

<tt>prepend\_state</tt> is a *shared* lift/decoder convention: it changes how all four forms read the physical state back out, and it is saved into the model file so that Part 2 lifts and decodes exactly the way Part 1 trained. The paper reports <tt>prepend\_state = false</tt>.

### Per-system settings

* **Duffing:** <tt>params.\{delta,alpha,beta,Ts\}</tt>, feature sizes (<tt>nz\_rbf</tt>/<tt>nz\_rq</tt>/<tt>nz\_mat</tt> $= 300$, <tt>deg\_poly\_x</tt> $= 4$). Paper config: polynomial $n = 15$, Chebyshev $T_0\ldots T_3$ ($p = 4$), $M = 800$ windows, $H = 20$, $\gamma = 10^{-4}$.
* **DC motor:** <tt>INPUT\_NONLINEARITY</tt> (1 $= 2\tanh(u)$, 2 $= 2\tanh(u\cos u)$, the hard non-monotone case), physical <tt>params.\{Ra,La,km,\dots\}</tt>, and the scaling <tt>params.Sx</tt>, <tt>params.Su</tt> bringing state and input to comparable magnitudes. Paper config: polynomial $n = 10$, Chebyshev $T_0\ldots T_8$ ($p = 9,$ the high modes are needed for the input nonlinearity).


## What Part 1 does




1. Generates $L$ multi-sine training trajectories.
1. Lifts them with the chosen state/input dictionaries.
1. Extracts multi-step windows of horizon $H$ and selects a diverse
subset of $M$ by **$k$-means on the window shape** $[z_0;\, z_{H/2};\, z_H]$ (start / midpoint / end of the lifted window), keeping the window nearest each cluster centre &mdash; so the fit sees windows that *evolve* differently, not just different initial conditions.
1. Fits all four forms by regularised least squares (Tikhonov $\gamma I$), both in **multi-step** mode (the $N = H\cdot M$ unfolded pairs) and a **one-step** baseline matched in sample count.
1. Evaluates open-loop on held-out test initial conditions (ICs), reporting per-trajectory **median / mean / worst** state error and the operator norm, plus a
KCF coupling sweep.
1. Saves everything (the four models, the decoder $D$, the
dictionaries, the scaling, and <tt>prepend\_state</tt>) to
<tt>koopman\_\<system\>\_model\_\<kernel\>.mat</tt>.

For the DC motor, dynamics are simulated in **physical** coordinates while training is in **scaled** coordinates ($x_{\text{scaled}} = S_x x$, $u_{\text{scaled}} = S_u u$); the decoder maps scaled lifted features back to scaled states, and Part 2 converts to physical units for plotting.


## What Part 2 does

Loads the trained models and runs receding-horizon MPC with a single-shooting <tt>fmincon</tt>/SQP solver and an analytic adjoint gradient. Prediction horizon $H_p$ is set at the top (Duffing $H_p = 15$, DC motor $H_p = 20$); note $H_p$ can differ from the training horizon $H$.

```
mpc_mode = 1   single form: compare its ONE-STEP vs MULTI-STEP model
           2   all four forms, MULTI-STEP   (default, the paper tables)
           3   all four forms, ONE-STEP
mpc_form       which form in mode 1 (1 Linear, 2 Bilinear, 3 GeKo, 4 KCF)
```

* **Duffing:** regulation to the origin; reports tracking ISE and per-step CPU (mean / worst).
* **DC motor:** reference tracking on the angular velocity $x_2$ (a mild sinusoid); same reporting. Bilinear is expected to fail here (the control-affine assumption is broken by $f(u)$), while GeKo and KCF track.


## Aborting a slow run
 
The input-lifted forms (GeKo, KCF) solve a larger program per step and can be slow, especially on long horizons or fine sampling. Part 2 therefore includes two interactive safeguards, controlled at the top of the script: 
* <tt>slow\_iter\_thresh\_s</tt> (default $60$ s): if the *first* MPC step exceeds this, the script pauses and asks whether to continue or abort.
* <tt>cumulative\_cap\_choice</tt> $\to$ <tt>cumulative\_thresh\_s</tt> (default $5$ min): once the total run time crosses this cap, the script
prompts once to continue or abort.

On abort the run stops cleanly and **returns the partial results collected so far** (the CPU and ISE statistics are computed over the steps that actually ran). This lets you interrupt a run that is taking too long without losing the work already done.


## Reproducing the paper tables

The published tables (open-loop and closed-loop) use <tt>KERNEL\_TYPE = 4</tt> (polynomial), <tt>prepend\_state = false</tt>, <tt>kcf\_couple\_deg = 2</tt>, the per-system dictionary sizes above,
$H = 20$, $M = 800$, $\gamma = 10^{-4}$. Run Part 1 then Part 2 (<tt>mpc\_mode = 2</tt>) for each system. Figures are regenerated by Part 1 (open-loop trajectories / error rollouts) and Part 2 (closed-loop).


## Notes

* The $k$-means window selection and <tt>fmincon</tt> warm-start make per-run timing mildly machine- and seed-dependent; <tt>rng(42)</tt> is fixed for reproducibility of the models themselves.
* The first MPC step incurs a one-time JIT/cold-start cost; the reported per-step CPU statistics are intended to reflect steady-state receding-horizon cost.


## Contact

This section of the KOT Toolbox is maintained by Mircea Lazar (TU Eindhoven), m.lazar@tue.nl. Questions and issues are welcome.

