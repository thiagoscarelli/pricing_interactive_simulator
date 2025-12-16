import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- App Configuration ---
st.set_page_config(page_title="Monopoly Pricing: Theory & Simulation", layout="wide")

# --- Section 1: Mathematical Exposition ---
st.title("Pricing with Binary Demand: Simulation Engine for Uniform Pricing, Price Experimentation, and Price Discrimination")
st.markdown("### 1. Theoretical Framework")

st.markdown(r"""
#### A. The Micro-Foundations of Binary Demand
For discrete goods (SaaS subscriptions, digital media), individual demand is often binary: $q_i \in \{0, 1\}$. We aggregate these individual choices over the market to derive demand.

Let consumer $i$ have a valuation (willingness to pay) $v_i$ drawn from a distribution with PDF $f(v)$ and CDF $F(v)$.
The consumer purchases if $v_i \geq P$. The individual demand function is an indicator:
$$
q_i(P) = \mathbb{I}(v_i \geq P)
$$

Aggregating over $N$ consumers, the expected quantity demanded is:
$$
Q(P) = N \cdot \mathbb{P}(v_i \geq P) = N [1 - F(P)]
$$
The slope of the demand curve is determined by the density of marginal consumers:
$$
\frac{dQ}{dP} = -N f(P)
$$

#### B. Elasticity and Hazard Rates
Substituting $Q(P)$ and $Q'(P)$ into the definition of price elasticity:
$$
\epsilon_d = \frac{P}{Q} \frac{dQ}{dP} = \frac{P}{N[1 - F(P)]} (-N f(P)) = - P \underbrace{\left( \frac{f(P)}{1 - F(P)} \right)}_{\text{Hazard Rate } h(P)}
$$
Thus, for binary demand, elasticity is the product of price and the **Hazard Rate** of the valuation distribution.

#### C. Revenue Maximization & The Inverse Elasticity Rule
The monopolist maximizes Profit $\Pi$ (assuming constant marginal cost $c$):
$$
\max_P \Pi(P) = (P - c) \cdot Q(P) = (P - c) \cdot N [1 - F(P)]
$$
The First Order Condition (FOC) requires $\Pi'(P) = 0$:
$$
[1 - F(P)] - (P - c)f(P) = 0
$$
Rearranging yields the optimal markup rule, relating back to the inverse hazard rate:
$$
P^* = c + \frac{1 - F(P^*)}{f(P^*)} = c + \frac{1}{h(P^*)}
$$
Equivalently, this satisfies the classic Lerner Index condition:
$$
\frac{P^* - c}{P^*} = -\frac{1}{\epsilon_d(P^*)}
$$

#### D. The Learning Problem
In reality, $f(v)$ is unknown. The monopolist faces a **pricing bandit** problem.
We compare three regimes:
1.  **Baseline Uniform Pricing:** The firm sets a fixed price $P_{base}$ based on a management guess or status quo.
2.  **Learned Uniform Pricing:** The firm runs $T$ experiments (A/B tests) to estimate demand $Q(P_t)$ and selects $\hat{P}^*$.
3.  **Third-Degree Price Discrimination:** If the firm can observe a signal correlated with valuation (e.g., "Student" vs "Professional"), it sets segment-specific prices $P_A$ and $P_B$ such that $P_k = c + 1/h_k(P_k)$.
""")

st.markdown("---")

# --- Section 2: Interactive Simulation ---
st.sidebar.header("1. Market Parameters")
dist_type = st.sidebar.selectbox(
    "Valuation Distribution", ["Gaussian (Normal)", "Exponential", "Uniform"]
)
N = st.sidebar.number_input("Total Population (N)", value=2000, step=100)
mc = st.sidebar.number_input("Marginal Cost (c)", value=10.0, step=1.0)

st.sidebar.subheader("Segment Parameters (set equal for no segmentation)")
if dist_type == "Gaussian (Normal)":
    mu1 = st.sidebar.slider("Segment A Mean (Students)", 20.0, 150.0, 30.0)
    mu2 = st.sidebar.slider("Segment B Mean (Pros)", 20.0, 150.0, 40.0)
    sigma = st.sidebar.slider("Std Dev (Common)", 5.0, 30.0, 15.0)
elif dist_type == "Exponential":
    scale1 = st.sidebar.slider("Segment A Scale", 10.0, 100.0, 30.0)
    scale2 = st.sidebar.slider("Segment B Scale", 10.0, 100.0, 40.0)
else:  # Uniform
    low = st.sidebar.slider("Lower Bound", 0.0, 200.0, 10.0)
    high = st.sidebar.slider("Upper Bound", 0.0, 200.0, 20.0)


# Data Generation
@st.cache_data
def generate_data(n, dist_choice, param_tuple, seed=42):
    """
    param_tuple: hashable tuple of distribution parameters
        Gaussian: (mu1, mu2, sigma)
        Exponential: (scale1, scale2)
        Uniform: (low, high)
    """
    np.random.seed(seed)
    n_a = n // 2
    n_b = n - n_a

    if dist_choice == "Gaussian (Normal)":
        mu1, mu2, sigma = param_tuple
        v_a = np.random.normal(mu1, sigma, n_a)
        v_b = np.random.normal(mu2, sigma, n_b)
    elif dist_choice == "Exponential":
        scale1, scale2 = param_tuple
        v_a = np.random.exponential(scale1, n_a)
        v_b = np.random.exponential(scale2, n_b)
    else:  # Uniform
        low, high = param_tuple
        v_a = np.random.uniform(low, high * 0.8, n_a)
        v_b = np.random.uniform(low * 1.2, high, n_b)

    v_a = np.maximum(v_a, 0)
    v_b = np.maximum(v_b, 0)
    return v_a, v_b

if dist_type == "Gaussian (Normal)":
    param_tuple = (mu1, mu2, sigma)
elif dist_type == "Exponential":
    param_tuple = (scale1, scale2)
else:
    param_tuple = (low, high)

v_a, v_b = generate_data(N, dist_type, param_tuple)
v_total = np.concatenate([v_a, v_b])


# Helper Functions
def calculate_surplus(price, valuations, cost):
    buyers = valuations[valuations >= price]
    q = len(buyers)
    revenue = q * price
    prod_cost = q * cost
    ps = revenue - prod_cost
    cs = np.sum(buyers - price)
    ts = ps + cs
    return {"p": price, "q": q, "ps": ps, "cs": cs, "ts": ts}


def optimize_grid(valuations, cost, resolution=100):
    if len(valuations) == 0:
        return {"p": cost, "ps": 0, "cs": 0, "ts": 0, "q": 0}
    max_v = np.max(valuations)
    grid = np.linspace(cost, max_v, resolution)
    best_res = None
    best_ps = -np.inf
    for p in grid:
        res = calculate_surplus(p, valuations, cost)
        if res["ps"] > best_ps:
            best_ps = res["ps"]
            best_res = res
    return best_res


# --- Regime Simulation ---

# Sidebar for Baseline
st.sidebar.divider()
st.sidebar.header("2. Baseline Strategy")
max_possible = float(np.percentile(v_total, 99))
p_baseline_input = st.sidebar.slider(
    "Status Quo Price",
    min_value=float(mc),
    max_value=max_possible,
    value=float(mc * 2.5),
)

# 1. Baseline Calculation
res_baseline = calculate_surplus(p_baseline_input, v_total, mc)

# 2. Learned Uniform (Experiments)
rounds = 10
experiment_grid = np.linspace(mc, max_possible, rounds)
history_ps = []
history_ts = []
history_cs = []
best_p_learned = mc
max_ps_seen = -np.inf

for p_test in experiment_grid:
    res_t = calculate_surplus(p_test, v_total, mc)
    history_ps.append(res_t["ps"])
    history_ts.append(res_t["ts"])
    history_cs.append(res_t["cs"])

    if res_t["ps"] > max_ps_seen:
        max_ps_seen = res_t["ps"]
        best_p_learned = p_test

res_learned = calculate_surplus(best_p_learned, v_total, mc)

# 3. Price Discrimination
res_a = optimize_grid(v_a, mc)
res_b = optimize_grid(v_b, mc)
res_discrim = {
    "ps": res_a["ps"] + res_b["ps"],
    "cs": res_a["cs"] + res_b["cs"],
    "ts": res_a["ts"] + res_b["ts"],
    "p": f"A:${res_a['p']:.0f} | B:${res_b['p']:.0f}",
}

# --- Display Metrics ---
st.divider()
st.subheader("Results Comparison")

c1, c2, c3 = st.columns(3)


def display_metric(col, title, res, is_baseline=False):
    col.markdown(f"### {title}")
    col.metric("Total Surplus (TS)", f"${res['ts']:,.0f}")

    col.markdown(f"""
    * **Producer Surplus:** :blue[${res["ps"]:,.0f}]
    * **Consumer Surplus:** :green[${res["cs"]:,.0f}]
    * **Price:** {res["p"] if isinstance(res["p"], str) else f"${res['p']:.2f}"}
    """)


display_metric(c1, "1. Baseline (Status Quo)", res_baseline, is_baseline=True)
display_metric(c2, "2. Learned Uniform", res_learned)
display_metric(c3, "3. Price Discrim.", res_discrim)

st.caption(
    "Note: 'Baseline' may occasionally beat 'Learned' in Total Surplus if the status quo price is accidentally efficient (close to MC), but it will rarely beat 'Learned' in Producer Surplus (Profit)."
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Aggregate Market (Uniform)",
        "Segmented Markets (3rd Degree)",
        "Learning Dynamics",
        "Distribution & Hazard Rate",
    ]
)

# PLOT 1: Aggregate Market (Baseline vs Learned)
with tab1:
    fig, ax = plt.subplots(figsize=(10, 6))

    all_vals_sorted = np.sort(v_total)[::-1]
    quantities = np.arange(1, len(all_vals_sorted) + 1)

    ax.plot(
        quantities,
        all_vals_sorted,
        color="black",
        lw=2,
        label="Aggregate Inverse Demand P(Q)",
    )
    ax.axhline(mc, color="red", linestyle="--", label="Marginal Cost")

    # 1. Plot Experiment Grid
    for p_exp in experiment_grid:
        ax.axhline(p_exp, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    # 2. Plot Baseline Price
    q_base = res_baseline["q"]
    ax.hlines(
        res_baseline["p"],
        0,
        q_base,
        color="orange",
        linestyle="-.",
        linewidth=2,
        label=f"Baseline Price (${res_baseline['p']:.0f})",
    )
    ax.vlines(q_base, 0, res_baseline["p"], color="orange", linestyle="-.", alpha=0.5)

    # 3. Plot Learned Price (Winner)
    q_learned = res_learned["q"]
    p_learned = res_learned["p"]
    ax.hlines(
        p_learned,
        0,
        q_learned,
        color="blue",
        linewidth=2,
        label=f"Learned Price (${p_learned:.0f})",
    )
    ax.vlines(q_learned, 0, p_learned, color="blue", linestyle=":", alpha=0.5)

    # Shading for Learned Outcome
    ax.fill_between(
        quantities[:q_learned], mc, p_learned, color="blue", alpha=0.1
    )  # PS
    ax.fill_between(
        quantities[:q_learned],
        p_learned,
        all_vals_sorted[:q_learned],
        color="green",
        alpha=0.1,
    )  # CS

    ax.set_title("Regime 1 & 2: Uniform Pricing (Aggregate Demand)")
    ax.set_xlabel("Quantity (Q)")
    ax.set_ylabel("Price ($) - P(Q)")
    ax.legend(loc="upper right")
    st.pyplot(fig)

# PLOT 2: Segmented Markets
with tab2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Segment A
    vals_a_sorted = np.sort(v_a)[::-1]
    q_a = np.arange(1, len(vals_a_sorted) + 1)
    ax1.plot(q_a, vals_a_sorted, "k-", label="Demand A")
    ax1.axhline(mc, color="red", linestyle="--")

    qa_opt = res_a["q"]
    pa_opt = res_a["p"]
    ax1.hlines(
        pa_opt, 0, qa_opt, color="purple", lw=2, label=f"Price A (${pa_opt:.0f})"
    )
    ax1.vlines(qa_opt, 0, pa_opt, color="purple", linestyle=":")
    ax1.fill_between(q_a[:qa_opt], mc, pa_opt, color="purple", alpha=0.2)
    ax1.fill_between(
        q_a[:qa_opt], pa_opt, vals_a_sorted[:qa_opt], color="green", alpha=0.1
    )

    ax1.set_title("Segment A (e.g., Students)")
    ax1.set_xlabel("Quantity")
    ax1.set_ylabel("Price")
    ax1.legend()

    # Segment B
    vals_b_sorted = np.sort(v_b)[::-1]
    q_b = np.arange(1, len(vals_b_sorted) + 1)
    ax2.plot(q_b, vals_b_sorted, "k-", label="Demand B")
    ax2.axhline(mc, color="red", linestyle="--")

    qb_opt = res_b["q"]
    pb_opt = res_b["p"]
    ax2.hlines(
        pb_opt, 0, qb_opt, color="darkblue", lw=2, label=f"Price B (${pb_opt:.0f})"
    )
    ax2.vlines(qb_opt, 0, pb_opt, color="darkblue", linestyle=":")
    ax2.fill_between(q_b[:qb_opt], mc, pb_opt, color="blue", alpha=0.2)
    ax2.fill_between(
        q_b[:qb_opt], pb_opt, vals_b_sorted[:qb_opt], color="green", alpha=0.1
    )

    ax2.set_title("Segment B (e.g., Professionals)")
    ax2.set_xlabel("Quantity")
    ax2.legend()

    st.pyplot(fig)

# PLOT 3: Learning Dynamics
with tab3:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    rounds_x = np.arange(1, 11)

    ax2.plot(rounds_x, history_ts, marker="o", label="Total Surplus", color="purple")
    ax2.plot(rounds_x, history_ps, marker="s", label="Producer Surplus", color="blue")
    ax2.plot(
        rounds_x,
        history_cs,
        marker="^",
        label="Consumer Surplus",
        color="green",
        linestyle="--",
    )

    best_idx = history_ps.index(max_ps_seen)
    ax2.axvline(rounds_x[best_idx], color="gold", linestyle=":", label="Optimal Chosen")

    ax2.set_xlabel("Experiment Round")
    ax2.set_ylabel("Value ($)")
    ax2.set_title("Experiment Results")
    ax2.legend()
    st.pyplot(fig2)

# PLOT 4: Distribution & Hazard Rate
with tab4:
    st.markdown(r"""
    **Visualizing the Optimal Price Condition**
    The optimal monopoly price $P^*$ occurs where the **Inverse Hazard Rate** equals the **Markup** over marginal cost:
    $$ P - c = \frac{1}{h(P)} \implies P - c = \frac{1 - F(P)}{f(P)} $$
    """)

    fig3, (ax_pdf, ax_haz) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 1. Estimate smooth PDF using KDE
    grid_x = np.linspace(0, max_possible, 200)
    kde = stats.gaussian_kde(v_total)
    pdf_est = kde(grid_x)

    # 2. Calculate CDF and Hazard Rate
    # Simple Riemann sum for CDF from PDF
    cdf_est = np.cumsum(pdf_est) * (grid_x[1] - grid_x[0])
    cdf_est = cdf_est / cdf_est[-1]  # normalize to 1

    # Hazard Rate = f(x) / (1 - F(x))
    survival = 1 - cdf_est
    survival = np.maximum(survival, 1e-3)  # avoid div by zero
    hazard = pdf_est / survival
    inv_hazard = 1 / hazard

    # Plot A: PDF & CDF
    ax_pdf.plot(grid_x, pdf_est, color="black", label="PDF f(p)")
    ax2_pdf = ax_pdf.twinx()
    ax2_pdf.plot(grid_x, cdf_est, color="gray", linestyle=":", label="CDF F(p)")
    ax_pdf.set_ylabel("Density")
    ax2_pdf.set_ylabel("Cumulative Probability")
    ax_pdf.set_title("Valuation Distribution (Aggregate)")
    ax_pdf.legend(loc="upper left")
    ax2_pdf.legend(loc="upper right")

    # Plot B: The Optimization Condition
    # We look for intersection of y = P - c and y = 1/h(P)
    markup_line = grid_x - mc

    ax_haz.plot(
        grid_x, inv_hazard, color="purple", label="Inverse Hazard (1/h(P))", linewidth=2
    )
    ax_haz.plot(
        grid_x, markup_line, color="red", linestyle="--", label="Net Price (P - c)"
    )

    # Limit y-axis for readability (Inverse Hazard blows up at high P)
    ax_haz.set_ylim(0, np.percentile(markup_line, 95) * 2)

    ax_haz.set_xlabel("Price ($) [Independent Variable]")
    ax_haz.set_ylabel("$")
    ax_haz.set_title("Optimal Price Determination")
    ax_haz.legend()

    # Annotate intersection roughly
    # Find index where lines cross
    idx = np.argwhere(np.diff(np.sign(markup_line - inv_hazard))).flatten()
    if len(idx) > 0:
        cross_x = grid_x[idx[0]]
        cross_y = markup_line[idx[0]]
        ax_haz.plot(cross_x, cross_y, "ko")
        ax_haz.annotate(
            f"Optimal Price P* ≈ {cross_x:.1f}",
            xy=(cross_x, cross_y),
            xytext=(cross_x + 10, cross_y - 10),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    st.pyplot(fig3)

######################################################################
