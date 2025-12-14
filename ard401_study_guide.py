import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ARD 401 - Mind Maps",
    page_icon="🧠",
    layout="wide"
)

# Simple, clean styling with dark background and white text
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .mind-map-box {
        background-color: #2d2d2d;
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #ffffff;
    }
    .unit-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .branch {
        margin-left: 20px;
        padding: 10px;
        background-color: #383838;
        border-left: 3px solid #667eea;
        margin-top: 8px;
        border-radius: 4px;
        color: #ffffff;
    }
    .sub-branch {
        margin-left: 20px;
        padding: 8px;
        background-color: #2d2d2d;
        border-left: 2px solid #764ba2;
        margin-top: 6px;
        color: #ffffff;
    }
    .key-point {
        background-color: #333333;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px;">
        <h1 style="margin: 0; color: white;">🧠 ARD 401 - Mind Maps</h1>
        <p style="margin: 10px 0; font-size: 1.1em; color: white;">Recommender Systems | Complete Visual Overview</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📌 Unit I", "📈 Unit II", "🌐 Unit III", "🛡️ Unit IV"])

# UNIT I
with tab1:
    st.markdown('<div class="unit-title"><h2>📌 UNIT I: Fundamentals & Collaborative Filtering</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mind-map-box">', unsafe_allow_html=True)
    st.markdown("""
## UNIT I: Fundamentals & CF
├── **Recommender Systems Basics**
│   ├── Goals: Personalization, Discovery, Engagement, Retention
│   ├── Challenges: Cold Start, Sparsity, Scalability
│   └── Types: Content-Based, Collaborative, Hybrid
│
├── **User-Based Collaborative Filtering**
│   ├── Concept: Similar users → Similar preferences
│   ├── Algorithm:
│   │   ├── 1. Calculate mean rating: r̄_u = Σr_ui / n
│   │   ├── 2. Find overlapping items between users
│   │   ├── 3. Compute Pearson correlation: -1 to +1
│   │   ├── 4. Select k-nearest neighbors (k=10-20)
│   │   └── 5. Weighted average: r̂_uj = r̄_u + Σ sim(u,v)×(r_vj - r̄_v) / Σ|sim|
│   ├── Key Values:
│   │   ├── Pearson: -1 to +1
│   │   ├── Similar users: 0.7-1.0
│   │   ├── Moderate: 0.4-0.7
│   │   └── Dissimilar: < 0.4
│   └── ⚠️ CRITICAL: ALWAYS mean-center (r_u - r̄_u)
│
├── **Item-Based Collaborative Filtering**
│   ├── Concept: Similar items → Rated similarly
│   ├── Formula: r̂_uj = Σ sim(i,j)×r_ui / Σ|sim|
│   ├── ✅ Advantages:
│   │   ├── More stable than user-based
│   │   ├── Better for new users
│   │   └── Cacheable (compute offline)
│   └── ⚠️ Exclude negative similarities (-0.94 to 1.0)
│
├── **Matrix Factorization (SVD)**
│   ├── Concept: R ≈ U × V^T (m×k user × n×k item)
│   ├── Prediction: r̂_ij = u_i · v_j
│   ├── Error: e_ij = r_ij - r̂_ij
│   ├── SGD Update:
│   │   ├── u_i ← u_i + γ(e_ij × v_j - λ × u_i)
│   │   └── v_j ← v_j + γ(e_ij × u_i - λ × v_j)
│   ├── Parameters:
│   │   ├── γ (learning rate): 0.001-0.1
│   │   ├── λ (regularization): 0.001-0.01 ⭐ NEVER FORGET!
│   │   ├── k (factors): 20-100
│   │   └── Convergence: 20-50 iterations
│   └── ⚠️ CRITICAL: ALWAYS include λ×u_i term (prevents overfitting)
│
└── **Key Challenges**
    ├── ❄️ Cold Start: New user/item → No ratings
    ├── 📉 Sparsity: 99.9% matrix empty
    ├── ⚡ Scalability: O(m²) complexity
    ├── ⚖️ Diversity: High accuracy = boring
    └── 👥 Bias: Popularity bias, user bias
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# UNIT II
with tab2:
    st.markdown('<div class="unit-title"><h2>📈 UNIT II: Evaluation & Context-Aware Systems</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mind-map-box">', unsafe_allow_html=True)
    st.markdown("""
## UNIT II: Evaluation Metrics
├── **Evaluation Paradigms**
│   ├── Offline: 80% train, 20% test → Fast, cheap
│   ├── Online A/B: Real users compete → Real behavior
│   └── User Study: Recruit participants (N=20-100) → Subjective
│
├── **Rating Prediction Metrics**
│   ├── MAE: Σ|r - r̂| / n → Typical: 0.3-0.7 stars
│   ├── RMSE: √[Σ(r - r̂)² / n] → Typical: 0.3-1.0 stars ⭐ MOST USED
│   └── MSE: Σ(r - r̂)² / n → Same as RMSE²
│
├── **Ranking Metrics** ⭐ MOST IMPORTANT
│   ├── Precision@k: (#rel in top-k) / k
│   │   └── Typical: 0.4-0.7 (What % of recs are good?)
│   ├── Recall@k: (#rel in top-k) / (total relevant)
│   │   └── Typical: 0.5-1.0 (What % of user items found?)
│   ├── NDCG@k: DCG / IDCG → ⭐ Position matters!
│   │   ├── Formula: DCG = Σ [2^rel_i - 1] / log₂(i+1)
│   │   ├── Typical: 0.5-0.8
│   │   └── ⚠️ CRITICAL: Use log₂(i+1), NOT log(i)
│   └── MAP: Σ(Precision at relevant) / |relevant|
│       └── Typical: 0.4-0.8
│
├── **NDCG Detailed Calculation** ⭐ COMPLEX!
│   ├── Step 1: Calculate DCG
│   │   ├── Position 1 (Relevant): 1/log₂(2) = 1.0
│   │   ├── Position 2 (Not): 0/log₂(3) = 0
│   │   ├── Position 3 (Relevant): 1/log₂(4) = 0.5
│   │   └── Sum = 1.5 (example)
│   ├── Step 2: Calculate IDCG (ideal ranking)
│   │   └── All relevant items first
│   └── Step 3: NDCG = DCG / IDCG
│
├── **Temporal Collaborative Filtering**
│   ├── Exponential Decay: w(t) = e^{-λ(t_current - t)}
│   ├── Parameters:
│   │   ├── λ = 0.01 typical
│   │   ├── Half-life ≈ 70 days
│   │   └── 1-day-old: 2.3× heavier than 95-day-old
│   └── Time-SVD: r̂_uit = μ + b_u(t) + b_i(t) + Σ_k u_uk × v_ik(t)
│
└── **Context-Aware Systems**
    ├── Multiple dimensions: Users × Items × Context
    ├── Example: Music with [Plot=5, Music=3, Effects=4]
    └── Weighted: 0.4×5 + 0.3×3 + 0.3×4 = 4.0
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# UNIT III
with tab3:
    st.markdown('<div class="unit-title"><h2>🌐 UNIT III: Structural Recommendations in Networks</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mind-map-box">', unsafe_allow_html=True)
    st.markdown("""
## UNIT III: Network Analysis
├── **PageRank Algorithm** ⭐ CORE ALGORITHM
│   ├── Concept: Important pages get links from important pages
│   ├── Formula: PR(p) = (1-d)/N + d × Σ_{q→p} [PR(q) / out(q)]
│   ├── Parameters:
│   │   ├── d (damping factor) = 0.85
│   │   ├── (1-d)/N = teleport probability ≈ 0.05
│   │   ├── PR(q) = PageRank of linking page
│   │   └── out(q) = number of outgoing links
│   ├── Calculation Example (3 pages):
│   │   ├── Init: PR(A)=PR(B)=PR(C)=0.333
│   │   ├── Iter 1: PR(A)=0.05, PR(B)=0.192, PR(C)=0.475
│   │   └── Convergence: ~20 iterations
│   └── ⚠️ CRITICAL: Σ PR = 1 (always normalize!)
│
├── **Link Prediction Metrics**
│   ├── Common Neighbors: |N(A) ∩ N(B)|
│   │   └── Simplest, (example: 2)
│   ├── Jaccard: |∩| / |∪|
│   │   └── Normalized, (example: 0.5)
│   ├── Adamic-Adar: Σ 1/log|N(w)| ⭐ Usually best!
│   │   ├── Example: C has 4 friends → weight = 0.722
│   │   ├── Example: D has 3 friends → weight = 0.910
│   │   └── Total: 1.632
│   └── Katz: Σ β^ℓ × paths (most sophisticated)
│
├── **Trust-Centric Recommendation**
│   ├── Concept: Trust relationships instead of similarity
│   ├── Formula: r̂_uj = Σ_v [trust(u,v) × r_vj] / Σ trust
│   ├── Advantages:
│   │   ├── ✅ Robust to attacks (attackers have NO trust)
│   │   ├── ✅ Better cold-start
│   │   ├── ✅ More transparent
│   │   └── ✅ Explicit relationships
│   └── Propagation: Direct, Transitive (diminished), Weighted
│
├── **HITS Algorithm**
│   ├── Hub Score: Pages linking to many authorities
│   ├── Authority Score: Pages receiving links from hubs
│   └── Iterative: Update scores → Normalize → Converge (~20 iter)
│
└── **Social Influence Models**
    ├── Linear Threshold: Adoption based on influenced neighbors
    ├── Cascade: Sequential adoption influence
    └── Independent: Each user makes independent decision
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# UNIT IV
with tab4:
    st.markdown('<div class="unit-title"><h2>🛡️ UNIT IV: Advanced Topics & Robustness</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="mind-map-box">', unsafe_allow_html=True)
    st.markdown("""
## UNIT IV: Advanced Topics
├── **Shilling Attack Detection** ⭐ ATTACKER VARIANCE 4× HIGHER!
│   ├── Normal User vs Attacker Comparison:
│   │   ├── Variance: 0.3-0.5 vs 1.2-2.0 ← FLAG!
│   │   ├── Distribution: [2,3,3,4,4] vs [5,5,5,1,1]
│   │   ├── Time: Spread over weeks vs Burst one day
│   │   └── Items: Real items vs Random/Targeted
│   ├── Detection Formula: var(user) = Σ(r - mean)² / n
│   │   └── Example: Normal var=0.4, Attacker var=3.2 (8× higher!)
│   └── ⚠️ Variance is KEY detection metric!
│
├── **Attack Types** (Impact %)
│   ├── Random: 0-5% (weakest)
│   ├── Average: 5-15%
│   ├── Bandwagon: 15-30%
│   ├── Love-Hate: 20-40% (strongest)
│   └── Sybil: Distributed coordinated
│
├── **Defense Strategies**
│   ├── Trust-Weighted CF: Use trust (attackers have NONE)
│   ├── Robust Matrix Factorization: L1 norm (outliers less influential)
│   ├── Outlier Detection: Remove suspicious accounts
│   └── Ensemble Methods: Multiple algos (fool one, not all)
│
├── **Multi-Armed Bandits**
│   ├── ε-Greedy:
│   │   ├── With prob ε: Explore random
│   │   ├── With prob 1-ε: Exploit best (ε=0.1 typical)
│   │   ├── Regret: O(T) linear
│   │   └── Simple but not optimal
│   └── UCB (Upper Confidence Bound): ⭐ Better!
│       ├── Select: μ̂_a + √(ln(t)/n_a)
│       ├── Auto-balances exploration/exploitation
│       ├── Regret: O(log T) optimal!
│       └── No ε parameter needed
│
├── **Learning to Rank**
│   ├── Pointwise: Individual ratings
│   │   ├── Input: (query, doc, rating)
│   │   └── Loss: MSE (regression)
│   ├── Pairwise: Item pairs ⭐ Most common!
│   │   ├── Input: (query, doc A > doc B)
│   │   └── Loss: Hinge loss
│   └── Listwise: Full lists
│       ├── Input: Query with full ranking
│       └── Loss: NDCG (when precision matters)
│
├── **Group Recommender Systems**
│   ├── Average: r_G = Σr_u / |G| → Fair but may satisfy nobody
│   ├── Least Misery: r_G = min(r_u) → Nobody dislikes
│   ├── Most Pleasure: r_G = max(r_u) → Ignores minority
│   └── Median: r_G = median(r_u) → Balanced
│
└── **Multi-Criteria Recommendation**
    ├── Multiple dimensions: Users × Items × Criteria
    ├── Example: Movie [Plot=5, Music=3, Effects=4]
    ├── Weights: w_plot=0.4, w_music=0.3, w_effects=0.3
    └── Overall: 0.4×5 + 0.3×3 + 0.3×4 = 4.0
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #cccccc; margin-top: 20px;">
        <p><strong>ARD 401 - Recommender Systems Mind Maps</strong></p>
        <p>Visual overview of all 4 units | Complete syllabus coverage</p>
    </div>
""", unsafe_allow_html=True)
