import streamlit as st

st.set_page_config(
    page_title="ARD 401 - Mind Maps",
    page_icon="🧠",
    layout="wide"
)

# Clean styling - dark background, white text, proper vertical formatting
st.markdown("""
    <style>
    body {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .mind-map-container {
        background-color: #1a1a1a;
        border-left: 5px solid #667eea;
        padding: 25px;
        border-radius: 8px;
        margin: 20px 0;
        color: #ffffff;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .unit-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="unit-header">
        <h1 style="margin: 0; color: white;">🧠 ARD 401 - Mind Maps</h1>
        <p style="margin: 10px 0; font-size: 1.1em; color: white;">Recommender Systems | Complete Vertical View</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📌 Unit I", "📈 Unit II", "🌐 Unit III", "🛡️ Unit IV"])

# UNIT I
with tab1:
    st.markdown('<div class="unit-header"><h2>📌 UNIT I: Fundamentals & Collaborative Filtering</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mind-map-container">
UNIT I: FUNDAMENTALS & CF
═════════════════════════════════════

📌 RECOMMENDER SYSTEMS BASICS
├─ Goals
│  ├─ Personalization
│  ├─ Discovery
│  ├─ Engagement
│  └─ Retention & Revenue
├─ Challenges
│  ├─ Cold Start (new users/items)
│  ├─ Sparsity (99.9% matrix empty)
│  ├─ Scalability (O(m²) complexity)
│  ├─ Diversity (avoid boring recs)
│  └─ Bias (popularity, user bias)
└─ Types
   ├─ Content-Based
   ├─ Collaborative Filtering
   └─ Hybrid

👥 USER-BASED COLLABORATIVE FILTERING
├─ Concept: Similar users → Similar preferences
├─ Algorithm (Step-by-step):
│  1. Calculate mean rating: r̄_u = Σr_ui / n
│  2. Find overlapping items (common rated items only)
│  3. Compute Pearson correlation on overlapping
│  4. Select k-nearest neighbors (k=10-20)
│  5. Weighted average: r̂_uj = r̄_u + Σ sim(u,v)×(r_vj - r̄_v) / Σ|sim|
├─ Key Similarity Ranges:
│  ├─ Pearson: -1 to +1
│  ├─ Similar users: 0.7 to 1.0
│  ├─ Moderate: 0.4 to 0.7
│  └─ Dissimilar: < 0.4
└─ ⚠️  CRITICAL: ALWAYS mean-center (r_u - r̄_u)

📦 ITEM-BASED COLLABORATIVE FILTERING
├─ Concept: Similar items → Rated similarly
├─ Formula: r̂_uj = Σ sim(i,j)×r_ui / Σ|sim|
├─ ✅ Advantages:
│  ├─ More stable than user-based
│  ├─ Better for new users (need only 1 rating)
│  ├─ Cacheable (compute offline)
│  └─ Predictable performance
└─ ⚠️  Note: Exclude negative similarities (-0.94 to 1.0)

⚡ MATRIX FACTORIZATION (SVD)
├─ Concept: R ≈ U × V^T
│  ├─ m × k user latent matrix
│  └─ n × k item latent matrix
├─ Prediction: r̂_ij = u_i · v_j
├─ Error Calculation: e_ij = r_ij - r̂_ij
├─ SGD Update (MOST IMPORTANT):
│  ├─ u_i ← u_i + γ(e_ij × v_j - λ × u_i)
│  ├─ v_j ← v_j + γ(e_ij × u_i - λ × v_j)
│  └─ ⭐ NEVER FORGET λ term (prevents overfitting!)
├─ Parameters:
│  ├─ γ (learning rate): 0.001-0.1
│  │  ├─ Too high → oscillates
│  │  └─ Too low → slow convergence
│  ├─ λ (regularization): 0.001-0.01
│  │  └─ Controls overfitting on sparse data
│  ├─ k (latent factors): 20-100
│  │  └─ Number of hidden dimensions
│  └─ Convergence: 20-50 iterations
└─ ⚠️  CRITICAL: Always include λ×u_i regularization!

🎯 KEY CHALLENGES
├─ ❄️  Cold Start
│  ├─ Problem: New user/item → No ratings exist
│  └─ Solution: Content-based, Hybrid, Knowledge-based
├─ 📉 Sparsity
│  ├─ Problem: 99.9% of matrix is empty
│  └─ Solution: Dimensionality reduction, Clustering
├─ ⚡ Scalability
│  ├─ Problem: O(m²) complexity (too slow)
│  └─ Solution: Item-based CF, Caching
├─ ⚖️  Diversity
│  ├─ Problem: High accuracy = boring recommendations
│  └─ Solution: Balance via regularization parameter λ
└─ 👥 Bias
   ├─ Problem: Popular items rated higher (natural bias)
   └─ Solution: Debiasing techniques, Fairness metrics
    </div>
    """, unsafe_allow_html=True)

# UNIT II
with tab2:
    st.markdown('<div class="unit-header"><h2>📈 UNIT II: Evaluation & Context-Aware Systems</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mind-map-container">
UNIT II: EVALUATION METRICS
════════════════════════════════════

📊 EVALUATION PARADIGMS
├─ Offline Evaluation
│  ├─ Method: Split data into 80% train, 20% test
│  ├─ ✅ Pros: Fast, cheap, repeatable
│  └─ ❌ Cons: Metrics ≠ real user behavior
├─ Online A/B Testing
│  ├─ Method: Real users see algorithm A vs B
│  ├─ ✅ Pros: Real behavior, business metrics
│  └─ ❌ Cons: Expensive, slow, risky
└─ User Study
   ├─ Method: Recruit N=20-100 participants
   ├─ ✅ Pros: Capture subjective aspects (satisfaction)
   └─ ❌ Cons: Small sample, low generalizability

📊 RATING PREDICTION METRICS (Regression)
├─ MAE (Mean Absolute Error)
│  ├─ Formula: Σ|r - r̂| / n
│  ├─ Typical: 0.3-0.7 stars
│  └─ Easy to interpret (average error in stars)
├─ RMSE (Root Mean Squared Error)
│  ├─ Formula: √[Σ(r - r̂)² / n]
│  ├─ Typical: 0.3-1.0 stars
│  └─ ⭐ MOST COMMONLY USED
└─ MSE (Mean Squared Error)
   ├─ Formula: Σ(r - r̂)² / n
   └─ Same as RMSE²

📊 RANKING METRICS (Most Important!)
├─ Precision@k
│  ├─ Formula: (#relevant in top-k) / k
│  ├─ Typical: 0.4-0.7
│  ├─ Question: What % of recommendations are good?
│  └─ k=10 is common
├─ Recall@k
│  ├─ Formula: (#relevant in top-k) / (total relevant)
│  ├─ Typical: 0.5-1.0
│  ├─ Question: What % of user's items did we find?
│  └─ Higher k = Higher recall
├─ NDCG@k ⭐ MOST SOPHISTICATED
│  ├─ Full name: Normalized Discounted Cumulative Gain
│  ├─ Formula: DCG / IDCG
│  ├─ Why it matters: Position matters!
│  │  ├─ Item at position 1 = worth more
│  │  ├─ Item at position 10 = worth less
│  │  └─ Log scale penalizes lower positions
│  ├─ DCG Calculation:
│  │  ├─ DCG = Σ [2^rel_i - 1] / log₂(i+1)
│  │  ├─ Relevant item at pos 1: (2¹-1) / log₂(2) = 1.0
│  │  ├─ Irrelevant item at pos 2: 0 / log₂(3) = 0
│  │  ├─ Relevant item at pos 3: (2¹-1) / log₂(4) = 0.5
│  │  └─ Sum these up = DCG value
│  ├─ IDCG: DCG if all items were ranked perfectly
│  ├─ NDCG = DCG / IDCG (normalized between 0-1)
│  ├─ Typical: 0.5-0.8
│  └─ ⚠️  CRITICAL: Use log₂(i+1), NOT log(i)!
└─ MAP (Mean Average Precision)
   ├─ Formula: Σ(Precision@k for each relevant) / |relevant|
   ├─ Typical: 0.4-0.8
   └─ Captures precision at each relevant position

⏰ TEMPORAL COLLABORATIVE FILTERING
├─ Why it matters: User preferences change over time
├─ Exponential Decay Model:
│  ├─ Formula: w(t) = e^{-λ(t_current - t)}
│  ├─ λ = 0.01 (typical value)
│  ├─ Half-life ≈ 70 days
│  └─ Example: 1-day-old rating 2.3× heavier than 95-day-old
├─ Time-SVD++ Model:
│  ├─ Formula: r̂_uit = μ + b_u(t) + b_i(t) + Σ_k u_uk × v_ik(t)
│  ├─ b_u(t) = user bias that changes over time
│  ├─ b_i(t) = item bias that changes over time
│  └─ Captures both user drift AND item popularity trends
└─ Key insight: Recent ratings matter more!

🌐 CONTEXT-AWARE SYSTEMS
├─ Multiple dimensions: Users × Items × Context
├─ Example contexts:
│  ├─ Location (home, work, traveling)
│  ├─ Time (morning, evening, weekend)
│  ├─ Device (phone, tablet, desktop)
│  ├─ Weather (sunny, rainy, snowy)
│  └─ Social (alone, with friends, at party)
├─ Multi-criteria example (Movie):
│  ├─ Plot rating: 5 stars
│  ├─ Music rating: 3 stars
│  ├─ Effects rating: 4 stars
│  ├─ Weights: [0.4, 0.3, 0.3]
│  └─ Overall: 0.4×5 + 0.3×3 + 0.3×4 = 4.0
└─ Approach: Factorize multi-dimensional tensor
    </div>
    """, unsafe_allow_html=True)

# UNIT III
with tab3:
    st.markdown('<div class="unit-header"><h2>🌐 UNIT III: Structural Recommendations in Networks</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mind-map-container">
UNIT III: NETWORKS & LINK PREDICTION
═════════════════════════════════════

🔗 PAGERANK ALGORITHM ⭐ CORE
├─ Concept: Important pages get links from important pages
├─ Real-world: Google uses PageRank for search ranking
├─ Formula: PR(p) = (1-d)/N + d × Σ_{q→p} [PR(q) / out(q)]
├─ Parameters:
│  ├─ d = damping factor = 0.85
│  │  ├─ Probability to follow a link = 85%
│  │  └─ Probability to teleport = 15%
│  ├─ (1-d)/N = teleport probability
│  │  ├─ With N=20 pages: (1-0.85)/20 ≈ 0.0075
│  │  └─ Each page gets equal 0.0075
│  ├─ PR(q) = PageRank of page q linking to p
│  │  └─ Vote from q depends on its own importance
│  └─ out(q) = number of outgoing links from q
│     └─ Divide PR equally among all outgoing links
├─ Calculation Example (3 pages):
│  ├─ Initial: PR(A)=PR(B)=PR(C)=1/3 ≈ 0.333
│  ├─ Iteration 1:
│  │  ├─ PR(A) = 0.05 + 0.85×(...calculations...) = 0.05
│  │  ├─ PR(B) = 0.05 + 0.85×(...calculations...) = 0.192
│  │  └─ PR(C) = 0.05 + 0.85×(...calculations...) = 0.475
│  ├─ Iteration 2: Recalculate using new PR values
│  └─ Convergence: ~20 iterations, then stabilizes
└─ ⚠️  CRITICAL: Always normalize so Σ PR = 1!
   └─ Sum of all PageRanks must equal 1.0

🔍 LINK PREDICTION METRICS
├─ Common Neighbors (CN)
│  ├─ Formula: |N(A) ∩ N(B)|
│  ├─ Simplest approach
│  ├─ Example: A and B have 2 mutual friends
│  └─ CN(A,B) = 2
├─ Jaccard Index
│  ├─ Formula: |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
│  ├─ Normalized version of CN
│  ├─ Range: 0 to 1
│  └─ Example: If A,B have 2 commons, 4 total = 2/4 = 0.5
├─ Adamic-Adar ⭐ USUALLY BEST
│  ├─ Formula: Σ_{w ∈ N(A)∩N(B)} [1 / log|N(w)|]
│  ├─ Key: Weight mutual friends by their degree
│  │  ├─ Friend with few friends → higher weight
│  │  └─ Friend with many friends → lower weight
│  ├─ Example:
│  │  ├─ Mutual friend C has 4 total friends
│  │  │  ├─ Weight = 1/log(4) = 0.722
│  │  ├─ Mutual friend D has 3 total friends
│  │  │  └─ Weight = 1/log(3) = 0.910
│  │  └─ AA(A,B) = 0.722 + 0.910 = 1.632
│  ├─ Intuition: Rare common connections are more valuable
│  └─ Typical performance: Better than CN and Jaccard
└─ Katz Index (Most sophisticated)
   ├─ Formula: Σ_ℓ β^ℓ × (# paths of length ℓ)
   ├─ Considers ALL paths between nodes
   ├─ β = damping factor (0 < β < 1)
   ├─ Path of length 1: Direct connection
   ├─ Path of length 2: Through 1 intermediate
   ├─ Path of length 3: Through 2 intermediates
   └─ Longer paths get exponentially less weight

👥 TRUST-CENTRIC RECOMMENDATION
├─ Concept: Use explicit trust instead of implicit similarity
├─ Formula: r̂_uj = Σ_v [trust(u,v) × r_vj] / Σ trust
├─ Why better than CF:
│  ├─ ✅ Robust to attacks (attackers have NO trust)
│  ├─ ✅ Better cold-start (explicit trust available)
│  ├─ ✅ More transparent (users understand why)
│  └─ ✅ Explicit relationships (more reliable)
├─ Trust Propagation:
│  ├─ Direct trust: A→B only
│  ├─ Transitive trust: A→B→C (diminished by distance)
│  │  ├─ trust(A,C) = trust(A,B) × trust(B,C) × decay
│  │  └─ Decay = e^{-λ×distance}
│  ├─ Weighted trust: Different trust levels per edge
│  └─ Filtered trust: Only high-trust edges matter
└─ Real-world: Epinions, Slashdot use trust networks

📊 HITS ALGORITHM
├─ Full name: Hypertext Induced Topic Search
├─ Two scores per node:
│  ├─ Hub Score: How many authorities does it link to?
│  │  └─ Good hub = links to many good authorities
│  └─ Authority Score: How many hubs link to it?
│     └─ Good authority = many good hubs link to it
├─ Iterative algorithm:
│  ├─ Step 1: Initialize all scores = 1/N
│  ├─ Step 2: For each iteration:
│  │  ├─ authority(p) = Σ hub(q) for all q→p
│  │  └─ hub(p) = Σ authority(r) for all p→r
│  ├─ Step 3: Normalize both scores (Σ = 1)
│  └─ Step 4: Repeat until convergence (~20 iterations)
└─ vs PageRank: HITS is query-specific, PR is global

🌐 SOCIAL INFLUENCE MODELS
├─ Linear Threshold Model
│  ├─ User adopts when influenced neighbors ≥ threshold
│  ├─ Example: Buy item if 3+ friends bought it
│  └─ Deterministic (threshold-based)
├─ Cascade Model
│  ├─ Sequential adoption spread through network
│  ├─ Example: User sees friend bought → might buy
│  └─ Probabilistic (influenced neighbors)
└─ Independent Cascade Model
   ├─ Each user makes independent decision
   ├─ Influenced by neighbors but randomized
   └─ More realistic for real social networks
    </div>
    """, unsafe_allow_html=True)

# UNIT IV
with tab4:
    st.markdown('<div class="unit-header"><h2>🛡️ UNIT IV: Advanced Topics & Robustness</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mind-map-container">
UNIT IV: ADVANCED & ROBUSTNESS
═══════════════════════════════════

🚨 SHILLING ATTACK DETECTION ⚠️  KEY METRIC: VARIANCE
├─ What is shilling: Fake accounts rating to manipulate recommendations
├─ Attacker vs Normal User Comparison:
│  ├─ VARIANCE (Most important!):
│  │  ├─ Normal user: 0.3-0.5 (consistent preferences)
│  │  └─ Attacker: 1.2-2.0 (random or biased) ⭐ 4× HIGHER!
│  ├─ Distribution pattern:
│  │  ├─ Normal: [2,3,3,4,4] = balanced, around mean
│  │  └─ Attacker: [5,5,5,1,1] = bimodal, polarized
│  ├─ Temporal pattern:
│  │  ├─ Normal: Spread over weeks/months
│  │  └─ Attacker: Burst in single day
│  └─ Item selection:
│     ├─ Normal: Items they've actually seen/used
│     └─ Attacker: Random or strategically targeted
├─ Detection Formula:
│  ├─ variance(user) = Σ(r - mean)² / n
│  └─ Example:
│     ├─ Normal: [2,3,3,4,4]
│     │  ├─ Mean = 3.2
│     │  └─ Variance = (0.04+0.04+0.04+0.64+0.64)/5 = 0.28
│     ├─ Attacker: [5,5,5,1,1]
│     │  ├─ Mean = 3.2
│     │  └─ Variance = (3.24+3.24+3.24+4.84+4.84)/5 = 3.88
│     └─ Ratio: 3.88/0.28 = 13.9× higher! ⚠️
└─ ⚠️  CRITICAL: High variance = likely attacker!

🎯 ATTACK TYPES (Impact %)
├─ Random Attack (0-5% impact) - Weakest
│  ├─ Rate random items with random ratings
│  └─ No pattern, easily detected
├─ Average Attack (5-15% impact)
│  ├─ Rate target item: 5 stars
│  ├─ Rate popular items: 3 stars (average)
│  └─ Slight variance, moderate impact
├─ Bandwagon Attack (15-30% impact)
│  ├─ Target item: 5 stars
│  ├─ Popular items: 5 stars
│  ├─ Unpopular items: 1 star
│  └─ Higher impact, moderate detection difficulty
├─ Love-Hate Attack (20-40% impact) - Strongest
│  ├─ Target item: 5 stars (maximize target)
│  ├─ Competitor items: 1 star (minimize competition)
│  ├─ Others: Strategic (1, 3, or 5 based on impact)
│  └─ Most dangerous, hardest to detect
└─ Sybil Attack - Distributed
   ├─ Multiple coordinated fake accounts
   ├─ Can execute complex strategies
   └─ Hardest to detect (network-level attack)

🛡️ DEFENSE STRATEGIES
├─ 1. Trust-Weighted Collaborative Filtering
│  ├─ Use explicit trust instead of similarity
│  ├─ Attackers have NO trust (no history)
│  ├─ Formula: r̂ = Σ trust(u,v) × r_vj / Σ trust
│  └─ Effectiveness: Very high (attackers isolated)
├─ 2. Robust Matrix Factorization
│  ├─ Use L1 norm instead of L2 norm
│  │  ├─ L1 penalty: λ|w| (linear)
│  │  └─ L2 penalty: λw² (quadratic, current)
│  ├─ L1 makes outliers less influential
│  └─ Attacks affect fewer items
├─ 3. Outlier Detection & Removal
│  ├─ Identify suspicious accounts via variance
│  ├─ Remove before training RS
│  └─ Risk: False positives (legitimate users flagged)
└─ 4. Ensemble Methods
   ├─ Multiple algorithms = multiple defense layers
   ├─ Attackers fool one, not all
   ├─ Final prediction = aggregate (average, median)
   └─ More robust but slower

🎰 MULTI-ARMED BANDITS (Exploration-Exploitation)
├─ Problem: Balance between trying new items vs recommending known good
├─ ε-Greedy Algorithm:
│  ├─ With probability ε: Explore random arm (ε=0.1 typical)
│  ├─ With probability 1-ε: Exploit best arm so far (0.9)
│  ├─ Simple and fast
│  ├─ Regret: O(T) linear - suboptimal
│  └─ Used in: Early-stage recommendations
└─ UCB (Upper Confidence Bound) - Better!
   ├─ Select arm maximizing: μ̂_a + √(ln(t) / n_a)
   ├─ Auto-balances: Uncertainty + empirical mean
   │  ├─ New arm (high uncertainty) = higher UCB
   │  └─ Tested arm (low uncertainty) = lower UCB
   ├─ No need for ε parameter (automatic)
   ├─ Regret: O(log T) optimal!
   └─ Used in: Contextual bandits, online learning

📊 LEARNING TO RANK (LTR)
├─ Problem: How to train model for ranking quality?
├─ Pointwise Approach:
│  ├─ Input: (query, document, relevance score)
│  ├─ Loss: MSE or cross-entropy (regression)
│  ├─ Treats each doc independently
│  ├─ ❌ Ignores relative ranking
│  └─ Use: Baseline, simple systems
├─ Pairwise Approach: ⭐ MOST COMMON
│  ├─ Input: (query, doc A > doc B)
│  │  └─ Pair where A is more relevant than B
│  ├─ Loss: Hinge loss (margin between pairs)
│  ├─ Learns relative ordering
│  ├─ ✅ Considers ranking structure
│  └─ Use: LambdaRank, RankNet
└─ Listwise Approach:
   ├─ Input: (query, full ranking list)
   ├─ Loss: NDCG (or other ranking metric)
   ├─ Optimizes full ranking quality
   ├─ ✅ Directly optimizes final metric
   └─ Use: LambdaMART, ListNet (when NDCG is critical)

👥 GROUP RECOMMENDER SYSTEMS
├─ Problem: Recommend to group of users, not single person
├─ Aggregation Strategies:
│  ├─ Average: r_G = Σr_u / |G|
│  │  ├─ Fair (treats all equally)
│  │  ├─ Example: [5,3,4]/3 = 4.0
│  │  └─ ❌ May satisfy nobody (4 stars to all)
│  ├─ Least Misery: r_G = min(r_u)
│  │  ├─ Conservative (nobody dislikes)
│  │  ├─ Example: min(5,3,4) = 3
│  │  └─ ❌ Often too low (limited choices)
│  ├─ Most Pleasure: r_G = max(r_u)
│  │  ├─ Optimistic (maximize happiness)
│  │  ├─ Example: max(5,3,4) = 5
│  │  └─ ❌ Ignores minority dislike
│  └─ Median: r_G = median(r_u)
│     ├─ Balanced compromise
│     ├─ Example: median(5,3,4) = 4
│     └─ ✅ Often best balance
└─ Variants: Threshold aggregation, weighted voting

📊 MULTI-CRITERIA RECOMMENDATION
├─ Problem: Single rating inadequate (multiple dimensions matter)
├─ Multiple dimensions:
│  ├─ Users × Items × Criteria
│  ├─ Example movie: [Plot, Music, Effects, Acting]
│  └─ Each rated separately
├─ Tensor Approach:
│  ├─ 3-way tensor: n_users × n_items × n_criteria
│  ├─ Factorize: U × I × C
│  └─ Predict each criterion separately
├─ Weighted Aggregation:
│  ├─ Example movie ratings:
│  │  ├─ Plot: 5 stars
│  │  ├─ Music: 3 stars
│  │  ├─ Effects: 4 stars
│  │  └─ Acting: 4 stars
│  ├─ Weights (user preferences): [0.4, 0.2, 0.2, 0.2]
│  ├─ Overall: 0.4×5 + 0.2×3 + 0.2×4 + 0.2×4 = 4.2
│  └─ Dynamic weights: Can change per user/context
└─ Benefits: Better satisfaction, domain-specific evaluation
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #888888; margin-top: 20px;">
        <p><strong>ARD 401 - Recommender Systems Mind Maps</strong></p>
        <p>All Units | Vertical Format | 100% Legible</p>
    </div>
""", unsafe_allow_html=True)
