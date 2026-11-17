"""
Precomputes lightweight summary artifacts for the Streamlit dashboard, reusing the
exact logic from SafeCart_Reproducible_Analysis.ipynb so every number matches the
notebook, the deck, and the report exactly. Run once; the dashboard loads the
outputs of this script instead of re-running models/optimization at page-load time.
"""
import pandas as pd, numpy as np, json, pickle, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/'
OUT_DIR = 'data/'

# ============================================================
# 1. Load + clean historical data (same as notebook)
# ============================================================
h = pd.read_csv(DATA_DIR + '02_SafeCart_Historical_Data.csv', low_memory=False)
dup_count = h.duplicated().sum()
h = h.drop_duplicates().reset_index(drop=True)
h['Event_Date_Time'] = pd.to_datetime(h['Event_Date_Time'])
h = h.sort_values('Event_Date_Time').reset_index(drop=True)
base_rate = float(h['Chargeback_Confirmed_30D'].mean())

# ============================================================
# 2. Current-process baseline
# ============================================================
ct = pd.crosstab(h['Manual_Review_Outcome'], h['Chargeback_Confirmed_30D'])
caught = int(ct.loc['Confirmed Fraud', 1])
missed = int(ct.loc['Cleared', 1] + ct.loc['Suspicious', 1])
total_fraud = int(ct[1].sum())
review_catch_rate = caught / total_fraud

# ============================================================
# 3. EDA univariate signals
# ============================================================
eda = {}
for col in ['Device_Known_Flag', 'Password_Reset_48H', 'Shipping_Address_New_Flag']:
    g = h.groupby(col)['Chargeback_Confirmed_30D'].mean()
    eda[col] = {str(k): float(v) for k, v in g.items()}

# ============================================================
# 4. Missingness
# ============================================================
missingness = {}
for col in ['IP_Distance_From_Usual_KM', 'Amount_vs_Customer_Avg']:
    m = h[col].isna()
    missingness[col] = {
        'pct_missing': float(m.mean() * 100),
        'fraud_rate_when_missing': float(h.loc[m, 'Chargeback_Confirmed_30D'].mean()),
        'fraud_rate_when_present': float(h.loc[~m, 'Chargeback_Confirmed_30D'].mean()),
    }

# ============================================================
# 5. Model comparison results (already validated across the whole project;
#    recorded here rather than re-run, since 5-fold CV across 8 model families
#    is the expensive part of the notebook and these numbers are already fixed)
# ============================================================
model_results = {
    "Logistic Regression": 0.808, "Linear SVM (calibrated)": 0.808,
    "Random Forest": 0.795, "Distance-to-centroid baseline": 0.794,
    "HistGradientBoosting": 0.777, "Gaussian Naive Bayes": 0.772,
    "Shallow Decision Tree (d=4)": 0.737, "k-Nearest Neighbors (k=50)": 0.734,
}

# ============================================================
# 6. Load final champion model, score current file (same as notebook)
# ============================================================
with open('final_champion.pkl', 'rb') as f:
    art = pickle.load(f)
model, scaler, cols = art['model'], art['scaler'], art['cols']

NUM_FEATS = ['Customer_Tenure_Months', 'Device_Known_Flag', 'New_Device_Age_Hours',
             'IP_Distance_From_Usual_KM', 'Failed_Logins_24H', 'Password_Reset_48H',
             'Transaction_Amount_EGP', 'Amount_vs_Customer_Avg', 'Shipping_Address_New_Flag']
ALL_FEATS = NUM_FEATS + ['IP_dist_missing', 'Amt_ratio_missing', 'hour', 'dow']
CAT_FEATS = ['Account_Value_Band', 'Payment_Method']

def engineer(df, ip_med, amt_med):
    df = df.copy()
    df['IP_dist_missing'] = df['IP_Distance_From_Usual_KM'].isna().astype(int)
    df['Amt_ratio_missing'] = df['Amount_vs_Customer_Avg'].isna().astype(int)
    df['IP_Distance_From_Usual_KM'] = df['IP_Distance_From_Usual_KM'].fillna(ip_med)
    df['Amount_vs_Customer_Avg'] = df['Amount_vs_Customer_Avg'].fillna(amt_med)
    df['hour'] = pd.to_datetime(df['Event_Date_Time']).dt.hour
    df['dow'] = pd.to_datetime(df['Event_Date_Time']).dt.dayofweek
    return df

cur = pd.read_csv(DATA_DIR + '03_SafeCart_Current_Operational_Cases.csv', low_memory=False)
cur = engineer(cur, art['ip_med'], art['amt_med'])
X_cur = pd.get_dummies(cur[ALL_FEATS + CAT_FEATS], columns=CAT_FEATS).reindex(columns=cols, fill_value=0)
cur['fraud_prob'] = model.predict_proba(scaler.transform(X_cur))[:, 1]

# ============================================================
# 7. LP-optimal policy (same as notebook)
# ============================================================
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

MISS, REVIEW, STEP, FALSEHOLD = 8000, 60, 8, 220
DETERRENCE_POINT = 0.5
p = cur['fraud_prob'].values
n = len(p)
c_approve = p * MISS
c_stepup = STEP + p * (1 - DETERRENCE_POINT) * MISS
c_review = REVIEW + p * (1 - review_catch_rate) * MISS
c_hold = (1 - p) * FALSEHOLD
C = np.column_stack([c_approve, c_stepup, c_review, c_hold]).flatten()
A_eq = lil_matrix((n, n * 4))
for i in range(n): A_eq[i, i*4:(i+1)*4] = 1
b_eq = np.ones(n)
A_ub = lil_matrix((2, n * 4))
for i in range(n):
    A_ub[0, i*4+2] = 1
    A_ub[1, i*4+3] = 1
b_ub = np.array([1800, 350])
res = linprog(C, A_eq=csr_matrix(A_eq), b_eq=b_eq, A_ub=csr_matrix(A_ub), b_ub=b_ub, bounds=(0, 1), method='highs')
X_lp = res.x.reshape(n, 4)
assign = X_lp.argmax(axis=1)
actions = np.array(['Approve', 'Step-up Authentication', 'Manual Review', 'Temporary Hold'])
cur['action'] = actions[assign]

do_nothing = float((p * MISS).sum())
lp_optimal = float(res.fun)
order = np.argsort(-p)
naive2 = np.zeros(n, dtype=int)
naive2[order[:350]] = 3
naive2[order[350:350+1800]] = 2
naive2_cost = float(np.choose(naive2, [c_approve, c_stepup, c_review, c_hold]).sum())

policy_comparison = {
    "Do nothing (approve all)": do_nothing,
    "Naive 2-bucket (no step-up)": naive2_cost,
    "LP-optimal (4 actions)": lp_optimal,
}
action_counts = cur['action'].value_counts().to_dict()

# ============================================================
# 8. Monte Carlo simulation -- save the RAW distribution for a real histogram
# ============================================================
np.random.seed(42)
is_stepup = cur['action'].values == 'Step-up Authentication'
is_review = cur['action'].values == 'Manual Review'
is_hold = cur['action'].values == 'Temporary Hold'
is_approve = cur['action'].values == 'Approve'
N_SIMS = 20000
det_draws = np.random.uniform(0.2, 0.8, N_SIMS)
total_costs = np.empty(N_SIMS)
for k, det in enumerate(det_draws):
    cost = (p[is_approve]*MISS).sum() + (STEP + p[is_stepup]*(1-det)*MISS).sum() \
         + (REVIEW + p[is_review]*(1-review_catch_rate)*MISS).sum() + ((1-p[is_hold])*FALSEHOLD).sum()
    total_costs[k] = cost
loss_prevented_pct = 100 * (do_nothing - total_costs) / do_nothing
np.save(OUT_DIR + 'monte_carlo_loss_prevented_pct.npy', loss_prevented_pct.astype(np.float32))

cost_floor = (p[is_approve]*MISS).sum() + (STEP + p[is_stepup]*1.0*MISS).sum() + (REVIEW+p[is_review]*(1-review_catch_rate)*MISS).sum() + ((1-p[is_hold])*FALSEHOLD).sum()
floor_pct = float(100*(do_nothing-cost_floor)/do_nothing)

# ============================================================
# 9. Fairness -- allocation share + conditional recall in Hold tier
# ============================================================
X_h_all = pd.get_dummies(h.assign(
    IP_dist_missing=h['IP_Distance_From_Usual_KM'].isna().astype(int),
    Amt_ratio_missing=h['Amount_vs_Customer_Avg'].isna().astype(int),
    hour=h['Event_Date_Time'].dt.hour, dow=h['Event_Date_Time'].dt.dayofweek,
)[ALL_FEATS + CAT_FEATS], columns=CAT_FEATS).reindex(columns=cols, fill_value=0)
h_ip = h['IP_Distance_From_Usual_KM'].fillna(art['ip_med'])
h_amt = h['Amount_vs_Customer_Avg'].fillna(art['amt_med'])
X_h_all['IP_Distance_From_Usual_KM'] = h_ip.values
X_h_all['Amount_vs_Customer_Avg'] = h_amt.values
h_scored = h.copy()
h_scored['p'] = model.predict_proba(scaler.transform(X_h_all))[:, 1]
hold_n = int(round(350 / 2500 * len(h_scored)))
order_h = np.argsort(-h_scored['p'].values)
h_scored['in_hold_tier'] = False
h_scored.iloc[order_h[:hold_n], h_scored.columns.get_loc('in_hold_tier')] = True

fairness = {}
for col in ['Customer_Gender', 'Nationality_Group']:
    rows = []
    for grp, d in h_scored.groupby(col):
        fraud_d = d[d['Chargeback_Confirmed_30D'] == 1]
        rows.append({
            'group': str(grp), 'n': int(len(d)), 'fraud_events': int(d['Chargeback_Confirmed_30D'].sum()),
            'pct_in_hold_tier': round(float(d['in_hold_tier'].mean()) * 100, 2),
            'fraud_recall_in_hold_tier_pct': round(float(fraud_d['in_hold_tier'].mean()) * 100, 2) if len(fraud_d) else None,
        })
    fairness[col] = rows

# ============================================================
# 10. Write the CSV the dashboard will page through, plus a summary JSON
# ============================================================
cur.to_csv(OUT_DIR + 'current_scored.csv', index=False)

summary = {
    'base_rate': base_rate, 'n_historical': int(len(h)), 'n_duplicates_dropped': int(dup_count),
    'total_fraud': total_fraud, 'caught_by_review': caught, 'missed_by_review': missed,
    'review_catch_rate': review_catch_rate,
    'eda': eda, 'missingness': missingness, 'model_results': model_results,
    'do_nothing_cost': do_nothing, 'naive2_cost': naive2_cost, 'lp_optimal_cost': lp_optimal,
    'action_counts': {k: int(v) for k, v in action_counts.items()},
    'mc_mean_pct': float(loss_prevented_pct.mean()), 'mc_p5_pct': float(np.percentile(loss_prevented_pct, 5)),
    'mc_p95_pct': float(np.percentile(loss_prevented_pct, 95)), 'floor_pct': floor_pct,
    'fairness': fairness,
    'current_file_stats': {'mean': float(cur['fraud_prob'].mean()), 'median': float(cur['fraud_prob'].median()), 'max': float(cur['fraud_prob'].max())},
}
with open(OUT_DIR + 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Precomputation complete.")
print(f"  Model results: {model_results}")
print(f"  Policy comparison: do_nothing={do_nothing:,.0f}  naive2={naive2_cost:,.0f}  lp_optimal={lp_optimal:,.0f}")
print(f"  Action counts: {action_counts}")
print(f"  Monte Carlo: mean={loss_prevented_pct.mean():.1f}%  [{np.percentile(loss_prevented_pct,5):.1f}%, {np.percentile(loss_prevented_pct,95):.1f}%]  floor={floor_pct:.1f}%")
print(f"  Saved: current_scored.csv, summary.json, monte_carlo_loss_prevented_pct.npy")
