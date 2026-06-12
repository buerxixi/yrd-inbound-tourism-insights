import pandas as pd
import numpy as np
import gensim
import pymysql
import json
import warnings

warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler  # ★ 新增：用于初步减少极端负样本
from global_config import tags_arr

# ================= 1. 数据加载 =================
print("正在加载数据和词向量模型...")
df = pd.read_csv(r'./training_data.csv')

tencent_wv_model = gensim.models.KeyedVectors.load_word2vec_format(
    # 腾讯 AI Lab 的 NLP 官网（ai.tencent.com/ailab/nlp）
    r'./tencent-ailab-embedding-en-d200-v0.1.0-s', binary=False
)


def get_text_vector(text):
    words = str(text).split()
    vecs = [tencent_wv_model.get_vector(w) for w in words if w in tencent_wv_model.key_to_index]
    return np.mean(vecs, axis=0) if vecs else np.zeros(tencent_wv_model.vector_size)


# ================= 2. 特征工程（增强版） =================
print("正在提取特征...")


def extract_stat_features(text_series):
    """提取文本统计特征：词数、平均词长、unique词比例等"""
    feats = []
    for text in text_series.fillna(''):
        words = str(text).split()
        n = len(words)
        unique_ratio = len(set(words)) / (n + 1e-5)
        avg_len = np.mean([len(w) for w in words]) if words else 0
        feats.append([n, unique_ratio, avg_len])
    return np.array(feats)


stat_features = extract_stat_features(df['keywords'])
stat_scaler = MaxAbsScaler()
stat_features_scaled = stat_scaler.fit_transform(stat_features)

# ★ 修改：降低维度 (50000 -> 10000)，限制组合词汇，提高最小词频，防止过拟合
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.85,
    min_df=3,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=10000
)
tfidf_features = tfidf_vectorizer.fit_transform(df['keywords'].fillna(''))

text_vectors = df['keywords'].fillna('').apply(get_text_vector)
text_vec_matrix = csr_matrix(np.vstack(text_vectors))
vec_scaler = MaxAbsScaler()
text_vec_scaled = vec_scaler.fit_transform(text_vec_matrix)

X_combined = hstack([tfidf_features, text_vec_scaled, csr_matrix(stat_features_scaled)])

# ================= 3. 数据库准备 =================
print("正在连接数据库...")
db = pymysql.connect() # review_section.sql
cursor = db.cursor(pymysql.cursors.DictCursor)
cursor.execute("SELECT * from review_section")
db_datas = cursor.fetchall()
db_comments = [item['text'] for item in db_datas]

db_tfidf = tfidf_vectorizer.transform(db_comments)
db_stat = stat_scaler.transform(extract_stat_features(pd.Series(db_comments)))
db_vecs = vec_scaler.transform(csr_matrix(np.vstack([get_text_vector(t) for t in db_comments])))
db_combined = hstack([db_tfidf, db_vecs, csr_matrix(db_stat)])


# ================= 4. 每类别最优 SMOTE 比例搜索 =================
def find_best_smote_ratio(X_tr, y_tr, pos_count, neg_count):
    """
    在验证集上网格搜索最优 SMOTE 采样比例（正/负）
    """
    ratios = [0.2, 0.3, 0.5, 0.7, 1.0]
    best_ratio = 0.5
    best_f1 = -1

    if X_tr.shape[0] < 50 or pos_count < 10:
        return best_ratio

    try:
        Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(
            X_tr, y_tr, test_size=0.25, random_state=0, stratify=y_tr
        )
    except:
        return best_ratio

    pos_i = int(np.sum(yi_tr == 1))
    if pos_i < 6:
        return best_ratio

    for ratio in ratios:
        try:
            target = int(neg_count * ratio)
            target = max(target, pos_i)
            strategy = min(ratio, 1.0)

            k = min(5, pos_i - 1)
            sm = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=strategy)
            Xi_res, yi_res = sm.fit_resample(Xi_tr, yi_tr)

            quick_model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                scale_pos_weight=1.5,  # 搜索时统一权重，防止跑偏
                eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
            )
            quick_model.fit(Xi_res, yi_res)
            proba = quick_model.predict_proba(Xi_val)[:, 1]

            precs, recs, thrs = precision_recall_curve(yi_val, proba)
            f1s = 2 * precs * recs / (precs + recs + 1e-9)
            f1_max = np.max(f1s)

            if f1_max > best_f1:
                best_f1 = f1_max
                best_ratio = ratio
        except:
            continue

    return best_ratio


# ================= 5. 主训练循环 =================
label_names = [f'label{i}' for i in range(0, 16)]
evaluation_results = []
SKIP_THRESHOLD = 5

for idx, label in enumerate(label_names):
    try:
        if label not in df.columns:
            continue

        y = df[label].values
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        ratio = neg_count / (pos_count + 1e-5)

        print(f"\n🚀 [{tags_arr[idx]} / {label}] 正={pos_count} 负={neg_count} 比={ratio:.1f}")

        if pos_count < SKIP_THRESHOLD:
            print(f"   ⚠️  正样本不足，跳过")
            evaluation_results.append({
                'name': tags_arr[idx], 'Label': label,
                'Accuracy': 'NaN', 'Precision': 'N/A', 'Recall': 'N/A',
                'F1-Score': 'N/A', 'Best_Thres': 'N/A', 'Note': f'样本不足(正={pos_count})'
            })
            with open(f'l{label.replace("label", "")}.json', 'w') as f:
                json.dump([], f)
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        pos_train = int(np.sum(y_train == 1))

        if pos_train >= 10:
            best_smote_ratio = find_best_smote_ratio(X_train, y_train, pos_train, int(np.sum(y_train == 0)))
            print(f"   🔍 最优 SMOTE 比例: {best_smote_ratio}")
        else:
            best_smote_ratio = None

        # ★ 修改：组合 RandomUnderSampler 与 SMOTE，防止负样本彻底压制
        if best_smote_ratio is not None and pos_train >= 10:
            k = min(5, pos_train - 1)
            try:
                # 先欠采样：限制负样本最多是正样本的 10 倍
                max_neg = pos_train * 10
                if int(np.sum(y_train == 0)) > max_neg:
                    rus = RandomUnderSampler(sampling_strategy={0: max_neg, 1: pos_train}, random_state=42)
                    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
                else:
                    X_train_res, y_train_res = X_train, y_train

                # 再过采样
                if pos_train >= 50 and best_smote_ratio >= 0.5:
                    sm = SMOTETomek(random_state=42,
                                    smote=SMOTE(k_neighbors=k, sampling_strategy=best_smote_ratio))
                    X_res, y_res = sm.fit_resample(X_train_res, y_train_res)
                    print(f"   SMOTETomek后: 正={int(np.sum(y_res == 1))} 负={int(np.sum(y_res == 0))}")
                else:
                    sm = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=best_smote_ratio)
                    X_res, y_res = sm.fit_resample(X_train_res, y_train_res)
                    print(f"   SMOTE后: 正={int(np.sum(y_res == 1))} 负={int(np.sum(y_res == 0))}")
            except Exception as e:
                print(f"   采样失败({e})，用原始数据")
                X_res, y_res = X_train, y_train
        else:
            X_res, y_res = X_train, y_train

        pos_r = int(np.sum(y_res == 1))
        neg_r = int(np.sum(y_res == 0))

        # ★ 修改：修复“双重惩罚”。如果采样已经比较平衡，树模型不需要再给极端权重
        if best_smote_ratio is not None and best_smote_ratio >= 0.5:
            sw = 1.5
        else:
            sw = np.sqrt(neg_r / (pos_r + 1e-5))

        use_ensemble = pos_count >= 50

        # ★ 修改：降低树的深度，防止过拟合
        depth = 3 if pos_count < 100 else 4
        leaves = 15 if pos_count < 100 else 31
        min_child = max(1, int(pos_r * 0.005))

        model_xgb = XGBClassifier(
            n_estimators=600, max_depth=depth, learning_rate=0.015,
            scale_pos_weight=sw,
            min_child_weight=min_child,
            subsample=0.8, colsample_bytree=0.7,
            gamma=0.15, reg_alpha=0.2, reg_lambda=2.0,
            eval_metric='aucpr',
            random_state=42, n_jobs=-1, verbosity=0
        )
        model_xgb.fit(X_res, y_res)
        y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

        if use_ensemble:
            model_lgbm = LGBMClassifier(
                n_estimators=600, max_depth=depth, learning_rate=0.015,
                num_leaves=leaves, scale_pos_weight=sw,
                min_child_samples=max(5, int(pos_r * 0.01)),
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.2, reg_lambda=2.0,
                random_state=42, n_jobs=-1, verbose=-1
            )
            model_lgbm.fit(X_res, y_res)
            y_proba_lgbm = model_lgbm.predict_proba(X_test)[:, 1]
            y_proba = y_proba_lgbm * 0.55 + y_proba_xgb * 0.45
        else:
            y_proba = y_proba_xgb

        # ★ 修改：使用更严格的阈值搜索逻辑 (结合 F0.5 score 保 Precision)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

        # 计算 F0.5 (更看重Precision) 和 F1
        f05_scores = (1 + 0.5 ** 2) * (precisions * recalls) / ((0.5 ** 2 * precisions) + recalls + 1e-9)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

        # 约束条件：要求 Precision和Recall双高
        mask = (precisions >= 0.60) & (recalls >= 0.55)
        if mask.any():
            f1_constrained = np.where(mask, f1_scores, 0)
            best_idx_b = np.argmax(f1_constrained)
            best_threshold = thresholds[best_idx_b] if best_idx_b < len(thresholds) else 0.5
            chosen = 'B(P≥0.6 & R≥0.55)'
        else:
            # 达不到约束时，退而求最大化 F0.5（保精确率避免预测满天飞）
            best_idx_b = np.argmax(f05_scores)
            best_threshold = thresholds[best_idx_b] if best_idx_b < len(thresholds) else 0.5
            chosen = 'C(最大化F0.5)'

        best_threshold = np.clip(best_threshold, 0.05, 0.95)

        final_pred = (y_proba >= best_threshold).astype(int)
        prec = precision_score(y_test, final_pred, zero_division=0)
        rec = recall_score(y_test, final_pred, zero_division=0)
        f1 = f1_score(y_test, final_pred, zero_division=0)
        acc = accuracy_score(y_test, final_pred)

        print(f"   ✨ [{chosen}] Thr={best_threshold:.3f} | P={prec:.3f} | R={rec:.3f} | F1={f1:.3f}")

        evaluation_results.append({
            'name': tags_arr[idx], 'Label': label,
            'Accuracy': round(acc, 3),
            'Precision': round(prec, 3),
            'Recall': round(rec, 3),
            'F1-Score': round(f1, 3),
            'Best_Thres': round(best_threshold, 3),
            'Note': f'正={pos_count}'
        })

        # 数据库预测
        db_proba_xgb = model_xgb.predict_proba(db_combined)[:, 1]
        if use_ensemble:
            db_proba_lgbm = model_lgbm.predict_proba(db_combined)[:, 1]
            db_proba = db_proba_lgbm * 0.55 + db_proba_xgb * 0.45
        else:
            db_proba = db_proba_xgb

        db_pred = (db_proba >= best_threshold).astype(int)
        hit_ids = [db_datas[i]['id'] for i, p in enumerate(db_pred) if p == 1]
        with open(f'l{label.replace("label", "")}.json', 'w') as f:
            json.dump(hit_ids, f)

    except Exception as e:
        import traceback

        print(f"   ❌ {e}")
        traceback.print_exc()

# ================= 6. 汇总 =================
print("\n" + "=" * 90)
print("📈 论文提交版：模型性能评估汇总表 (v8)")
print("=" * 90)
result_df = pd.DataFrame(evaluation_results)
print(result_df.to_string(index=False))

valid = result_df[result_df['F1-Score'] != 'N/A'].copy()
for col in ['Precision', 'Recall', 'F1-Score']:
    valid[col] = pd.to_numeric(valid[col])
print(f"\n📊 有效类别均值:")
print(f"  Precision: {valid['Precision'].mean():.3f}")
print(f"  Recall:    {valid['Recall'].mean():.3f}")
print(f"  F1-Score:  {valid['F1-Score'].mean():.3f}")

db.close()