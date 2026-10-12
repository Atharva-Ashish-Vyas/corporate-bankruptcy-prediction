"""Reproducible experiment. Never select a model or threshold using test results."""
from pathlib import Path
import json
import time
import platform
import warnings
import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from .data import load_dataset, FEATURES, sha256
from .modeling import candidates, choose_threshold, metrics

SEED = 42
N_FOLDS = 3
COLORS = {"ink": "#142C42", "teal": "#007F78", "red": "#C35A43", "muted": "#617384"}

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")

def save_plot(fig, root, stem):
    fig.tight_layout()
    fig.savefig(root / "reports" / "figures" / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / "reports" / "figures" / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

def make_figures(root, data, audit, cv, y, score, threshold, oof_y, oof_score, importance):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": COLORS["ink"],
                         "text.color": COLORS["ink"], "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(7, 3.3))
    counts = data.bankrupt.value_counts().reindex([0, 1])
    ax.barh(["Non-bankrupt", "Bankrupt"], counts, color=[COLORS["muted"], COLORS["red"]])
    for idx, value in enumerate(counts): ax.text(value + 80, idx, f"{value:,}", va="center")
    ax.set(xlabel="Financial statements after exact duplicate removal", xlim=(0, counts.max()*1.15))
    save_plot(fig, root, "class_balance")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    missing = data[FEATURES].isna().mean().sort_values().tail(8)
    ax.barh(missing.index, missing.values * 100, color=COLORS["teal"])
    ax.set(xlabel="Missing values (%)", title="Features with the most missing values")
    save_plot(fig, root, "missing_values")
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ordered = cv.sort_values("cv_ap_mean")
    ax.barh(ordered.model, ordered.cv_ap_mean, xerr=ordered.cv_ap_std, color=COLORS["teal"], capsize=3)
    ax.axvline(data.bankrupt.mean(), color=COLORS["muted"], linestyle="--", label="Class prevalence")
    ax.set(xlabel="3-fold cross-validation average precision", xlim=(0, 1))
    ax.legend(frameon=False)
    save_plot(fig, root, "model_comparison")
    precision, recall, _ = precision_recall_curve(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(root/"reports"/"pr_curve.csv", index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(root/"reports"/"roc_curve.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].step(recall, precision, where="post", color=COLORS["teal"])
    axes[0].axhline(np.mean(y), color=COLORS["muted"], linestyle="--")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Test AP = {average_precision_score(y, score):.3f}", xlim=(0,1), ylim=(0,1))
    axes[1].plot(fpr, tpr, color=COLORS["teal"])
    axes[1].plot([0,1], [0,1], "--", color=COLORS["muted"])
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title=f"Test ROC AUC = {roc_auc_score(y, score):.3f}", xlim=(0,1), ylim=(0,1))
    save_plot(fig, root, "pr_roc")
    cm = np.array(metrics(y, score, threshold)["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Non-bankrupt", "Bankrupt"], yticklabels=["Non-bankrupt", "Bankrupt"], xlabel="Predicted", ylabel="Actual")
    for (i,j), value in np.ndenumerate(cm):
        ax.text(j,i,f"{value:,}", ha="center", va="center", fontsize=26, color="white" if value > cm.max()/2 else COLORS["ink"])
    save_plot(fig, root, "confusion_matrix")
    pp, rr, tt = precision_recall_curve(oof_y, oof_score)
    f2 = np.divide(5*pp[:-1]*rr[:-1], 4*pp[:-1]+rr[:-1], out=np.zeros_like(tt), where=(4*pp[:-1]+rr[:-1])>0)
    threshold_data = pd.DataFrame({"threshold":tt,"precision":pp[:-1],"recall":rr[:-1],"f2":f2})
    threshold_data.to_csv(root/"reports"/"threshold_curve.csv",index=False)
    fig, ax = plt.subplots(figsize=(7.5,4))
    ax.plot(tt,pp[:-1],label="Precision",color=COLORS["muted"])
    ax.plot(tt,rr[:-1],label="Recall",color=COLORS["teal"])
    ax.plot(tt,f2,label="F2",color=COLORS["red"])
    ax.axvline(threshold,color=COLORS["ink"],linestyle="--",label=f"Chosen {threshold:.3f}")
    ax.set(xlabel="Threshold",ylabel="Development OOF score",ylim=(0,1),xlim=(0,1))
    ax.legend(frameon=False)
    save_plot(fig,root,"threshold_tradeoff")
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    top = importance.head(10).iloc[::-1]
    ax.barh(top.feature,top.importance_mean,xerr=top.importance_std,color=COLORS["teal"],capsize=2)
    ax.set(xlabel="Decrease in test average precision after shuffling",title="Permutation importance (descriptive)")
    save_plot(fig,root,"feature_importance")

def bootstrap_intervals(y, scores, threshold, n=500):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    classes = [np.flatnonzero(y == c) for c in [0,1]]
    observed = {k: [] for k in ["precision","recall","f2","average_precision","roc_auc"]}
    for _ in range(n):
        idx = np.concatenate([rng.choice(c,size=len(c),replace=True) for c in classes])
        result = metrics(y[idx],scores[idx],threshold)
        for key in observed: observed[key].append(result[key])
    return {key: np.percentile(values,[2.5,97.5]).tolist() for key,values in observed.items()}

def train(root):
    root = Path(root)
    for directory in ["reports/figures","models","examples"]: (root/directory).mkdir(parents=True,exist_ok=True)
    data,audit=load_dataset(root/"data/raw/3year.arff")
    x,y=data[FEATURES],data.bankrupt
    dev_idx,test_idx=train_test_split(np.arange(len(data)),test_size=.2,stratify=y,random_state=SEED)
    xd,yd=x.iloc[dev_idx],y.iloc[dev_idx]
    xt,yt=x.iloc[test_idx],y.iloc[test_idx]
    split=pd.DataFrame({"source_row":data.index,"partition":"development"})
    split.loc[test_idx,"partition"]="test"
    split.to_csv(root/"reports/split_manifest.csv",index=False)
    write_json(root/"reports/data_audit.json",audit)
    fold_indices=list(StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED).split(xd,yd))
    comparison=[]; oof={}; model_map=candidates(SEED)
    write_json(root/"reports/experiment_config.json",{"seed":SEED,"test_fraction":.2,"cv_folds":N_FOLDS,"selection_metric":"mean CV average precision","threshold_rule":"maximum development OOF F2","models":{k:str(v) for k,v in model_map.items()}})
    with threadpool_limits(limits=2):
        for name,base in model_map.items():
            print(f"Cross-validating {name}...",flush=True)
            start=time.perf_counter(); scores=np.zeros(len(yd)); aps=[]; aucs=[]
            for fold,(fit_idx,val_idx) in enumerate(fold_indices):
                estimator=clone(base)
                estimator.fit(xd.iloc[fit_idx],yd.iloc[fit_idx])
                p=estimator.predict_proba(xd.iloc[val_idx])[:,1]
                scores[val_idx]=p
                aps.append(average_precision_score(yd.iloc[val_idx],p))
                aucs.append(roc_auc_score(yd.iloc[val_idx],p))
            oof[name]=scores
            row={"model":name,"cv_ap_mean":float(np.mean(aps)),"cv_ap_std":float(np.std(aps,ddof=1)),"cv_roc_auc_mean":float(np.mean(aucs)),"seconds":round(time.perf_counter()-start,2)}
            comparison.append(row)
            print(f"  AP {row['cv_ap_mean']:.4f} +/- {row['cv_ap_std']:.4f}",flush=True)
        cv=pd.DataFrame(comparison).sort_values("cv_ap_mean",ascending=False).reset_index(drop=True)
        selected=cv.iloc[0].model
        threshold=choose_threshold(yd.to_numpy(),oof[selected])
        print(f"Selected {selected}; OOF threshold {threshold:.6f}",flush=True)
        final=clone(model_map[selected]).fit(xd,yd)
        test_scores=final.predict_proba(xt)[:,1]
        result=metrics(yt,test_scores,threshold)
        baseline=metrics(yt,np.zeros(len(yt)),.5)
        intervals=bootstrap_intervals(yt,test_scores,threshold)
        print("Computing descriptive permutation importance...",flush=True)
        imp=permutation_importance(final,xt,yt,scoring="average_precision",n_repeats=3,random_state=SEED,n_jobs=1)
    importance=pd.DataFrame({"feature":FEATURES,"importance_mean":imp.importances_mean,"importance_std":imp.importances_std}).sort_values("importance_mean",ascending=False)
    importance.to_csv(root/"reports/feature_importance.csv",index=False)
    cv.to_csv(root/"reports/model_comparison.csv",index=False)
    pd.DataFrame({"source_row":xd.index,"actual":yd.to_numpy(),"oof_score":oof[selected]}).to_csv(root/"reports/oof_predictions.csv",index=False)
    test_predictions=pd.DataFrame({"source_row":xt.index,"actual":yt.to_numpy(),"risk_score":test_scores,"predicted":(test_scores>=threshold).astype(int)})
    test_predictions.to_csv(root/"reports/test_predictions.csv",index=False)
    report={"selected_model":selected,"seed":SEED,"audit":audit,
            "split":{"development":len(yd),"development_positive":int(yd.sum()),"test":len(yt),"test_positive":int(yt.sum())},
            "cv":cv.to_dict(orient="records"),"test":result,"test_default_threshold":metrics(yt,test_scores,.5),
            "baseline":baseline,"bootstrap_95_intervals":intervals,"bootstrap_replicates":500,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__}}
    write_json(root/"reports/metrics.json",report)
    bundle={"model":final,"threshold":threshold,"features":FEATURES,"name":selected,"seed":SEED}
    model_path=root/"models/bankruptcy_model.joblib"
    joblib.dump(bundle,model_path,compress=3)
    write_json(root/"models/manifest.json",{"sha256":sha256(model_path),"model":selected,"threshold":threshold,"versions":report["versions"]})
    xt.head(10).to_csv(root/"examples/sample_companies.csv",index=False)
    test_predictions.head(10).to_csv(root/"examples/sample_expected_results.csv",index=False)
    demos=[]
    for actual,predicted,label in [(1,1,"Detected bankruptcy"),(0,0,"Correct non-bankruptcy"),(1,0,"Missed bankruptcy"),(0,1,"False alarm")]:
        matching=test_predictions.query("actual == @actual and predicted == @predicted")
        if len(matching):
            row=matching.iloc[0]; features=xt.loc[int(row.source_row)].replace({np.nan:None}).to_dict()
            demos.append({"label":label,"source_row":int(row.source_row),"actual":actual,"features":features})
    write_json(root/"examples/demo_cases.json",demos)
    make_figures(root,data,audit,cv,yt.to_numpy(),test_scores,threshold,yd.to_numpy(),oof[selected],importance)
    print(json.dumps({"selected_model":selected,"test":result,"audit":audit},indent=2),flush=True)
