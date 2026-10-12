import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from bankruptcy.data import FEATURES, load_dataset, validate_features
from bankruptcy.modeling import choose_threshold, metrics
from bankruptcy.predict import load_bundle, predict_frame

ROOT=Path(__file__).resolve().parents[1]

def test_cleaning_and_partition_leakage():
    data,audit=load_dataset(ROOT/"data/raw/3year.arff")
    assert audit["raw_rows"]==10503 and audit["exact_duplicates_removed"]==87
    assert len(data)==10416 and data.bankrupt.sum()==493
    split=pd.read_csv(ROOT/"reports/split_manifest.csv").set_index("source_row")
    assert split.index.is_unique and set(split.index)==set(data.index)
    a=data.loc[split.index[split.partition=="development"],FEATURES]
    b=data.loc[split.index[split.partition=="test"],FEATURES]
    assert set(pd.util.hash_pandas_object(a,index=False)).isdisjoint(set(pd.util.hash_pandas_object(b,index=False)))

def test_schema_reorders_and_rejects_bad_inputs():
    valid=pd.DataFrame([[1.0]*64],columns=FEATURES)
    assert list(validate_features(valid[FEATURES[::-1]]))==FEATURES
    missing=valid.copy();missing.loc[0,"X4"]=np.nan
    assert np.isnan(validate_features(missing).loc[0,"X4"])
    with pytest.raises(ValueError,match="Missing"):
        validate_features(valid.drop(columns="X1"))
    bad=valid.astype(object);bad.loc[0,"X1"]="hello"
    with pytest.raises(ValueError,match="nonnumeric"):
        validate_features(bad)
    with pytest.raises(ValueError,match="all 64"):
        validate_features(valid*float("nan"))

def test_threshold_and_constant_score_metrics():
    y=np.array([0,0,0,1]); score=np.zeros(4)
    result=metrics(y,score,.5)
    assert result["average_precision"]==.25 and result["recall"]==0
    assert result["roc_auc"]==.5
    scores=np.array([.1,.2,.3,.8])
    assert choose_threshold(y,scores)==.8

def test_saved_model_batch_and_single_agree():
    bundle=load_bundle(ROOT)
    frame=pd.read_csv(ROOT/"examples/sample_companies.csv")
    batch=predict_frame(bundle,frame)
    single=predict_frame(bundle,frame.iloc[[0]])
    assert np.isclose(batch.iloc[0].risk_score,single.iloc[0].risk_score)
    expected=pd.read_csv(ROOT/"examples/sample_expected_results.csv")
    assert np.allclose(batch.risk_score,expected.risk_score)
    assert np.array_equal(batch.predicted_bankrupt,expected.predicted)

def test_report_confusion_totals():
    report=json.loads((ROOT/"reports/metrics.json").read_text())
    cm=np.array(report["test"]["confusion_matrix"])
    assert cm.sum()==report["split"]["test"]
    assert cm[1].sum()==report["split"]["test_positive"]
    assert np.isclose(report["test"]["recall"],cm[1,1]/cm[1].sum())

def test_web_demo_valid_and_invalid_upload():
    from app import create_app
    client=create_app().test_client()
    assert client.get("/api/health").status_code==200
    assert client.get("/").status_code==200
    case=client.get("/api/example/0")
    assert case.status_code==200 and "risk_score" in case.json["result"]
    assert client.get("/api/example/99").status_code==404
    response=client.post("/api/predict",data={"file":(io.BytesIO((ROOT/"examples/sample_companies.csv").read_bytes()),"input.csv")},content_type="multipart/form-data")
    assert response.status_code==200 and len(response.json["rows"])==10
    bad=client.post("/api/predict",data={"file":(io.BytesIO(b"X1\n1\n"),"bad.csv")},content_type="multipart/form-data")
    assert bad.status_code==400 and "X64" in bad.json["error"]
