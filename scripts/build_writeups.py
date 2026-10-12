"""Regenerate both write-ups from verified results. pip install reportlab"""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"deliverables"
R=json.loads((ROOT/"reports/metrics.json").read_text())
M=R["test"]; D=R["test_default_threshold"]
TEAM=json.loads((ROOT/"docs/team_details.json").read_text())
INK=colors.HexColor('#142C42');TEAL=colors.HexColor('#007F78');GRAY=colors.HexColor('#526677');RULE=colors.HexColor('#D5DEDF')

def style(size=9.7,leading=13.3):
    return ParagraphStyle('body',fontName='Helvetica',fontSize=size,leading=leading,textColor=INK,spaceAfter=5)
B=style(); SMALL=style(8,10.6)
HEAD=ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=11.3,leading=15,textColor=TEAL,spaceBefore=9,spaceAfter=4)
TITLE=ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=INK,spaceAfter=7)
def p(text,s=B): return Paragraph(text,s)
def h(text):return p(text,HEAD)
def team_line():
    first=TEAM.get('student_1','') or '____________________'
    second=TEAM.get('student_2','') or '____________________'
    id1=TEAM.get('student_id_1','') or '____________'
    id2=TEAM.get('student_id_2','') or '____________'
    return f'Team: {first} ({id1}) &nbsp; / &nbsp; {second} ({id2})'
def title(subtitle):
    return [p('Corporate Bankruptcy Prediction',TITLE),p('UE24CS352A - Machine Learning | '+subtitle,SMALL),p(team_line(),SMALL),Spacer(1,5*mm)]
def table(rows,widths,font=8.5):
    t=Table([[p('<font color="white"><b>'+str(c)+'</b></font>' if i==0 else str(c),style(font,font+3)) for c in row] for i,row in enumerate(rows)],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.4,INK),('LINEBELOW',(0,1),(-1,-1),.3,RULE)]))
    for c in t._cellvalues[0]:
        c.style.textColor=colors.white
    return t
def footer(canvas,doc):
    canvas.setStrokeColor(RULE);canvas.line(18*mm,15*mm,A4[0]-18*mm,15*mm)
    canvas.setFont('Helvetica',7.5);canvas.setFillColor(GRAY)
    canvas.drawString(18*mm,10.5*mm,'Corporate Bankruptcy Prediction | Reproducible mini-project')
    canvas.drawRightString(A4[0]-18*mm,10.5*mm,str(doc.page))
def build(name,story):
    OUT.mkdir(exist_ok=True)
    doc=SimpleDocTemplate(str(OUT/name),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,bottomMargin=20*mm,title='Corporate Bankruptcy Prediction',author='Project team')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def two_pages():
    s=title('Project write-up')
    s += [p('<b>Abstract.</b> This project predicts bankruptcy within three years using financial ratios from Polish companies. Six classifiers are compared using development-set cross-validation. Histogram gradient boosting is selected, and a screening threshold is chosen from out-of-fold predictions. On the independent test set, the model detects 84 of 99 bankrupt statements, with 34.3% precision and 0.750 average precision. The remaining false alarms and missed cases define the main limitation.'),
          h('1. Problem and objective'),p('The task is binary classification: label 1 indicates bankruptcy during the three-year horizon and label 0 indicates no bankruptcy in that horizon. Because positive cases are rare, high accuracy alone is insufficient. The objective is to compare ranking quality and evaluate a threshold that places greater emphasis on recall.'),
          h('2. Dataset and data preparation'),p('The project uses UCI\'s <b>3year.arff</b>, containing 10,503 financial statements, 64 financial ratios and 495 positive labels [1]. It removes 87 exact duplicate rows before splitting, leaving <b>10,416 statements and 493 positives (4.73%)</b>. No identical feature vectors with conflicting labels were found. Missing values remain available to the model pipeline.'),
          p('For logistic regression, naive Bayes and SVM, preprocessing uses median imputation with missingness indicators, an inverse-hyperbolic-sine transform and standard scaling. Forest models use imputation and indicators. Histogram gradient boosting handles missing values natively. Each fitted transformation is learned inside its training fold.'),
          h('3. Experimental design'),p('A stratified split with seed 42 reserves <b>2,084 rows (99 positives)</b> for final testing. The remaining <b>8,332 rows (394 positives)</b> are evaluated with identical three-fold stratified splits. Each algorithm uses one fixed configuration. The highest mean cross-validation average precision selects the model. The selected model\'s out-of-fold scores determine a threshold that maximizes F2; neither choice uses test labels [2].'),
          h('4. Cross-validation comparison')]
    rows=[['Model','Mean AP','Fold SD','Mean ROC AUC']]
    for row in R['cv']:
        rows.append([row['model'],f"{row['cv_ap_mean']:.3f}",f"{row['cv_ap_std']:.3f}",f"{row['cv_roc_auc_mean']:.3f}"])
    s.append(table(rows,[84*mm,27*mm,27*mm,36*mm]))
    s += [p('AP = average precision. Fold standard deviation describes the three development folds; it is not a confidence interval. Histogram boosting uses 250 iterations, learning rate 0.07, up to 15 leaves, L2 regularization 1, and balanced class weights.',SMALL),PageBreak(),h('5. Independent test results'),
          p(f'The chosen threshold is <b>{M["threshold"]:.4f}</b>. After fitting on the development set, the frozen model achieves <b>{M["recall"]:.1%} recall, {M["precision"]:.1%} precision, {M["average_precision"]:.3f} AP and {M["roc_auc"]:.3f} ROC AUC</b>. Its confusion matrix is TN=1,824, FP=161, FN=15, TP=84. The all-negative baseline has 95.25% accuracy and zero recall.'),
          table([['Threshold rule','Recall','Precision','F2','FP / FN'],[f'OOF F2 choice ({M["threshold"]:.3f})',f'{M["recall"]:.1%}',f'{M["precision"]:.1%}',f'{M["f2"]:.3f}','161 / 15'],['Default 0.500',f'{D["recall"]:.1%}',f'{D["precision"]:.1%}',f'{D["f2"]:.3f}','36 / 31']],[61*mm,28*mm,28*mm,23*mm,34*mm]),
          Spacer(1,3*mm),Image(str(ROOT/'reports/figures/pr_roc.png'),width=174*mm,height=69.6*mm),
          p('Figure 1. Held-out precision-recall and ROC curves. The dashed precision baseline is test prevalence. The chosen threshold gains 16 detections and 125 false alarms relative to 0.5. Default 0.5 has higher test F2 in this split; the original threshold is retained to avoid selection on the test set.',SMALL),
          p('Using 500 stratified bootstrap resamples of test predictions, approximate 95% intervals are 77.8-91.9% for recall, 31.1-38.7% for precision, and 0.683-0.820 for AP. These intervals condition on the fixed model and split and do not include retraining or distribution-shift uncertainty.'),
          h('6. Implementation, interpretation and conclusion'),p('The repository provides modular training and inference, a saved model and checksum, exact split and prediction records, an executed notebook, automated tests, and a local Flask demo with CSV input. The demo includes correctly classified examples and real errors. Permutation importance identifies X27 (operating profit / financial expenses), X34 (operating expenses / total liabilities), and X46 (liquidity ratio) as influential, without establishing causality.'),
          p('The experiment supports a useful ranking signal in this dataset, while 161 false alarms and 15 missed cases remain. Scores are uncalibrated. Historical Polish data may not transfer to new settings; missing company IDs prevent verified company-disjoint validation, and a random split is not a temporal forecasting test. Future work should add group/date information, assess threshold stability, and validate calibration on independent data.'),
          h('References'),p('[1] S. Tomczak, <i>Polish Companies Bankruptcy</i>, UCI, 2016. doi:10.24432/C5F600. CC BY 4.0.<br/>[2] Scikit-learn documentation: model evaluation and cross-validation, scikit-learn.org.<br/>[3] N. A. Patil, <i>Corporate Bankruptcy Prediction</i>, supplied reference report; codeatri/Bankruptcy_Prediction, GitHub. Background only; this implementation and reported experiment are independent.',SMALL)]
    build('writeup_two_pages.pdf',s)

def one_page():
    s=title('Concise submission summary')
    s += [h('Problem and dataset'),p('Predict bankruptcy within three years from 64 financial ratios. The UCI Polish Companies Bankruptcy third-year file contains 10,503 statements and 495 positive labels. Removing 87 exact duplicates leaves 10,416 statements and 493 positives (4.73%). Missing feature values are retained for model-specific handling.'),
          h('Method and implementation'),p('A stratified 80/20 split (seed 42) reserves 2,084 test statements, including 99 positives. Logistic regression, Gaussian naive Bayes, RBF SVM, random forest, extra trees, and histogram gradient boosting are compared using identical three-fold development splits. Learned preprocessing stays inside each training fold. Mean cross-validation average precision selects the model; development out-of-fold F2 selects the decision threshold.'),
          p('Histogram gradient boosting leads with mean development AP 0.619 (fold SD 0.037). The selected configuration uses 250 iterations, learning rate 0.07, 15 leaves, L2 regularization 1, balanced class weights and native missing-value handling. The frozen threshold is 0.1980.'),
          h('Verified test results'),table([['Measure','Selected model','Always non-bankrupt'],['Recall','84.8% (84 / 99)','0.0%'],['Precision','34.3%','0.0%'],['Average precision / ROC AUC','0.750 / 0.968','0.048 / 0.500'],['Accuracy','91.55%','95.25%'],['False positives / false negatives','161 / 15','0 / 99']],[68*mm,53*mm,53*mm]),
          Spacer(1,2*mm),p('The default 0.5 threshold gives 68.7% recall and 65.4% precision. Lowering it to the development-selected value gains 16 detections and 125 false alarms. Although 0.5 has higher test F2, the selected threshold remains fixed to preserve the evaluation protocol.'),
          h('Conclusion and limitations'),p('The model provides a useful ranking signal on this historical sample, with substantial false alarms. Scores are uncalibrated. Company IDs are unavailable, so exact duplicate removal cannot ensure company-level independence. A random holdout does not establish performance in later years or other countries. Future work should test group/temporal splits and threshold stability.'),
          p('Deliverables include reproducible source code, an executed notebook, saved model, split and prediction evidence, evaluation figures, automated tests, a CSV-capable local demo, and presentation materials.'),
          h('Sources'),p('Tomczak (2016), UCI Polish Companies Bankruptcy, doi:10.24432/C5F600 (CC BY 4.0). Scikit-learn evaluation and cross-validation documentation. Background: supplied Patil report and github.com/codeatri/Bankruptcy_Prediction. All results above come from this independent implementation.',SMALL)]
    build('writeup_one_page.pdf',s)

if __name__=='__main__':
    two_pages();one_page();print('Write-ups generated in deliverables/.')
