# Wav2Sleep Ablation Studies - Research Protocol

## 📋 **Systematic Ablation Design**

Following the research evaluation protocol:

### **1. Model Capacity Evaluation**
**Objective:** Analyze how model complexity affects performance  
**Method:** Vary hidden representation dimension (32, 64, and 128)  
**Analysis:** Performance vs complexity trade-offs

### **2. Regularization Effect Analysis**
**Objective:** Evaluate the effect of regularization  
**Method:** Test dropout rates of 0.1 and 0.3  
**Analysis:** Overfitting vs underfitting balance

### **3. Missing Modality Robustness Evaluation**
**Objective:** Compare performance across modality availability scenarios  
**Method:** Test three configurations:
- All modalities present (ECG + PPG + respiration)
- Only ECG and PPG available  
- Only ECG available  
**Analysis:** Graceful degradation and clinical deployment feasibility

### **4. Attention-Based Visualization (Extension)**
**Objective:** Analyze which physiological modalities the transformer attends to during different sleep stages  
**Method:** Extract and visualize attention weights from CLS-token transformer  
**Analysis:** Clinical interpretability and modality importance

---

## 📁 **Available Example Scripts**

| File | Ablations Covered | Dependencies | Runtime |
|------|-------------------|--------------|---------|
| `wav2sleep_quick_demo.py` | 1, 2, 3 | PyTorch only | 30 sec |
| `sleep_multiclass_wav2sleep_corrected.py` | 1, 2, 3, 4 | PyHealth | 5-10 min |
| `sleep_multiclass_wav2sleep.py` | 1, 2, 3, 4 + extras | PyHealth | 15-20 min |

---

## 🚀 **Running the Ablation Studies**

### **Method 1: Quick Demo (Recommended First Run)**
```bash
cd PyHealth
python examples/wav2sleep/wav2sleep_quick_demo.py
```

### **Method 2: Complete Research Protocol**
```bash
# Ensure PyHealth environment
python examples/wav2sleep/sleep_multiclass_wav2sleep_corrected.py
```

---

## 📊 **Expected Ablation Results**

### **1. Model Capacity Analysis Results:**
```
Hidden Dimension | Accuracy | F1-Score | Parameters | Analysis
32               | 0.xxx    | 0.xxx    | ~14K       | Optimal efficiency
64               | 0.xxx    | 0.xxx    | ~48K       | Balanced performance  
128              | 0.xxx    | 0.xxx    | ~178K      | Diminishing returns
```

### **2. Regularization Analysis Results:**
```
Dropout Rate | Accuracy | F1-Score | Effect
0.1          | 0.xxx    | 0.xxx    | Optimal regularization
0.3          | 0.xxx    | 0.xxx    | Over-regularization
```

### **3. Missing Modality Robustness Results:**
```
Configuration      | Accuracy | F1-Score | Performance Drop
ECG+PPG+Resp (All) | 0.xxx    | 0.xxx    | Baseline (0%)
ECG+PPG            | 0.xxx    | 0.xxx    | -X.X% from baseline  
ECG Only           | 0.xxx    | 0.xxx    | -X.X% from baseline
```

### **4. Attention Visualization Results:**
- Sleep stage-specific attention heatmaps
- Modality importance rankings per stage
- Clinical interpretation guidelines

---

## 🔧 **Setup Instructions**

### **Option A: Minimal Setup (Fastest)**
```bash
# Install only PyTorch
pip install torch matplotlib scikit-learn numpy
python examples/wav2sleep/wav2sleep_quick_demo.py
```

### **Option B: Full PyHealth Integration**
```bash
# Create clean environment  
python3 -m venv wav2sleep_ablation
source wav2sleep_ablation/bin/activate

# Install PyHealth in development mode
pip install -e . --no-deps
pip install torch matplotlib scikit-learn numpy pandas

# Run complete experiments
python examples/wav2sleep/sleep_multiclass_wav2sleep_corrected.py
```

---

## 🧪 **Research Protocol Validation**

### **Systematic Evaluation Checklist:**
- [ ] **Model capacity:** Hidden dimensions [32, 64, 128] tested
- [ ] **Regularization:** Dropout rates [0.1, 0.3] evaluated  
- [ ] **Missing modalities:** All/ECG+PPG/ECG-only compared
- [ ] **Statistical significance:** Performance differences quantified
- [ ] **Clinical relevance:** Missing modality scenarios realistic
- [ ] **Attention analysis:** Transformer interpretability demonstrated

### **Expected Research Outcomes:**
1. **Optimal model configuration** identified through systematic testing
2. **Missing modality robustness** quantified for clinical deployment
3. **Regularization effects** characterized for training stability
4. **Clinical interpretability** enhanced through attention visualization

---

## 📈 **Success Metrics**

### **Quantitative Measures:**
- Performance differences ≥ 2% considered significant
- Missing modality performance drop < 10% acceptable
- Model capacity sweet spot identification
- Regularization optimal point determination

### **Qualitative Measures:**  
- Clear performance trends across configurations
- Clinically interpretable attention patterns
- Robust behavior with missing physiological signals
- Systematic experimental design validation

The ablation studies follow rigorous research methodology to provide comprehensive model evaluation for the Wav2Sleep architecture.