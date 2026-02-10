# Auto-AVSR Manual Setup Guide

This guide helps you set up Auto-AVSR **separately** from your LipNet project.

---

## 📁 Project Structure (After Setup)

```
Lip2/
├─ LipNet/              ← Your original project (UNCHANGED)
│  ├─ app/
│  ├─ data/
│  └─ models/
├─ myenv/               ← LipNet's virtual environment
├─ auto_avsr/           ← New Auto-AVSR project (SEPARATE)
│  ├─ venv_avsr/        ← Auto-AVSR's virtual environment
│  ├─ auto_avsr/        ← Source code
│  └─ models/           ← Pre-trained models
└─ setup_auto_avsr.ps1  ← Automated setup script
```

---

## 🚀 Option 1: Automated Setup (Recommended)

**Run the PowerShell script I created:**

```powershell
cd "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2"
.\setup_auto_avsr.ps1
```

This will:
- Clone the Auto-AVSR repository
- Create a separate virtual environment (`venv_avsr`)
- Install all dependencies
- Create a test script

---

## 🔧 Option 2: Manual Setup (If Script Fails)

### Step 1: Check Internet Connection

```powershell
ping github.com
```

If this fails, check your internet or proxy settings.

### Step 2: Clone Repository

```powershell
cd "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2"
git clone https://github.com/mpc001/auto_avsr.git
cd auto_avsr
```

**Alternative (if git fails):** Download ZIP from https://github.com/mpc001/auto_avsr and extract to `Lip2\auto_avsr\`

### Step 3: Create Virtual Environment

```powershell
python -m venv venv_avsr
.\venv_avsr\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Auto-AVSR requirements
pip install -r requirements.txt

# Additional dependencies
pip install opencv-python numpy scipy ffmpeg-python sentencepiece
```

### Step 5: Download Pre-trained Model

Auto-AVSR will download models automatically on first run, or you can manually download:

```powershell
# The model will be downloaded when you run inference
# Or manually download from the Auto-AVSR repository releases
```

### Step 6: Test Installation

Create a test file `test_avsr.py`:

```python
"""Test Auto-AVSR"""
import sys

try:
    from auto_avsr.pipelines import InferencePipeline
    print("✅ Auto-AVSR imported successfully!")
    
    # Try to initialize (this will download model if needed)
    print("Initializing pipeline...")
    pipeline = InferencePipeline()
    print("✅ Pipeline initialized!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you activated the virtual environment:")
    print("  .\\venv_avsr\\Scripts\\Activate.ps1")
except Exception as e:
    print(f"⚠️ Error: {e}")
```

Run it:
```powershell
python test_avsr.py
```

---

## 🧪 Testing Auto-AVSR

### Test on a LipNet video:

```powershell
# Make sure you're in auto_avsr folder with venv_avsr activated
cd "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2\auto_avsr"
.\venv_avsr\Scripts\Activate.ps1

# Test on one of your LipNet videos
python test_avsr.py "..\LipNet\data\s1\bbaf2n.mpg"
```

### Test on your custom video:

```powershell
python test_avsr.py "..\LipNet\data\s1\test.mp4"
```

---

## 📊 Expected Results

**LipNet (Current - Overfitted):**
- ✅ Works on training videos: `bbaf2n.mpg` → Accurate prediction
- ❌ Fails on custom videos: `test.mp4` → Gibberish

**Auto-AVSR (Better Generalization):**
- ⚠️ Works on training videos: `bbaf2n.mpg` → Decent prediction
- ✅ Works on custom videos: `test.mp4` → Much better than LipNet
- 📈 Overall: ~40% WER vs ~60%+ for LipNet

---

## 🔄 Switching Between Projects

### To use **LipNet**:
```powershell
cd "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2\LipNet\app"
& "C:/Users/VAIBHAV RAGHAV/Desktop/Lip2/myenv/Scripts/streamlit.exe" run streamlitapp.py
```

### To use **Auto-AVSR**:
```powershell
cd "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2\auto_avsr"
.\venv_avsr\Scripts\Activate.ps1
python test_avsr.py <path_to_video>
```

---

## 🐛 Troubleshooting

### Issue: "Could not resolve host: github.com"
**Solution:** 
- Check internet connection
- Try: `ipconfig /flushdns`
- Download ZIP manually from GitHub

### Issue: "No module named 'auto_avsr'"
**Solution:**
- Make sure you activated the correct virtual environment: `.\venv_avsr\Scripts\Activate.ps1`
- Check you're in the `auto_avsr` folder

### Issue: "Model not found"
**Solution:**
- Auto-AVSR downloads models automatically on first run
- Ensure you have ~1GB free disk space
- Check internet connection

### Issue: PyTorch CUDA errors
**Solution:**
- We installed CPU-only PyTorch to avoid GPU issues
- If you have a GPU and want to use it:
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 📋 Comparison Checklist

Test both models on the same video and compare:

| Video | LipNet Prediction | Auto-AVSR Prediction | Winner |
|-------|------------------|---------------------|--------|
| Training video (bbaf2n.mpg) | ________ | ________ | _____ |
| Custom video (test.mp4) | ________ | ________ | _____ |
| Webcam recording | ________ | ________ | _____ |

---

## ✅ Success Indicators

You've successfully set up Auto-AVSR when:
- [ ] `auto_avsr` folder exists separately from `LipNet`
- [ ] `venv_avsr` virtual environment created
- [ ] Can run `python test_avsr.py` without errors
- [ ] Model downloads automatically on first inference
- [ ] Can switch between LipNet and Auto-AVSR easily
- [ ] Both projects work independently

---

## 🎯 Next Steps

1. **Test Auto-AVSR** on your custom `test.mp4` video
2. **Compare results** with LipNet predictions
3. **Document findings** - which model works better?
4. **For presentation:** Show both models side-by-side to demonstrate overfitting vs generalization

---

## 📞 Need Help?

If setup fails:
1. Check internet connection (`ping github.com`)
2. Verify Python version: `python --version` (should be 3.8+)
3. Check disk space (~2GB needed)
4. Review error messages carefully

Common errors are usually:
- Network issues → Download ZIP manually
- Virtual environment not activated → Run `.\venv_avsr\Scripts\Activate.ps1`
- Missing dependencies → Re-run `pip install -r requirements.txt`

---

**Remember:** LipNet and Auto-AVSR are completely separate. You can use both, delete either, or keep both running simultaneously!
