# AutoAVSR - Automatic Audio-Visual Speech Recognition

This repository contains the Auto-AVSR (Automatic Audio-Visual Speech Recognition) model for lip reading with enhanced generalization capabilities.

---

## 📁 Project Structure

```
lip-reading-ai/
├─ auto_avsr/                    ← Main Auto-AVSR project
│  ├─ venv_avsr/                 ← Virtual environment
│  ├─ datamodule/                ← Data processing modules
│  ├─ espnet/                    ← Neural network components
│  ├─ preparation/               ← Data preparation tools
│  ├─ spm/                       ← SentencePiece models
│  ├─ tutorials/                 ← Jupyter notebooks
│  ├─ streamlit_app.py           ← Web interface
│  ├─ test_auto_avsr.py          ← Test script
│  └─ train.py                   ← Training script
├─ .gitignore                    ← Git ignore file
└─ README.md                     ← This file
```

---

## 🚀 Quick Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/Vaibhavraghav108/lip-reading-ai.git
cd lip-reading-ai/auto_avsr
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv_avsr
# On Windows:
.\venv_avsr\Scripts\Activate.ps1
# On Linux/Mac:
source venv_avsr/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Auto-AVSR requirements
pip install -r requirements.txt

# Additional dependencies
pip install opencv-python numpy scipy ffmpeg-python sentencepiece
```

### Step 4: Download Pre-trained Model

Auto-AVSR will download models automatically on first run, or you can manually download:

```bash
# The model will be downloaded when you run inference
# Or manually download from the Auto-AVSR repository releases
```

### Step 5: Test Installation

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
```bash
python test_avsr.py
```

---

## 🧪 Testing Auto-AVSR

### Test on a video file:

```bash
# Make sure you're in auto_avsr folder with venv_avsr activated
cd auto_avsr
# On Windows:
.\venv_avsr\Scripts\Activate.ps1
# On Linux/Mac:
source venv_avsr/bin/activate

# Test on a video file
python test_auto_avsr.py path/to/your/video.mp4
```

### Run Streamlit Web Interface:

```bash
streamlit run streamlit_app.py
```

---

## 🎯 Features

**Auto-AVSR Advantages:**
- ✅ Trained on 438 hours of data vs traditional models with ~34 hours
- ✅ Uses modern Transformer architecture vs older CNN+LSTM
- ✅ Achieves ~36% WER (Word Error Rate) on LRS3 benchmark
- ✅ Better generalization to unseen speakers and videos
- ✅ State-of-the-art visual speech recognition performance

**Technical Architecture:**
- **Visual Frontend:** ResNet-18 backbone with 3D Convolutional layers
- **Transformer Encoder:** 12 layers of multi-head self-attention
- **Transformer Decoder:** Autoregressive text generation with attention

---

## 📊 Model Performance

| Dataset | Word Error Rate (WER) | Notes |
|---------|----------------------|-------|
| LRS3 Test | ~36% | Benchmark dataset |
| Custom Videos | Varies | Depends on video quality |
| Training Videos | ~25% | Expected performance |

---

## 🐛 Troubleshooting

### Issue: "Could not resolve host: github.com"
**Solution:** 
- Check internet connection
- Try: `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (Mac)
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
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 🔧 Advanced Usage

### Training Custom Models

```bash
python train.py --config-path configs --config-name lrs3
```

### Data Preparation

```bash
cd preparation
python preprocess_lrs2lrs3.py --data-dir /path/to/dataset
```

### Feature Extraction

See the Jupyter notebooks in the `tutorials/` folder for detailed examples.

---

## 📋 Requirements

- Python 3.8+
- PyTorch (CPU or GPU version)
- OpenCV
- FFmpeg
- ~2GB free disk space (including model weights)

---

## ✅ Success Indicators

You've successfully set up Auto-AVSR when:
- [ ] `auto_avsr` folder exists with all dependencies
- [ ] `venv_avsr` virtual environment created
- [ ] Can run `python test_auto_avsr.py` without errors
- [ ] Model downloads automatically on first inference
- [ ] Streamlit web interface loads correctly

---

## 📞 Need Help?

If setup fails:
1. Check internet connection (`ping github.com`)
2. Verify Python version: `python --version` (should be 3.8+)
3. Check disk space (~2GB needed)
4. Review error messages carefully

Common errors are usually:
- Network issues → Download ZIP manually
- Virtual environment not activated → Run activation command
- Missing dependencies → Re-run `pip install -r requirements.txt`

---

## 📄 License

This project uses the Auto-AVSR model which is available under its original license. See the `LICENSE` file for more details.