import streamlit as st
import os
import sys
import torch
import argparse
from pathlib import Path

# Add auto_avsr to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess
from lightning import ModelModule
from datamodule.transforms import VideoTransform
import torchvision
import gdown

# Set page config
st.set_page_config(
    page_title="Auto-AVSR Lip Reading",
    page_icon="🎯",
    layout="wide"
)

class AutoAVSRPipeline:
    def __init__(self, model_path="vsr_trlrs3_base.pth"):
        self.model_path = model_path
        self.landmarks_detector = None
        self.video_process = None
        self.modelmodule = None
        self.video_transform = VideoTransform(subset="test")
        
        # Setup args for model
        parser = argparse.ArgumentParser()
        self.args, _ = parser.parse_known_args(args=[])
        setattr(self.args, 'modality', 'video')
        
    def download_model(self):
        """Download model from Google Drive if not exists"""
        if not os.path.exists(self.model_path):
            st.info("📥 Downloading Auto-AVSR model (1GB)... This may take a few minutes.")
            file_id = "12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                gdown.download(url, self.model_path, quiet=False)
                progress_bar.progress(100)
                status_text.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                return False
        return True
    
    @st.cache_resource
    def load_detector(_self):
        """Load MediaPipe face detector and video processor"""
        detector = LandmarksDetector()
        video_proc = VideoProcess(convert_gray=False)
        return detector, video_proc
    
    @st.cache_resource
    def load_model(_self, model_path, _args):
        """Load Auto-AVSR model"""
        if not os.path.exists(model_path):
            return None
        try:
            ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
            modelmodule = ModelModule(_args)
            modelmodule.model.load_state_dict(ckpt)
            modelmodule.eval()
            return modelmodule
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    def predict(self, video_path):
        """Run prediction on video"""
        try:
            # Load detector and video processor
            if self.landmarks_detector is None or self.video_process is None:
                with st.spinner("📷 Loading MediaPipe detector..."):
                    self.landmarks_detector, self.video_process = self.load_detector()
            
            # Load model
            if self.modelmodule is None:
                if not self.download_model():
                    return None
                with st.spinner("🧠 Loading Auto-AVSR model..."):
                    self.modelmodule = self.load_model(self.model_path, self.args)
                    if self.modelmodule is None:
                        st.error("❌ Failed to load model")
                        return None
            
            # Load video
            with st.spinner(f"📹 Loading video..."):
                video_path_str = str(Path(video_path).resolve())
                if not os.path.exists(video_path_str):
                    st.error(f"❌ Video not found: {video_path_str}")
                    return None
                
                video, _, _ = torchvision.io.read_video(video_path_str, pts_unit="sec")
                video = video.numpy()
            
            # Detect landmarks
            with st.spinner("🔍 Detecting facial landmarks..."):
                landmarks = self.landmarks_detector(video)
                if landmarks is None:
                    st.error("❌ No face detected in video")
                    return None
            
            # Process video
            with st.spinner("⚙️ Processing video..."):
                video = self.video_process(video, landmarks)
                video = torch.tensor(video)
                video = video.permute((0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
                video = self.video_transform(video)
            
            # Run inference
            with st.spinner("🧠 Running inference..."):
                with torch.no_grad():
                    transcript = self.modelmodule(video)
            
            return transcript
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return AutoAVSRPipeline()

def get_video_files(directory):
    """Get all video files from directory"""
    video_extensions = ('.mpg', '.mp4', '.avi', '.mov')
    video_files = []
    
    if os.path.exists(directory):
        for file in sorted(os.listdir(directory)):
            if file.lower().endswith(video_extensions):
                video_files.append(file)
    
    return video_files

def main():
    # Header
    st.title("🎯 Auto-AVSR Visual Speech Recognition")
    st.markdown("### Transformer-based Lip Reading with Auto-AVSR")
    st.markdown("---")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Auto-AVSR** is a state-of-the-art visual speech recognition model:
        
        - 🧠 **Architecture:** Transformer-based
        - 📚 **Training Data:** LRS3 (438+ hours)
        - 🎯 **Advantage:** Generalizes to unseen videos
        - 📊 **Performance:** ~36% WER on LRS3
        
        **vs LipNet:**
        - LipNet: 34h GRID (overfitted)
        - Auto-AVSR: 438h LRS3 (generalized)
        """)
        
        st.markdown("---")
        st.markdown("**Model Details:**")
        st.info("Model: `vsr_trlrs3_base.pth` (1GB)\n\nAutomatically downloaded on first use.")
    
    # Video directory
    video_dir = os.path.join("..", "LipNet", "data", "s1")
    
    # Get video files
    video_files = get_video_files(video_dir)
    
    if not video_files:
        st.error(f"❌ No video files found in: {video_dir}")
        st.info("Please add videos to the directory.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Select Video")
        
        # Video dropdown
        selected_video = st.selectbox(
            "Choose a video file:",
            video_files,
            help="Select from training videos (.mpg) or custom videos (.mp4)"
        )
        
        if selected_video:
            video_path = os.path.join(video_dir, selected_video)
            
            # Display video info
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / 1024  # KB
                st.success(f"✅ Selected: `{selected_video}` ({file_size:.1f} KB)")
                
                # Show video type
                if selected_video.endswith('.mpg'):
                    st.info("📊 **Training Video** - From GRID dataset")
                else:
                    st.info("🎬 **Custom Video** - External source")
            
            # Display video
            try:
                st.video(video_path)
            except Exception as e:
                st.warning(f"⚠️ Could not display video preview: {e}")
        
        # Predict button
        if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
            if selected_video:
                video_path = os.path.join(video_dir, selected_video)
                
                # Run prediction
                pipeline = get_pipeline()
                prediction = pipeline.predict(video_path)
                
                if prediction:
                    # Store in session state
                    st.session_state.prediction = prediction
                    st.session_state.predicted_video = selected_video
    
    with col2:
        st.subheader("💬 Prediction Result")
        
        # Display prediction
        if 'prediction' in st.session_state and 'predicted_video' in st.session_state:
            st.success(f"**Video:** {st.session_state.predicted_video}")
            
            # Display prediction in a nice box
            st.markdown("""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin: 10px 0;
            ">
                <h3 style="color: #1f1f1f; margin: 0;">Predicted Text:</h3>
                <p style="font-size: 18px; color: #333; margin: 10px 0; font-weight: 500;">
                    {}</p>
            </div>
            """.format(st.session_state.prediction.upper()), unsafe_allow_html=True)
            
            # Show comparison with LipNet
            st.markdown("---")
            st.info("""
            **💡 Tip:** Compare this result with LipNet!
            
            - **Training videos (.mpg):** Both models should work
            - **Custom videos (.mp4):** Auto-AVSR works, LipNet fails
            
            This demonstrates the difference between overfitting and generalization.
            """)
        else:
            st.info("👆 Select a video and click 'Generate Prediction' to see results")
            
            # Show sample predictions
            st.markdown("---")
            st.markdown("**📊 Sample Predictions:**")
            st.code("""
Training Video (bbaf2n.mpg):
→ "BUT PEOPLE HAVE TO GO DOWN"

Custom Video (bro.mp4):
→ "ME ASK YOU A COMPUTER QUESTION..."

Custom Video (emma.mp4):
→ "OH BOY THAT IS THIS ONE..."
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎓 Built with Auto-AVSR | Transformer-based Visual Speech Recognition</p>
        <p>📚 Trained on LRS3 Dataset (438+ hours) | 🔬 State-of-the-art Lip Reading</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
