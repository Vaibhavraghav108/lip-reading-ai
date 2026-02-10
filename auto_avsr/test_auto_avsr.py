"""
Simple Auto-AVSR Inference Script
Test visual speech recognition on any video file
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
import argparse

from lightning import ModelModule
from datamodule.transforms import VideoTransform

class SimpleVSRPipeline:
    """Simplified VSR pipeline for testing"""
    
    def __init__(self, model_path, detector="mediapipe"):
        print("🔧 Initializing VSR pipeline...")
        
        # Setup args
        parser = argparse.ArgumentParser()
        self.args, _ = parser.parse_known_args(args=[])
        setattr(self.args, 'modality', 'video')
        
        # Initialize detector
        print(f"📷 Loading {detector} detector...")
        if detector == "mediapipe":
            try:
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
                print("✅ MediaPipe detector loaded")
            except Exception as e:
                print(f"❌ Failed to load MediaPipe: {e}")
                print("💡 Try: pip install mediapipe")
                raise
        elif detector == "retinaface":
            try:
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                print(f"🖥️  Using device: {device}")
                self.landmarks_detector = LandmarksDetector(device=device)
                self.video_process = VideoProcess(convert_gray=False)
                print("✅ RetinaFace detector loaded")
            except Exception as e:
                print(f"❌ Failed to load RetinaFace: {e}")
                print("💡 Try: pip install retinaface-pytorch")
                raise
        
        self.video_transform = VideoTransform(subset="test")
        
        # Load model
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("\n📥 Downloading pre-trained model...")
            self.download_model(model_path)
        
        print(f"🧠 Loading model from {model_path}...")
        try:
            ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.modelmodule = ModelModule(self.args)
            self.modelmodule.model.load_state_dict(ckpt)
            self.modelmodule.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def download_model(self, save_path):
        """Download pre-trained VSR model"""
        import gdown
        
        # Google Drive file ID for vsr_trlrs3_base.pth
        file_id = "12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"Downloading from Google Drive...")
        print("⏳ This may take a few minutes (~500MB)...")
        
        try:
            gdown.download(url, save_path, quiet=False)
            print(f"✅ Model downloaded to {save_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n💡 Manual download:")
            print(f"   1. Visit: https://drive.google.com/file/d/{file_id}/view")
            print(f"   2. Download and save to: {save_path}")
            raise
    
    def load_video(self, video_path):
        """Load video file"""
        print(f"📹 Loading video: {video_path}")
        video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        return video.numpy()
    
    def predict(self, video_path):
        """Run prediction on a video file"""
        video_path = os.path.abspath(video_path)
        
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\n{'='*60}")
        print(f"🎬 Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Load and process video
        video = self.load_video(video_path)
        print(f"📊 Video shape: {video.shape} (frames, height, width, channels)")
        
        # Detect landmarks
        print("🔍 Detecting facial landmarks...")
        landmarks = self.landmarks_detector(video)
        
        # Process video
        print("⚙️  Processing video...")
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        
        # Run inference
        print("🧠 Running inference...")
        with torch.no_grad():
            transcript = self.modelmodule(video)
        
        print(f"\n{'='*60}")
        print(f"💬 PREDICTION:")
        print(f"{'='*60}")
        print(f"{transcript}")
        print(f"{'='*60}\n")
        
        return transcript


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("🎯 Auto-AVSR Visual Speech Recognition Test")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        print("❌ Error: No video file specified")
        print("\n📖 Usage:")
        print("   python test_auto_avsr.py <path_to_video>")
        print("\n📝 Examples:")
        print('   python test_auto_avsr.py "..\\LipNet\\data\\s1\\bbaf2n.mpg"')
        print('   python test_auto_avsr.py "..\\LipNet\\data\\s1\\test.mp4"')
        print()
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = "vsr_trlrs3_base.pth"
    
    try:
        # Initialize pipeline (using mediapipe as it's easier to install)
        pipeline = SimpleVSRPipeline(model_path, detector="mediapipe")
        
        # Run prediction
        result = pipeline.predict(video_path)
        
        print("✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\n🐛 Debugging tips:")
        print("  1. Make sure virtual environment is activated:")
        print("     .\\venv_avsr\\Scripts\\Activate.ps1")
        print("  2. Check all dependencies installed:")
        print("     pip install torch torchvision pytorch-lightning sentencepiece av opencv-python mediapipe")
        print("  3. Verify video file exists and is accessible")
        print("  4. Try a different video format (.mp4 recommended)")
        sys.exit(1)


if __name__ == "__main__":
    main()
