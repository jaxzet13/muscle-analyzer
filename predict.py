from cog import BasePredictor, Input, Path
from ultralytics import YOLO
import cv2
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        self.model = YOLO("best.pt")  # Replace with your model path

    def predict(self, image: Path) -> Path:
        img = cv2.imread(str(image))
        H, W, _ = img.shape

        results = self.model(img)

        for result in results:
            if result.masks is not None:
                for j, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy() * 255
                    mask_resized = cv2.resize(mask_np, (W, H))
                    mask_resized = mask_resized.astype(np.uint8)
                    output_path = f"output_mask_{j}.png"
                    cv2.imwrite(output_path, mask_resized)
                    return Path(output_path)

        return Path("no_mask.png")
