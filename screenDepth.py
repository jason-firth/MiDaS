import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA (GPU support) is not available. Exiting.")
    exit()

import cv2
import numpy as np 
from midas.model_loader import default_models, load_model
import mss

@torch.no_grad()
def process(device, model, image, target_size):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    return prediction

def create_side_by_side(image, depth):
    # Resize the depth map to match the height of the input image
    depth_resized = cv2.resize(depth, (image.shape[1], image.shape[0]))

    depth_min = depth_resized.min()
    depth_max = depth_resized.max()
    normalized_depth = 255 * (depth_resized - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side= np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    return np.concatenate((image, right_side), axis=1)

def run():
    optimize=False
    side=False
    height=None
    square=False
    grayscale=False
    model_type="dpt_beit_large_512"
    model_path = default_models[model_type]
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    frame_count = 0  # Initialize frame counter

    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust these values according to your screen resolution
        while True:
            # Capture every other frame to improve framerate
            frame = sct.grab(monitor)
            frame_count += 1
            if frame_count % 2 == 0:
                continue
            
            frame = np.array(frame)
            
            # Resize the input frame
            frame = cv2.resize(frame, (net_w, net_h))
            
            original_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            image = transform({"image": original_image_rgb/255})["image"]
            try:
                prediction = process(device, model, image, (net_w, net_h))
            except RuntimeError as e:
                print("Error during inference:", e)
                continue
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            content = create_side_by_side(original_image_bgr, prediction)
            
            # Resize the output window
            resized_content = cv2.resize(content, (content.shape[1] // 2, content.shape[0] // 2))
            
            cv2.imshow("Depth", resized_content/255)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
