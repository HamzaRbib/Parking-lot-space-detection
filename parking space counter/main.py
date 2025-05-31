import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not
import sys
import matplotlib.pyplot as plt

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# File paths
mask_path = r"data\mask_1920_1080.png"
video_path = r"data\parking_1920_1080.mp4"
output_path = r"parking_output.mp4"

# Load mask
mask = cv2.imread(mask_path, 0)
if mask is None:
    print("Error: Could not load mask image. Please check the path:", mask_path)
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file. Please check the path:", video_path)
    sys.exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video output settings - using H.264 codec for better performance
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Get parking spots
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
frame_nmr = 0
step = 30  # Check every 30 frames
display_step = 2  # Show every 2nd frame to reduce display overhead

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete.")
            break

        # Process spots status less frequently
        if frame_nmr % step == 0 and previous_frame is not None:
            # Pre-calculate all spot crops for better performance
            spot_crops = []
            prev_crops = []
            for spot in spots:
                x1, y1, w, h = spot
                spot_crops.append(frame[y1:y1 + h, x1:x1 + w])
                prev_crops.append(previous_frame[y1:y1 + h, x1:x1 + w])
            
            # Calculate differences in batch
            for spot_indx, (spot_crop, prev_crop) in enumerate(zip(spot_crops, prev_crops)):
                if spot_crop.shape == prev_crop.shape:
                    diffs[spot_indx] = calc_diff(spot_crop, prev_crop)

        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                max_diff = np.amax(diffs)
                arr_ = [j for j in np.argsort(diffs) if max_diff > 0 and diffs[j] / max_diff > 0.4]

            # Process spots in batch
            for spot_indx in arr_:
                x1, y1, w, h = spots[spot_indx]
                spot_crop = frame[y1:y1 + h, x1:x1 + w]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status

            previous_frame = frame.copy()

        # Draw spots and status
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spot
            color = (0, 255, 0) if spot_status else (0, 0, 255)
            label = "Empty" if spot_status else "Occupied"
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(frame, label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw availability counter
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
                    (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show frame less frequently to reduce display overhead
        if frame_nmr % display_step == 0:
            try:
                cv2.imshow("Parking Lot Status", frame)
            except cv2.error as e:
                print("Warning: Could not display frame. Continuing without display.")
                print("Error details:", str(e))

        # Write frame to output
        out.write(frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

        frame_nmr += 1

except Exception as e:
    print("An error occurred:", str(e))
finally:
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processed video saved to:", output_path)