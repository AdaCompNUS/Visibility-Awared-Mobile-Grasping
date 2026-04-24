#!/usr/bin/env python

# FILE: gui_utils.py
# Utility module for GUI interactions

import cv2
import numpy as np


class ClickPointCollector:
    """
    Simple GUI for collecting click points from user.
    """

    def __init__(self):
        self.click_points = []
        self.click_labels = []  # 1 for positive, 0 for negative
        self.finished = False

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for collecting click points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click = positive point
            self.click_points.append([x, y])
            self.click_labels.append(1)
            print(f"Added positive point: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click = negative point
            self.click_points.append([x, y])
            self.click_labels.append(0)
            print(f"Added negative point: ({x}, {y})")

    def collect_points(self, image_rgb):
        """
        Show image and collect click points from user.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3)

        Returns:
            tuple: (points, labels) where points is list of [x,y] and labels is list of 0/1
        """
        # Reset state
        self.click_points = []
        self.click_labels = []
        self.finished = False

        print("Interactive point collection started:")
        print("- Left click: Add positive point (include in segment)")
        print("- Right click: Add negative point (exclude from segment)")
        print("- Press 's' to finish and segment")
        print("- Press 'r' to reset all points")
        print("- Press 'q' to quit without segmenting")

        # Convert RGB to BGR for OpenCV display
        display_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        window_name = "Click Points - Left: Positive, Right: Negative"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while not self.finished:
            # Create display image with points overlaid
            img_display = display_image.copy()

            # Draw points
            for i, (point, label) in enumerate(
                zip(self.click_points, self.click_labels)
            ):
                color = (
                    (0, 255, 0) if label == 1 else (0, 0, 255)
                )  # Green for positive, Red for negative
                cv2.circle(
                    img_display, tuple(point), 3, color, -1
                )  # Smaller filled circle
                cv2.circle(
                    img_display, tuple(point), 5, (255, 255, 255), 1
                )  # Smaller white outline
                # Add number label
                cv2.putText(
                    img_display,
                    str(i + 1),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )  # Smaller text

            # Add instruction text
            cv2.putText(
                img_display,
                f"Points: {len(self.click_points)} | Press 's' to segment",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, img_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                # Finish and return points
                if len(self.click_points) == 0:
                    print("No points selected! Please click at least one point.")
                    continue
                self.finished = True
                print(f"Finished with {len(self.click_points)} points")

            elif key == ord("r"):
                # Reset points
                self.click_points = []
                self.click_labels = []
                print("Reset all points")

            elif key == ord("q"):
                # Quit without segmenting
                cv2.destroyAllWindows()
                return None, None

        cv2.destroyAllWindows()
        return self.click_points.copy(), self.click_labels.copy()


class MaskVisualizer:
    """
    Utility for visualizing segmentation masks.
    """

    @staticmethod
    def show_mask_overlay(
        image_rgb, mask, window_name="Segmentation Result", wait_key=True
    ):
        """
        Show image with mask overlay.

        Args:
            image_rgb: Original RGB image
            mask: Binary mask (0s and 1s)
            window_name: Window title
            wait_key: Whether to wait for key press
        """
        # Convert RGB to BGR for display
        display_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Assert proper dimensions
        h, w = display_image.shape[:2]
        assert len(mask.shape) == 2, f"Mask must be 2D, got {mask.shape}"
        assert mask.shape == (
            h,
            w,
        ), f"Mask shape {mask.shape} must match image shape ({h}, {w})"

        # Create colored mask overlay
        mask_colored = np.zeros_like(display_image)
        mask_colored[mask > 0] = [0, 255, 255]  # Yellow for masked area

        # Blend original image with mask
        alpha = 0.6
        result = cv2.addWeighted(display_image, 1 - alpha, mask_colored, alpha, 0)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result)

        if wait_key:
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    @staticmethod
    def save_mask_visualization(image_rgb, mask, filename):
        """
        Save image with mask overlay to file.

        Args:
            image_rgb: Original RGB image
            mask: Binary mask
            filename: Output filename
        """
        # Convert RGB to BGR
        display_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Assert proper dimensions
        h, w = display_image.shape[:2]
        assert len(mask.shape) == 2, f"Mask must be 2D, got {mask.shape}"
        assert mask.shape == (
            h,
            w,
        ), f"Mask shape {mask.shape} must match image shape ({h}, {w})"

        # Create mask overlay
        mask_colored = np.zeros_like(display_image)
        mask_colored[mask > 0] = [0, 255, 255]  # Yellow

        # Blend and save
        alpha = 0.6
        result = cv2.addWeighted(display_image, 1 - alpha, mask_colored, alpha, 0)
        cv2.imwrite(filename, result)
        print(f"Mask visualization saved to: {filename}")
