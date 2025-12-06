"""临时脚本：使用 OpenCV 打开样例图像，方便观察比例尺与靶球细节。"""
from __future__ import annotations

from pathlib import Path

import cv2 as cv


def main() -> None:
    image_path = Path(__file__).resolve().parents[1] / "data_samples" / "10um_Default_0532_Mode2D.tiff"
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    cv.namedWindow("OCT image viewer", cv.WINDOW_NORMAL)
    cv.imshow("OCT image viewer", image)
    print("Inspect the image (use mouse to check scale/spot details). Press any key to close.")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
