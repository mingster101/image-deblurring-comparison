import os
import shutil

src_root = "Motion_Deblurring/Datasets/GOPRO_Large"
dst_root = "Motion_Deblurring/Datasets/GOPRO_Large_flat"

for split in ["train", "test"]:

    src_dir = os.path.join(src_root, split)

    dst_blur = os.path.join(dst_root, split, "blur")
    dst_sharp = os.path.join(dst_root, split, "sharp")

    os.makedirs(dst_blur, exist_ok=True)
    os.makedirs(dst_sharp, exist_ok=True)

    for scene in os.listdir(src_dir):

        scene_path = os.path.join(src_dir, scene)

        blur_dir = os.path.join(scene_path, "blur")
        sharp_dir = os.path.join(scene_path, "sharp")

        if not os.path.exists(blur_dir):
            continue

        for img in os.listdir(blur_dir):

            shutil.copy(
                os.path.join(blur_dir, img),
                os.path.join(dst_blur, f"{scene}_{img}")
            )

            shutil.copy(
                os.path.join(sharp_dir, img),
                os.path.join(dst_sharp, f"{scene}_{img}")
            )