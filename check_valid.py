import os

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

root_dir = 'D:\Research\VLM_project\dataset\QuickDraw\photo'   # folder cha

def check_valid(root_dir):
    for cls in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTS:
                print('path: {} - filename: {}'.format(cls_path, f))
                
def rename(root_dir):
    for cls in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            ext = os.path.splitext(f)[1].lower()
            if  ext not in IMAGE_EXTS and f.endswith(".php"):
                old_path = os.path.join(cls_path, f)
                new_path = os.path.join(cls_path, f.replace(".php", ".png"))
                os.rename(old_path, new_path)

check_valid(root_dir)
# rename(root_dir)