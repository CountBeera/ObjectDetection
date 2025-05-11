# test_pycoco.py
from pycocotools.coco import COCO

try:
    annFile = r"C:\Intern\archive\coco2017\annotations\instances_val2017.json"  # Adjust path if needed
    coco = COCO(annFile)
    print("pycocotools is installed and working.")
except Exception as e:
    print("pycocotools test failed:", e)
