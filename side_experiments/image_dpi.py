from PIL import Image
PAGE_PATH = "/Users/ishanghorpade/Documents/GitHub/handwriting_work/dataset/scans/session1/page1.png"
img = Image.open(PAGE_PATH)
print(img.info.get("dpi"))  # sometimes embedded in the file metadata
