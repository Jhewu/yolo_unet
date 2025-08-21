from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo12x.pt")  # load an official model
    results = model(["BraTS-SSA-00041-0000-t1c.png", "BraTS-SSA-00041-0001-t1c.png", "BraTS-SSA-00041-0002-t1c.png", "BraTS-SSA-00041-0003-t1c.png", "BraTS-SSA-00041-0004-t1c.png"], 
                    device="cpu")  # predict on an image