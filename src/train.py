from ultralytics import YOLO

dataset_path = r"C:\MyPythonProjects\project_wildlife\manul.v1-roboflow\data.yaml"


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=dataset_path,
        epochs=50,
        imgsz=416,
        device="cpu",
        workers=1,
        batch=4,
        name="manul_cpu_run",
    )


if __name__ == "__main__":
    main()
