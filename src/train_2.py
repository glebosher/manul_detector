from ultralytics import YOLO

# Путь к уже обученной модели из первого прогона
old_weights_path = r"C:\MyPythonProjects\runs\detect\manul_cpu_run\weights\best.pt"

# Путь к новому файлу data.yaml 
new_yaml_path = r"C:\MyPythonProjects\project_wildlife\manul.v2i.yolov8\data.yaml"


def main():
    # Загружаем как основу собственную модель
    model = YOLO(old_weights_path)

    # Продолжаем обучение на новом датасете
    model.train(
        data=new_yaml_path,
        epochs=30,         
        imgsz=416,        
        device='cpu',      
        name="manul_cpu_run_v2" 
    )


if __name__ == "__main__":
    main()
