from roboflow import Roboflow
rf = Roboflow(api_key="ncNHMLI9owV4BSC8MnHg")
project = rf.workspace().project("brain-tumor-detection-fzgsl")
model = project.version(1).model

print(model.predict("3.jpg",confidence=30,overlap=30).json())


model.predict("3.jpg",confidence=30,overlap=30).save("prediction3.jpg")