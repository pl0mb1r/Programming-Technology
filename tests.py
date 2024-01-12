from CarDamageReporter import CarDamageReporter
from CarDamageDetector import CarDamageDetector

# Example usage:
model_path = 'path/to/your/model.h5'
image_path = 'path/to/your/car_image.jpg'

damage_detector = CarDamageDetector(model_path)
is_damaged, damage_probability = damage_detector.predict_damage(image_path)

damage_reporter = CarDamageReporter()
report = damage_reporter.generate_report(is_damaged, damage_probability)

print(report)
