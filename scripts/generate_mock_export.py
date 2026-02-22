import json
import time
import random

def generate_mock_export(filename="mock_export.jsonl", num_records=5):
    """Generates a mock Vertex AI Data Labeling export file."""
    
    diagnoses = ["Normal Sinus Rhythm", "Atrial Fibrillation", "Myocardial Infarction", "PVC"]
    reviewers = ["worker_A", "worker_B", "worker_C"]
    
    with open(filename, "w") as f:
        for i in range(num_records):
            record_id = f"RECORD_{1000+i}"
            
            # Simulate 3 reviews per record
            for reviewer in reviewers:
                # 80% chance of agreement (consensus)
                if random.random() > 0.2:
                    label = diagnoses[i % len(diagnoses)]
                else:
                    label = random.choice(diagnoses)
                
                entry = {
                    "imageGcsUri": f"gs://cardio-ai-raw-data/ecgs/{record_id}.png",
                    "labelAnnotations": [
                        {
                            "displayName": label,
                            "annotationSpecId": reviewer
                        }
                    ]
                }
                f.write(json.dumps(entry) + "\n")
                
    print(f"Generated {filename} with {num_records * 3} reviews.")

if __name__ == "__main__":
    generate_mock_export()
