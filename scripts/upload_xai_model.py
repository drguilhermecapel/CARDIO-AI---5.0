import argparse
from google.cloud import aiplatform

def upload_model_with_xai(
    project_id,
    location,
    display_name,
    model_artifact_uri,
    serving_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest"
):
    aiplatform.init(project=project_id, location=location)

    # Configure Explanation Spec (Integrated Gradients)
    # Suitable for differentiable models (CNN/Transformers)
    explanation_parameters = aiplatform.explain.ExplanationParameters(
        {"integrated_gradients_attribution": {"step_count": 50}}
    )

    # Metadata describes the input/output tensors for the explanation
    # Assuming the model signature input is 'input_1' and output is 'pathology'
    explanation_metadata = aiplatform.explain.ExplanationMetadata(
        inputs={
            "input_1": aiplatform.explain.ExplanationMetadata.InputMetadata(
                {"input_tensor_name": "input_1", "encoding": "IDENTITY", "modality": "image"} 
                # treating signal as 1D image for visualization tools
            )
        },
        outputs={
            "pathology": aiplatform.explain.ExplanationMetadata.OutputMetadata(
                {"output_tensor_name": "pathology", "encoding": "IDENTITY"}
            )
        },
    )

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=serving_image_uri,
        explanation_parameters=explanation_parameters,
        explanation_metadata=explanation_metadata,
    )

    print(f"Model uploaded with XAI: {model.resource_name}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--location', default='us-central1')
    parser.add_argument('--display_name', default='ecg-hybrid-xai')
    parser.add_argument('--model_uri', required=True, help='gs:// path to saved_model')
    
    args = parser.parse_args()
    upload_model_with_xai(
        args.project_id, args.location, args.display_name, args.model_uri
    )
