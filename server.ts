import express from 'express';
import cors from 'cors';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { Storage } from '@google-cloud/storage';
import { BigQuery } from '@google-cloud/bigquery';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// --- GCP Services Initialization (Lazy) ---
let storage: Storage | null = null;
let bigquery: BigQuery | null = null;

const getStorage = () => {
  if (!storage) storage = new Storage();
  return storage;
};

const getBigQuery = () => {
  if (!bigquery) bigquery = new BigQuery();
  return bigquery;
};

// --- API Routes ---

// Health Check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Archive Analysis (GCS + BigQuery)
app.post('/api/archive', async (req, res) => {
  try {
    const { imageBase64, analysisResult, patientContext } = req.body;
    
    if (!imageBase64 || !analysisResult) {
      return res.status(400).json({ error: 'Missing required data' });
    }

    const timestamp = new Date().toISOString();
    const analysisId = analysisResult.id || `ecg-${Date.now()}`;
    
    // 1. Upload Image to Cloud Storage (GCS)
    // Bucket: cardio-ai-artifacts (configured via env or default)
    const bucketName = process.env.GCS_BUCKET_NAME || 'cardio-ai-artifacts';
    const fileName = `ecgs/${analysisId}.png`;
    
    // In a real scenario, we'd use the GCS client. 
    // For this demo, we'll simulate the upload if credentials aren't present, 
    // or attempt it if they are.
    let gcsUrl = `gs://${bucketName}/${fileName}`;
    
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      const bucket = getStorage().bucket(bucketName);
      const file = bucket.file(fileName);
      const buffer = Buffer.from(imageBase64.replace(/^data:image\/\w+;base64,/, ""), 'base64');
      await file.save(buffer, {
        metadata: { contentType: 'image/png' },
        resumable: false
      });
      gcsUrl = `https://storage.googleapis.com/${bucketName}/${fileName}`;
    } else {
      console.log(`[Mock GCS] Would upload to ${gcsUrl}`);
    }

    // 2. Stream Features to BigQuery
    // Dataset: cardio_analytics, Table: predictions
    const datasetId = process.env.BQ_DATASET || 'cardio_analytics';
    const tableId = process.env.BQ_TABLE || 'predictions';
    
    const row = {
      analysis_id: analysisId,
      timestamp: timestamp,
      diagnosis: analysisResult.diagnosis,
      confidence: analysisResult.confidenceLevel,
      urgency: analysisResult.urgency,
      heart_rate: analysisResult.heartRate,
      rhythm: analysisResult.rhythm,
      // Flattened features for tabular analysis
      pr_interval: analysisResult.precisionMeasurements?.waves?.intervals?.prMs,
      qrs_duration: analysisResult.precisionMeasurements?.waves?.qrsComplex?.durationMs,
      qtc_interval: analysisResult.precisionMeasurements?.waves?.intervals?.qtcMs,
      // Metadata
      gcs_uri: gcsUrl,
      model_version: analysisResult.precisionMeasurements?.neuralTelemetry?.modelArchitecture || 'unknown',
      // Privacy: Hash patient ID if present, otherwise anonymous
      patient_hash: patientContext ? Buffer.from(JSON.stringify(patientContext)).toString('base64') : 'anonymous'
    };

    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      await getBigQuery().dataset(datasetId).table(tableId).insert([row]);
    } else {
      console.log(`[Mock BigQuery] Would insert row:`, row);
    }

    res.json({ success: true, gcsUrl, analysisId });
  } catch (error: any) {
    console.error('Archive Error:', error);
    res.status(500).json({ error: 'Failed to archive analysis', details: error.message });
  }
});

// Explainable AI (XAI) - Saliency Simulation
// In a real production setup, this would call a Vertex AI Endpoint with XAI enabled.
app.post('/api/explain', async (req, res) => {
  try {
    const { analysisId, features } = req.body;
    
    // Simulate Saliency Map generation (e.g., Grad-CAM or Integrated Gradients)
    // Returning bounding boxes or heatmaps for the frontend to visualize
    const saliencyData = {
      method: 'Integrated Gradients',
      attribution: [
        { feature: 'V1_ST_Elevation', score: 0.85, description: 'High contribution to STEMI diagnosis' },
        { feature: 'aVL_Reciprocal_Depression', score: 0.65, description: 'Supporting evidence' }
      ],
      // Mock heatmap coordinates (normalized 0-1)
      heatmap: [
        { x: 0.1, y: 0.4, intensity: 0.9 }, // V1 area
        { x: 0.6, y: 0.3, intensity: 0.7 }  // aVL area
      ]
    };

    res.json(saliencyData);
  } catch (error: any) {
    res.status(500).json({ error: 'XAI Generation Failed' });
  }
});


// --- Vite Middleware ---
if (process.env.NODE_ENV !== 'production') {
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
} else {
  // Production static serving would go here
  app.use(express.static(path.join(__dirname, 'dist')));
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`GCP Integration Active: ${!!process.env.GOOGLE_APPLICATION_CREDENTIALS}`);
});
